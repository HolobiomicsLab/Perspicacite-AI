"""Multi-KB retrieval: fan a query across several KB collections, merge + dedup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from perspicacite.llm.embeddings import EmbeddingFailedError
from perspicacite.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger("perspicacite.retrieval.multi_kb")


def canonical_embedding_fingerprint(name: str | None) -> str | None:
    """Normalize an embedding-model fingerprint to its primary model name.

    Historic KBs stamped one of:
      - "text-embedding-3-small"                 (MCP / LiteLLMEmbeddingProvider)
      - "text-embedding-3-small|all-MiniLM-L6-v2" (web / FallbackEmbeddingProvider)
      - "text-embedding-3-small+code:foo"        (TypedEmbeddingProvider)

    All three share the same primary embedding model (whatever sits left
    of the first ``|`` or ``+``), and only the primary actually produces
    vectors in the common case. Comparing the raw strings makes the
    multi-KB compat check refuse otherwise-compatible KBs.
    """
    if not name:
        return None
    # Strip the fallback suffix ("primary|fallback") first, then the
    # typed-provider suffix ("primary+code:foo+..."). Both separators
    # behave additively so order doesn't matter beyond left-to-right.
    s = str(name)
    for sep in ("|", "+"):
        if sep in s:
            s = s.split(sep, 1)[0]
    return s.strip() or None


def check_embedding_compat(kb_metas: Sequence[Any]) -> str | None:
    """Return None if all KBs share an embedding model; else a message listing the models.

    Comparison happens on the canonical primary-model fingerprint, so two
    KBs stamped ``"text-embedding-3-small"`` and
    ``"text-embedding-3-small|all-MiniLM-L6-v2"`` (same primary, different
    fallback config) are correctly treated as compatible.
    """
    raw_models = [getattr(m, "embedding_model", None) for m in kb_metas]
    canon = {canonical_embedding_fingerprint(m) for m in raw_models}
    canon.discard(None)
    if len(canon) <= 1:
        return None
    return (
        "Cannot query these knowledge bases together: they use different embedding models "
        f"({', '.join(sorted(str(m) for m in raw_models if m))}). Re-embedding is not supported."
    )


class MultiKBRetriever:
    """Fan a query across multiple KB collections, merge results, and dedup by paper_id.

    Quacks like the DynamicKnowledgeBase slice that RAG modes use: `.search`,
    `.search_two_pass`, and tolerant `.collection_name` / `._initialized` attrs.
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        kb_metas: Sequence[Any],
        default_top_k: int = 10,
        default_min_score: float = 0.0,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.kb_metas = list(kb_metas)
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score
        # tolerate modes assigning these:
        self.collection_name = "+".join(getattr(m, "collection_name", "") for m in self.kb_metas)
        self._initialized = True

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.default_top_k
        min_score = self.default_min_score if min_score is None else min_score
        query_embeddings = await self.embedding_service.embed_query([query])
        query_embedding = query_embeddings[0]
        merged: dict[str, dict[str, Any]] = {}  # paper_id -> best dict
        orderless: list[dict[str, Any]] = []
        for meta in self.kb_metas:
            coll = getattr(meta, "collection_name", None)
            if not coll:
                continue
            try:
                results = await self.vector_store.search(
                    collection=coll,
                    query_embedding=query_embedding,
                    top_k=top_k * 2,
                )
            except EmbeddingFailedError:
                # A degenerate query fails against every collection, so skipping
                # them one by one would return an empty result reported as success.
                raise
            except Exception as e:
                logger.warning("multi_kb_search_collection_failed", collection=coll, error=str(e))
                continue
            kb_display = getattr(meta, "name", None) or coll
            for r in results:
                chunk = getattr(r, "chunk", None)
                meta_obj = getattr(chunk, "metadata", None) if chunk is not None else None
                pid = getattr(meta_obj, "paper_id", None)
                score = float(getattr(r, "score", 0.0) or 0.0)
                if score < min_score:
                    continue
                d = {
                    "text": getattr(chunk, "text", "") if chunk is not None else "",
                    "score": score,
                    "paper_id": pid,
                    "metadata": meta_obj,
                    "kb_name": kb_display,
                }
                if pid:
                    prev = merged.get(pid)
                    if prev is None or score > prev["score"]:
                        merged[pid] = d
                else:
                    orderless.append(d)
        combined = list(merged.values()) + orderless
        combined.sort(key=lambda x: x["score"], reverse=True)
        result = combined[:top_k]
        logger.info(
            "multi_kb_search",
            kbs=[getattr(m, "name", "?") for m in self.kb_metas],
            hits=len(result),
        )
        return result

    async def search_two_pass(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        *,
        max_papers_cap: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Multi-KB two-pass retrieval: chunk hits → identify top papers
        (across all KBs), then pull every chunk of those papers.

        Returns paper-level dicts matching the single-KB DynamicKnowledgeBase
        contract: ``paper_id, paper_score, title, authors, year, doi,
        kb_name, chunks, full_text``. This unblocks the multi-KB code path
        in BasicRAGMode (which previously returned "Unknown title, score 0.00"
        because the v1 delegate handed back chunk-shaped dicts).
        """
        top_k = top_k or self.default_top_k
        min_score = self.default_min_score if min_score is None else min_score
        hard_cap = min(max_papers_cap or top_k, 5)

        # ── Pass 1: fan chunk-level vector search across collections ─────
        chunk_hits = await self.search(query, top_k=top_k, min_score=min_score)
        if not chunk_hits:
            return []

        # ── Group by paper_id, keep best score per paper + winning kb_name ─
        paper_scores: dict[str, float] = {}
        paper_meta: dict[str, Any] = {}
        paper_kb: dict[str, str] = {}  # paper_id → kb_display name
        paper_collection: dict[str, str] = {}  # paper_id → chroma collection
        # Map kb display name → collection name for the pass-2 fetch.
        kb_to_collection: dict[str, str] = {}
        for meta in self.kb_metas:
            kb_to_collection[getattr(meta, "name", "")] = getattr(meta, "collection_name", "")

        for hit in chunk_hits:
            pid = hit.get("paper_id")
            score = float(hit.get("score") or 0.0)
            kb_disp = hit.get("kb_name") or ""
            if not pid:
                continue
            if pid not in paper_scores or score > paper_scores[pid]:
                paper_scores[pid] = score
                paper_meta[pid] = hit.get("metadata")
                paper_kb[pid] = kb_disp
                paper_collection[pid] = kb_to_collection.get(kb_disp, "")

        if not paper_scores:
            return []

        # Take the top ``hard_cap`` papers across all KBs.
        candidate_order = sorted(
            paper_scores.keys(), key=lambda p: -paper_scores[p]
        )[:hard_cap]

        # ── Pass 2: fetch all chunks for each top paper from its own collection ─
        results: list[dict[str, Any]] = []
        for pid in candidate_order:
            coll = paper_collection.get(pid)
            if not coll:
                continue
            try:
                paper_chunks = await self.vector_store.get_chunks_by_paper_ids(coll, [pid])
            except Exception as e:
                logger.warning("multi_kb_pass2_failed", paper_id=pid, error=str(e))
                paper_chunks = []
            # Dedup overlap (chunks of one paper share overlap-window text).
            from perspicacite.rag.utils import deduplicate_chunk_overlaps
            deduped = deduplicate_chunk_overlaps(paper_chunks, overlap_words=50)

            meta = paper_meta.get(pid)
            full_text = " ".join(d.get("text", "") for d in deduped)
            results.append({
                "paper_id": pid,
                "paper_score": paper_scores[pid],
                "title": getattr(meta, "title", None) if meta else None,
                "authors": getattr(meta, "authors", None) if meta else None,
                "year": getattr(meta, "year", None) if meta else None,
                "doi": getattr(meta, "doi", None) if meta else None,
                "kb_name": paper_kb.get(pid),
                "chunks": deduped,
                "full_text": full_text,
            })

        logger.info(
            "multi_kb_search_two_pass",
            kbs=[getattr(m, "name", "?") for m in self.kb_metas],
            papers=len(results),
        )
        return results


async def query_chunks_across_collections(
    *,
    vector_store: Any,
    embedding_service: Any,
    collection_names: list[str],
    query: str,
    top_k: int,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Fan vector_store.search across collections, merge by paper_id (best score),
    tag each hit with kb_name (= collection_name), sort, return top_k."""
    if not collection_names:
        return []
    query_embedding = (await embedding_service.embed_query([query]))[0]

    async def _one(coll: str) -> list[Any]:
        try:
            results: list[Any] = await vector_store.search(
                collection=coll, query_embedding=query_embedding, top_k=top_k * 2
            )
            return results
        except EmbeddingFailedError:
            # Degenerate query: every collection fails identically, and an empty
            # merged result would be indistinguishable from "nothing matched".
            raise
        except Exception as e:
            logger.warning("fanout_search_failed", collection=coll, error=str(e))
            return []

    per = await asyncio.gather(*(_one(c) for c in collection_names))
    merged: dict[str, dict[str, Any]] = {}
    orderless: list[dict[str, Any]] = []
    for coll, hits in zip(collection_names, per, strict=False):
        for r in hits:
            chunk = getattr(r, "chunk", None)
            meta = getattr(chunk, "metadata", None) if chunk is not None else None
            pid = getattr(meta, "paper_id", None)
            score = float(getattr(r, "score", 0.0) or 0.0)
            if score < min_score:
                continue
            d = {
                "text": getattr(chunk, "text", "") if chunk is not None else "",
                "score": score,
                "paper_id": pid,
                "metadata": meta,
                "kb_name": coll,
            }
            if pid:
                prev = merged.get(pid)
                if prev is None or score > prev["score"]:
                    merged[pid] = d
            else:
                orderless.append(d)
    combined = list(merged.values()) + orderless
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


async def get_chunks_by_paper_ids_across(
    vector_store: Any,
    *,
    collection_names: list[str],
    paper_ids: list[str],
) -> list[Any]:
    """Fan get_chunks_by_paper_ids across collections in parallel.
    Returns concatenated DocumentChunk list (caller dedups if needed)."""
    if not collection_names or not paper_ids:
        return []

    async def _one(coll: str) -> list[Any]:
        try:
            chunks: list[Any] = await vector_store.get_chunks_by_paper_ids(coll, paper_ids)
            return chunks
        except Exception as e:
            logger.warning("fanout_get_chunks_failed", collection=coll, error=str(e))
            return []

    per = await asyncio.gather(*(_one(c) for c in collection_names))
    out: list[Any] = []
    for chunks in per:
        out.extend(chunks)
    return out
