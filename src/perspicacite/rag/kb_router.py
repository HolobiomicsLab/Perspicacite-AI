"""Auto-routing: pick the most-relevant KBs for a query.

Until now, multi-KB retrieval required the caller to enumerate `kb_names`
explicitly. This module lets the user point at "all KBs" (or a curated
subset) and have Perspicacité decide which ones to actually query, based
on each KB's description plus a sample of paper titles.

Two routing methods, picked by the caller:

- ``"bm25"`` (default) — fast, no LLM call. Builds a BM25 corpus over
  per-KB "context strings" (description + top-K paper titles) and scores
  the query against it. Sub-second.
- ``"llm"`` — one LLM call asks a cheap model to read all KB context
  strings and return JSON with per-KB scores + one-sentence reasons.
  More accurate on semantic mismatches; costs one LLM call regardless
  of KB count.

The router never *queries* a KB itself; it returns a ranked list of
``KBRouteHit`` entries. The caller (chat router, MCP tool) then passes
the selected names to whichever retrieval flow it normally uses
(``MultiKBRetriever`` / ``query_chunks_across_collections``).

The "context string" each KB is scored against blends:
  - the KB's own ``description`` field (when present)
  - up to ``sample_papers`` paper titles read from its Chroma collection

so even KBs created with a generic description like "Imported from
references.bib" get scored on real content.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import bm25s

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.rag.kb_router")


# (KB-router-scope BM25 index cache. Keys: corpus fingerprint.)
# Rebuilding a Lucene-style BM25 index on every call is wasteful when the
# corpus (per-KB context strings) only changes when a KB is created or
# edited. We cache by a stable sha1 over the sorted (name, text) pairs so
# any in-place mutation invalidates automatically.
_BM25_CACHE: dict[str, tuple[bm25s.BM25, list[str], list[list[str]]]] = {}


def _corpus_fingerprint(kb_contexts: dict[str, str]) -> str:
    """Stable fingerprint of (name, text) pairs — invalidates cache on edits."""
    h = hashlib.sha1()
    for name in sorted(kb_contexts):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(kb_contexts[name].encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def _bm25_cache_clear() -> None:
    """Clear the in-memory KB-router BM25 cache (used by tests)."""
    _BM25_CACHE.clear()


def _build_bm25_index(corpus_tokens: list[list[str]], *, fingerprint: str) -> bm25s.BM25:
    """Build a Lucene-style BM25 index over already-tokenised corpus docs.

    ``fingerprint`` is accepted for symmetry with the cache layer and to
    let tests monkey-patch this builder while keeping the call shape stable.
    """
    # ``show_progress=False`` silences bm25s' tqdm bars — they're noise in
    # CLI/MCP output and useless on 2–20-doc KB-routing corpora.
    retriever = bm25s.BM25(method="lucene")
    retriever.index(corpus_tokens, show_progress=False)
    return retriever


@dataclass
class KBRouteHit:
    kb_name: str
    score: float
    reason: str | None = None
    sampled_titles: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __iter__(self):
        """Allow ``for name, score in route_kbs(...)`` destructuring.

        Audit 2026-05-15 finding #7: the harness naturally tried to
        unpack hits and silently got wrong answers. Yielding only the
        two most-commonly-needed fields keeps the cheap-tuple ergonomics
        without removing access to the richer attributes.
        """
        yield self.kb_name
        yield self.score


# Cap how many paper titles we sample per KB when building the context
# string. Higher = better routing on KBs with weak descriptions but more
# tokens for the LLM path. 12 keeps the prompt comfortable.
DEFAULT_SAMPLE_PAPERS = 12


async def _build_kb_context(
    kb: Any,
    *,
    vector_store: Any,
    sample_papers: int = DEFAULT_SAMPLE_PAPERS,
) -> tuple[str, int]:
    """Return ``(context_string, num_titles_used)`` for one KB.

    Format:
        <description sentence>
        Top papers:
        - <title 1>
        - <title 2>
        ...
    """
    parts: list[str] = []
    desc = (kb.description or "").strip()
    # Skip noisy generic descriptions — they're worse signal than nothing
    generic = {
        "imported from references.bib",
        "smoke test",
        "audit test kb",
        "test",
        "demo",
    }
    if desc and desc.lower() not in generic:
        parts.append(desc)
    titles_used = 0
    try:
        rows = await vector_store.list_paper_metadata(kb.collection_name)
    except Exception as exc:
        logger.info("kb_router_list_papers_failed", kb=kb.name, error=str(exc))
        rows = []
    titles: list[str] = []
    for r in rows[:sample_papers * 2]:
        t = (r.get("title") or "").strip()
        if t and t not in titles:
            titles.append(t)
        if len(titles) >= sample_papers:
            break
    if titles:
        parts.append("Top papers:")
        for t in titles:
            parts.append(f"- {t}")
        titles_used = len(titles)
    if not parts:
        # Fall back to the KB name (very weak signal but better than empty)
        parts.append(kb.name.replace("_", " "))
    return "\n".join(parts), titles_used


def _tokenize(text: str) -> list[str]:
    import re
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]+", text) if len(w) > 1]


def _bm25_score_corpus(
    *,
    kb_names: list[str],
    corpus_tokens: list[list[str]],
    query_tokens: list[str],
    fingerprint: str,
) -> list[float]:
    """Run a single-query bm25s retrieval and return per-doc normalised scores.

    The returned list is aligned with ``kb_names`` and normalised to [0, 1]
    by dividing by the per-batch max, with a unique-token-overlap fallback
    when every BM25 score is 0 (small-corpus IDF collapse). Looks up — and
    populates on miss — the ``_BM25_CACHE``.
    """
    # Guard: bm25s also misbehaves on empty docs.
    corpus_tokens = [t or ["__empty__"] for t in corpus_tokens]
    cached = _BM25_CACHE.get(fingerprint)
    if cached is None:
        retriever = _build_bm25_index(corpus_tokens, fingerprint=fingerprint)
        _BM25_CACHE[fingerprint] = (retriever, list(kb_names), corpus_tokens)
    else:
        retriever, _cached_names, corpus_tokens = cached

    q_tokens = query_tokens or ["__empty__"]
    n_docs = len(kb_names)
    k = max(1, n_docs)
    # bm25s.retrieve returns (results_2d, scores_2d). results_2d[0] is the row
    # of top-k document indices for our single query; scores_2d[0] are the
    # matching scores. We re-key them back to the original kb_names order.
    results, scores_2d = retriever.retrieve([q_tokens], k=k, show_progress=False)
    raw_by_idx = [0.0] * n_docs
    for idx, s in zip(results[0], scores_2d[0], strict=True):
        raw_by_idx[int(idx)] = float(s)

    max_s = max(raw_by_idx) if raw_by_idx else 0.0
    if max_s > 0:
        return [s / max_s for s in raw_by_idx]
    # Small-N IDF-collapse fallback: identical motivation to the rank-bm25
    # version (e.g. N=2 with a term in exactly one doc → all-zero scores).
    q_unique = set(q_tokens)
    overlaps = [
        len(q_unique.intersection(doc_toks))
        for doc_toks in corpus_tokens
    ]
    max_o = max(overlaps) if overlaps else 0
    if max_o > 0:
        return [o / max_o for o in overlaps]
    return [0.0] * n_docs


async def _route_bm25(
    *, query: str, kb_contexts: list[tuple[str, str, int]], top_k: int,
    score_threshold: float,
) -> list[KBRouteHit]:
    """BM25 routing. ``kb_contexts`` is [(kb_name, context_string, num_titles), ...]."""
    kb_names = [c[0] for c in kb_contexts]
    corpus_tokens = [_tokenize(c[1]) for c in kb_contexts]
    fingerprint = _corpus_fingerprint({c[0]: c[1] for c in kb_contexts})
    normalized = _bm25_score_corpus(
        kb_names=kb_names,
        corpus_tokens=corpus_tokens,
        query_tokens=_tokenize(query),
        fingerprint=fingerprint,
    )
    hits = [
        KBRouteHit(
            kb_name=name,
            score=float(norm),
            reason=None,
            sampled_titles=ntitles,
        )
        for (name, _ctx, ntitles), norm in zip(kb_contexts, normalized, strict=True)
    ]
    hits.sort(key=lambda h: -h.score)
    return [h for h in hits if h.score >= score_threshold][:top_k]


def route_kbs(
    *,
    query: str,
    kb_contexts: dict[str, str],
    top_k: int = 3,
    score_threshold: float = 0.0,
) -> list[KBRouteHit]:
    """Synchronous BM25 routing over a pre-built ``{kb_name: context}`` map.

    Thin wrapper around the same scoring path used by :func:`auto_route_kbs`
    for callers that already have plain context strings in hand (tests, the
    capsule-cycle plan's bm25s migration). Shares the corpus-fingerprint
    cache so repeated calls over the same KB set skip re-indexing.
    """
    if not kb_contexts:
        return []
    kb_names = list(kb_contexts.keys())
    corpus_tokens = [_tokenize(kb_contexts[n]) for n in kb_names]
    fingerprint = _corpus_fingerprint(kb_contexts)
    normalized = _bm25_score_corpus(
        kb_names=kb_names,
        corpus_tokens=corpus_tokens,
        query_tokens=_tokenize(query),
        fingerprint=fingerprint,
    )
    hits = [
        KBRouteHit(kb_name=name, score=float(score), reason=None, sampled_titles=0)
        for name, score in zip(kb_names, normalized, strict=True)
    ]
    hits.sort(key=lambda h: -h.score)
    return [h for h in hits if h.score >= score_threshold][:top_k]


_LLM_SYSTEM = (
    "You are routing a user's research question to the most relevant "
    "knowledge bases. Below are knowledge bases, each with a short "
    "description and a sample of paper titles. Score each KB from 0.0 "
    "(not relevant) to 1.0 (clearly relevant) for the question, and "
    "give a one-sentence reason.\n\n"
    "Return JSON only, in this exact shape:\n"
    '{"hits": [{"kb_name": "...", "score": 0.0-1.0, "reason": "..."}, ...]}'
)


async def _route_llm(
    *,
    query: str,
    kb_contexts: list[tuple[str, str, int]],
    top_k: int,
    score_threshold: float,
    llm_client: Any,
    model: str,
    provider: str,
) -> list[KBRouteHit]:
    """LLM routing — one batched call scores every KB.

    The KB context block is large (~description + 12 titles per KB)
    and identical across every routing query until the user creates
    or modifies a KB. We mark it as a cacheable prefix so Anthropic's
    prompt cache amortises the cost across a session — only the user
    question changes between hits.
    """
    from perspicacite.llm.client import build_cached_messages
    kbs_block = "Knowledge bases:\n\n" + "\n\n".join(
        f"### {name}\n{ctx}"
        for name, ctx, _ in kb_contexts
    )
    messages = build_cached_messages(
        system=_LLM_SYSTEM,
        cacheable_context=kbs_block,
        user_message=f"Question:\n{query}",
        provider=provider,
    )
    try:
        text = await llm_client.complete(
            messages=messages,
            model=model,
            provider=provider,
            max_tokens=1500,
            temperature=0.2,
            stage="kb_router",
        )
    except Exception as exc:
        logger.warning("kb_router_llm_failed_falling_back_to_bm25", error=str(exc))
        return await _route_bm25(
            query=query, kb_contexts=kb_contexts, top_k=top_k,
            score_threshold=score_threshold,
        )
    # Strip code fences if the model added them
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        # ```json ... ``` or ``` ... ```
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.strip().rstrip("`")
    try:
        obj = json.loads(cleaned)
    except Exception as exc:
        logger.warning(
            "kb_router_llm_unparseable_falling_back_to_bm25",
            error=str(exc), sample=cleaned[:200],
        )
        return await _route_bm25(
            query=query, kb_contexts=kb_contexts, top_k=top_k,
            score_threshold=score_threshold,
        )
    # Map num_titles back from kb_contexts for the returned hits
    name_to_titles = {name: nt for name, _ctx, nt in kb_contexts}
    hits: list[KBRouteHit] = []
    for h in obj.get("hits") or []:
        name = h.get("kb_name")
        if not name or name not in name_to_titles:
            continue
        try:
            score = float(h.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        hits.append(KBRouteHit(
            kb_name=name,
            score=max(0.0, min(1.0, score)),
            reason=(h.get("reason") or None),
            sampled_titles=name_to_titles[name],
        ))
    hits.sort(key=lambda h: -h.score)
    return [h for h in hits if h.score >= score_threshold][:top_k]


async def auto_route_kbs(
    *,
    query: str,
    kb_metas: list[Any],
    vector_store: Any,
    method: str = "bm25",
    top_k: int = 3,
    score_threshold: float = 0.1,
    llm_client: Any = None,
    llm_model: str = "claude-haiku-4-5",
    llm_provider: str = "anthropic",
    sample_papers: int = DEFAULT_SAMPLE_PAPERS,
) -> list[KBRouteHit]:
    """Pick the top-N most relevant KBs for ``query``.

    Args:
        query: User research question.
        kb_metas: KnowledgeBase metas to consider. Caller supplies the
            candidate set (could be every KB, a curated subset, KBs the
            user has access to, etc.).
        vector_store: For sampling per-KB paper titles.
        method: ``"bm25"`` (default; no LLM) or ``"llm"``.
        top_k: Max KBs to return.
        score_threshold: Drop KBs whose normalized score < this. For
            ``bm25`` scores are normalized to [0,1] by dividing by the
            max in the batch.
        llm_client: Required when ``method="llm"``.
        llm_model / llm_provider: Override the default cheap router model.
        sample_papers: Per-KB paper-title sample size used to build the
            context string scored against the query.

    Returns:
        A ranked list of :class:`KBRouteHit`. Empty when no KB passed
        the threshold.
    """
    if not kb_metas:
        return []
    # Build per-KB context strings (description + sampled titles).
    kb_contexts: list[tuple[str, str, int]] = []
    for kb in kb_metas:
        ctx, ntitles = await _build_kb_context(
            kb, vector_store=vector_store, sample_papers=sample_papers,
        )
        kb_contexts.append((kb.name, ctx, ntitles))

    if method == "llm" and llm_client is not None:
        return await _route_llm(
            query=query, kb_contexts=kb_contexts, top_k=top_k,
            score_threshold=score_threshold, llm_client=llm_client,
            model=llm_model, provider=llm_provider,
        )
    return await _route_bm25(
        query=query, kb_contexts=kb_contexts, top_k=top_k,
        score_threshold=score_threshold,
    )
