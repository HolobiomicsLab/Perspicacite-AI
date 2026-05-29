"""Dynamic Knowledge Base for session-specific document storage.

Creates temporary vector collections for relevant papers,
scoped to a single research session.
"""

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from perspicacite.logging import get_logger
from perspicacite.rag.paper_metadata_codec import decode_paper_metadata_json
from perspicacite.rag.query_scope import PaperScopeResult, merge_scope_with_candidates
from perspicacite.retrieval.chroma_store import _metadata_to_chunk

if TYPE_CHECKING:
    from perspicacite.models.documents import DocumentChunk
    from perspicacite.models.papers import Paper
    from perspicacite.models.search import SearchFilters

logger = get_logger("perspicacite.rag.dynamic_kb")


@dataclass
class KnowledgeBaseConfig:
    """Configuration for dynamic knowledge base."""

    # Collection settings
    collection_prefix: str = "session_"
    vector_size: int = 768  # Default for voyage-multilingual-2

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_method: str = "token"  # "token", "semantic", "agentic"

    # Retrieval settings
    top_k: int = 5
    # Keep low: Chroma distances vary by metric; we rank by score and take top_k.
    min_relevance_score: float = 0.0


class DynamicKnowledgeBase:
    """
    Session-scoped knowledge base for relevant papers.

    Creates a temporary vector collection, adds documents from
    relevant papers, and provides retrieval capabilities.

    The collection is ephemeral - created per session and
    optionally cleaned up after use.
    """

    def __init__(
        self,
        vector_store: Any,  # VectorStoreInterface
        embedding_service: Any,  # EmbeddingServiceInterface
        config: KnowledgeBaseConfig | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config or KnowledgeBaseConfig()

        self.session_id = str(uuid.uuid4())[:8]
        self.collection_name = f"{self.config.collection_prefix}{self.session_id}"
        self._initialized = False
        self._paper_ids: set[str] = set()
        # Optional descriptive tags assigned by callers after construction
        # (base.py tags kb_name for single-KB provenance parity; run_ingest
        # sets name/description). None unless a caller sets them.
        self.kb_name: str | None = None
        self.name: str | None = None
        self.description: str | None = None

    async def initialize(self) -> None:
        """Create the session collection."""
        if self._initialized:
            return

        try:
            await self.vector_store.create_collection(
                name=self.collection_name,
                embedding_dim=self.config.vector_size,
            )
            self._initialized = True
            logger.info(
                "kb_initialized",
                session=self.session_id,
                collection=self.collection_name,
            )

        except Exception as e:
            logger.error("kb_init_error", error=str(e))
            raise

    async def add_papers(
        self,
        papers: list["Paper"],
        include_full_text: bool = True,
    ) -> int:
        """
        Add papers to the knowledge base.

        Args:
            papers: Relevant papers to add
            include_full_text: Whether to include full text or just metadata

        Returns:
            Number of documents added
        """
        if not self._initialized:
            await self.initialize()

        total_added = 0

        for paper in papers:
            if paper.id in self._paper_ids:
                continue  # Skip duplicates

            try:
                docs_added = await self._add_paper(paper, include_full_text)
                total_added += docs_added
                self._paper_ids.add(paper.id)

            except Exception as e:
                logger.error("add_paper_error", paper_id=paper.id, error=str(e))

        logger.info(
            "papers_added",
            session=self.session_id,
            papers=len(papers),
            documents=total_added,
        )
        return total_added

    async def _add_paper(
        self,
        paper: "Paper",
        include_full_text: bool,
    ) -> int:
        """Add a single paper to the collection."""
        from perspicacite.models.documents import ChunkMetadata, DocumentChunk
        from perspicacite.rag.chunking import SimpleChunker, create_chunker

        # Skip papers with no searchable content
        has_full_text = bool(paper.full_text and paper.full_text.strip())
        has_abstract = bool(paper.abstract and paper.abstract.strip()
                           and paper.abstract.strip().lower() != "no abstract available")
        if not has_full_text and not has_abstract:
            logger.info(
                "skip_paper_no_content",
                paper_id=paper.id,
                title=paper.title[:60] if paper.title else "",
            )
            return 0

        # JSON-encode ``paper.metadata`` once per paper so every chunk
        # carries the same payload through chroma. Non-bundle papers
        # (no ``metadata`` dict) leave the field as None.
        paper_md_json: str | None = None
        if isinstance(getattr(paper, "metadata", None), dict) and paper.metadata:
            try:
                paper_md_json = json.dumps(paper.metadata, default=str)
            except (TypeError, ValueError):
                paper_md_json = None

        chunks: list[DocumentChunk] = []

        # Create metadata chunk (header only — the abstract is embedded as its
        # own chunk below, so passage similarity matches clean prose, not the
        # metadata envelope).
        metadata_text = f"""Title: {paper.title}
Authors: {', '.join(str(a) for a in paper.authors)}
Year: {paper.year or 'Unknown'}
DOI: {paper.doi or 'Unknown'}"""

        # Format authors as comma-separated string for metadata
        authors_str = ", ".join(str(a) for a in paper.authors) if paper.authors else None

        paper_content_type = getattr(paper, "content_type", None)
        paper_license = getattr(paper, "license", None)
        chunks.append(DocumentChunk(
            id=f"{paper.id}_metadata",
            text=metadata_text,
            metadata=ChunkMetadata(
                paper_id=paper.id,
                chunk_index=0,
                source=paper.source,
                title=paper.title,
                authors=authors_str,
                year=paper.year,
                doi=paper.doi,
                url=paper.url,
                abstract=paper.abstract,
                content_type=paper_content_type,
                license_id=paper_license,
                section="metadata",
                paper_metadata_json=paper_md_json,
            ),
        ))

        # Abstract as its own chunk: clean prose (no metadata envelope) so
        # passage-similarity queries match the abstract cleanly. Carries the
        # same paper metadata so retrieval surfaces license_id / doi / etc.
        if paper.abstract and paper.abstract.strip():
            chunks.append(DocumentChunk(
                id=f"{paper.id}_abstract",
                text=paper.abstract,
                metadata=ChunkMetadata(
                    paper_id=paper.id,
                    chunk_index=1,
                    source=paper.source,
                    title=paper.title,
                    authors=authors_str,
                    year=paper.year,
                    doi=paper.doi,
                    url=paper.url,
                    abstract=paper.abstract,
                    content_type=paper_content_type,
                    license_id=paper_license,
                    section="abstract",
                    paper_metadata_json=paper_md_json,
                ),
            ))

        # Add full text if available and requested
        if include_full_text and paper.full_text:
            chunker = create_chunker(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                method=self.config.chunking_method,
            )

            # Advanced methods are async; token is sync
            if isinstance(chunker, SimpleChunker):
                text_chunks = chunker.chunk_text(paper.full_text)
            else:
                text_chunks = await chunker.chunk_text_async(paper.full_text)

            for i, chunk_text in enumerate(text_chunks):
                chunks.append(DocumentChunk(
                    id=f"{paper.id}_chunk_{i}",
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        paper_id=paper.id,
                        chunk_index=i + 1,  # +1 because metadata is chunk 0
                        source=paper.source,
                        title=paper.title,
                        authors=authors_str,
                        year=paper.year,
                        doi=paper.doi,
                        url=paper.url,
                        content_type=paper_content_type,
                        license_id=paper_license,
                        paper_metadata_json=paper_md_json,
                    ),
                ))

        # Contextual retrieval — dispatch by tier:
        #   none     → no extra prefix (structural-only)
        #   abstract → paper.abstract (or leading slice) repeated per chunk, 0 LLM
        #   summary  → 1 LLM call → reused for every chunk of this paper
        #   chunk    → 1 LLM call per chunk (Anthropic-style)
        # Stored chunk.text is never modified — only the embedding sees
        # the prefix.
        per_chunk_context: list[str] | None = None
        kb_cfg = getattr(self.config, "_kb_config", None) or getattr(self.config, "kb_config", None)
        if kb_cfg is None:
            kb_cfg = self.config
        tier = getattr(kb_cfg, "contextual_retrieval_tier", "none")
        # Legacy shorthand: contextual_retrieval=True ⇒ tier="chunk"
        if getattr(kb_cfg, "contextual_retrieval", False) and tier == "none":
            tier = "chunk"
        llm_client = getattr(self, "llm_client", None)
        doc_text = paper.full_text or paper.abstract or ""

        if tier == "abstract" and doc_text:
            try:
                from perspicacite.retrieval.contextual import abstract_prefix_for_paper
                prefix = abstract_prefix_for_paper(
                    paper_text=doc_text, abstract=paper.abstract,
                    max_chars=getattr(kb_cfg, "contextual_retrieval_max_chars", 400),
                )
                if prefix:
                    per_chunk_context = [prefix] * len(chunks)
                    logger.info(
                        "contextual_abstract_applied",
                        paper_id=paper.id, chars=len(prefix), chunks=len(chunks),
                    )
            except Exception as exc:
                logger.warning("contextual_abstract_failed", paper_id=paper.id, error=str(exc))

        elif tier == "summary" and llm_client is not None and doc_text:
            try:
                from perspicacite.retrieval.contextual import generate_document_summary
                summary = await generate_document_summary(
                    paper_id=paper.id, document_text=doc_text,
                    llm_client=llm_client,
                    model=getattr(kb_cfg, "contextual_retrieval_model", "claude-haiku-4-5"),
                    provider=getattr(kb_cfg, "contextual_retrieval_provider", "anthropic"),
                    max_chars=getattr(kb_cfg, "contextual_retrieval_max_chars", 400),
                )
                if summary:
                    per_chunk_context = [summary] * len(chunks)
                    logger.info(
                        "contextual_summary_applied",
                        paper_id=paper.id, chars=len(summary), chunks=len(chunks),
                    )
            except Exception as exc:
                logger.warning("contextual_summary_failed", paper_id=paper.id, error=str(exc))

        elif tier == "chunk" and llm_client is not None and doc_text:
            try:
                from perspicacite.retrieval.contextual import generate_chunk_contexts_bulk
                per_chunk_context = await generate_chunk_contexts_bulk(
                    paper_id=paper.id, chunks=chunks, document_text=doc_text,
                    llm_client=llm_client,
                    model=getattr(kb_cfg, "contextual_retrieval_model", "claude-haiku-4-5"),
                    provider=getattr(kb_cfg, "contextual_retrieval_provider", "anthropic"),
                    max_chars=getattr(kb_cfg, "contextual_retrieval_max_chars", 400),
                )
                logger.info(
                    "contextual_chunk_applied",
                    paper_id=paper.id,
                    contexts=sum(1 for c in per_chunk_context if c),
                    chunks=len(chunks),
                )
            except Exception as exc:
                logger.warning(
                    "contextual_chunk_failed_falling_back",
                    paper_id=paper.id, error=str(exc),
                )

        # Sub-project B: stamp each chunk with the model that will produce its
        # vector so downstream code can route queries through the matching
        # embedder.  stamp_embedding_models_on_chunks returns new (frozen)
        # chunks — the original list is not mutated.
        chunks = stamp_embedding_models_on_chunks(chunks, embedder=self.embedding_service)

        # Add to vector store (embeddings generated internally)
        await self.vector_store.add_documents(
            collection=self.collection_name,
            chunks=chunks,
            per_chunk_context=per_chunk_context,
        )

        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        filters: "SearchFilters | None" = None,
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results (default: config.top_k)
            min_score: Minimum relevance score
            filters: Optional ``SearchFilters`` (year_min/year_max/...).
                Translated to Chroma where-clauses inside the vector
                store. See Wave 4.2.

        Returns:
            List of search results with text and metadata
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized")

        top_k = top_k or self.config.top_k
        min_score = min_score or self.config.min_relevance_score

        # Embed query with query-specific prompt when available (instruct models such as
        # stella_en_1.5B_v5 require a task prefix for query encoding; embed_query() handles
        # this transparently while plain embed() is still used for document ingest).
        query_embeddings = await self.embedding_service.embed_query([query])
        query_embedding = query_embeddings[0]

        # Search
        results = await self.vector_store.search(
            collection=self.collection_name,
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get extra for filtering
            filters=filters,
        )

        # Filter by score and deduplicate by paper
        filtered = []
        seen_papers = set()

        for r in results:
            # Handle RetrievedChunk objects
            score = r.score
            paper_id = r.chunk.metadata.paper_id if r.chunk.metadata else None

            if score < min_score:
                continue

            if not paper_id or paper_id in seen_papers:
                continue

            seen_papers.add(paper_id)
            filtered.append({
                "text": r.chunk.text,
                "score": score,
                "paper_id": paper_id,
                "metadata": r.chunk.metadata,
                # F-15: propagate kb_name when the retriever was tagged by
                # BaseRAGMode._build_kb_retriever (single-KB path). The
                # MultiKBRetriever sets this per chunk already.
                "kb_name": getattr(self, "kb_name", None),
            })

            if len(filtered) >= top_k:
                break

        return filtered

    async def search_two_pass(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        *,
        paper_scope: PaperScopeResult | None = None,
        max_papers_cap: int | None = None,
    ) -> list[dict[str, Any]]:
        """Two-pass retrieval: identify relevant papers, then fetch all their chunks.

        Pass 1 uses existing ``search()`` to find top papers (deduplicated by
        paper_id).  Pass 2 fetches ALL chunks for those papers via
        ``get_chunks_by_paper_ids()`` and removes chunk overlaps.

        Optional ``paper_scope`` merges user-referenced papers (quoted title / DOI)
        with vector hits and caps how many full papers are loaded.

        Returns:
            Paper-level dicts with keys: paper_id, paper_score, title, authors,
            year, doi, chunks, full_text.
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized")

        top_k = top_k or self.config.top_k
        min_score = min_score or self.config.min_relevance_score
        hard_cap = min(max_papers_cap or top_k, 50)  # was 5; cap controlled by max_papers_retrieval

        # ── Pass 1: identify relevant papers ───────────────────────────
        hit_chunks = await self.search(query, top_k=top_k, min_score=min_score)

        paper_scores: dict[str, float] = {}
        paper_meta: dict[str, Any] = {}

        if hit_chunks:
            for hit in hit_chunks:
                pid = hit["paper_id"]
                score = hit["score"]
                if pid not in paper_scores or score > paper_scores[pid]:
                    paper_scores[pid] = score
                    meta = hit["metadata"]
                    paper_meta[pid] = meta

        if paper_scope and paper_scope.forced_paper_ids and not hit_chunks:
            for pid in paper_scope.forced_paper_ids:
                paper_scores[pid] = 0.55
                row = await self.vector_store.peek_paper_metadata_row(
                    self.collection_name, pid
                )
                if row:
                    paper_meta[pid] = _metadata_to_chunk(row)

        if not paper_scores:
            return []

        candidate_order = sorted(
            paper_scores.keys(), key=lambda p: -paper_scores[p]
        )
        paper_ids = merge_scope_with_candidates(
            candidate_order, paper_scores, paper_scope, hard_cap
        )

        for pid in paper_ids:
            if pid not in paper_meta:
                row = await self.vector_store.peek_paper_metadata_row(
                    self.collection_name, pid
                )
                if row:
                    paper_meta[pid] = _metadata_to_chunk(row)

        # ── Pass 2: fetch all chunks for those papers ──────────────────
        all_chunks = await self.vector_store.get_chunks_by_paper_ids(
            self.collection_name, paper_ids
        )
        if not all_chunks:
            # Fallback: return pass-1 results as-is (if any)
            if hit_chunks:
                return [
                    {
                        "paper_id": hit["paper_id"],
                        "paper_score": hit["score"],
                        "title": getattr(hit["metadata"], "title", None),
                        "authors": getattr(hit["metadata"], "authors", None),
                        "year": getattr(hit["metadata"], "year", None),
                        "doi": getattr(hit["metadata"], "doi", None),
                        "paper_metadata": decode_paper_metadata_json(hit["metadata"]),
                        "chunks": [{"chunk_index": 0, "text": hit["text"]}],
                        "full_text": hit["text"],
                        "kb_name": getattr(self, "kb_name", None),
                    }
                    for hit in hit_chunks
                ]
            return []

        # Remove chunk overlaps
        from perspicacite.rag.utils import deduplicate_chunk_overlaps

        deduped = deduplicate_chunk_overlaps(
            all_chunks, overlap_words=self.config.chunk_overlap
        )

        # Group by paper_id and build result
        from collections import OrderedDict

        grouped: OrderedDict[str, list[dict]] = OrderedDict()
        for d in deduped:
            grouped.setdefault(d["paper_id"], []).append(d)

        # Preserve merged ordering
        results: list[dict[str, Any]] = []
        for pid in paper_ids:
            chunks_list = grouped.get(pid, [])
            full_text = " ".join(c["text"] for c in chunks_list)
            meta = paper_meta.get(pid)
            results.append({
                "paper_id": pid,
                "paper_score": paper_scores[pid],
                "title": getattr(meta, "title", None),
                "authors": getattr(meta, "authors", None),
                "year": getattr(meta, "year", None),
                "doi": getattr(meta, "doi", None),
                "paper_metadata": decode_paper_metadata_json(meta),
                "chunks": chunks_list,
                "full_text": full_text,
                # F-15: tag with the KB this retriever wraps so SourceReference
                # carries kb_name through to the caller (single-KB path).
                "kb_name": getattr(self, "kb_name", None),
            })

        logger.info(
            "search_two_pass_complete",
            query=query[:80],
            papers=len(results),
            chunks=len(deduped),
        )
        return results

    async def cleanup(self) -> None:
        """Clean up the session collection."""
        if not self._initialized:
            return

        try:
            await self.vector_store.delete_collection(self.collection_name)
            self._initialized = False
            logger.info("kb_cleaned", session=self.session_id)

        except Exception as e:
            logger.error("kb_cleanup_error", error=str(e))

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


def stamp_embedding_models_on_chunks(
    chunks: list["DocumentChunk"],
    *,
    embedder: object,  # EmbeddingProvider or TypedEmbeddingProvider — kept untyped to avoid circular imports
) -> list["DocumentChunk"]:
    """Return new chunks with ``metadata.embedding_model`` set to the
    actual model that would handle each chunk's content type.

    For a :class:`TypedEmbeddingProvider`, this reads the routing map
    and picks the right inner provider's ``model_name`` per chunk.
    For any other provider, every chunk gets ``embedder.model_name``.

    The original chunks are NOT mutated (``ChunkMetadata`` is frozen);
    new chunks are returned via ``model_copy``.

    Sub-project B (2026-05-15 design): records the embedder identity
    on each chunk so the KB can later route queries through the
    matching model.
    """
    from perspicacite.llm.embeddings import TypedEmbeddingProvider
    from perspicacite.models.documents import DocumentChunk as _DocumentChunk

    if isinstance(embedder, TypedEmbeddingProvider):
        by_type = embedder._by_type
        default = embedder._default

        def _model_for(ctype: str | None) -> str:
            if ctype and ctype in by_type:
                return by_type[ctype].model_name
            return default.model_name
    else:
        single_name = embedder.model_name  # type: ignore[attr-defined]

        def _model_for(ctype: str | None) -> str:
            del ctype
            return single_name

    out: list[_DocumentChunk] = []
    for c in chunks:
        md_updated = c.metadata.model_copy(
            update={"embedding_model": _model_for(c.metadata.content_type)}
        )
        out.append(_DocumentChunk(id=c.id, text=c.text, metadata=md_updated))
    return out


class DynamicKBFactory:
    """Factory for creating dynamic knowledge bases."""

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        default_config: KnowledgeBaseConfig | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.default_config = default_config

    def create_kb(self, config: KnowledgeBaseConfig | None = None) -> DynamicKnowledgeBase:
        """Create a new knowledge base."""
        return DynamicKnowledgeBase(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            config=config or self.default_config,
        )
