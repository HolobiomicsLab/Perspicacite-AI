"""Dynamic Knowledge Base for session-specific document storage.

Creates temporary vector collections for relevant papers,
scoped to a single research session.
"""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from perspicacite.logging import get_logger
from perspicacite.rag.query_scope import PaperScopeResult, merge_scope_with_candidates
from perspicacite.retrieval.chroma_store import _metadata_to_chunk

if TYPE_CHECKING:
    from perspicacite.models.papers import Paper

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
        from perspicacite.models.documents import DocumentChunk, ChunkMetadata
        from perspicacite.rag.chunking import create_chunker, SimpleChunker

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

        chunks: list[DocumentChunk] = []

        # Create metadata chunk
        abstract_display = paper.abstract or "No abstract available"
        metadata_text = f"""Title: {paper.title}
Authors: {', '.join(str(a) for a in paper.authors)}
Year: {paper.year or 'Unknown'}
DOI: {paper.doi or 'Unknown'}

Abstract:
{abstract_display}"""

        # Format authors as comma-separated string for metadata
        authors_str = ", ".join(str(a) for a in paper.authors) if paper.authors else None
        
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
                    ),
                ))

        # Add to vector store (embeddings generated internally)
        await self.vector_store.add_documents(
            collection=self.collection_name,
            chunks=chunks,
        )

        return len(chunks)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results (default: config.top_k)
            min_score: Minimum relevance score

        Returns:
            List of search results with text and metadata
        """
        if not self._initialized:
            raise RuntimeError("Knowledge base not initialized")

        top_k = top_k or self.config.top_k
        min_score = min_score or self.config.min_relevance_score

        # Embed query (embed() expects a list, returns a list)
        query_embeddings = await self.embedding_service.embed([query])
        query_embedding = query_embeddings[0]

        # Search
        results = await self.vector_store.search(
            collection=self.collection_name,
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get extra for filtering
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
        hard_cap = min(max_papers_cap or top_k, 5)

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
                        "chunks": [{"chunk_index": 0, "text": hit["text"]}],
                        "full_text": hit["text"],
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
                "chunks": chunks_list,
                "full_text": full_text,
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
