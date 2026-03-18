"""Hybrid retrieval combining vector similarity and BM25."""

from perspicacite.llm.embeddings import EmbeddingProvider
from perspicacite.logging import get_logger
from perspicacite.models.search import RetrievedChunk, SearchFilters
from perspicacite.retrieval.bm25 import BM25Index
from perspicacite.retrieval.chroma_store import ChromaVectorStore

logger = get_logger("perspicacite.retrieval.hybrid")


class HybridRetriever:
    """
    Combines vector similarity and BM25 for better recall.

    Uses Reciprocal Rank Fusion (RRF) to merge results:
    score = sum(1 / (k + rank_i)) for each retriever i

    Where k is a constant (typically 60) that dampens the impact of high rankings.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index,
        embedding_provider: EmbeddingProvider,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Chroma vector store
            bm25_index: BM25 keyword index
            embedding_provider: For generating query embeddings
            vector_weight: Weight for vector scores (not used in RRF)
            bm25_weight: Weight for BM25 scores (not used in RRF)
            rrf_k: RRF constant (higher = less emphasis on top ranks)
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_provider = embedding_provider
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search using hybrid retrieval.

        Args:
            query: Search query
            collection: Collection name
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            Fused and ranked results
        """
        logger.debug(
            "hybrid_search_start",
            query=query[:50],
            collection=collection,
        )

        # 1. Vector search
        query_embedding = await self.embedding_provider.embed([query])
        vector_results = await self.vector_store.search(
            collection=collection,
            query_embedding=query_embedding[0],
            top_k=top_k * 3,  # Over-fetch for fusion
            filters=filters,
        )

        # 2. BM25 search
        bm25_results = await self.bm25_index.search(
            query=query,
            top_k=top_k * 3,
        )

        # 3. RRF fusion
        fused_results = self._rrf_fusion(
            vector_results,
            bm25_results,
            top_k=top_k,
        )

        logger.debug(
            "hybrid_search_complete",
            vector_hits=len(vector_results),
            bm25_hits=len(bm25_results),
            fused=len(fused_results),
        )

        return fused_results

    def _rrf_fusion(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Fuse results using Reciprocal Rank Fusion.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        # Build score map by chunk ID
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        # Add vector scores
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)
            chunks[chunk_id] = result

        # Add BM25 scores
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (self.rrf_k + rank)
            if chunk_id not in chunks:
                chunks[chunk_id] = result

        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        # Build results
        fused = []
        for chunk_id in sorted_ids[:top_k]:
            chunk = chunks[chunk_id]
            fused.append(
                RetrievedChunk(
                    chunk=chunk.chunk,
                    score=scores[chunk_id],
                    retrieval_method="hybrid",
                )
            )

        return fused

    def _weighted_fusion(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Alternative: Fuse using weighted scores.

        Normalizes scores from each retriever and weights them.
        Less effective than RRF but simpler.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of results

        Returns:
            Fused and ranked results
        """
        # Normalize scores for each result set
        if vector_results:
            max_vector = max(r.score for r in vector_results)
            if max_vector > 0:
                for r in vector_results:
                    r.score = r.score / max_vector

        if bm25_results:
            max_bm25 = max(r.score for r in bm25_results)
            if max_bm25 > 0:
                for r in bm25_results:
                    r.score = r.score / max_bm25

        # Build combined scores
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for r in vector_results:
            scores[r.chunk.id] = scores.get(r.chunk.id, 0) + r.score * self.vector_weight
            chunks[r.chunk.id] = r

        for r in bm25_results:
            scores[r.chunk.id] = scores.get(r.chunk.id, 0) + r.score * self.bm25_weight
            if r.chunk.id not in chunks:
                chunks[r.chunk.id] = r

        # Sort and return top-k
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

        fused = []
        for chunk_id in sorted_ids[:top_k]:
            chunk = chunks[chunk_id]
            fused.append(
                RetrievedChunk(
                    chunk=chunk.chunk,
                    score=scores[chunk_id],
                    retrieval_method="hybrid",
                )
            )

        return fused
