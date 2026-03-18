"""Cross-encoder reranker for two-stage retrieval."""

import asyncio
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.search import RetrievedChunk

logger = get_logger("perspicacite.retrieval.reranker")


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.

    Two-stage retrieval:
    1. Fast retrieval (vector/hybrid) → top_k * 3 candidates
    2. Cross-encoder scoring → top_k final results

    Cross-encoder is more accurate because it sees query + document together,
    unlike bi-encoders which encode them separately.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 8,
    ):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any | None = None

    def _get_model(self) -> Any:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    "loading_cross_encoder",
                    model=self.model_name,
                )
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks by cross-encoder score.

        Args:
            query: Original query
            chunks: Candidate chunks from first-stage retrieval
            top_k: Number of results to return

        Returns:
            Reranked chunks
        """
        if not chunks:
            return []

        if len(chunks) <= top_k:
            # Not enough to rerank
            return chunks[:top_k]

        logger.debug(
            "rerank_start",
            candidates=len(chunks),
            top_k=top_k,
        )

        # Prepare pairs for scoring
        pairs = [(query, chunk.chunk.text) for chunk in chunks]

        try:
            # Score in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            model = self._get_model()

            all_scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                scores = await loop.run_in_executor(
                    None,
                    lambda: model.predict(batch),
                )
                all_scores.extend(scores.tolist())

            # Attach scores and sort
            scored_chunks = list(zip(chunks, all_scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Build results with reranked scores
            reranked = []
            for chunk, score in scored_chunks[:top_k]:
                reranked.append(
                    RetrievedChunk(
                        chunk=chunk.chunk,
                        score=float(score),
                        retrieval_method="reranked",
                    )
                )

            logger.debug(
                "rerank_complete",
                input_candidates=len(chunks),
                output_top_k=len(reranked),
            )

            return reranked

        except Exception as e:
            logger.error(
                "rerank_failed",
                error=str(e),
            )
            # Return original chunks if reranking fails
            return chunks[:top_k]


class NoOpReranker:
    """
    No-op reranker that just returns top_k results.

    Used when reranking is disabled or model not available.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Return top_k chunks unchanged."""
        return chunks[:top_k]
