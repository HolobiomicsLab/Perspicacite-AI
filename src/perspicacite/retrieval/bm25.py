"""BM25 keyword-based retrieval."""

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from perspicacite.logging import get_logger
from perspicacite.models.documents import DocumentChunk
from perspicacite.models.search import RetrievedChunk

logger = get_logger("perspicacite.retrieval.bm25")


class BM25Index:
    """
    BM25 index for keyword-based retrieval.

    Built alongside the vector index for hybrid search.
    """

    def __init__(self):
        self.index: BM25Okapi | None = None
        self.documents: list[DocumentChunk] = []
        self.tokenized_corpus: list[list[str]] = []

    async def build(self, chunks: list[DocumentChunk]) -> None:
        """
        Build BM25 index from document chunks.

        Args:
            chunks: Document chunks to index
        """
        if not chunks:
            logger.warning("bm25_build_empty_corpus")
            return

        self.documents = chunks

        # Tokenize documents (simple whitespace tokenization)
        self.tokenized_corpus = [
            _tokenize(chunk.text) for chunk in chunks
        ]

        # Build BM25 index
        self.index = BM25Okapi(self.tokenized_corpus)

        logger.info(
            "bm25_index_built",
            document_count=len(chunks),
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Search by BM25 score.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of retrieved chunks with BM25 scores
        """
        if self.index is None or not self.documents:
            logger.warning("bm25_search_no_index")
            return []

        # Tokenize query
        tokenized_query = _tokenize(query)

        # Get scores for all documents
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Normalize scores to [0, 1]
        max_score = max(scores) if scores.any() else 1.0

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                normalized_score = scores[idx] / max_score if max_score > 0 else 0
                results.append(
                    RetrievedChunk(
                        chunk=self.documents[idx],
                        score=float(normalized_score),
                        retrieval_method="bm25",
                    )
                )

        logger.debug(
            "bm25_search_complete",
            query=query[:50],
            results=len(results),
        )
        return results

    def save(self, path: str) -> None:
        """
        Save index to disk.

        Args:
            path: Path to save index
        """
        if self.index is None:
            raise ValueError("Cannot save empty index")

        data = {
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
            "index_params": {
                "k1": self.index.k1,
                "b": self.index.b,
                "epsilon": self.index.epsilon,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("bm25_index_saved", path=path)

    def load(self, path: str) -> None:
        """
        Load index from disk.

        Args:
            path: Path to load index from
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]

        # Rebuild index
        self.index = BM25Okapi(
            self.tokenized_corpus,
            k1=data["index_params"]["k1"],
            b=data["index_params"]["b"],
            epsilon=data["index_params"]["epsilon"],
        )

        logger.info(
            "bm25_index_loaded",
            path=path,
            documents=len(self.documents),
        )


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenization for BM25.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    if not text:
        return []

    # Lowercase and split on whitespace
    # Remove punctuation
    import re

    text = text.lower()
    # Keep alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Split on whitespace
    tokens = text.split()

    # Filter out very short tokens
    tokens = [t for t in tokens if len(t) > 1]

    return tokens
