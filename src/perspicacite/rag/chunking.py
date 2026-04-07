"""Text chunking for RAG operations.

Dispatches to the appropriate chunking strategy:
- token: Fast word-based splitting (default, no dependencies)
- semantic: Sentence embedding-based cohesive chunks
- agentic: LLM-based intelligent partitioning
"""

from __future__ import annotations

from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.rag.chunking")


class SimpleChunker:
    """Fast word-based text chunker (no external dependencies)."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks by word count."""
        if not text:
            return []
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
        return chunks


class AdvancedChunkerAdapter:
    """Wraps pipeline.chunking_advanced.AdvancedChunker to match the simple interface.

    Returns list[str] instead of list[DocumentChunk] so _add_paper doesn't need changes.
    """

    def __init__(
        self,
        method: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        *,
        llm_client: Any = None,
    ):
        self.method = method
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.llm_client = llm_client

    async def chunk_text_async(self, text: str) -> list[str]:
        """Chunk text using the advanced chunker. Returns list of strings."""
        from perspicacite.pipeline.chunking_advanced import AdvancedChunker as _AdvancedChunker

        # Map word-based chunk_size to approximate token count (~1.3 tokens per word)
        approx_tokens = int(self.chunk_size * 0.75)
        approx_overlap = int(self.overlap * 0.75)

        chunker = _AdvancedChunker(
            method=self.method,
            max_tokens=approx_tokens,
            overlap_tokens=approx_overlap,
            section_aware=True,
        )

        # Create a minimal paper stub for metadata
        from perspicacite.models.papers import Paper, PaperSource

        paper = Paper(
            id="chunk",
            title="",
            authors=[],
            source=PaperSource.WEB_SEARCH,
        )

        doc_chunks = await chunker.chunk_text(text, paper, llm_client=self.llm_client)
        return [c.text for c in doc_chunks]

    def chunk_text(self, text: str) -> list[str]:
        """Synchronous fallback — not supported for advanced methods."""
        raise RuntimeError(
            f"Chunking method '{self.method}' requires async. "
            "Use chunk_text_async() or switch to 'token' method."
        )


def create_chunker(
    chunk_size: int = 1000,
    overlap: int = 200,
    method: str = "token",
    *,
    llm_client: Any = None,
) -> SimpleChunker | AdvancedChunkerAdapter:
    """Create a chunker for the given method.

    Args:
        chunk_size: Target chunk size (words for token, approx tokens for semantic/agentic)
        overlap: Overlap between chunks
        method: "token", "semantic", or "agentic"
        llm_client: LLM client for agentic chunking

    Returns:
        Chunker with .chunk_text() or .chunk_text_async() method
    """
    if method == "token":
        return SimpleChunker(chunk_size=chunk_size, overlap=overlap)
    elif method in ("semantic", "agentic"):
        return AdvancedChunkerAdapter(
            method=method,
            chunk_size=chunk_size,
            overlap=overlap,
            llm_client=llm_client,
        )
    else:
        logger.warning(f"Unknown chunking method '{method}', falling back to token")
        return SimpleChunker(chunk_size=chunk_size, overlap=overlap)
