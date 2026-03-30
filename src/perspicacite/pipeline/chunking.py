"""Text chunking for document processing.

This module provides basic chunking strategies. For advanced strategies
(semantic, agentic with LLM), use chunking_advanced module.
"""

import re
from typing import Literal

from perspicacite.logging import get_logger
from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.kb import ChunkConfig
from perspicacite.models.papers import Paper, PaperSource

logger = get_logger("perspicacite.pipeline.chunking")


async def chunk_text(
    text: str,
    paper: Paper,
    config: ChunkConfig | None = None,
) -> list[DocumentChunk]:
    """
    Split text into chunks.

    Args:
        text: Text to chunk
        paper: Source paper for metadata
        config: Chunking configuration

    Returns:
        List of document chunks
    """
    if config is None:
        config = ChunkConfig()

    if config.method == "token":
        return _chunk_by_tokens(text, paper, config)
    elif config.method == "semantic":
        return _chunk_by_semantic(text, paper, config)
    elif config.method == "section_aware":
        return _chunk_by_section(text, paper, config)
    else:
        raise ValueError(f"Unknown chunking method: {config.method}")


def _chunk_by_tokens(
    text: str,
    paper: Paper,
    config: ChunkConfig,
) -> list[DocumentChunk]:
    """Chunk by token count (approximated by characters)."""
    # Rough approximation: 1 token ≈ 4 characters
    char_per_chunk = config.chunk_size * 4
    overlap_chars = config.chunk_overlap * 4

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + char_per_chunk, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending
            search_text = text[end - 100 : end + 100]
            sentence_end = search_text.rfind(". ")
            if sentence_end != -1:
                end = end - 100 + sentence_end + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk = DocumentChunk(
                id=f"{paper.id}_{chunk_index}",
                text=chunk_text,
                metadata=ChunkMetadata(
                    paper_id=paper.id,
                    chunk_index=chunk_index,
                    source=paper.source,
                    title=paper.title,
                    authors=_format_authors(paper.authors),
                    year=paper.year,
                    doi=paper.doi,
                    url=paper.url,
                ),
            )
            chunks.append(chunk)
            chunk_index += 1

        start = end - overlap_chars
        if start >= end:
            start = end

    return chunks


def _chunk_by_semantic(
    text: str,
    paper: Paper,
    config: ChunkConfig,
) -> list[DocumentChunk]:
    """
    Chunk by semantic boundaries (paragraphs and sections).

    For now, this is a simplified implementation that splits by paragraphs.
    A full implementation would use embeddings to find semantic boundaries.
    """
    # Split by paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    # Rough char target
    char_target = config.chunk_size * 4

    for para in paragraphs:
        para_length = len(para)

        if current_length + para_length > char_target and current_chunk:
            # Save current chunk
            chunk_text = "\n\n".join(current_chunk)
            chunk = DocumentChunk(
                id=f"{paper.id}_{chunk_index}",
                text=chunk_text,
                metadata=ChunkMetadata(
                    paper_id=paper.id,
                    chunk_index=chunk_index,
                    source=paper.source,
                    title=paper.title,
                    authors=_format_authors(paper.authors),
                    year=paper.year,
                    doi=paper.doi,
                    url=paper.url,
                ),
            )
            chunks.append(chunk)
            chunk_index += 1

            # Start new chunk with overlap
            overlap_count = min(len(current_chunk), 2)  # Keep last 2 paragraphs
            current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
            current_length = sum(len(p) for p in current_chunk)

        current_chunk.append(para)
        current_length += para_length

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        chunk = DocumentChunk(
            id=f"{paper.id}_{chunk_index}",
            text=chunk_text,
            metadata=ChunkMetadata(
                paper_id=paper.id,
                chunk_index=chunk_index,
                source=paper.source,
                title=paper.title,
                authors=_format_authors(paper.authors),
                year=paper.year,
                doi=paper.doi,
                url=paper.url,
            ),
        )
        chunks.append(chunk)

    return chunks


def _chunk_by_section(
    text: str,
    paper: Paper,
    config: ChunkConfig,
) -> list[DocumentChunk]:
    """
    Chunk respecting section boundaries.

    Detects sections by common headers (Abstract, Introduction, Methods, etc.)
    and chunks within each section.
    """
    # Common section headers
    section_pattern = r"\n\s*(Abstract|Introduction|Methods?|Results?|Discussion|Conclusion|References?|Acknowledgments?)\s*\n"

    # Split by sections
    parts = re.split(f"({section_pattern})", text, flags=re.IGNORECASE)

    chunks = []
    chunk_index = 0

    # If no sections found, fall back to token chunking
    if len(parts) < 3:
        return _chunk_by_tokens(text, paper, config)

    current_section = "Introduction"
    for i, part in enumerate(parts):
        if not part.strip():
            continue

        # Check if this is a section header
        if re.match(section_pattern, f"\n{part}\n", re.IGNORECASE):
            current_section = part.strip()
            continue

        # Chunk this section
        section_config = ChunkConfig(
            method="token",
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        section_chunks = _chunk_by_tokens(part, paper, section_config)

        # Update section metadata
        for chunk in section_chunks:
            chunk.metadata.section = current_section
            chunk.id = f"{paper.id}_{chunk_index}"
            chunk.metadata.chunk_index = chunk_index
            chunk_index += 1

        chunks.extend(section_chunks)

    return chunks


def _format_authors(authors: list) -> str | None:
    """Format authors list to string."""
    if not authors:
        return None

    names = []
    for author in authors:
        if hasattr(author, "family") and author.family:
            names.append(author.family)
        elif hasattr(author, "name"):
            names.append(author.name.split()[-1])

    if not names:
        return None

    if len(names) == 1:
        return names[0]
    elif len(names) == 2:
        return f"{names[0]} & {names[1]}"
    else:
        return f"{names[0]} et al."
