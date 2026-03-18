"""Document chunk models."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from perspicacite.models.papers import PaperSource


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    model_config = {"frozen": True}

    paper_id: str
    chunk_index: int
    section: Optional[str] = None
    page_number: Optional[int] = None
    source: PaperSource = PaperSource.BIBTEX
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"ChunkMetadata(paper_id='{self.paper_id}', "
            f"chunk_index={self.chunk_index}, section='{self.section}')"
        )


class DocumentChunk(BaseModel):
    """A chunk of a document with metadata."""

    model_config = {"frozen": False}

    id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None

    def __repr__(self) -> str:
        text_preview = self.text[:50].replace("\n", " ")
        return f"DocumentChunk(id='{self.id}', text='{text_preview}...')"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.model_dump(),
            "embedding": self.embedding,
        }
