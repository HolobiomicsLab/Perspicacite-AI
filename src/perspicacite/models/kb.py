"""Knowledge base models."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChunkConfig(BaseModel):
    """Configuration for text chunking."""

    method: Literal["token", "semantic", "section_aware"] = "token"
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    preserve_sections: bool = True

    def __repr__(self) -> str:
        return (
            f"ChunkConfig(method='{self.method}', "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )


class KnowledgeBase(BaseModel):
    """A knowledge base of papers."""

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = None
    collection_name: str  # Chroma collection name
    embedding_model: str = "text-embedding-3-small"
    chunk_config: ChunkConfig = Field(default_factory=ChunkConfig)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    paper_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)

    def __repr__(self) -> str:
        return (
            f"KnowledgeBase(name='{self.name}', "
            f"papers={self.paper_count}, chunks={self.chunk_count})"
        )


class KBStats(BaseModel):
    """Statistics for a knowledge base."""

    name: str
    description: Optional[str] = None
    paper_count: int
    chunk_count: int
    embedding_model: str
    created_at: datetime
    updated_at: datetime
    size_mb: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"KBStats(name='{self.name}', "
            f"papers={self.paper_count}, size_mb={self.size_mb:.1f})"
        )
