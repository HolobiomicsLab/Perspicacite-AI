"""Search and retrieval models."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from perspicacite.models.documents import DocumentChunk
from perspicacite.models.papers import PaperSource


class SearchFilters(BaseModel):
    """Filters for search queries."""

    year_min: Optional[int] = None
    year_max: Optional[int] = None
    authors: Optional[list[str]] = None
    journals: Optional[list[str]] = None
    sources: Optional[list[PaperSource]] = None
    has_full_text: Optional[bool] = None

    def __repr__(self) -> str:
        filters = []
        if self.year_min:
            filters.append(f"year>={self.year_min}")
        if self.year_max:
            filters.append(f"year<={self.year_max}")
        if self.authors:
            filters.append(f"authors={self.authors}")
        if self.journals:
            filters.append(f"journals={self.journals}")
        return f"SearchFilters({', '.join(filters)})"

    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return all([
            self.year_min is None,
            self.year_max is None,
            self.authors is None,
            self.journals is None,
            self.sources is None,
            self.has_full_text is None,
        ])


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its relevance score."""

    chunk: DocumentChunk
    score: float = Field(le=1.0)
    retrieval_method: Literal["vector", "bm25", "hybrid"] = "vector"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(score={self.score:.3f}, "
            f"method='{self.retrieval_method}', "
            f"chunk_id='{self.chunk.id}')"
        )


class SearchQuery(BaseModel):
    """A search query."""

    text: str
    kb_name: str = "default"
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    filters: Optional[SearchFilters] = None
    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True

    def __repr__(self) -> str:
        return (
            f"SearchQuery(text='{self.text[:50]}...', "
            f"kb='{self.kb_name}', mode='{self.mode}')"
        )
