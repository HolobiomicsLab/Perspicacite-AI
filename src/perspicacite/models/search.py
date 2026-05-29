"""Search and retrieval models."""

from typing import Literal

from pydantic import BaseModel, Field

from perspicacite.models.documents import DocumentChunk
from perspicacite.models.papers import PaperSource


class SearchFilters(BaseModel):
    """Filters for search queries."""

    year_min: int | None = None
    year_max: int | None = None
    authors: list[str] | None = None
    journals: list[str] | None = None
    sources: list[PaperSource] | None = None
    has_full_text: bool | None = None
    # 2026-05-15: composite skill-bundle KBs tag each chunk with a
    # `source_skill` metadata field. Set this to restrict retrieval to
    # one skill inside a composite KB. See
    # docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md.
    source_skill: str | None = None

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
        if self.source_skill:
            filters.append(f"source_skill={self.source_skill}")
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
            self.source_skill is None,
        ])


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its relevance score."""

    model_config = {"extra": "allow"}  # Allow extra fields like wrrf_score

    chunk: DocumentChunk
    score: float = Field(le=1.0)
    retrieval_method: Literal["vector", "bm25", "hybrid", "reranked"] = "vector"

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
    filters: SearchFilters | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool = True

    def __repr__(self) -> str:
        return (
            f"SearchQuery(text='{self.text[:50]}...', "
            f"kb='{self.kb_name}', mode='{self.mode}')"
        )
