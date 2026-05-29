"""Document chunk models."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from perspicacite.models.papers import PaperSource


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""

    model_config = {"frozen": True}

    paper_id: str
    chunk_index: int
    section: str | None = None
    page_number: int | None = None
    source: PaperSource = PaperSource.BIBTEX
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    license_id: str | None = None  # OA license id, surfaced on passage results
    # Local-doc / smart-chunking extensions (all optional):
    content_type: str | None = None  # "pdf" | "markdown" | "code" | "text"
    language: str | None = None  # python | typescript | ...
    heading_path: list[str] | None = None  # markdown heading stack
    source_file_path: str | None = None  # absolute path for local files
    # ASB-aligned provenance (Cycle A 2026-05-13) — all optional, additive.
    source_section: str | None = None
    page: int | None = None
    char_span: tuple[int, int] | None = None
    figure_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    resource_refs: list[str] = Field(default_factory=list)
    parent_paper_id: str | None = None
    is_external: bool = False
    # Sub-project A (code-aware chunking) extensions — all optional.
    symbol_name: str | None = None
    symbol_kind: str | None = None  # "function" | "class" | "method" | "cell" | "module"
    parent_class: str | None = Field(
        None,
        description="If symbol_kind is a method, the enclosing class name. None otherwise.",
    )
    start_line: int | None = None   # 1-indexed inclusive
    end_line: int | None = None     # 1-indexed inclusive
    docstring: str | None = None    # ≤500 chars, truncated
    imports: list[str] = Field(default_factory=list)

    # Sub-project B (per-type embedding routing) — records which embedder
    # actually produced the chunk's vector. None when not yet embedded.
    embedding_model: str | None = None

    # Cite-graph enrichment fields (2026-05-15 spec).
    source_via: Literal["bundle", "enrichment", "cite_graph", "cite_graph_script"] | None = None
    cited_tool: str | None = None
    discovery_score: float | None = None

    # Carries the upstream ``Paper.metadata`` dict, JSON-encoded so it
    # round-trips through Chroma's scalar-only per-doc metadata. None
    # for non-bundle papers. Decoded back to a dict at the retrieval
    # boundary (see DynamicKnowledgeBase.search_two_pass).
    paper_metadata_json: str | None = None

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
    embedding: list[float] | None = None

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
