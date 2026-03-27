"""API request/response models."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from perspicacite.models.kb import ChunkConfig
from perspicacite.models.messages import Message
from perspicacite.models.papers import Paper
from perspicacite.models.rag import RAGMode, SourceReference


class ChatRequest(BaseModel):
    """Request for chat endpoint."""

    messages: list[Message]
    kb_name: str = "default"
    mode: RAGMode = RAGMode.AGENTIC
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    stream: bool = True
    use_web_search: bool = False
    conversation_id: Optional[str] = None
    max_iterations: Optional[int] = None


class ChatResponse(BaseModel):
    """Response from chat endpoint (non-streaming)."""

    message: Message
    sources: list[SourceReference] = Field(default_factory=list)
    conversation_id: str
    mode: RAGMode


class KBCreateRequest(BaseModel):
    """Request to create a knowledge base."""

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = None
    source_type: Literal["bibtex", "papers", "empty"] = "empty"
    source_path: Optional[str] = None  # BibTeX file path
    papers: Optional[list[str]] = None  # DOIs or URLs
    embedding_model: str = "text-embedding-3-small"
    chunk_config: ChunkConfig = Field(default_factory=ChunkConfig)


class KBAddPapersRequest(BaseModel):
    """Request to add papers to a KB."""

    papers: list[str]  # DOIs, URLs, or PDF paths
    auto_chunk: bool = True


class SearchRequest(BaseModel):
    """Request for literature search."""

    query: str
    apis: list[str] = Field(default_factory=lambda: ["semantic_scholar", "openalex", "pubmed"])
    max_results: int = Field(default=20, ge=1, le=100)
    year_min: Optional[int] = None
    year_max: Optional[int] = None


class SearchResponse(BaseModel):
    """Response from literature search."""

    papers: list[Paper]
    total_found: int
    apis_used: list[str]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str  # Machine-readable error code
    detail: str  # Human-readable message
    status_code: int  # HTTP status code
    request_id: str  # For debugging/support


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: dict[str, bool]


class InfoResponse(BaseModel):
    """System info response."""

    version: str
    available_providers: list[dict[str, Any]]
    available_kbs: list[str]
    config: dict[str, Any]  # Sanitized config (no secrets)
