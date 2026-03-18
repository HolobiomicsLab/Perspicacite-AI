"""RAG models."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from perspicacite.models.search import SearchFilters


class RAGMode(str, Enum):
    """RAG mode for different research depths."""

    QUICK = "quick"
    STANDARD = "standard"
    ADVANCED = "advanced"
    DEEP = "deep"
    CITATION = "citation"
    AGENTIC = "agentic"


class SourceReference(BaseModel):
    """Reference to a source paper."""

    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_text: Optional[str] = None

    def __repr__(self) -> str:
        return f"SourceReference(title='{self.title[:40]}...', score={self.relevance_score:.2f})"

    def to_citation(self, style: str = "nature") -> str:
        """Format as citation string."""
        author_part = self.authors or "Unknown"
        if "," in author_part:
            # Multiple authors, use et al.
            author_part = author_part.split(",")[0].strip() + " et al."
        year_part = f", {self.year}" if self.year else ""
        return f"[{author_part}{year_part}]"


class RAGRequest(BaseModel):
    """Request for RAG query."""

    query: str
    kb_name: str = "default"
    mode: RAGMode = RAGMode.STANDARD
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    max_iterations: Optional[int] = None
    use_web_search: bool = False
    filters: Optional[SearchFilters] = None
    conversation_id: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"RAGRequest(query='{self.query[:50]}...', "
            f"mode='{self.mode.value}', kb='{self.kb_name}')"
        )


class RAGResponse(BaseModel):
    """Response from RAG query."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    mode: RAGMode
    iterations: int = 1
    confidence: Optional[float] = None
    research_plan: Optional[list[str]] = None
    web_search_used: bool = False
    tokens_used: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"RAGResponse(mode='{self.mode.value}', "
            f"sources={len(self.sources)}, iterations={self.iterations})"
        )


class StreamEvent(BaseModel):
    """Structured SSE event for streaming responses."""

    event: Literal[
        "status",  # Processing status updates
        "content",  # Answer text delta
        "source",  # Source reference
        "reasoning",  # Chain-of-thought (Deep/Citation modes)
        "plan",  # Research plan step
        "tool_call",  # Tool invocation
        "tool_result",  # Tool result
        "error",  # Error message
        "done",  # Stream complete
    ]
    data: str  # JSON-encoded payload

    def __repr__(self) -> str:
        return f"StreamEvent(event='{self.event}', data='{self.data[:50]}...')"

    @classmethod
    def status(cls, message: str) -> "StreamEvent":
        """Create a status event."""
        import json

        return cls(event="status", data=json.dumps({"message": message}))

    @classmethod
    def content(cls, delta: str) -> "StreamEvent":
        """Create a content delta event."""
        import json

        return cls(event="content", data=json.dumps({"delta": delta}))

    @classmethod
    def done(
        cls,
        conversation_id: str,
        tokens_used: int,
        mode: str,
        iterations: int,
    ) -> "StreamEvent":
        """Create a done event."""
        import json

        return cls(
            event="done",
            data=json.dumps(
                {
                    "conversation_id": conversation_id,
                    "tokens_used": tokens_used,
                    "mode": mode,
                    "iterations": iterations,
                }
            ),
        )
