"""RAG models."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from perspicacite.models.search import SearchFilters


class RAGMode(str, Enum):
    """RAG modes for benchmark comparison.

    BASIC: Simple retrieval + generation (single query, no refinement)
    ADVANCED: Query rephrasing + hybrid retrieval + WRRF scoring + optional refinement
    DEEP_RESEARCH: Multi-cycle research with planning, web search, reflection (canonical name)
    PROFOUND: Deprecated alias for DEEP_RESEARCH — kept for backward compat
    AGENTIC: Intent-based agentic RAG with tool use
    LITERATURE_SURVEY: Systematic field mapping with theme identification and structured output
    CONTRADICTION: Identify agreement / disagreement / open questions across papers
    REASONING: Indicium claim-graph reasoning (knob: reasoning_strategy)
    """

    BASIC = "basic"
    ADVANCED = "advanced"
    DEEP_RESEARCH = "deep_research"    # canonical name
    PROFOUND = "profound"              # deprecated alias — kept for backward compat
    AGENTIC = "agentic"
    LITERATURE_SURVEY = "literature_survey"
    CONTRADICTION = "contradiction"
    REASONING = "reasoning"


class SourceReference(BaseModel):
    """Reference to a source paper."""

    # Allow both field name and alias for construction / serialisation.
    # populate_by_name=True means callers can use either "discovery_sources"
    # (new canonical name) or "sources_all" (legacy alias) interchangeably.
    model_config = ConfigDict(populate_by_name=True)

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    url: str | None = None
    # Which provider returned this paper (e.g. "google_scholar", "openalex",
    # "core", "europe_pmc", "scilex"). None for KB-sourced papers. Also
    # surfaced in the UI's source card next to the "details" button.
    source: str | None = None
    # When ``source`` is the SciLEx meta-wrapper, this lists the APIs that
    # were actually queried (e.g. ["semantic_scholar", "openalex", "pubmed"]).
    # SciLEx doesn't expose per-paper provenance so we can only say which
    # APIs were called, not which one returned this specific paper.
    source_apis: list[str] | None = None
    # Metadata enrichment provenance — which secondary sources contributed
    # to *enriching* this record after the initial search hit. Common
    # values: "crossref" (canonical bibliographic patch), "openalex"
    # (abstract / OA URL fill-in), "unpaywall" (PDF availability + OA
    # status). Distinct from ``discovery_sources`` (which databases RETURNED
    # the paper) vs ``enrichment_sources`` (which databases CLEANED IT UP).
    # Renders as a separate chip group with a subtle visual treatment.
    enrichment_sources: list[str] | None = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_text: str | None = None
    # Full abstract text, used by the paper-detail side panel for
    # Google-Scholar / web-fallback papers without a DOI (the side
    # panel can't fall back to a /api/paper lookup in that case). Kept
    # separate from ``chunk_text`` which is sometimes a truncated chunk
    # rather than a full abstract.
    abstract: str | None = None
    kb_name: str | None = None
    # The internal paper_id as stored in the KB (e.g. "scifact:4983") or the
    # paper's canonical external identifier.  Allows SSE clients (eval harness,
    # UI) to identify KB papers that have no DOI.  None for external-only
    # results that were never ingested into a KB.
    paper_id: str | None = None
    # All upstream providers that returned THIS specific paper (deduped).
    # Renamed from legacy ``sources_all`` → ``discovery_sources`` (matches
    # Paper.discovery_sources). The old name lives on as a Pydantic alias
    # so existing JSON payloads (and JS reading src.sources_all when
    # dumped with by_alias=True) keep working until UI catches up.
    discovery_sources: list[str] | None = Field(
        default=None,
        alias="sources_all",
        description="DBs that returned this paper (deduped). Multi-DB matches render as a chip group.",
    )
    # Carries the underlying paper's ``Paper.metadata`` dict (or chunk
    # metadata) as a free-form mapping. Surfaces ASB skill / workflow-card
    # fields to the response builders (build_asb_response_metadata).
    metadata: dict[str, Any] | None = None

    @field_validator("authors", mode="before")
    @classmethod
    def _coerce_authors(cls, v):
        """Accept str (legacy), None, or list — always store list[str].

        Audit 2026-05-15 finding #2: the field was previously
        ``Optional[str]`` which broke construction from upstream
        ``normalize_paper_dict`` (returns ``list[str]``). This validator
        keeps backward compat for the comma-joined-string call sites
        that pre-dated the fix.
        """
        if v is None:
            return []
        if isinstance(v, list):
            return [str(a).strip() for a in v if str(a).strip()]
        if isinstance(v, str):
            # Split on " and " (BibTeX-style) then commas.
            parts: list[str] = []
            for chunk in v.replace(" and ", ",").split(","):
                chunk = chunk.strip()
                if chunk:
                    parts.append(chunk)
            return parts
        return [str(v)]

    def __repr__(self) -> str:
        return f"SourceReference(title='{self.title[:40]}...', score={self.relevance_score:.2f})"

    def to_citation(self, style: str = "nature") -> str:
        """Format as citation string."""
        if not self.authors:
            author_part = "Unknown"
        elif len(self.authors) == 1:
            author_part = self.authors[0]
        else:
            author_part = f"{self.authors[0]} et al."
        year_part = f", {self.year}" if self.year else ""
        return f"[{author_part}{year_part}]"


class RAGRequest(BaseModel):
    """Request for RAG query."""

    # extra="allow" lets callers attach transient runtime fields like
    # ``telemetry_sink`` without subclassing. These extra fields are never
    # serialised / validated, keeping the schema stable.
    # arbitrary_types_allowed=True because app_state holds an AppState /
    # MinimalAppState instance that is not itself a Pydantic model.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    query: str
    kb_name: str = "default"
    mode: RAGMode = RAGMode.BASIC
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    max_iterations: int | None = None
    use_web_search: bool = False
    filters: SearchFilters | None = None
    conversation_id: str | None = None
    refined_query: str | None = None
    kb_scope: str | None = None
    # v1: optional evaluator LLM (same client; different model/provider per call)
    evaluator_provider: str | None = None
    evaluator_model: str | None = None
    databases: list[str] | None = Field(
        default=None,
        description=(
            "List of databases to search. None means the legacy default "
            "(semantic_scholar, openalex, pubmed) — every call site already "
            "guards with ``request.databases or [...]``."
        ),
    )
    conversation_history: list[dict[str, str]] | None = Field(
        default=None,
        description="Recent chat turns (role/content) for query rewrite and generation context",
    )
    max_papers_retrieval: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Hard cap on papers loaded in two-pass; None uses mode default",
    )
    bm25_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    vector_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    recency_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    recency_half_life_years: float | None = Field(default=None, gt=0.0)
    kb_names: list[str] | None = None
    reasoning_strategy: (
        Literal["provenance", "contradiction", "graph", "evidence_graded"] | None
    ) = Field(
        default=None,
        description=(
            "Strategy for RAGMode.REASONING. None = strongest shipped layer "
            "(see rag/modes/reasoning/__init__.py). Unshipped strategies raise "
            "NotImplementedError pointing to the planned sprint."
        ),
    )
    task_id: str | None = Field(
        default=None,
        description="Optional task ID for MCP cancellation tracking",
    )
    app_state: Any = Field(
        default=None,
        description="AppState / MinimalAppState; threaded by RAGEngine",
        exclude=True,
    )

    # === Per-call overrides for budget / parallelism ===
    # Each is None by default, in which case the mode uses its
    # config-file default. Bounded to safe ranges.
    max_total_seconds: float | None = Field(
        default=None, ge=30.0, le=1800.0,
        description="Overrides per-mode max_total_seconds (30-1800s)",
    )
    batch_size: int | None = Field(
        default=None, ge=1, le=100,
        description="Overrides literature_survey batch_size (1-100)",
    )
    crossref_concurrency: int | None = Field(
        default=None, ge=1, le=10,
        description="Overrides Crossref enrichment concurrency (1-10)",
    )

    # === Per-call screening / download knobs ===
    # Each is None by default; modes that perform screening fall back to
    # their existing config defaults. Clamping happens at the MCP boundary
    # (single source of truth) so internal callers are trusted.
    screen_method: str | None = Field(
        default=None,
        description='Screening method: "bm25" | "rerank" | "llm". None = mode default.',
    )
    screen_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Relevance threshold in [0, 1]. None = mode default.",
    )
    max_papers_to_download: int | None = Field(
        default=None, ge=1, le=50,
        description="Hard cap on papers downloaded for full-text analysis (1-50).",
    )
    # max_iterations already exists; existing validator stays.

    use_hyde: bool = Field(
        default=False,
        description=(
            "When True, generate a hypothetical paper abstract (HyDE) from the "
            "claim before vector search. Bridges colloquial claim language to "
            "domain-specific paper vocabulary. Off by default. Only active in "
            "basic mode; other modes ignore this flag."
        ),
    )

    # === Per-call retrieval-strategy overrides ===
    # None = use the mode/config default; True/False overrides for this call.
    # Rationale: the optimal choice depends on the KB's embedding strength —
    # weak embedders (MiniLM) benefit from BM25 hybrid + cross-encoder rerank,
    # strong instruction-tuned embedders (Qwen3, codestral-embed, OpenAI-3-large)
    # are hurt by both. A single server hosting mixed-strength KBs can flip these
    # per request instead of needing one server per profile.
    # See docs/embedding_reranker_policy.md.
    use_hybrid: bool | None = Field(
        default=None,
        description=(
            "Override hybrid (vector+BM25) retrieval for this call. None = mode "
            "default (rag_modes.<mode>.use_hybrid). False = pure vector — "
            "recommended for strong instruction-tuned embedders."
        ),
    )
    use_reranker: bool | None = Field(
        default=None,
        description=(
            "Override cross-encoder reranking of web-search results for this "
            "call. None = server default (rag_modes.reranker_enabled). False = "
            "no rerank — recommended for strong instruction-tuned embedders."
        ),
    )

    def __repr__(self) -> str:
        return (
            f"RAGRequest(query='{self.query[:50]}...', "
            f"mode='{self.mode.value}', kb='{self.kb_name}')"
        )


class FigureRef(BaseModel):
    """A figure attached to a RAG response for display in the GUI / MCP."""
    id: str
    paper_id: str
    label: str | None = None      # e.g. "Figure 3"
    caption: str | None = None
    source_url: str | None = None  # paper DOI / page URL
    page: int | None = None
    thumbnail_b64: str | None = None  # small base64 PNG for inline display


class CodeExcerpt(BaseModel):
    """A code-chunk excerpt attached to a RAG response (sub-project C)."""
    id: str                            # e.g. "github:owner/repo@SHA:path#Lstart-Lend"
    paper_id: str
    file_path: str
    symbol_name: str | None = None  # None for module chunks
    symbol_kind: str                   # "function" | "class" | "method" | "cell" | "module"
    language: str                      # "python" | "r" | etc.
    start_line: int
    end_line: int
    text: str
    source_url: str                    # e.g. GitHub blob URL with #L<s>-L<e>


class RAGResponse(BaseModel):
    """Response from RAG query."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    mode: RAGMode
    iterations: int = 1
    confidence: float | None = None
    research_plan: list[str] | None = None
    web_search_used: bool = False
    tokens_used: int | None = None
    figures: list[FigureRef] = Field(default_factory=list)
    code_excerpts: list[CodeExcerpt] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

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
        "code_excerpt",  # Code-chunk attachment (sub-project C, 2026-05-15)
        "figure_ref",  # Figure-reference attachment (sub-project C, 2026-05-15)
        "metadata",  # Run summary: iteration_count, completion_reason
        "diagnostic",  # Early-return diagnostics: papers_searched, filtered_as_known, …
        "revision",  # Post-answer corrected text (e.g. copyright filter); client may render or ignore
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
    def status_kind(cls, message: str, kind: str, **extra: Any) -> "StreamEvent":
        """Create a structured status event with a discriminator ``kind``.

        The chat router forwards status events as-is (status_data is spread
        into the SSE payload), so adding fields here lets the frontend
        render rich cards (query rephrasing, provider progress, batch
        progress) without expanding the StreamEvent literal type.

        Conventions:
        - ``kind="query_rephrased"`` → extras: ``original``, ``rewritten``,
          ``by`` ("conversation_history" | "keyword_optimizer")
        - ``kind="provider_progress"`` → extras: ``phase`` ("start"|"done"),
          ``provider``, optional ``count``
        - ``kind="batch_progress"`` → extras: ``current``, ``total``,
          ``stage``
        """
        import json

        payload: dict[str, Any] = {"message": message, "kind": kind}
        payload.update(extra)
        return cls(event="status", data=json.dumps(payload))

    @classmethod
    def content(cls, delta: str) -> "StreamEvent":
        """Create a content delta event."""
        import json

        return cls(event="content", data=json.dumps({"delta": delta}))

    @classmethod
    def source(cls, source: "SourceReference") -> "StreamEvent":
        """Create a source event."""
        import json

        return cls(event="source", data=json.dumps(source.model_dump()))

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

    @classmethod
    def code_excerpt(cls, payload: dict) -> "StreamEvent":
        """Create a code-excerpt event (sub-project C)."""
        import json
        return cls(event="code_excerpt", data=json.dumps(payload))

    @classmethod
    def figure_ref(cls, payload: dict) -> "StreamEvent":
        """Create a figure-ref event (sub-project C)."""
        import json
        return cls(event="figure_ref", data=json.dumps(payload))

    @classmethod
    def metadata(cls, **payload: Any) -> "StreamEvent":
        """Create a metadata event carrying run summary fields.

        Typical fields: ``iteration_count``, ``completion_reason``.
        """
        import json
        return cls(event="metadata", data=json.dumps(payload))

    @classmethod
    def diagnostic(cls, **payload: Any) -> "StreamEvent":
        """Create a diagnostic event for early-return paths.

        Typical fields: ``papers_searched``, ``filtered_as_known``,
        ``kb_chunks_found``, ``cancellation_reason``.
        """
        import json
        return cls(event="diagnostic", data=json.dumps(payload))
