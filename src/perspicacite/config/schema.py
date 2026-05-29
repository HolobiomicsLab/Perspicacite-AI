"""Pydantic models for configuration."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=5468, ge=1024, le=65535)
    reload: bool = False

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5468",
        ]
    )


class MCPConfig(BaseModel):
    """MCP server configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=5001, ge=1024, le=65535)
    transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http"


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: Path = Field(default=Path("~/.local/share/perspicacite/memory.db"))
    chroma_path: Path = Field(default=Path("./chroma_db"))

    @field_validator("path", "chroma_path")
    @classmethod
    def expand_home(cls, v: Path) -> Path:
        """Expand ~ to home directory."""
        return v.expanduser()


class CiteGraphConfig(BaseModel):
    """Cite-graph enrichment knobs (2026-05-15 spec)."""

    min_year_offset: int = Field(
        default=7, ge=0, le=100,
        description="Drop citing papers older than now - min_year_offset years.",
    )
    min_citations: int = Field(
        default=1, ge=0,
        description="Drop citing papers with fewer than this many citations.",
    )
    max_papers: int = Field(
        default=50, ge=1, le=1000,
        description="Hard cap on papers ingested per enrichment run.",
    )
    venue_denylist: list[str] = Field(
        default_factory=list,
        description="Venue/journal names to drop (e.g., predatory journals).",
    )
    include_scripts: bool = Field(
        default=False,
        description=(
            "When True, also pull ≤3 GitHub scripts per citing paper "
            "(deferred to follow-up; v1 ignores this flag)."
        ),
    )
    w_citations: float = Field(default=0.30, ge=0.0, le=1.0)
    w_recency:   float = Field(default=0.20, ge=0.0, le=1.0)
    w_oa:        float = Field(default=0.20, ge=0.0, le=1.0)
    w_match:     float = Field(default=0.30, ge=0.0, le=1.0)
    multi_source_bonus: float = Field(
        default=0.15, ge=0.0, le=0.5,
        description=(
            "Score bonus for citing papers confirmed by ≥2 of the three "
            "citation-graph sources (OpenAlex, Semantic Scholar, COCI). "
            "Rewards cross-validated citations without penalising COCI's "
            "lower recall."
        ),
    )


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base defaults."""

    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    chunking_method: Literal["token", "semantic", "agentic"] = "token"
    default_top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_two_pass: bool = Field(
        default=True, description="Enable two-pass retrieval for full paper context"
    )
    markdown_heading_aware: bool = Field(
        default=True,
        description="Use heading-aware chunking for markdown files",
    )
    code_language_aware: bool = Field(
        default=True,
        description="Use language-aware chunking for source-code files",
    )
    code_chunking: Literal["auto", "ast", "splitter"] = Field(
        default="auto",
        description=(
            "Code-chunking strategy. 'auto' prefers AST/Tree-sitter and "
            "falls back to the splitter. 'ast' fails loud (logs and falls "
            "back) when AST/TS is unavailable. 'splitter' keeps today's "
            "language-aware splitter behaviour."
        ),
    )
    # Contextual retrieval — three tiers, pick by cost / quality:
    #
    #   "none"     — structural prefix only (title + section), free.
    #                Already happens unconditionally.
    #   "abstract" — also prepend the paper's abstract (or first ~500
    #                chars of full_text) to every chunk of that paper.
    #                Zero LLM calls; uses Crossref/OpenAlex metadata
    #                we already have. Same prefix on every chunk of a
    #                paper.
    #   "summary"  — one LLM call per paper to produce a 50-100 word
    #                summary, cached, applied to every chunk of that
    #                paper. 1/N the cost of "chunk" for N chunks.
    #   "chunk"    — Anthropic-style: one LLM call per chunk produces
    #                a contextual sentence specific to that chunk.
    #                Most expensive, best recall (30-40% lift on
    #                technical content per Anthropic's benchmark).
    #
    # The `contextual_retrieval: bool` field below stays for backwards
    # compat — True maps to "chunk", False maps to "none". The new
    # `contextual_retrieval_tier` overrides when set.
    contextual_retrieval_tier: Literal["none", "abstract", "summary", "chunk"] = Field(
        default="none",
        description="Contextual-retrieval cost/quality tier.",
    )
    contextual_retrieval: bool = Field(
        default=False,
        description="(legacy) shorthand for contextual_retrieval_tier='chunk'.",
    )
    contextual_retrieval_model: str = Field(
        default="claude-haiku-4-5",
        description="Cheap fast model for contextual-retrieval prefix generation.",
    )
    contextual_retrieval_provider: str = Field(
        default="anthropic",
        description="Provider for the contextual-retrieval model.",
    )
    contextual_retrieval_max_chars: int = Field(
        default=400,
        ge=0,
        le=2000,
        description="Max chars of LLM-generated context prepended per chunk.",
    )

    # ---- embedding cache (Wave 2.2) --------------------------------
    # Cache embedding vectors keyed by (model, text). Embeddings are
    # deterministic per (model, text), so the cache is safe to keep
    # forever by default. See
    # docs/superpowers/specs/2026-05-14-embedding-cache-design.md.
    embedding_cache_enabled: bool = Field(
        default=True,
        description=(
            "Cache embedding vectors on disk so repeated ingests don't "
            "re-embed identical chunks. Default on; per-call bypass via "
            "provider.embed(..., cache=False)."
        ),
    )
    embedding_cache_path: Path = Field(
        default=Path("data/embedding_cache.db"),
        description=(
            "SQLite file backing the embedding cache. Covered by the "
            "data/*.db .gitignore rule."
        ),
    )
    embedding_cache_ttl_days: int = Field(
        default=0,
        ge=0,
        description=(
            "Days before a cached embedding expires. 0 = forever "
            "(default — embeddings are deterministic per model+text)."
        ),
    )

    # ---- checkpoint / resume (Wave 3.3) ----------------------------
    checkpoint_dir: Path = Field(
        default=Path("data/checkpoints"),
        description=(
            "Directory for ingest checkpoint files (Wave 3.3). "
            "Each multi-paper ingest writes <kb>__<op>.json here "
            "and removes it on clean completion."
        ),
    )

    # ---- per-KB append-only event log (Wave 4.3) -------------------
    log_dir: Path = Field(
        default=Path("data/kb_logs"),
        description=(
            "Directory for per-KB append-only event logs (Wave 4.3). "
            "Each KB writes <kb_name>.jsonl with paper_added / "
            "paper_skipped / paper_failed events for audit + rollback."
        ),
    )

    # ---- MCP resources (Wave 5.1) ----------------------------------
    mcp_resource_max_events: int = Field(
        default=1000,
        description=(
            "Max KB-log events returned by the "
            "perspicacite://kb/{name}/log MCP resource."
        ),
    )

    # ---- ORCID disambiguation (Wave 4.4) ---------------------------
    orcid_cache_path: Path = Field(
        default=Path("data/orcid_cache.db"),
        description=(
            "SQLite cache for name→ORCID resolutions. Covered by the "
            "data/*.db .gitignore rule."
        ),
    )
    orcid_cache_ttl_days: int = Field(
        default=30,
        ge=0,
        description="Days before a cached resolution expires. 0 = forever.",
    )
    orcid_confidence_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum (top1 - top2) / top1 spread between the best and "
            "second-best OpenAlex candidates. Below this, resolution "
            "returns None (ambiguous)."
        ),
    )

    # ---- multimodal visual extraction (Wave 4.1) -------------------
    # Render each PDF page and ask a vision-capable LLM to extract
    # figures / tables / formulas. Off by default — opt-in safety.
    # See docs/superpowers/specs/2026-05-14-multimodal-pdf-extraction-design.md.
    visual_extraction_enabled: bool = Field(
        default=False,
        description=(
            "When True, run MultimodalPDFExtractor on each ingested PDF "
            "to produce figure / table / formula chunks. Default off."
        ),
    )
    visual_extraction_model: str = Field(
        default="claude-sonnet-4-5",
        description="Vision-capable model used for extraction.",
    )
    visual_extraction_provider: str = Field(
        default="anthropic",
        description=(
            "Provider for the extraction model. Must support image "
            "content blocks (anthropic, openai, gemini, ...)."
        ),
    )
    visual_extraction_dpi: int = Field(
        default=150,
        ge=72,
        le=300,
        description=(
            "Page render DPI. Higher = clearer image, more tokens. "
            "150 is a good default for typical scientific PDFs."
        ),
    )

    library_paper_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional curated map of library name → canonical paper DOI. "
            "First lookup source for the cite-graph resolver. "
            "Example: {'openff-evaluator': '10.1021/acs.jctc.8b00640'}."
        ),
    )
    cite_graph: CiteGraphConfig = Field(default_factory=CiteGraphConfig)
    ingest_mode: Literal["auto", "full_text", "abstract_only"] = Field(
        default="auto",
        description=(
            "Content acquisition mode for KB ingestion.\n"
            "  'auto'          — current behaviour: try structured → PDF → abstract.\n"
            "  'full_text'     — fail papers that have no full text.\n"
            "  'abstract_only' — skip PDF/structured fetches entirely; use abstract\n"
            "                    from OpenAlex/Crossref discovery. ~80% faster for\n"
            "                    large corpora; retrieval depth is shallower."
        ),
    )


class CopyrightFilterConfig(BaseModel):
    """Runtime check on synthesis output to catch verbatim copies of
    source-paper text. Defense-in-depth on top of the safeguard prompt
    that already nudges the LLM to paraphrase.

    Modes:
    - ``log``: warn-only, answer returned unchanged (always-on default).
    - ``quote``: wrap spans in “…” + citation (no LLM call).
    - ``strip``: replace spans with ``[content paraphrased]``.
    - ``rewrite``: LLM paraphrases the flagged spans (one extra call).
    """

    enabled: bool = Field(default=True, description="Run the copyright filter on synthesis output.")
    action: Literal["log", "quote", "strip", "rewrite"] = Field(
        default="log",
        description="What to do when a verbatim span is detected.",
    )
    min_ngram: int = Field(
        default=8, ge=3, le=30,
        description="Minimum word-ngram length to flag as verbatim.",
    )
    rewrite_model: str = "claude-haiku-4-5"
    rewrite_provider: str = "anthropic"


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider.

    Most fields apply to LiteLLM-backed providers (anthropic, openai,
    deepseek, ...). The ``executable``/``prompt_via``/... block is
    used only by the subprocess agent-CLI providers (``agent_cli``
    and its preset ``claude_cli``); other providers ignore them.
    """

    base_url: str = ""
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)

    # ----- agent_cli subprocess fields (see llm/agent_cli.py) -----
    # Path or name of the binary to spawn. ``None`` means "not an
    # agent CLI provider" — LiteLLM handles this entry instead.
    executable: str | None = Field(
        default=None,
        description=(
            "Agent-CLI binary (e.g. 'claude', 'codex', 'openclaw'). "
            "When set, this provider routes through the subprocess "
            "agent_cli client instead of LiteLLM."
        ),
    )
    prompt_via: str = Field(
        default="stdin",
        description="How to deliver the prompt to the CLI: 'stdin' or 'arg'.",
    )
    prompt_flag: str | None = Field(
        default=None,
        description=(
            "When prompt_via='arg', the flag to put before the prompt "
            "(e.g. '--message'). ``None`` = pass as positional arg."
        ),
    )
    system_flag: str | None = Field(
        default=None,
        description="CLI flag for the system prompt, e.g. '--append-system-prompt'.",
    )
    model_flag: str | None = Field(
        default=None,
        description="CLI flag for the model name, e.g. '--model'.",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Always-appended args (e.g. ['-p', '--output-format', 'json']).",
    )
    output_format: str = Field(
        default="text",
        description="'text' to return stdout verbatim, 'json' to parse JSON.",
    )
    result_json_path: str | None = Field(
        default=None,
        description=(
            "Dotted JSON path for the assistant text "
            "(e.g. 'result' or 'message.content[0].text'). Only used "
            "when output_format='json'."
        ),
    )
    output_file_flag: str | None = Field(
        default=None,
        description=(
            "When set, allocate a tempfile per call and pass it via "
            "this flag (e.g. Codex's '--output-last-message'). Read "
            "the result from that file instead of stdout — cleaner "
            "for CLIs that pollute stdout with banner / progress."
        ),
    )
    usage_input_tokens_path: str | None = Field(
        default=None,
        description=(
            "Dotted JSON path to input-token count in the CLI's JSON "
            "output (e.g. 'usage.input_tokens'). Only used when "
            "output_format='json'. None → counts stay at 0 in "
            "provenance (today's behaviour). Wave 2.3."
        ),
    )
    usage_output_tokens_path: str | None = Field(
        default=None,
        description=(
            "Dotted JSON path to output-token count. Same rules as "
            "usage_input_tokens_path."
        ),
    )
    cost_usd_path: str | None = Field(
        default=None,
        description=(
            "F4 (audit 2026-05-15): dotted JSON path to a pre-computed "
            "total cost in USD (e.g. 'total_cost_usd' for Claude Code). "
            "When set, the agent_cli adapter feeds this number directly "
            "to BudgetTracker, bypassing the (provider, model) PRICING_TABLE "
            "lookup. Recommended for subscription-mode CLIs whose effective "
            "cost the CLI computes itself."
        ),
    )
    cache_read_tokens_path: str | None = Field(
        default=None,
        description=(
            "F4 (audit 2026-05-15): dotted JSON path to "
            "`cache_read_input_tokens` (Anthropic prompt-cache hits). "
            "Recorded into provenance for cache-hit-rate analysis. "
            "None → not recorded."
        ),
    )
    cache_creation_tokens_path: str | None = Field(
        default=None,
        description=(
            "F4 (audit 2026-05-15): dotted JSON path to "
            "`cache_creation_input_tokens` (cache writes)."
        ),
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory for the subprocess. ``None`` = inherit.",
    )
    env_extra: dict[str, str] = Field(
        default_factory=dict,
        description="Extra env vars for the subprocess.",
    )
    model_aliases: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map user-facing model names to CLI aliases. Substring "
            "match — so 'claude-sonnet': 'sonnet' collapses any "
            "claude-sonnet-* to 'sonnet'."
        ),
    )


class BudgetConfig(BaseModel):
    """Per-process LLM spend caps (Wave 2.4).

    Default off — set ``enabled: true`` to activate. When enabled,
    any breach raises ``BudgetExceededError`` (under ``action='abort'``)
    or logs a warning (under ``action='warn'``).
    """

    enabled: bool = Field(default=False, description="Master on/off switch.")
    max_input_tokens: int | None = Field(
        default=None, description="Total input tokens across all calls. None = no cap.",
    )
    max_output_tokens: int | None = Field(
        default=None, description="Total output tokens. None = no cap.",
    )
    max_usd: float | None = Field(
        default=None, description="Estimated dollar spend. None = no cap.",
    )
    action: Literal["abort", "warn"] = Field(
        default="abort",
        description="'abort' raises BudgetExceededError; 'warn' logs and continues.",
    )


class LLMConfig(BaseModel):
    """LLM configuration."""

    default_provider: str = "deepseek"
    default_model: str = "deepseek-chat"  # DeepSeek V3
    # v1 core/core.py get_response: truncate mandatory + base system prompt to this length (chars)
    max_context_window: int = Field(default=10000, ge=2000, le=500000)

    embedding_models_per_type: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional per-content-type embedding model routing. "
            "Keys are content types ('code', 'text', 'markdown', etc.); "
            "values are model strings passed to the embedding factory. "
            "When empty (default), every chunk goes through a single "
            "embedder selected from KnowledgeBaseConfig.embedding_model. "
            "Example: {'code': 'mistral/codestral-embed', 'text': "
            "'text-embedding-3-small'}."
        ),
    )

    # Per-stage model overrides. When set, the corresponding call site
    # uses this model/provider pair instead of (default_provider,
    # default_model). When ``None`` the stage falls back to the default
    # pair, preserving today's behaviour. This lets the user pay
    # Sonnet/Opus prices only on synthesis modes that benefit from
    # them, and use Haiku / a local Ollama for cheap roles.
    #
    # Recommended starting point (kept as a comment so we don't ship
    # an opinionated default that surprises existing users):
    #   default_provider: "anthropic"
    #   default_model:    "claude-sonnet-4-5"      # heavy synthesis (profound, agentic)
    #   models:
    #     synthesis_basic:    "claude-haiku-4-5"   # basic / advanced RAG modes
    #     routing:            "claude-haiku-4-5"   # kb_router LLM
    #     screening:          "claude-haiku-4-5"   # screen_papers LLM
    #     rephrase:           "claude-haiku-4-5"   # rephrase_query
    #     contextual:         "claude-haiku-4-5"   # contextual retrieval per-chunk
    #     search_optimize:    "claude-haiku-4-5"   # query optimizer
    #     grounding:          "claude-haiku-4-5"   # GUI grounding extractor
    #     claim_graph:        "deepseek/deepseek-v4-pro"  # claim extraction (structured output)
    models: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Per-stage model overrides. Keys: synthesis_basic, "
            "synthesis_heavy, routing, screening, rephrase, contextual, "
            "search_optimize, grounding, claim_graph. "
            "Value is the model name; uses default_provider unless the "
            "name contains a provider prefix like 'anthropic/claude-...'. "
            "Empty dict = every stage uses the global default pair."
        ),
    )
    # Optional per-stage provider override (parallel to ``models``).
    # Use when you want one stage on Ollama, another on Anthropic, etc.
    providers_per_stage: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-stage provider override. Value may be a single "
            "provider string (today's behaviour) or a list of "
            "providers — the client tries each in order on failure "
            "(see fallback-chain spec Wave 3.2). "
            "Same keys as `models`. Falls back to `default_provider`."
        ),
    )
    # MCP sampling: when True, LLM calls made inside MCP tool bodies
    # first try the connected client's sampling/createMessage protocol
    # (uses the client's subscription / credentials, not ours), then
    # fall back to the configured provider on failure. Off by default
    # because not every MCP client implements sampling — Claude Code
    # currently does not (anthropics/claude-code#1785), but Claude
    # Desktop has partial support. Turn on once you've tested with
    # your client.
    use_mcp_sampling: bool = Field(
        default=False,
        description=(
            "Try MCP sampling/createMessage first; fall back to "
            "default_provider on capability error. Only effective "
            "inside MCP tools that wrap their body with use_mcp_context."
        ),
    )
    default_timeout_s: float = Field(
        default=60.0,
        ge=1.0,
        description=(
            "Global LiteLLM call timeout in seconds, applied when the "
            "per-provider config has no timeout set and the caller does "
            "not pass a timeout= kwarg. Per-call kwargs always win."
        ),
    )
    budget: BudgetConfig = Field(
        default_factory=BudgetConfig,
        description="Per-process token / dollar caps. See BudgetConfig.",
    )
    pricing_overrides: dict[str, dict[str, list[float] | tuple[float, float]]] = Field(
        default_factory=dict,
        description=(
            "Optional per-(provider, model) pricing overrides in $/M tokens "
            "as [input, output]. Falls through to the default PRICING_TABLE "
            "in perspicacite.llm.budget."
        ),
    )

    # ---- free-tier auto-mode -------------------------------------------
    # When free_auto_mode=True and the caller does not specify an explicit
    # model/provider, complete() uses free_tier_fallback_models as the
    # *primary* chain rather than default_model. Models are tried in order;
    # on any error the next model in the list is attempted automatically.
    # Callers that pass an explicit model= still use that model (no override).
    # Requires OPENROUTER_API_KEY (a free account at openrouter.ai is enough
    # — no credits needed for :free-suffix models).
    free_auto_mode: bool = Field(
        default=False,
        description=(
            "When True, complete() without an explicit model uses "
            "free_tier_fallback_models as the primary rotation chain instead "
            "of default_model. Ideal for zero-cost deployments or as a safe "
            "default when no paid API key is available."
        ),
    )
    # Free-tier model pool — used in two ways:
    #  1. As the primary chain when free_auto_mode=True (tried in order, rotate on error).
    #  2. As a fallback chain when free_auto_mode=False and the primary fails with a
    #     quota-exceeded, invalid-model-ID, or auth error.
    # All entries are called via the 'openrouter' provider.
    # Requires OPENROUTER_API_KEY (free account is sufficient).
    # Recommended pool (best quality → broadest availability):
    #   - "deepseek/deepseek-v4-flash:free"           1M ctx, scientific text
    #   - "qwen/qwen3-coder:free"                     1M ctx, reasoning/code
    #   - "nvidia/nemotron-3-super-120b-a12b:free"    1M ctx, strong general
    #   - "meta-llama/llama-3.3-70b-instruct:free"   131K ctx, reliable
    #   - "openrouter/free"                           auto-picks best available
    free_tier_fallback_models: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of OpenRouter free-tier model IDs. Used as the "
            "primary rotation chain when free_auto_mode=True, or as a "
            "fallback chain when the primary model fails with a quota, "
            "invalid-model-ID, or auth error. All entries routed via "
            "'openrouter'. Requires OPENROUTER_API_KEY (free account ok)."
        ),
    )

    # ---- disk cache (Wave 2.1) -------------------------------------
    # Cache complete() responses on disk keyed by
    # (provider, model, messages, temperature, max_tokens). Pays back
    # on every dev iteration and on slow agent-CLI paths (6–16 s →
    # <10 ms). See docs/superpowers/specs/2026-05-14-llm-disk-cache-design.md.
    cache_enabled: bool = Field(
        default=True,
        description=(
            "Cache LLM responses on disk so repeated identical calls "
            "return instantly. Default on; bypass per-call with "
            "client.complete(..., cache=False)."
        ),
    )
    cache_path: Path = Field(
        default=Path("data/llm_cache.db"),
        description=(
            "SQLite file backing the cache. Created on first use. "
            "Already covered by the `data/*.db` .gitignore rule."
        ),
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=0,
        description=(
            "Cached responses expire after this many hours. 0 means "
            "never expire (kept until manually cleared)."
        ),
    )

    providers: dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {
            "anthropic": LLMProviderConfig(
                base_url="https://api.anthropic.com",
                timeout=120,
                max_retries=3,
            ),
            "openai": LLMProviderConfig(
                base_url="https://api.openai.com/v1",
                timeout=60,
                max_retries=3,
            ),
            "deepseek": LLMProviderConfig(
                base_url="https://api.deepseek.com",
                timeout=60,
                max_retries=3,
            ),
            "minimax": LLMProviderConfig(
                base_url="https://api.minimaxi.com/anthropic",  # Anthropic-compatible API for Chinese users
                timeout=120,
                max_retries=3,
            ),
            # Claude Code CLI subprocess provider — spawns `claude -p`
            # for each call. Uses the user's Pro/Max subscription.
            # The agent-CLI flag block is left empty because
            # ClaudeCLIClient bakes Claude Code's defaults in.
            "claude_cli": LLMProviderConfig(
                base_url="",
                timeout=180,
                max_retries=1,  # subprocess retries are pointless
                executable="claude",  # marks this as an agent_cli provider
            ),
            # Generic agent-CLI provider — fully config-driven.
            # Populate executable + flags in user config to point at
            # any single-shot completion CLI (Codex, OpenClaw, Hermes,
            # opencode, OpenHands, ...). Empty by default so callers
            # without a config get a clear error instead of trying to
            # exec a missing binary.
            "agent_cli": LLMProviderConfig(
                base_url="",
                timeout=180,
                max_retries=1,
            ),
        }
    )

    context: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_tokens": 8000,
            "chat_history_turns": 10,
            "summarize_threshold": 20,
        }
    )


class RAGModeSettings(BaseModel):
    """Settings for a RAG mode."""

    max_iterations: int = Field(default=1, ge=1)
    tools: list[str] = Field(default_factory=list)
    rerank: bool = True
    query_expansion: bool = False
    enable_planning: bool = False
    enable_reflection: bool = False
    build_citation_graph: bool = False
    use_hybrid: bool = False  # Use hybrid retrieval (vector + BM25)
    max_papers: int = Field(
        default=10, ge=1, le=50
    )  # Max papers to include in response (agentic mode)
    # v1 core: optional separate model for refine_response / evaluate_response
    evaluator_provider: str | None = None
    evaluator_model: str | None = None
    # v1 core refine_response / profonde (clamped in mode code to 1–3 where applicable)
    refinement_iterations: int = Field(default=2, ge=1, le=5)
    # v1 profonde: mid-cycle plan review after consecutive step failures
    enable_plan_review: bool = True
    # v1 get_response / profonde relevancy features
    use_relevancy_optimization: bool = True
    # Cap on per-paper LLM extraction calls during final map-reduce answer synthesis
    map_reduce_max_papers: int = Field(default=8, ge=1, le=64)
    # deep_research: allow live web / academic-API searches in research cycles.
    # Set to false for KB-only evaluation environments to prevent rate-limit errors.
    use_websearch: bool = True


class RAGModesConfig(BaseModel):
    """RAG mode configurations for benchmark comparison."""

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model used for reranking",
    )
    reranker_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for cross-encoder reranking of web-search results "
            "and agentic relevance scoring. Set False (or leave reranker_model "
            "empty) to disable. RECOMMENDED OFF when retrieval is driven by a "
            "strong instruction-tuned embedder (Qwen3-Embedding, OpenAI "
            "text-embedding-3-large, codestral-embed): empirically a general "
            "cross-encoder DEMOTES already-correct top hits from such embedders "
            "(SciFact dev: ms-marco CE −1.4pp / bge-reranker-v2-m3 −1.0pp on "
            "Qwen3-8B full; up to −4.7pp on QADS-masked vectors). Keep ON for "
            "weak local embedders like all-MiniLM-L6-v2, where CE adds ~+10pp. "
            "See docs/embedding_reranker_policy.md."
        ),
    )

    # KB auto-routing — triggered by `kb_name="auto"` in /api/chat or
    # MCP route_kbs(). The router scores each KB's description +
    # sampled paper titles against the query and picks top-N to query
    # in parallel via the multi-KB path.
    route_method: Literal["bm25", "llm"] = Field(
        default="bm25",
        description="bm25 = free + fast; llm = one cheap LLM call per route, better on semantics.",
    )
    route_top_k: int = Field(
        default=3, ge=1, le=10,
        description="Max KBs the router will return.",
    )
    route_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Drop KBs whose normalized score is below this.",
    )

    # Basic: Single query, no refinement, fastest
    basic: RAGModeSettings = Field(
        default_factory=lambda: RAGModeSettings(
            max_iterations=1,
            tools=["kb_search"],
            rerank=False,
            query_expansion=False,
            enable_planning=False,
            enable_reflection=False,
        )
    )

    # Advanced: Query rephrasing, WRRF scoring, optional refinement
    advanced: RAGModeSettings = Field(
        default_factory=lambda: RAGModeSettings(
            max_iterations=1,
            tools=["kb_search"],
            rerank=True,
            query_expansion=True,  # Generate similar queries
            enable_planning=False,
            enable_reflection=True,  # Optional refinement
            use_hybrid=True,  # Enable hybrid retrieval by default
        )
    )

    # Profond: Multi-cycle research with planning (from v1).
    # Default 1 cycle — users opt in to deeper research via the
    # per-request override in the composer.
    profound: RAGModeSettings = Field(
        default_factory=lambda: RAGModeSettings(
            max_iterations=1,
            tools=["kb_search", "web_search"],
            rerank=True,
            query_expansion=True,
            enable_planning=True,  # Research planning
            enable_reflection=True,  # Plan review and adjustment
        )
    )

    # deep_research is the renamed user-facing name for "profound" mode.
    # When None, engine falls back to the profound field for backward compat.
    deep_research: RAGModeSettings | None = Field(
        default=None,
        description=(
            "Settings for deep_research mode (renamed from profound). "
            "When None, falls back to the profound field for backward compat."
        ),
    )

    # Agentic: Intent-based with tool selection
    agentic: RAGModeSettings = Field(
        default_factory=lambda: RAGModeSettings(
            max_iterations=5,
            tools=["kb_search", "lotus_search", "literature_search", "fetch_pdf"],
            rerank=True,
            query_expansion=True,
            enable_planning=True,
            enable_reflection=True,
        )
    )

    # Contradiction: Identify agreement / disagreement / open questions across papers
    contradiction: RAGModeSettings = Field(
        default_factory=lambda: RAGModeSettings(
            max_iterations=1,
            tools=["kb_search"],
            rerank=True,
            query_expansion=True,
            enable_planning=False,
            enable_reflection=False,
            use_hybrid=True,
        )
    )


class SciLexAPIConfig(BaseModel):
    """Configuration for a SciLEx API."""

    enabled: bool = True
    rate_limit: int = Field(default=100, ge=1)


class SciLexConfig(BaseModel):
    """SciLEx integration configuration."""

    enabled: bool = True
    config_path: Path | None = None
    pubmed_email: str = ""

    apis: dict[str, SciLexAPIConfig] = Field(
        default_factory=lambda: {
            "semantic_scholar": SciLexAPIConfig(enabled=True, rate_limit=100),
            "openalex": SciLexAPIConfig(enabled=True, rate_limit=100),
            "pubmed": SciLexAPIConfig(enabled=True, rate_limit=10),
            "arxiv": SciLexAPIConfig(enabled=True, rate_limit=100),
            "ieee": SciLexAPIConfig(enabled=False, rate_limit=100),
            "springer": SciLexAPIConfig(enabled=False, rate_limit=100),
            "elsevier": SciLexAPIConfig(enabled=False, rate_limit=100),
            "hal": SciLexAPIConfig(enabled=True, rate_limit=100),
            "dblp": SciLexAPIConfig(enabled=True, rate_limit=100),
            "istex": SciLexAPIConfig(enabled=True, rate_limit=100),
        }
    )

    collection: dict[str, Any] = Field(
        default_factory=lambda: {
            "default_max_papers": 100,
            "quality_threshold": 0.7,
            "deduplicate": True,
            "download_pdfs": True,
        }
    )


class WebSearchConfig(BaseModel):
    """Web search configuration."""

    providers: list[str] = Field(default_factory=lambda: ["google_scholar", "semantic_scholar"])
    cache_ttl: int = Field(default=3600, ge=0)  # seconds


class ZoteroConfig(BaseModel):
    """Zotero integration configuration.

    Minimal config to enable: just ``enabled: true``. By default we talk
    to the desktop app's local API (no api_key needed). For the Zotero
    cloud API, set ``api_key`` and ``library_id`` and switch ``base_url``
    back to empty.
    """

    enabled: bool = False
    # API key — REQUIRED for cloud (api.zotero.org), OPTIONAL for the
    # local desktop API (base_url on loopback).
    api_key: str = ""
    # Library id — REQUIRED for cloud, OPTIONAL for the local API where
    # the client can auto-discover. Provide a specific id to scope to a
    # particular group library.
    library_id: str = ""
    library_type: str = "user"  # "user" or "group"
    collection_key: str = ""
    # Base URL for the Zotero API. Empty means cloud (api.zotero.org).
    # The default below targets the desktop app's local API so that
    # ``enabled: true`` alone is enough on a typical workstation. Set to
    # empty (``""``) to switch back to the cloud API; in that case
    # ``api_key`` and ``library_id`` become required.
    # The desktop checkbox is in Zotero → Settings → Advanced →
    # "Allow other applications on this computer to communicate with Zotero".
    base_url: str = "http://localhost:23119/api"


class LocalDocsConfig(BaseModel):
    """Server-side local-document ingestion configuration.

    `allowed_roots` is the allow-list for the `/api/kb/{name}/local-paths`
    endpoint and the `ingest_local_documents` MCP tool. If empty, those
    endpoints/tools refuse all calls (server-side path entry is disabled).
    The web multipart upload path (`/api/kb/{name}/local-files`) is
    unaffected.
    """

    allowed_roots: list[Path] = Field(default_factory=list)

    @field_validator("allowed_roots", mode="before")
    @classmethod
    def _expand_roots(cls, v: Any) -> list[Path]:
        if v is None:
            return []
        out: list[Path] = []
        for p in v:
            out.append(Path(p).expanduser().resolve())
        return out


class CapsuleConfig(BaseModel):
    """Per-paper capsule storage and build behaviour."""

    enabled: bool = True
    auto_build_on_ingest: bool = True
    root: Path = Path("./data/capsules")
    min_version: str = "0.1"


class MultimodalMode(str, Enum):
    """Multimodal RAG mode (sub-project C, 2026-05-15)."""
    OFF = "off"     # never attach figures to the LLM call
    AUTO = "auto"   # current behaviour: attach when chunk.figure_refs is non-empty
    FORCE = "force" # also pull top-N figures by caption relevance (v1: same as AUTO; force-mode retrieval ships in a follow-up)


class MultimodalConfig(BaseModel):
    """Multimodal RAG: figures-in-prompt + inline thumbnails in answers."""

    enabled: bool = True
    max_images: int = 6
    vision_allowlist: list[str] = Field(
        default_factory=lambda: [
            "anthropic/claude-",
            "claude-",
            "openai/gpt-4o",
            "gpt-4o",
        ]
    )
    mode: MultimodalMode = Field(
        default=MultimodalMode.AUTO,
        description=(
            "Multimodal retrieval mode. 'off' never attaches figures, "
            "'auto' attaches when retrieved chunks reference figures, "
            "'force' also pulls top-N by caption relevance. In v1, "
            "'force' is treated as 'auto' (caption-rank retrieval ships "
            "in a follow-up)."
        ),
    )
    show_code: bool = Field(
        default=False,
        description=(
            "When True, RAGResponse.code_excerpts is populated with "
            "AST-chunk excerpts from cited code chunks, each linked "
            "to its source URL (GitHub blob URL with line range)."
        ),
    )


class ExternalResourcesConfig(BaseModel):
    """V1 mining + V2 fetch-on-demand for paper-referenced external resources."""

    mine: bool = True                      # V1 — always-on (Cycle A wires this)
    fetch_on_demand: bool = True           # V2 — gated by user/MCP action
    cache_dir: Path = Path("./data/cache")
    cache_ttl_days: int = 30
    zenodo_max_bytes_per_file: int = 500_000
    zenodo_max_bytes_per_record: int = 5_000_000
    text_file_extensions: list[str] = Field(default_factory=lambda: [
        ".md", ".rst", ".txt",
        ".py", ".R", ".r", ".jl",
        ".ipynb",
        ".yml", ".yaml", ".toml", ".json", ".csv",
    ])


class PDFDownloadConfig(BaseModel):
    """PDF download configuration."""

    unpaywall_email: str | None = Field(
        default=None, description="Email for Unpaywall API. Required for querying open access PDFs."
    )
    alternative_endpoint: str | None = Field(
        default=None,
        description=(
            "Alternative endpoint for PDF downloads — a private or "
            "institutional repository the user maintains (campus proxy, "
            "internal aggregator of pre-cleared PDFs, on-prem PDF cache). "
            "User-provided; empty by default. Receives ``<base>/<doi>``."
        ),
    )

    # Publisher API keys for institutional access
    # Open access (no key needed)
    # - arXiv: fully open, no registration needed

    # Institutional access (API keys needed)
    wiley_tdm_token: str | None = Field(
        default=None,
        description="Wiley TDM (Text and Data Mining) API client token. Register at https://developer.wiley.com/",
    )
    elsevier_api_key: str | None = Field(
        default=None,
        description="Elsevier ScienceDirect API key. Register at https://dev.elsevier.com/",
    )
    aaas_api_key: str | None = Field(
        default=None, description="AAAS (Science) API key for institutional access."
    )
    rsc_api_key: str | None = Field(
        default=None,
        description="Royal Society of Chemistry API key. Register at https://api.rsc.org/",
    )
    springer_api_key: str | None = Field(
        default=None,
        description="Springer Nature API key. Register at https://dev.springernature.com/",
    )
    # ACS typically uses IP-based access, no API key

    semantic_scholar_api_key: str | None = Field(
        default=None,
        description="Semantic Scholar API key. Register at https://www.semanticscholar.org/product/api#api-key",
    )

    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    # Cookie jar for institutional-access PDF fetch.
    # Path to a Netscape-format cookies.txt exported from a browser
    # logged into the user's library proxy (e.g. via the "Get
    # cookies.txt" / "EditThisCookie" extension). When set, those
    # cookies are attached to every PDF-download request so the
    # publisher serves the entitled PDF over the user's session.
    # This is the server-side equivalent of what the Zotero Connector
    # extension does in the browser. Cookies expire — re-export as
    # needed.
    cookies_path: str | None = Field(
        default=None,
        description="Path to Netscape-format cookies.txt for institutional PDF access.",
    )
    cookie_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Only attach the cookie jar to requests whose host matches one "
            "of these substrings (e.g. ['sciencedirect.com', 'wiley.com', "
            "'proxy.lib.example.edu']). Empty list = attach to all PDF "
            "requests (broadest access, slight risk of cookie leakage to "
            "third-party hosts in the redirect chain)."
        ),
    )

    # PDF byte cache. When enabled, every successfully-downloaded PDF
    # is written to ``cache_dir`` keyed by DOI, and subsequent fetches
    # for the same DOI serve from disk instead of re-hitting the
    # publisher. Reduces network load on re-ingest, makes ingestion
    # idempotent + offline-replayable, and gives downstream tools
    # (Zotero attachment upload, export-kb) something to attach.
    cache_pdfs: bool = Field(
        default=True,
        description=(
            "Cache successfully-downloaded PDF bytes to disk. When True, "
            "re-ingesting the same DOI serves the cached file instead of "
            "re-fetching from the publisher."
        ),
    )
    cache_dir: str = Field(
        default="data/papers",
        description=(
            "Directory where cached PDFs live. Relative to the working "
            "directory unless absolute. Created on first write."
        ),
    )

    # When a PDF exceeds this size, push_to_zotero(..., attach_pdf=True)
    # skips the PDF upload and falls back to capturing the publisher
    # landing page as an HTML snapshot instead (much smaller, and the
    # user usually has the PDF locally anyway when it's a 50+ MB review
    # article). 0 disables the cap. Default 30 MB keeps Zotero free-tier
    # users (300 MB total) under-quota while still uploading typical
    # 1–5 MB articles unchanged.
    max_pdf_attach_bytes: int = Field(
        default=30 * 1024 * 1024,
        description=(
            "Skip PDF attachment in push_to_zotero when the cached PDF "
            "exceeds this size (in bytes). Falls back to HTML capture. "
            "Set 0 to disable."
        ),
    )


class GitHubConfig(BaseModel):
    """GitHub fetcher knobs (2026-05-15 spec).

    Controls how the GitHub-repo / skill-bundle ingest pipeline talks to
    api.github.com and caches downloaded tarballs on disk. See
    docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md.
    """

    token_env_var: str = Field(default="GITHUB_TOKEN", min_length=1)
    cache_dir: Path = Path("data/github_cache")
    cache_max_mb: int = Field(default=2048, gt=0)
    default_branch: str = Field(default="HEAD", min_length=1)
    user_agent: str = Field(default="Perspicacite/2.0", min_length=1)
    api_base: str = "https://api.github.com"

    @field_validator("api_base")
    @classmethod
    def _api_base_must_be_http(cls, v: str) -> str:
        """Reject non-HTTP(S) bases. ``AnyHttpUrl`` coerces to a URL
        object and breaks downstream string concatenation; a regex on
        ``str`` keeps the field shape stable."""
        if not isinstance(v, str) or not (
            v.startswith("https://") or v.startswith("http://")
        ):
            raise ValueError(
                f"api_base must start with http:// or https://, got {v!r}"
            )
        return v


class BundlesConfig(BaseModel):
    """Skill-bundle ingest knobs (2026-05-15 spec).

    KB-naming templates used by ``ingest_skill_bundle`` /
    ``ingest_skill_bundles_batch``. ``default_kb_name_template`` applies
    in per-skill mode (one KB per bundle); ``composite_kb_name_template``
    applies in composite mode (many bundles → one KB).
    """

    default_kb_name_template: str = "{name}"  # for per-skill mode
    composite_kb_name_template: str = "composite-{domain}"

    @field_validator("default_kb_name_template")
    @classmethod
    def _per_skill_template_requires_name(cls, v: str) -> str:
        """Per-skill mode formats this with ``name=<bundle.name>``. A
        template without ``{name}`` would silently produce the same KB
        name for every bundle and collapse them all into one collection."""
        if "{name}" not in v:
            raise ValueError(
                f"default_kb_name_template must contain '{{name}}', got {v!r}"
            )
        return v

    @field_validator("composite_kb_name_template")
    @classmethod
    def _composite_template_requires_domain(cls, v: str) -> str:
        """Composite mode formats this with ``domain=<user-supplied>``.
        Without ``{domain}`` the template can't differentiate composites
        for different research domains."""
        if "{domain}" not in v:
            raise ValueError(
                f"composite_kb_name_template must contain '{{domain}}', got {v!r}"
            )
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = True
    token: str | None = None  # Set via env: PERSPICACITE_AUTH_TOKEN


class UIConfig(BaseModel):
    """UI configuration."""

    theme: Literal["light", "dark", "system"] = "system"
    citation_format: Literal["nature", "apa", "mla", "ieee"] = "nature"


class QueryOptimizationConfig(BaseModel):
    """Settings for the LLM-assisted query rewrite step in literature search.

    The shared `query_optimizer` runs one Haiku call before the aggregator
    fan-out; the GUI-only grounding extractor runs one additional Haiku call
    to turn the prior assistant turn into a short context phrase (or None on
    topic pivot).
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Default for the per-call optimize_query argument. When False, "
            "search_literature and the basic RAG mode skip the rewrite step "
            "and pass the user's query verbatim."
        ),
    )
    timeout_s: float = Field(
        default=5.0, ge=0.5, le=30.0,
        description="Hard ceiling for the Haiku rewrite call (seconds).",
    )
    max_context_chars: int = Field(
        default=300, ge=0, le=2000,
        description=(
            "Truncation cap (head-keep) for the `context` parameter passed "
            "to the rewrite step."
        ),
    )
    grounding_enabled: bool = Field(
        default=True,
        description=(
            "Whether the GUI chat router runs the grounding extractor step "
            "before invoking the search path. Independent of `enabled` — set "
            "False to disable only the GUI auto-grounding."
        ),
    )
    grounding_timeout_s: float = Field(
        default=4.0, ge=0.5, le=30.0,
        description="Hard ceiling for the grounding-extractor Haiku call.",
    )
    grounding_max_prior_chars: int = Field(
        default=200, ge=0, le=2000,
        description="Truncation cap for the prior-turn excerpt fed to the extractor.",
    )
    grounding_max_query_chars: int = Field(
        default=200, ge=0, le=2000,
        description="Truncation cap for the new query fed to the extractor.",
    )


class SearchConfig(BaseModel):
    """Search provider routing configuration."""

    provider_timeout_s: float = Field(
        default=20.0, ge=1.0,
        description=(
            "Timeout (seconds) for 'reliable' tier providers. "
            "external = 1.5×, flaky = 2.25× this value."
        ),
    )
    max_results_per_provider: int = Field(
        default=25, ge=1, le=200,
        description="Max results fetched per provider before merge.",
    )
    enabled_providers: list[str] = Field(
        default_factory=list,
        description=(
            "Allowlist of provider names. Empty list = all registered "
            "providers enabled. Options: scilex, pubmed, europepmc, "
            "pubchem, core, inspire, ads."
        ),
    )
    core_api_key: str = Field(
        default="",
        description="CORE API v3 key (optional; raises rate limit when set).",
    )
    ads_api_key: str = Field(
        default="",
        description="NASA ADS token (required for ADS provider; skipped if absent).",
    )
    query_optimization: QueryOptimizationConfig = Field(
        default_factory=QueryOptimizationConfig,
        description="LLM-assisted query rewrite + GUI grounding extractor.",
    )


class CustomDatabase(BaseModel):
    """A user-defined database entry, surfaced in the composer DB picker.

    Display-only — favicon is fetched from `homepage` via the existing
    DatabaseGlyph component. Search integration is not auto-wired.
    """

    id: str = Field(..., description="Stable id used in selection lists.")
    label: str = Field(..., description="Human-readable name shown in tooltips.")
    short: str = Field(default="", description="Optional 2-char abbreviation for the glyph fallback.")
    homepage: str = Field(..., description="Base URL — favicon is auto-fetched from this domain.")
    blurb: str = Field(default="", description="Optional one-line description.")


class GoogleScholarConfig(BaseModel):
    """Google Scholar search via headless Chromium (optional [browser] dep)."""

    enabled: bool = Field(
        default=False,
        description=(
            "Enable Google Scholar provider. Requires `playwright` optional dep: "
            "`uv pip install -e \".[browser]\" && playwright install chromium`."
        ),
    )
    headless: bool = Field(
        default=True,
        description="Run Chromium headless. Set False for debugging.",
    )
    delay_seconds: float = Field(
        default=2.0, ge=0.5, le=30.0,
        description="Polite delay between requests (seconds). Do not lower below 1.0.",
    )
    max_results: int = Field(
        default=20, ge=1, le=50,
        description="Hard cap on results per search call.",
    )
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        description="Browser User-Agent string sent to Scholar.",
    )
    openrouter_fallback_enabled: bool = Field(
        default=False,
        description=(
            "Call OpenRouter web_search when Scholar returns a CAPTCHA. "
            "Requires openrouter_api_key or OPENROUTER_API_KEY env var."
        ),
    )
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key. Also read from OPENROUTER_API_KEY env var.",
    )
    openrouter_fallback_model: str = Field(
        default="deepseek/deepseek-chat",
        description=(
            "OpenRouter model for CAPTCHA fallback. 'deepseek/deepseek-chat' (DeepSeek-V3) "
            "is cheap and works with Exa-backed search. For native search use: "
            "'anthropic/claude-haiku-4-5' or 'openai/gpt-4o-mini'."
        ),
    )
    openrouter_fallback_domains: list[str] = Field(
        default_factory=lambda: [
            "arxiv.org",
            "biorxiv.org",
            "chemrxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "europepmc.org",
            "semanticscholar.org",
            "crossref.org",
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "wiley.com",
        ],
        description="Exa search restricted to these academic domains.",
    )
    serpapi_api_key: str = Field(
        default="",
        description=(
            "SerpApi key for the Google Scholar engine. When set (and the "
            "google_scholar provider is enabled), the reliable SerpApi backend "
            "is used instead of headless-Chromium scraping. Also read from "
            "SERPAPI_API_KEY / SERPAPI_KEY env vars. Free tier: 100 searches/mo."
        ),
    )


class AnchorConfig(BaseModel):
    """Quote-anchoring / claim-verification policy (R3)."""

    strict: bool = Field(
        default=False,
        description=(
            "When True, drop claims whose extracted quote cannot be verified "
            "against any source passage. Default False keeps them tagged "
            "(fail-open), mirroring ASB's --anchor-strict."
        ),
    )
    near_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0,
        description=(
            "Longest-common-substring ratio (over quote length) at or above "
            "which a non-exact match is accepted as 'repaired'."
        ),
    )
    audit_dir: str | None = Field(
        default=None,
        description=(
            "Directory for per-build anchor audit sidecars (JSONL). None "
            "disables the file sidecar; the structlog event is always emitted."
        ),
    )


class Config(BaseModel):
    """Main configuration for Perspicacité v2."""

    version: str = "2.0.0"
    config_name: str = "default"

    server: ServerConfig = Field(default_factory=ServerConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag_modes: RAGModesConfig = Field(default_factory=RAGModesConfig)
    scilex: SciLexConfig = Field(default_factory=SciLexConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    pdf_download: PDFDownloadConfig = Field(default_factory=PDFDownloadConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    zotero: ZoteroConfig = Field(default_factory=ZoteroConfig)
    local_docs: LocalDocsConfig = Field(default_factory=LocalDocsConfig)
    capsule: CapsuleConfig = Field(default_factory=CapsuleConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    external_resources: ExternalResourcesConfig = Field(default_factory=ExternalResourcesConfig)
    copyright_filter: CopyrightFilterConfig = Field(default_factory=CopyrightFilterConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    bundles: BundlesConfig = Field(default_factory=BundlesConfig)
    google_scholar: GoogleScholarConfig = Field(default_factory=GoogleScholarConfig)
    anchor: AnchorConfig = Field(default_factory=AnchorConfig)
    # User-defined databases shown in the composer's DB picker. These
    # are display-only: the frontend renders them with a favicon pulled
    # from `homepage`. Wiring a custom DB into the search pipeline is
    # a separate, provider-implementation concern.
    custom_databases: list["CustomDatabase"] = Field(
        default_factory=list,
        description=(
            "User-defined databases that appear in the composer DB picker. "
            "Each entry needs an id, label, and homepage URL — the favicon "
            "is auto-fetched from the homepage. Searches are not yet wired "
            "to custom entries; they're for visual/config presence only."
        ),
    )

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        """Validate configuration consistency."""
        # Ensure chunk_overlap < chunk_size
        if self.knowledge_base.chunk_overlap >= self.knowledge_base.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version string."""
        if not v.startswith("2."):
            raise ValueError("Config version must be 2.x")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, masking secrets."""
        d = self.model_dump()
        # Mask auth token if present
        if self.auth.token:
            d["auth"]["token"] = "***"
        return d
