"""Application state for the web app.

Holds the singleton instance of `AppState` that all routers share. Routers
import `app_state` from this module; the FastAPI lifespan calls
`await app_state.initialize()` once on startup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from perspicacite.jobs.registry import JobRegistry
from perspicacite.memory.session_store import SessionStore
from perspicacite.provenance.store import ProvenanceStore

if TYPE_CHECKING:
    # Imported only for type annotations. The real imports happen lazily inside
    # initialize() to avoid import cycles and heavy startup imports; `from
    # __future__ import annotations` keeps these references string-only at
    # runtime, so there's no cost or cycle here.
    from perspicacite.llm import AsyncLLMClient
    from perspicacite.llm.embeddings import EmbeddingProvider
    from perspicacite.pipeline.download import PDFDownloader
    from perspicacite.pipeline.parsers.pdf import PDFParser
    from perspicacite.rag.agentic import AgenticOrchestrator
    from perspicacite.rag.engine import RAGEngine
    from perspicacite.retrieval import ChromaVectorStore

logger = logging.getLogger(__name__)


# F-31: Map provider name -> env var that must be present for the provider to
# function. Used by the startup preflight in AppState.initialize. We don't
# guess models or API URLs here; that's the provider's job. We only refuse to
# serve when the default provider's auth credential is missing.
_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic":      "ANTHROPIC_API_KEY",
    "openrouter":     "OPENROUTER_API_KEY",
    "openai":         "OPENAI_API_KEY",
    "deepseek":       "DEEPSEEK_API_KEY",
    "minimax":        "MINIMAX_API_KEY",
}


def _preflight_llm_keys(config: Any) -> None:
    """Refuse to serve when the configured default LLM provider's API key is
    missing from the process environment.

    Operators can bypass with PERSPICACITE_ALLOW_MISSING_LLM_KEYS=1 (for
    offline dev / mocked-LLM tests). Bypass logs a loud WARN so the choice
    is visible in the startup log.
    """
    llm_cfg = getattr(config, "llm", None)
    if llm_cfg is None:
        return
    default_provider = (llm_cfg.default_provider or "anthropic").lower()
    required_env = _PROVIDER_KEY_ENV.get(default_provider)
    if not required_env:
        # Unknown provider name — let the provider itself complain later.
        return
    if os.environ.get(required_env):
        return
    if os.environ.get("PERSPICACITE_ALLOW_MISSING_LLM_KEYS"):
        logger.warning(
            "preflight_llm_key_missing_bypassed provider=%s env=%s — calls will fail",
            default_provider, required_env,
        )
        return
    raise RuntimeError(
        f"LLM preflight failed: default_provider='{default_provider}' but "
        f"{required_env} is not set in the environment. Export the key (in "
        f"~/.zshrc or the launching shell); if it's already set there, run "
        f"`source ~/.zshrc` (or `source ~/.profile`) in this shell. Or set "
        f"PERSPICACITE_ALLOW_MISSING_LLM_KEYS=1 to bypass for offline / "
        f"mocked-LLM development."
    )


class AppState:
    """Application state with agentic orchestrator and RAG engine."""

    def __init__(self, config_path: str | None = None):
        # Required components: None until initialize() runs, non-None forever
        # after. Exposed via properties below that return the concrete type and
        # raise if read before init — so call sites don't carry `| None` noise
        # and a pre-init access fails loudly instead of as an opaque NoneType
        # AttributeError deep in a request handler.
        self._llm_client: AsyncLLMClient | None = None
        self._vector_store: ChromaVectorStore | None = None
        self._rag_engine: RAGEngine | None = None  # Multi-mode RAG engine
        self._pdf_downloader: PDFDownloader | None = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._pdf_parser: PDFParser | None = None
        # Genuinely-optional components: may legitimately stay None (degraded
        # mode) and are guarded with `is None` at their use sites — these stay
        # plain attributes so `if not app_state.session_store:` checks work.
        self.session_store: SessionStore | None = None
        self.provenance_store: ProvenanceStore | None = None
        self.job_registry: JobRegistry | None = None
        # Set in initialize(), but kept a plain `| None` attribute (not a raising
        # property) because tests monkeypatch `app_state.orchestrator` directly —
        # a raising getter breaks monkeypatch's save-old-value read. Call sites
        # guard with `is None`.
        self.orchestrator: AgenticOrchestrator | None = None
        self.initialized = False
        # Config path — set by CLI before lifespan fires so AppState reads
        # the same config file as the CLI (previously it re-called load_config()
        # with no path and always loaded the default config.yml).
        self._config_path: str | None = config_path

    # --- Required-component accessors -------------------------------------
    # Each returns the concrete component (never None) once initialize() has
    # run. Setters accept the value (used by initialize() and by tests that
    # inject mocks). Reading before init raises a clear RuntimeError.
    def _require(self, value: Any, name: str) -> Any:
        if value is None:
            raise RuntimeError(
                f"AppState.{name} accessed before initialize() — the FastAPI "
                f"lifespan must run AppState.initialize() before handlers use it."
            )
        return value

    @property
    def llm_client(self) -> AsyncLLMClient:
        return self._require(self._llm_client, "llm_client")

    @llm_client.setter
    def llm_client(self, value: AsyncLLMClient | None) -> None:
        self._llm_client = value

    @property
    def vector_store(self) -> ChromaVectorStore:
        return self._require(self._vector_store, "vector_store")

    @vector_store.setter
    def vector_store(self, value: ChromaVectorStore | None) -> None:
        self._vector_store = value

    @property
    def rag_engine(self) -> RAGEngine:
        return self._require(self._rag_engine, "rag_engine")

    @rag_engine.setter
    def rag_engine(self, value: RAGEngine | None) -> None:
        self._rag_engine = value

    @property
    def pdf_downloader(self) -> PDFDownloader:
        return self._require(self._pdf_downloader, "pdf_downloader")

    @pdf_downloader.setter
    def pdf_downloader(self, value: PDFDownloader | None) -> None:
        self._pdf_downloader = value

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._require(self._embedding_provider, "embedding_provider")

    @embedding_provider.setter
    def embedding_provider(self, value: EmbeddingProvider | None) -> None:
        self._embedding_provider = value

    @property
    def pdf_parser(self) -> PDFParser:
        return self._require(self._pdf_parser, "pdf_parser")

    @pdf_parser.setter
    def pdf_parser(self, value: PDFParser | None) -> None:
        self._pdf_parser = value

    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return

        logger.info("Initializing Perspicacité v2 Agentic System...")

        # Load config — use the path set by the CLI so multi-server setups
        # (e.g. different ports / embedding models) work correctly.
        from perspicacite.config.loader import load_config
        from perspicacite.llm import AsyncLLMClient
        from perspicacite.llm.embeddings import create_embedding_provider
        from perspicacite.rag.agentic import AgenticOrchestrator, LLMAdapter
        from perspicacite.rag.tools import ToolRegistry
        from perspicacite.retrieval import ChromaVectorStore

        config = load_config(self._config_path)

        # F-31: Preflight check — refuse to serve when the configured default
        # LLM provider's API key isn't in env. Silent 401s deep in the RAG
        # pipeline are the second-worst kind of bug (after silent 200s).
        _preflight_llm_keys(config)

        # Initialize LLM
        self.llm_client = AsyncLLMClient(config.llm)
        logger.info("LLM client initialized")

        # Initialize embeddings (uses factory so local sentence-transformers
        # models like "all-MiniLM-L6-v2" work without an OpenAI key, and
        # API-based models still get a local fallback).
        self.embedding_provider = create_embedding_provider(
            model=config.knowledge_base.embedding_model,
        )
        logger.info("Embedding provider initialized")

        # Pre-warm the cross-encoder reranker if it isn't cached yet.
        # First-query loads otherwise pay a one-off download (~90 MB for
        # ms-marco-MiniLM) AND can trigger huggingface.co rate-limit retry
        # loops (HEAD 429 → 31 s × 5 = 155 s freeze at "Screen for
        # relevance"). Doing it once at boot is a single visible cost
        # users can wait through; subsequent restarts skip the network
        # entirely (cache hit + local_files_only path in screening.py).
        # Skip entirely when reranking is disabled (reranker_enabled=False or
        # empty reranker_model) — e.g. strong-embedder configs that don't rerank.
        _reranker_model = getattr(config.rag_modes, "reranker_model", None)
        _reranker_enabled = getattr(config.rag_modes, "reranker_enabled", True)
        if _reranker_enabled and _reranker_model:
            _prewarm_reranker(_reranker_model)
        else:
            logger.info(
                "prewarm_reranker_skipped: reranking disabled "
                "(reranker_enabled=%s, reranker_model=%r)",
                _reranker_enabled,
                _reranker_model,
            )

        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            persist_dir=str(config.database.chroma_path), embedding_provider=self.embedding_provider
        )
        logger.info("Vector store initialized")

        # Store config for later use
        self.config = config

        # Initialize tool registry (LOTUS deactivated for now)
        tool_registry = ToolRegistry()
        # Register the live web-search tool so Profound (and other modes that
        # check ``"web_search" in tools.list_tools()``) actually find it. The
        # tool reads ``self`` (AppState) to reach the aggregator config.
        try:
            from perspicacite.rag.tools import WebSearchTool

            # WebSearchTool.execute has a richer signature (explicit `query`)
            # than the Tool ABC's `execute(self, **kwargs)`; it is always
            # invoked through the registry with kwargs, so the Liskov mismatch
            # is intentional and safe here.
            tool_registry.register(WebSearchTool(app_state=self))  # type: ignore[arg-type]
            logger.info("Tool registry initialized (web_search registered, LOTUS deactivated)")
        except Exception as exc:  # pragma: no cover - best-effort
            # NB: module logger is a stdlib logging.Logger, not structlog — it
            # takes %-args, not kwargs. `error=str(exc)` would raise TypeError.
            logger.warning("web_search_tool_register_failed error=%s", str(exc))
            logger.info("Tool registry initialized (LOTUS deactivated)")

        # Create LLM adapter for agentic components
        llm_adapter = LLMAdapter(
            client=self.llm_client,
            model=config.llm.default_model,
            provider=config.llm.default_provider,
        )

        # Initialize agentic orchestrator
        self.orchestrator = AgenticOrchestrator(
            llm_client=llm_adapter,
            tool_registry=tool_registry,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            max_iterations=5,
            use_two_pass=getattr(config.knowledge_base, "use_two_pass", True),
            map_reduce_max_papers=getattr(config.rag_modes.agentic, "map_reduce_max_papers", 8),
            app_state=self,
        )
        logger.info("Agentic orchestrator initialized")

        # Initialize session store FIRST so RAGEngine can receive it
        db_path = Path("./data/perspicacite.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_store = SessionStore(db_path)
        await self.session_store.init_db()
        logger.info("Session store initialized")

        # Initialize RAG engine for multi-mode support
        from perspicacite.rag.engine import RAGEngine

        self.rag_engine = RAGEngine(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            tool_registry=tool_registry,
            config=config,
            session_store=self.session_store,
            app_state=self,
        )
        logger.info("RAG engine initialized (supports all modes)")

        sidecar_dir = self.session_store.db_path.parent / "provenance"
        self.provenance_store = ProvenanceStore(
            db_path=self.session_store.db_path,
            sidecar_dir=sidecar_dir,
        )
        logger.info("Provenance store initialized")
        self.rag_engine.provenance_store = self.provenance_store

        self.job_registry = JobRegistry(db_path=self.session_store.db_path)
        logger.info("Job registry initialized")

        # Initialize PDF downloader and parser. The downloader picks up
        # the optional cookies.txt path + cookie-domain allowlist from
        # config.pdf_download so PDF requests can ride a logged-in
        # browser session (institutional access via SSO/proxy). The
        # cookie jar is loaded lazily per-request inside
        # PDFDownloader.download.
        from perspicacite.pipeline.download import PDFDownloader
        from perspicacite.pipeline.parsers.pdf import PDFParser

        self.pdf_downloader = PDFDownloader(
            timeout=getattr(config.pdf_download, "timeout", 30.0),
            max_retries=getattr(config.pdf_download, "max_retries", 3),
            cookies_path=getattr(config.pdf_download, "cookies_path", None),
            cookie_domains=getattr(config.pdf_download, "cookie_domains", []),
        )
        self.pdf_parser = PDFParser()
        logger.info("PDF downloader and parser initialized")

        # Cookie freshness check (Priority 7): surface a warning at boot
        # if any configured paywall domain has expired or missing cookies,
        # so the user knows to re-run `perspicacite import-browser-cookies`
        # *before* the first paywall hit (rather than silently getting
        # HTML access-check pages back from the publisher).
        _warn_stale_cookies(
            cookies_path=getattr(config.pdf_download, "cookies_path", None),
            cookie_domains=getattr(config.pdf_download, "cookie_domains", []) or [],
        )

        self.initialized = True
        logger.info("System initialization complete!")


def _prewarm_reranker(model_name: str) -> None:
    """Ensure the cross-encoder reranker is cached on disk before any
    query hits the rerank step.

    The runtime load path in ``search/screening.py`` already tries
    ``local_files_only=True`` first to avoid HuggingFace HEAD 429s on
    every restart. That works on the second-and-later boot, but on the
    *first* boot the cache is empty, the local-only load fails, and the
    fallback then hits the network — which is exactly the scenario we
    want to control. Calling this at startup downloads the model once,
    visible in the boot log, so the first user query doesn't pay it.

    No-op when the cache already contains the model. Errors are logged
    but non-fatal — the rerank path still works without the prewarm.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.info(
            "prewarm_reranker_skipped: sentence_transformers not installed"
        )
        return

    # Fast cache probe: try local-only first; if it succeeds, the model
    # is already present and we skip the network load entirely.
    try:
        CrossEncoder(model_name, local_files_only=True)
        logger.info("prewarm_reranker_cached: %s", model_name)
        return
    except Exception as exc:
        logger.debug("prewarm_reranker_cache_miss model=%s error=%s", model_name, str(exc))

    import time
    _t0 = time.monotonic()
    logger.info("prewarm_reranker_downloading: %s (one-time, ~90 MB)", model_name)
    try:
        CrossEncoder(model_name)
        logger.info(
            "prewarm_reranker_ready: %s (%.1fs)",
            model_name, time.monotonic() - _t0,
        )
    except Exception as exc:
        logger.warning(
            "prewarm_reranker_failed: %s — %s — rerank will retry at first use",
            model_name, exc,
        )


def _warn_stale_cookies(*, cookies_path: str | None, cookie_domains: list[str]) -> None:
    """Log a warning per stale cookie domain at startup.

    Non-fatal — the server boots either way. Three reasons we warn at
    boot rather than at first download attempt:

    1. The first download attempt could be hours later; the failure
       mode (publisher returns HTML access-check page instead of PDF)
       is non-obvious, easy to attribute to "the script is broken".
    2. Re-importing cookies takes 5 seconds; better to know upfront.
    3. Boot logs are the natural place to surface integration-health
       state, and the operator sees them every restart.
    """
    if not cookies_path or not cookie_domains:
        return
    from http.cookiejar import MozillaCookieJar
    from pathlib import Path

    from perspicacite.pipeline.download.cookies import (
        check_cookie_freshness_for_domains,
    )

    p = Path(cookies_path).expanduser()
    if not p.exists():
        logger.warning(
            "pdf_cookies_missing — paywalled-publisher fetches will fail. "
            "Run `perspicacite import-browser-cookies` to fix.",
            extra={"cookies_path": str(p)},
        )
        return
    try:
        jar = MozillaCookieJar(str(p))
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as exc:
        logger.warning(
            "pdf_cookies_load_failed at startup; freshness check skipped",
            extra={"path": str(p), "error": str(exc)},
        )
        return
    report = check_cookie_freshness_for_domains(jar, cookie_domains)
    stale = [w for w in report if w.status != "ok"]
    if not stale:
        logger.info(
            "pdf_cookies_health_ok",
            extra={"path": str(p), "domains_checked": len(report)},
        )
        return
    for w in stale:
        logger.warning(
            "pdf_cookies_stale",
            extra={
                "domain": w.domain,
                "status": w.status,
                "matched_hosts": w.matched_hosts,
                "advice": w.advice,
            },
        )


app_state = AppState()
