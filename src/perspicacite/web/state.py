"""Application state for the web app.

Holds the singleton instance of `AppState` that all routers share. Routers
import `app_state` from this module; the FastAPI lifespan calls
`await app_state.initialize()` once on startup.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from perspicacite.config.paths import resolve_session_db_path
from perspicacite.jobs.registry import JobRegistry
from perspicacite.memory.session_store import SessionStore
from perspicacite.provenance.store import ProvenanceStore

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
        self.llm_client = None
        self.embedding_provider = None
        self.vector_store = None
        self.orchestrator = None
        self.rag_engine = None  # Multi-mode RAG engine
        self.session_store: SessionStore | None = None
        self.provenance_store: ProvenanceStore | None = None
        self.job_registry: JobRegistry | None = None
        self.pdf_downloader = None
        self.pdf_parser = None
        self.initialized = False
        # Config path — set by CLI before lifespan fires so AppState reads
        # the same config file as the CLI (previously it re-called load_config()
        # with no path and always loaded the default config.yml).
        self._config_path: str | None = config_path

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
        # models like "all-MiniLM-L6-v2" work without an OpenAI key).
        # use_local_fallback=False: the local fallback only ever wraps API models,
        # but it has a DIFFERENT dimension (MiniLM 384 vs OpenAI 3072), so on any API
        # error it silently emits 384-dim vectors that a 3072-dim collection rejects —
        # poisoning ingests. Fail loud on API errors instead. (Local models are
        # unaffected: the factory returns them without a fallback wrapper.)
        self.embedding_provider = create_embedding_provider(
            model=config.knowledge_base.embedding_model,
            use_local_fallback=False,
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
            tool_registry.register(WebSearchTool(app_state=self))
            logger.info("Tool registry initialized (web_search registered, LOTUS deactivated)")
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning("web_search_tool_register_failed: %s", exc)
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
            export_apa_docx=getattr(config.rag_modes.agentic, "export_apa_docx", False),
            export_apa_docx_dir=getattr(config.rag_modes.agentic, "export_apa_docx_dir", "output"),
            app_state=self,
        )
        logger.info("Agentic orchestrator initialized")

        # Initialize session store FIRST so RAGEngine can receive it
        db_path = resolve_session_db_path(config)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_store = SessionStore(db_path)
        await self.session_store.init_db()
        # Log both stores: an instance whose metadata and vectors live in
        # different places lists knowledge bases that return nothing.
        logger.info(
            "Session store initialized at %s (chroma: %s)",
            db_path,
            config.database.chroma_path,
        )

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
    except Exception:
        pass

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
