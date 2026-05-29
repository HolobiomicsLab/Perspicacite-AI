"""Literature Survey RAG Mode - Systematic research field mapping.

This mode is designed for comprehensive literature surveys, not quick answers.
It systematically maps a research field by:
1. Broad search across multiple APIs
2. Abstract analysis in batches (50-100 papers)
3. Theme clustering and identification
4. AI recommendations for deep analysis
5. User-selected full-text analysis (up to 50 papers)
6. Structured survey report with PDF export
"""

import asyncio
import contextlib
import json
import re
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference
from perspicacite.provenance.context import get_collector
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.telemetry import emit_phase
from perspicacite.retrieval.recency import apply_recency_weighting
from perspicacite.search.scilex_adapter import SciLExAdapter

logger = get_logger("perspicacite.rag.modes.literature_survey")


def _target_kb(request: Any) -> str:
    """Return the KB to use for any storage-targeting decisions.

    Literature Survey does NOT retrieve from a KB (it uses external SciLEx
    search), but the API still accepts ``request.kb_names`` for parity with
    other RAG modes. When multiple KBs are supplied, storage / provenance
    must converge on a single target — by convention the first entry.

    Falls back to ``request.kb_name`` when ``kb_names`` is None or empty.
    """
    names = getattr(request, "kb_names", None)
    if names:
        return names[0]
    return request.kb_name


def _apply_recency_to_candidates(
    candidates: list[Any],
    recency_weight: float | None,
    half_life_years: float | None,
) -> list[Any]:
    """Apply recency weighting to a list of PaperCandidate objects.

    PaperCandidate stores its score in ``relevance_score`` (not ``score`` /
    ``paper_score``), so we can't pass the objects directly to the generic
    helpers.  This wrapper converts each candidate to a plain dict with a
    ``_candidate`` back-reference, delegates to ``apply_recency_weighting``,
    writes the adjusted score back, and returns the re-sorted list.
    No-op when *recency_weight* is None or 0.
    """
    if not recency_weight or recency_weight <= 0 or not candidates:
        return candidates

    # Build proxy dicts that the recency helper understands, carrying a
    # reference to the original candidate so we can write the score back.
    proxies = [
        {"year": c.year, "score": float(c.relevance_score or 0.0), "_candidate": c}
        for c in candidates
    ]
    apply_recency_weighting(proxies, recency_weight, half_life_years)

    # Write adjusted scores back and return re-sorted candidates
    for proxy in proxies:
        proxy["_candidate"].relevance_score = proxy["score"]

    return [proxy["_candidate"] for proxy in proxies]


@dataclass
class Theme:
    """A research theme identified from papers."""
    name: str
    description: str
    papers: list[dict[str, Any]] = field(default_factory=list)
    key_insights: list[str] = field(default_factory=list)


@dataclass
class PaperCandidate:
    """A paper candidate for the survey."""
    id: str
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    doi: str | None
    citation_count: int = 0
    relevance_score: float = 0.0
    themes: list[str] = field(default_factory=list)
    recommended: bool = False
    reason: str = ""  # Why recommended


@dataclass
class SurveySession:
    """Persistent session for literature survey."""
    session_id: str
    query: str
    papers: list[PaperCandidate] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    selected_papers: list[str] = field(default_factory=list)  # Paper IDs
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dict for persistence."""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "papers_count": len(self.papers),
            "themes_count": len(self.themes),
            "selected_count": len(self.selected_papers),
            "created_at": self.created_at.isoformat(),
        }


class LiteratureSurveyRAGMode(BaseRAGMode):
    """
    Literature Survey RAG Mode - Systematic research field mapping.
    
    Key characteristics:
    - Comprehensive coverage (50-100 papers analyzed from abstracts)
    - Dynamic theme identification (3-8 themes)
    - AI recommendations for deep analysis
    - User selection (up to 50 papers)
    - Structured PDF output
    """

    def __init__(self, config: Any):
        super().__init__(config)

        # Configuration
        self.batch_size = 20  # Papers per batch for abstract analysis
        self.max_deep_analysis = 50  # Safety cap for full-text download
        self.relevance_threshold = 2.0  # Lower than agentic for broader coverage
        self.max_themes = 8
        self.min_themes = 3

        # Issue 3: keep up to seed_known_max known papers as anchor seeds when
        # the broad search returns ONLY already-indexed papers. Prevents the
        # empty-results → KB-vector-fallback → poor diversity loop.
        _ls_settings = getattr(config.rag_modes, "literature_survey", None) or {}
        if hasattr(_ls_settings, "model_dump"):
            _ls_settings = _ls_settings.model_dump()
        elif hasattr(_ls_settings, "dict"):
            _ls_settings = _ls_settings.dict()
        elif not isinstance(_ls_settings, dict):
            _ls_settings = {}
        self.seed_known_max: int = int(_ls_settings.get("seed_known_max", 5))

        # SciLEx for multi-API search
        self.scilex_adapter = SciLExAdapter()

        # Session management
        self.sessions: dict[str, SurveySession] = {}

        # Injected by RAGEngine when a SessionStore is available.
        # Used by _store_references_to_all_kbs (Task 3) to write reference rows.
        # _prepare_kb_context (Task 2) and _store_references_to_all_kbs (Task 3)
        # are called from execute() / execute_stream() in Task 4.
        self.session_store: Any = None

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """
        Execute literature survey.
        
        This is a multi-phase process:
        1. Broad search
        2. Abstract analysis (batch by batch)
        3. Theme identification
        4. Recommendations
        5. User selection (handled via UI/API)
        6. Deep analysis
        7. Survey generation
        """
        session_id = str(uuid.uuid4())
        session = SurveySession(session_id=session_id, query=request.query)
        self.sessions[session_id] = session

        logger.info("literature_survey_start", query=request.query, session_id=session_id)

        # Prepare KB context: retrieve semantically similar papers from all
        # provided KBs and collect ALL known paper_ids for pre-filtering.
        kb_context_block, known_paper_ids = await self._prepare_kb_context(
            request, vector_store, embedding_provider
        )

        # Phase 1: Broad search
        logger.info("phase_1_search")
        papers = await self._broad_search(request.query, request.databases, app_state=getattr(request, "app_state", None))

        # Pre-filter: remove papers already in any provided KB
        _papers_searched = len(papers)
        papers = self._filter_known_papers(papers, known_paper_ids)
        _filtered_as_known = _papers_searched - len(papers)

        if not papers:
            _cancellation_reason = (
                "all_known_no_fallback" if _filtered_as_known > 0 else "no_papers"
            )
            return RAGResponse(
                answer="No papers found for this topic. Try broadening your search terms.",
                sources=[],
                mode=RAGMode.LITERATURE_SURVEY,
                metadata={
                    "session_id": session_id,
                    "phase": "search_failed",
                    "diagnostic": {
                        "papers_searched": _papers_searched,
                        "filtered_as_known": _filtered_as_known,
                        "kb_fallback_tried": False,
                        "kb_chunks_found": 0,
                        "cancellation_reason": _cancellation_reason,
                    },
                },
            )

        # Convert to candidates
        session.papers = self._convert_to_candidates(papers)
        logger.info("papers_found", count=len(session.papers))

        # Provenance: record broad search
        _c = get_collector()
        if _c is not None:
            _c.add_trace(
                "broad_search",
                detail={"count": len(session.papers), "kb_name": _target_kb(request)},
            )

        # Phase 2 & 3: Batch abstract analysis + theme identification
        logger.info("phase_2_3_analysis")
        session.themes = await self._analyze_abstracts_batch(
            session.papers, request.query, llm
        )
        logger.info("themes_identified", count=len(session.themes))

        # Apply recency weighting on candidates using relevance_score as the score field
        session.papers = _apply_recency_to_candidates(
            session.papers,
            request.recency_weight,
            getattr(request, "recency_half_life_years", None),
        )

        # Provenance: per-paper retrieval events after scoring
        if _c is not None:
            for rank, cand in enumerate(session.papers):
                _c.add_retrieval(
                    paper_id=cand.id,
                    doi=cand.doi,
                    title=cand.title,
                    score=float(cand.relevance_score or 0.0),
                    kb_name=None,
                    content_type=None,
                    pipeline_step=None,
                    rank=rank,
                    stage_label="survey.broad_search",
                )
            _c.add_trace("cluster", detail={"themes": len(session.themes)})

        # Phase 4: Generate recommendations
        logger.info("phase_4_recommendations")
        await self._generate_recommendations(session.papers, session.themes, llm)

        # Provenance: record recommendations stage
        if _c is not None:
            _c.add_trace("recommend")

        # Build interim listing summary (for API compatibility / metadata).
        summary = self._generate_interim_summary(session, known_context=kb_context_block)

        # Store references to extra KBs (indices 1..n) for future re-ingestion.
        # Falls back to [request.kb_name] when kb_names is absent; in that case
        # len(all_kb_names) == 1 so _store_references_to_all_kbs is a no-op.
        all_kb_names = list(request.kb_names or [request.kb_name])
        recommended_papers = [p for p in session.papers if p.recommended]
        await self._store_references_to_all_kbs(
            recommended_papers, all_kb_names, request.query
        )

        # Generate the actual synthesized survey report so execute() returns
        # a scoreable answer rather than a bare paper listing. The interim
        # summary is preserved in metadata for UI compatibility.
        survey_papers = recommended_papers[:20] if recommended_papers else session.papers[:20]
        try:
            answer = await self._generate_survey_report(session, survey_papers, llm)
        except Exception as _survey_exc:
            logger.warning(
                "literature_survey_report_failed_fallback",
                error=str(_survey_exc),
            )
            answer = summary  # graceful fallback to interim summary

        return RAGResponse(
            answer=answer,
            sources=self._convert_to_sources(session.papers),
            mode=RAGMode.LITERATURE_SURVEY,
            metadata={
                "session_id": session_id,
                "phase": "awaiting_selection",
                "papers_count": len(session.papers),
                "themes_count": len(session.themes),
                "recommended_count": sum(1 for p in session.papers if p.recommended),
                "interim_summary": summary,
            }
        )

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncGenerator[Any, None]:
        """Stream literature survey progress."""
        from perspicacite.models.rag import StreamEvent

        session_id = str(uuid.uuid4())
        session = SurveySession(session_id=session_id, query=request.query)
        self.sessions[session_id] = session

        # Store active request so nested helpers can read per-call overrides
        # (e.g. batch_size, crossref_concurrency) without signature changes.
        self._current_request = request

        # Prepare KB context
        kb_context_block, known_paper_ids = await self._prepare_kb_context(
            request, vector_store, embedding_provider
        )

        # Diagnostic counters — populated as the pipeline progresses and emitted
        # in every early-return path so callers can diagnose empty results.
        _diag: dict = {
            "papers_searched": 0,
            "filtered_as_known": 0,
            "kb_fallback_tried": False,
            "kb_chunks_found": 0,
        }

        _phase_sink = getattr(request, "telemetry_sink", None)
        yield StreamEvent.status("Literature Survey: Initializing...")

        # Phase 1: Search
        emit_phase(_phase_sink, phase="collect", state="running")
        yield StreamEvent.status("Literature Survey: Searching across academic databases...")
        _bs_telemetry = getattr(request, "telemetry_sink", None) or []
        papers = await self._broad_search(
            request.query, request.databases, telemetry=_bs_telemetry,
            app_state=getattr(request, "app_state", None),
        )
        # When _bs_telemetry is a CallbackTelemetrySink (MCP path), events
        # already flowed to ctx.report_progress live — skip the drain.
        if isinstance(_bs_telemetry, list):
            for _ev in _bs_telemetry:
                _k = _ev.get("kind")
                if _k == "query_rephrased":
                    yield StreamEvent.status_kind(
                        f"Rewrote search query: '{_ev.get('original','')}' → '{_ev.get('rewritten','')}'",
                        kind="query_rephrased",
                        original=_ev.get("original", ""),
                        rewritten=_ev.get("rewritten", ""),
                        by=_ev.get("by", "keyword_optimizer"),
                    )
                elif _k == "provider_progress" and _ev.get("phase") == "start":
                    _provs = ", ".join(
                        p.replace("_", " ").title() for p in _ev.get("providers", [])
                    )
                    _sq = _ev.get("searched_query") or ""
                    _msg = (
                        f"Querying databases: {_provs} — keywords: '{_sq}'"
                        if _sq
                        else f"Querying databases: {_provs}…"
                    )
                    yield StreamEvent.status_kind(
                        _msg,
                        kind="provider_progress",
                        phase="start",
                        providers=_ev.get("providers", []),
                        searched_query=_sq,
                    )
                elif _k == "provider_progress" and _ev.get("phase") == "done":
                    _bp = _ev.get("by_provider", {}) or {}
                    _msg = ", ".join(
                        f"{src.replace('_',' ').title()}: {n}"
                        for src, n in sorted(_bp.items(), key=lambda kv: -kv[1])
                    ) if _bp else f"Total {_ev.get('total', 0)} hits"
                    yield StreamEvent.status_kind(
                        f"Database results — {_msg}",
                        kind="provider_progress",
                        phase="done",
                        total=_ev.get("total", 0),
                        by_provider=_bp,
                    )

        # Pre-filter: remove papers already in any provided KB
        _diag["papers_searched"] = len(papers)
        papers = self._filter_known_papers(papers, known_paper_ids)
        _diag["filtered_as_known"] = _diag["papers_searched"] - len(papers)

        if not papers:
            # If a KB was explicitly provided and the broad search is empty (because
            # all results were already in the KB), fall back to surveying KB papers
            # directly.  This is the "survey what I have" use case.
            kb_names_used: list[str] = list(getattr(request, "kb_names", None) or [])
            if kb_names_used:
                yield StreamEvent.status(
                    "Literature Survey: Using KB papers as survey corpus (all new results already known)"
                )
                _diag["kb_fallback_tried"] = True
                try:
                    retriever = self._build_kb_retriever(request, vector_store, embedding_provider)
                    kb_results = await retriever.search(request.query, top_k=30)
                    # Build PaperCandidate objects from KB chunks (chunk_text as abstract)
                    seen_pids: set[str] = set()
                    kb_candidates: list[PaperCandidate] = []
                    for r in kb_results:
                        pid = r.get("paper_id") or ""
                        if pid and pid in seen_pids:
                            continue
                        if pid:
                            seen_pids.add(pid)
                        meta = r.get("metadata")
                        chunk_text = r.get("chunk_text") or ""
                        if not chunk_text.strip():
                            continue
                        title = (getattr(meta, "title", None) or "Unknown title")
                        year = getattr(meta, "year", None)
                        doi = getattr(meta, "doi", None)
                        authors_raw = getattr(meta, "authors", None) or []
                        authors = (
                            [a if isinstance(a, str) else getattr(a, "name", str(a)) for a in authors_raw]
                            if authors_raw else []
                        )
                        kb_candidates.append(PaperCandidate(
                            id=pid or str(uuid.uuid4()),
                            title=title,
                            authors=authors,
                            year=year,
                            abstract=chunk_text[:800],  # first 800 chars as abstract
                            doi=doi,
                            citation_count=0,
                        ))
                    _diag["kb_chunks_found"] = len(kb_candidates)
                    if kb_candidates:
                        session.papers = kb_candidates
                        emit_phase(_phase_sink, phase="collect", state="done")
                        yield StreamEvent.status(
                            f"Literature Survey: Surveying {len(session.papers)} KB papers"
                        )
                    else:
                        emit_phase(_phase_sink, phase="collect", state="done")
                        yield StreamEvent.diagnostic(
                            **_diag,
                            cancellation_reason="kb_fallback_empty",
                        )
                        yield StreamEvent.status("Literature Survey: No papers found")
                        yield StreamEvent.content(
                            "No papers found for this topic. Try broadening your search terms or adding papers to your KB."
                        )
                        yield StreamEvent.done(
                            conversation_id=session_id,
                            tokens_used=0,
                            mode="literature_survey",
                            iterations=1,
                        )
                        return
                except Exception as _kb_exc:
                    logger.warning("survey_kb_fallback_failed", error=str(_kb_exc))
                    emit_phase(_phase_sink, phase="collect", state="done")
                    yield StreamEvent.diagnostic(
                        **_diag,
                        cancellation_reason="kb_fallback_failed",
                        kb_fallback_error=str(_kb_exc),
                    )
                    yield StreamEvent.status("Literature Survey: No papers found")
                    yield StreamEvent.content(
                        "No papers found for this topic. Try broadening your search terms."
                    )
                    yield StreamEvent.done(
                        conversation_id=session_id,
                        tokens_used=0,
                        mode="literature_survey",
                        iterations=1,
                    )
                    return
            else:
                emit_phase(_phase_sink, phase="collect", state="done")
                _cancellation_reason = (
                    "all_known_no_fallback" if _diag["filtered_as_known"] > 0 else "no_papers"
                )
                yield StreamEvent.diagnostic(
                    **_diag,
                    cancellation_reason=_cancellation_reason,
                )
                yield StreamEvent.status("Literature Survey: No papers found")
                yield StreamEvent.content("No papers found for this topic. Try broadening your search terms.")
                yield StreamEvent.done(
                    conversation_id=session_id,
                    tokens_used=0,
                    mode="literature_survey",
                    iterations=1,
                )
                return
        else:
            session.papers = self._convert_to_candidates(papers)
            emit_phase(_phase_sink, phase="collect", state="done")
            yield StreamEvent.status(f"Literature Survey: Found {len(session.papers)} papers")

        # Provenance: record broad search
        _c = get_collector()
        if _c is not None:
            _c.add_trace(
                "broad_search",
                detail={"count": len(session.papers), "kb_name": _target_kb(request)},
            )

        # Phase 2: Batch analysis with live progress events.
        # Run the analyzer as a background task; use an asyncio.Queue to pipe
        # per-batch progress out to the SSE stream so the UI shows
        # "Analyzing batch 3/5 (20 papers)" updates in real time.
        _progress_q: asyncio.Queue = asyncio.Queue()

        async def _progress_cb(
            current: int,
            total: int,
            batch_size: int,
            stage: str = "abstract_analysis",
        ) -> None:
            await _progress_q.put({
                "kind": "batch_progress",
                "stage": stage,
                "current": current,
                "total": total,
                "batch_size": batch_size,
            })

        # Cancellation check — respect MCP cancel_task requests
        from perspicacite.rag.cancellation import is_cancelled as _is_cancelled
        _tid = getattr(request, "task_id", None)
        if _tid and _is_cancelled(_tid):
            logger.info("literature_survey_cancelled", task_id=_tid, stage="pre_analysis")
            yield StreamEvent(event="error", data=json.dumps({"reason": "cancelled", "task_id": _tid}))
            return

        emit_phase(_phase_sink, phase="extract_themes", state="running")
        yield StreamEvent.status("Literature Survey: Analyzing abstracts in batches...")

        analysis_task = asyncio.create_task(
            self._analyze_abstracts_batch(
                session.papers, request.query, llm,
                progress_cb=_progress_cb,
            )
        )

        # Drain the queue concurrently with the analysis task. When the task
        # is done AND the queue is empty, we exit the loop.
        while not analysis_task.done() or not _progress_q.empty():
            try:
                ev = await asyncio.wait_for(_progress_q.get(), timeout=0.5)
            except TimeoutError:
                # Also check for cancellation between batches
                if _tid and _is_cancelled(_tid):
                    analysis_task.cancel()
                    try:
                        await analysis_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass  # swallow any other exception from the cancelled task
                    logger.info("literature_survey_cancelled", task_id=_tid, stage="mid_analysis")
                    yield StreamEvent(event="error", data=json.dumps({"reason": "cancelled", "task_id": _tid}))
                    return
                continue
            yield StreamEvent.status_kind(
                f"Analyzing batch {ev['current']}/{ev['total']} ({ev['batch_size']} papers)…",
                kind="batch_progress",
                stage=ev["stage"],
                current=ev["current"],
                total=ev["total"],
                batch_size=ev["batch_size"],
            )

        session.themes = await analysis_task
        emit_phase(_phase_sink, phase="extract_themes", state="done")
        yield StreamEvent.status(
            f"Literature Survey: Identified {len(session.themes)} research themes"
        )

        # Apply recency weighting on candidates using relevance_score as the score field
        session.papers = _apply_recency_to_candidates(
            session.papers,
            request.recency_weight,
            getattr(request, "recency_half_life_years", None),
        )

        # Provenance: per-paper retrieval events + cluster trace
        if _c is not None:
            for rank, cand in enumerate(session.papers):
                _c.add_retrieval(
                    paper_id=cand.id,
                    doi=cand.doi,
                    title=cand.title,
                    score=float(cand.relevance_score or 0.0),
                    kb_name=None,
                    content_type=None,
                    pipeline_step=None,
                    rank=rank,
                    stage_label="survey.broad_search",
                )
            _c.add_trace("cluster", detail={"themes": len(session.themes)})

        # Emit per-paper source events so the frontend renders the
        # source-pill grid like other modes (Basic, Advanced, Profond).
        # We cap at 20 to match what's actually shown in the UI.
        for _src in self._convert_to_sources(session.papers[:20]):
            yield StreamEvent.source(_src)

        # Phase 3: Recommendations (deepen)
        emit_phase(_phase_sink, phase="deepen", state="running")
        yield StreamEvent.status("Literature Survey: Generating recommendations...")
        await self._generate_recommendations(session.papers, session.themes, llm)
        emit_phase(_phase_sink, phase="deepen", state="done")

        # Provenance: record recommendations stage
        if _c is not None:
            _c.add_trace("recommend")

        # Build the interim listing (used as fallback / UI metadata)
        summary = self._generate_interim_summary(session, known_context=kb_context_block)

        # Store references to extra KBs before generating the report.
        all_kb_names = list(request.kb_names or [request.kb_name])
        recommended_papers = [p for p in session.papers if p.recommended]
        await self._store_references_to_all_kbs(
            recommended_papers, all_kb_names, request.query
        )

        # Generate a synthesized survey report so the stream yields a scoreable
        # answer rather than a bare paper listing.
        survey_papers = recommended_papers[:20] if recommended_papers else session.papers[:20]
        try:
            survey_answer = await self._generate_survey_report(session, survey_papers, llm)
        except Exception as _exc:
            logger.warning("literature_survey_stream_report_failed", error=str(_exc))
            survey_answer = summary  # graceful fallback

        yield StreamEvent.content(survey_answer)

        # Emit metadata for UI
        yield StreamEvent(
            event="status",
            data=json.dumps({
                "message": "Literature Survey: Complete",
                "session_id": session_id,
                "papers_count": len(session.papers),
                "themes_count": len(session.themes),
                "recommended_count": sum(1 for p in session.papers if p.recommended),
                "interim_summary": summary,
            })
        )

        yield StreamEvent.done(
            conversation_id=session_id,
            tokens_used=0,
            mode="literature_survey",
            iterations=1,
        )



    async def _broad_search(
        self,
        query: str,
        databases: list[str] | None = None,
        telemetry: list[dict[str, Any]] | None = None,
        app_state: Any = None,
    ) -> list[Any]:
        """
        Broad search across multiple APIs.

        Uses SciLEx to search across selected databases. ``telemetry`` lets the
        streaming caller surface query rewriting + per-DB results to SSE.
        """
        # Default databases if none specified
        if not databases:
            databases = ["semantic_scholar", "openalex", "pubmed"]

        # Rewrite the query via the shared optimizer (Haiku) before searching.
        _app_state = app_state

        # Optimizer call in its own try/except
        effective_query = query
        if _app_state is not None and getattr(_app_state, "config", None) is not None:
            import perspicacite.search.query_optimizer as _qo_mod
            try:
                opt = await _qo_mod.optimize_query(
                    query=query,
                    context=None,
                    app_state=_app_state,
                    optimize_enabled=True,  # always rewrite for web search (item 2)
                )
                effective_query = opt.searched_query
                if opt.applied:
                    logger.info(
                        "literature_survey_query_rewritten",
                        original=query,
                        rewritten=effective_query,
                    )
                    if telemetry is not None:
                        telemetry.append({
                            "kind": "query_rephrased",
                            "by": "keyword_optimizer",
                            "original": query,
                            "rewritten": effective_query,
                        })
            except Exception as _opt_exc:
                logger.warning("literature_survey_optimizer_failed", error=str(_opt_exc))
                # effective_query already = query, no reassignment needed

        # Route through the unified pipeline (aggregator → Crossref enrich).
        # rerank=False: survey keeps its own LLM-based relevance analyser
        # (_analyze_abstracts_batch) which scores papers 1-5 after broad
        # collection — MiniLM reranking here would prematurely bias the
        # corpus before the theme clustering pass sees it.
        try:
            from perspicacite.rag.resolve_papers import resolve_papers_pipeline
            papers = await resolve_papers_pipeline(
                query=effective_query,
                databases=databases,
                max_docs=100,
                app_state=_app_state,
                telemetry=telemetry,
                enrich=True,
                rerank=False,  # survey keeps its own analyser
                optimize_query=False,  # already optimised above
            )
            return papers
        except Exception as e:
            logger.error("broad_search_failed", error=str(e))
            return []

    def _convert_to_candidates(self, papers: list[Any]) -> list[PaperCandidate]:
        """Convert SciLEx Paper models to candidates.
        
        Only includes papers with abstracts - these are required for
        AI relevance analysis and theme categorization.
        """
        candidates = []
        skipped_count = 0
        for p in papers:
            # Skip papers without abstracts - can't analyze relevance without content
            if not p.abstract or not p.abstract.strip():
                skipped_count += 1
                continue

            candidate = PaperCandidate(
                id=p.id or str(uuid.uuid4()),
                title=p.title or "Untitled",
                authors=[a.name for a in p.authors] if p.authors else [],
                year=p.year,
                abstract=p.abstract,
                doi=p.doi,
                citation_count=p.citation_count or 0,
            )
            candidates.append(candidate)

        if skipped_count > 0:
            logger.info("papers_without_abstracts_skipped", count=skipped_count)

        return candidates

    async def _prepare_kb_context(
        self,
        request: Any,
        vector_store: Any,
        embedding_provider: Any,
        top_k: int = 10,
    ) -> tuple[str, set[str]]:
        """Retrieve known papers from all provided KBs.

        Performs two operations:
        1. Fetches ALL paper_ids from every KB's ChromaDB collection (for
           pre-filtering broad search results).
        2. Runs a semantic top-K search across KBs (via _build_kb_retriever)
           and formats a human-readable context block for the survey summary.

        Returns:
            context_block: Formatted string listing known papers (for summary).
            all_known_ids: Full set of paper_ids/DOIs already in any provided KB.

        Both return values are empty if kb_names is absent or empty.
        Never raises — errors are caught and logged.
        """
        from perspicacite.models.kb import chroma_collection_name_for_kb

        kb_names: list[str] = list(getattr(request, "kb_names", None) or [])
        if not kb_names:
            return "", set()

        # ── A. Collect ALL paper_ids from ChromaDB across every KB ──────────────
        all_known_ids: set[str] = set()
        for kb_name in kb_names:
            col = chroma_collection_name_for_kb(kb_name)
            try:
                rows = await vector_store.list_paper_ids_in_collection(col)
                # rows: list[tuple[paper_id, title, chunk_count]]
                all_known_ids.update(pid for pid, _, _ in rows)
            except Exception as exc:
                logger.warning(
                    "survey_kb_id_fetch_error", kb=kb_name, error=str(exc)
                )

        # ── B. Semantic top-K retrieval for the context block ───────────────────
        context_block = ""
        try:
            retriever = self._build_kb_retriever(request, vector_store, embedding_provider)
            results = await retriever.search(request.query, top_k=top_k)
            if results:
                lines: list[str] = []
                seen_pids: set[str] = set()
                for r in results:
                    pid = r.get("paper_id") or ""
                    if pid and pid in seen_pids:
                        continue
                    if pid:
                        seen_pids.add(pid)
                    meta = r.get("metadata")
                    title = (getattr(meta, "title", None) or "Unknown title")
                    year = getattr(meta, "year", None) or ""
                    doi = getattr(meta, "doi", None) or ""
                    kb_tag = r.get("kb_name") or ""
                    line = f"- {title} ({year})"
                    if kb_tag:
                        line += f" [KB: {kb_tag}]"
                    if doi:
                        line += f" DOI: {doi}"
                    lines.append(line)
                if lines:
                    context_block = (
                        "Papers already in your knowledge base(s) — "
                        "excluded from new-paper analysis:\n"
                        + "\n".join(lines)
                    )
        except Exception as exc:
            logger.warning("survey_kb_context_retrieval_error", error=str(exc))

        logger.info(
            "survey_kb_context_prepared",
            known_ids_total=len(all_known_ids),
            context_lines=len(context_block.splitlines()),
            kb_names=kb_names,
        )
        return context_block, all_known_ids

    def _filter_known_papers(
        self,
        papers: list[Any],
        known_paper_ids: set[str],
    ) -> list[Any]:
        """Remove papers already present in any provided KB.

        When ALL results are already in the KB (a common case for well-indexed
        topics), keep up to ``self.seed_known_max`` of them as anchor seeds to
        preserve survey diversity.  Falls back to pure KB vector search only
        when the broad search returns zero results.

        A paper is excluded when its ``id`` or ``doi`` appears in
        ``known_paper_ids``.  Papers with no identifiers are kept.
        """
        if not known_paper_ids or not papers:
            return papers

        before_count = len(papers)
        new_papers = [
            p for p in papers
            if (getattr(p, "id", None) not in known_paper_ids)
            and (not getattr(p, "doi", None) or getattr(p, "doi", None) not in known_paper_ids)
        ]
        filtered_count = before_count - len(new_papers)
        if filtered_count:
            logger.info("survey_known_papers_filtered", count=filtered_count)

        if new_papers:
            return new_papers

        # All results are known — keep top-N as anchor seeds for survey diversity.
        # Only fall back to pure KB vector search when the broad search itself
        # returns zero results (empty papers list), not when all are known.
        seeds = [p for p in papers if (
            getattr(p, "id", None) in known_paper_ids
            or (getattr(p, "doi", None) and getattr(p, "doi", None) in known_paper_ids)
        )][:self.seed_known_max]
        logger.info(
            "survey_using_known_seeds",
            seed_count=len(seeds),
            seed_known_max=self.seed_known_max,
        )
        return seeds

    async def _store_references_to_all_kbs(
        self,
        papers: list[Any],
        kb_names: list[str],
        survey_query: str,
    ) -> int:
        """Store reference rows in SQLite for every KB beyond the first.

        ``kb_names[0]`` (the primary KB) is intentionally skipped here — callers
        are expected to invoke ``add_dois_to_kb`` directly for it using the DOIs
        from ``response.sources``.  Indices 1..n receive a lightweight
        ``kb_paper_references`` row per paper so a future ``add_dois_to_kb`` /
        rebuild can fully ingest them.

        Only papers with a non-null ``doi`` are stored (papers without a DOI
        cannot be looked up by a future ingestion command anyway).

        Returns the total number of NEW rows written.
        Never raises.
        """
        if self.session_store is None or len(kb_names) < 2:
            return 0

        extra_kbs = kb_names[1:]
        total = 0
        query_snippet = str(survey_query)[:200]

        for kb_name in extra_kbs:
            for paper in papers:
                doi = getattr(paper, "doi", None)
                if not doi:
                    continue  # skip: no DOI means can't re-ingest via add_dois_to_kb
                try:
                    authors = [str(a) for a in (getattr(paper, "authors", []) or [])]
                    abstract_raw = getattr(paper, "abstract", None)
                    abstract = abstract_raw[:500] if abstract_raw else None  # cap to avoid DB bloat
                    new = await self.session_store.store_paper_reference(
                        kb_name=kb_name,
                        doi=doi,
                        title=str(getattr(paper, "title", "") or "Untitled"),
                        authors=authors,
                        year=getattr(paper, "year", None),
                        abstract=abstract,
                        survey_query=query_snippet,
                    )
                    if new:
                        total += 1
                        logger.info(
                            "survey_reference_stored",
                            kb=kb_name,
                            doi=doi,
                        )
                except Exception as exc:
                    logger.warning(
                        "survey_reference_store_error",
                        kb=kb_name,
                        paper=str(getattr(paper, "title", "?"))[:50],
                        error=str(exc),
                    )

        logger.info(
            "survey_references_complete",
            extra_kbs=extra_kbs,
            total_new=total,
        )
        return total

    async def _analyze_abstracts_batch(
        self,
        papers: list[PaperCandidate],
        query: str,
        llm: Any,
        progress_events: list[dict[str, Any]] | None = None,
        progress_cb: Any = None,
    ) -> list[Theme]:
        """
        Analyze abstracts in batches and identify themes.

        Process:
        1. Score each paper's relevance (1-5)
        2. Accumulate insights across batches
        3. Identify themes from patterns

        Args:
            progress_events: Optional list the function appends per-batch
                progress dicts to (kind, current, total, batch_size). Allows
                the streaming caller to drain progress AFTER the await — used
                when an async generator wrapper is not in play.
            progress_cb: Optional ``async def cb(current, total, batch_size)``
                invoked before each batch. Preferred over ``progress_events``
                because it fires DURING execution, enabling live SSE updates.
        """
        logger.info("theme_analysis_start", total_papers=len(papers))

        # Filter papers with abstracts
        papers_with_abstracts = [p for p in papers if p.abstract]

        logger.info("theme_analysis_papers_with_abstracts", count=len(papers_with_abstracts))

        if not papers_with_abstracts:
            logger.warning("no_abstracts_found")
            return []

        # === Parallel batch analysis ===
        # Previously this loop awaited each `_analyze_batch` SEQUENTIALLY,
        # which dominated literature_survey latency (~3-6 min per batch ×
        # 4 batches = 20+ min wall time, even though each batch is just one
        # LLM call). Run them concurrently with a small semaphore so the
        # provider sees ~3 parallel completions, which is well within
        # OpenRouter / DeepSeek rate caps for a normal account.
        all_analyses: list[dict[str, Any]] = []
        # Per-call batch_size override; fall back to config-file default.
        _req = getattr(self, "_current_request", None)
        batch_size = (
            getattr(_req, "batch_size", None) or self.batch_size
        )
        total_batches = (len(papers_with_abstracts) + batch_size - 1) // batch_size
        batches = [
            papers_with_abstracts[i:i + batch_size]
            for i in range(0, len(papers_with_abstracts), batch_size)
        ]
        # Concurrency cap. 3 is conservative; raise carefully if rate
        # limits permit. Each call sends ~10-25 abstract previews.
        sem = asyncio.Semaphore(3)
        completed = {"n": 0}
        cb_lock = asyncio.Lock()

        async def _one_batch(idx: int, batch: list[PaperCandidate]) -> list[dict[str, Any]]:
            async with sem:
                try:
                    result = await self._analyze_batch(batch, query, llm)
                except Exception as e:
                    logger.warning("batch_analysis_exception", idx=idx + 1, error=str(e))
                    result = []
            # Emit progress AFTER each batch lands so the UI ticks in real
            # time despite parallel execution.
            async with cb_lock:
                completed["n"] += 1
                done_n = completed["n"]
                if progress_events is not None:
                    progress_events.append({
                        "kind": "batch_progress",
                        "stage": "abstract_analysis",
                        "current": done_n,
                        "total": total_batches,
                        "batch_size": len(batch),
                    })
                if progress_cb is not None:
                    try:
                        await progress_cb(done_n, total_batches, len(batch))
                    except Exception as _cb_exc:
                        logger.warning("batch_progress_cb_failed", error=str(_cb_exc))
            return result

        logger.info(
            "abstract_batches_parallel_start",
            total_batches=total_batches,
            parallelism=3,
        )
        batch_results = await asyncio.gather(
            *(_one_batch(i, b) for i, b in enumerate(batches))
        )
        for r in batch_results:
            all_analyses.extend(r)

        # Update papers with scores
        logger.info("batch_analysis_complete", successful_analyses=len(all_analyses), total_papers=len(papers_with_abstracts))

        for analysis in all_analyses:
            for p in papers_with_abstracts:
                if p.id == analysis.get("paper_id"):
                    p.relevance_score = analysis.get("relevance_score", 0)
                    break

        # === Relevance pre-filter for clustering ===
        # Only feed papers scoring >= 3/5 (on-topic) into theme identification.
        # Previously off-topic noise (papers the LLM scored 1-2) was diluting
        # the concept pool and producing themes that drifted away from the
        # user's query (e.g. "molecular networking" surfacing unrelated themes).
        # We still keep low-relevance papers in `papers` for stats / assignment
        # fallback, but they no longer shape the theme taxonomy.
        on_topic_analyses = [
            a for a in all_analyses
            if int(a.get("relevance_score", 0) or 0) >= 3
        ]
        logger.info(
            "theme_clustering_input",
            on_topic=len(on_topic_analyses),
            total=len(all_analyses),
            dropped_low_relevance=len(all_analyses) - len(on_topic_analyses),
        )
        # If filter wipes everything out (e.g. very strict LLM scoring) fall
        # back to using all analyses so we still produce *some* themes.
        clustering_analyses = on_topic_analyses if on_topic_analyses else all_analyses

        # Identify themes from on-topic analyses only
        themes = await self._identify_themes(clustering_analyses, query, llm)
        logger.info("themes_identified", count=len(themes), theme_names=[t.name for t in themes])

        # Assign papers to themes (all papers have abstracts). Pipe the
        # progress callback through so the parallel classifier emits live
        # "Theme assignment: 12/100" events to the SSE stream.
        async def _theme_assign_progress(done: int, tot: int) -> None:
            if progress_cb is not None:
                # Use a 4th positional arg as stage marker so the streaming
                # caller can route this to a separate progress card.
                try:
                    await progress_cb(done, tot, 0, "theme_assignment")
                except TypeError:
                    # Caller hasn't upgraded to the 4-arg signature — fall
                    # back to the 3-arg form so older wrappers don't crash.
                    with contextlib.suppress(Exception):
                        await progress_cb(done, tot, 0)
                except Exception:
                    pass

        await self._assign_papers_to_themes(
            papers_with_abstracts, themes, llm,
            progress_cb=_theme_assign_progress,
        )

        # Log theme statistics
        for theme in themes:
            logger.info("theme_stats", name=theme.name, paper_count=len(theme.papers))

        return themes

    async def _analyze_batch(
        self,
        batch: list[PaperCandidate],
        query: str,
        llm: Any
    ) -> list[dict[str, Any]]:
        """Analyze a single batch of papers."""
        # Format papers for prompt (shorter abstracts to save tokens)
        papers_text = "\n\n".join([
            f"PAPER {i+1} (ID: {p.id}):\nTitle: {p.title}\nAbstract: {p.abstract[:300]}"
            for i, p in enumerate(batch)
        ])

        prompt = f"""Analyze these papers for the topic: "{query}"

For each paper return JSON with:
- paper_id: use the ID shown
- relevance_score: 1-5, STRICTLY calibrated as follows:
    5 = paper is squarely about "{query}" (core method, central application, or direct contribution)
    4 = paper directly studies "{query}" but is one step removed (e.g. an application of it)
    3 = paper uses or touches "{query}" but it is not the focus
    2 = paper mentions "{query}" only in passing or treats a tangentially related topic
    1 = paper is off-topic w.r.t. "{query}" even if surface keywords match
- key_concepts: 3-6 SHORT noun phrases ONLY tightly related to "{query}". Skip generic
  concepts like "statistics", "machine learning", "case study" unless they are central.
- methodology: brief methods used
- contribution: main contribution

Be ruthless with 1s and 2s — off-topic papers should NOT score 3+. We are
building a focused survey on "{query}" and noise damages the themes.

PAPERS:
{papers_text}

JSON ONLY (no other text):
{{
  "analyses": [
    {{"paper_id": "...", "relevance_score": 4, "key_concepts": ["..."], "methodology": "...", "contribution": "..."}}
  ]
}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            # Bumped 4000 → 8000: a 25-paper batch can need 6-7k tokens for
            # the full analyses array with key_concepts + methodology +
            # contribution per paper. Truncation here cascades into
            # "Expecting ',' delimiter" JSON errors that wipe the entire
            # batch's relevance scores.
            response = await llm.complete(
                messages, temperature=0.3, max_tokens=8000, stage="survey.cluster"
            )

            if not response:
                logger.warning("batch_analysis_empty_response")
                return []

            # Parse JSON with better error handling. The previous greedy
            # regex `\{.*\}` matched FROM the first `{` TO the LAST `}`
            # which can span unrelated text on multi-block responses; this
            # is fine for our prompt but breaks when the closing brace is
            # truncated. We try a salvage pass that closes orphan brackets
            # when the strict parse fails.
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = self._fix_json(json_match.group())
                try:
                    data = json.loads(json_str)
                    return data.get("analyses", [])
                except json.JSONDecodeError as _de:
                    salvaged = self._salvage_truncated_json(json_str)
                    if salvaged is not None:
                        logger.info(
                            "batch_analysis_json_salvaged",
                            recovered=len(salvaged),
                            error=str(_de),
                        )
                        return salvaged
                    raise
            return []
        except Exception as e:
            logger.error("batch_analysis_failed", error=str(e), response_preview=response[:200] if 'response' in locals() else "N/A")
            return []

    def _salvage_truncated_json(self, json_str: str) -> list[dict[str, Any]] | None:
        """Best-effort recovery from a truncated LLM analyses array."""
        from perspicacite.rag.utils.json_salvage import salvage_truncated_array
        return salvage_truncated_array(json_str, "analyses")

    def _fix_json(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses."""
        import re

        from perspicacite.rag.utils.json_salvage import clean_control_chars
        # Strip raw control chars some providers emit inside string values.
        json_str = clean_control_chars(json_str)
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Remove any markdown code block markers
        json_str = json_str.replace("```json", "").replace("```", "")
        return json_str.strip()

    async def _identify_themes(
        self,
        analyses: list[dict[str, Any]],
        query: str,
        llm: Any
    ) -> list[Theme]:
        """Identify research themes from all analyses."""
        logger.info("identifying_themes", analyses_count=len(analyses))

        # If no analyses, create generic themes based on the query
        if not analyses:
            logger.warning("no_analyses_for_themes", creating_generic_themes=True)
            return [
                Theme(name=f"{query.title()} Research", description=f"Research related to {query}"),
                Theme(name="Methods and Approaches", description="Methodologies and techniques"),
                Theme(name="Applications", description="Practical applications and use cases"),
            ]

        # Aggregate key concepts
        all_concepts = []
        for a in analyses:
            all_concepts.extend(a.get("key_concepts", []))

        # If no concepts found, create generic themes
        if not all_concepts:
            logger.warning("no_concepts_found", creating_generic_themes=True)
            return [
                Theme(name=f"{query.title()} Research", description=f"Research related to {query}"),
                Theme(name="Related Topics", description="Related research areas"),
            ]

        concepts_text = ", ".join(set(all_concepts))
        logger.info("theme_concepts_aggregated", unique_concepts=len(set(all_concepts)))

        # Anchored on the user query. The old prompt asked for "3-8 themes"
        # which over-encourages padding even when only one or two are
        # genuinely relevant. The new prompt:
        #   - foregrounds the query as the topic anchor
        #   - asks for 1-5 themes (fewer if the concept pool is narrow)
        #   - explicitly rejects themes that aren't tightly related to the query
        #   - asks the LLM to drop noisy / tangential concepts
        prompt = f"""You are clustering research concepts into themes for a
literature survey on the topic: "{query}".

INSTRUCTIONS:
- Identify between 1 and 5 themes that DIRECTLY advance understanding of "{query}".
- Prefer FEWER, HIGH-QUALITY themes. If the corpus only supports one or two
  tightly-relevant themes, return only one or two.
- IGNORE concepts that are tangential to "{query}" (e.g. unrelated diseases,
  unrelated methods, generic statistics). Do NOT invent a catch-all theme to
  absorb them.
- Each theme must be specific to "{query}" — reject generic themes like
  "Methods", "Applications", "Future Work".
- The theme NAME should make the connection to "{query}" obvious.

CONCEPTS FROM ON-TOPIC PAPERS:
{concepts_text}

Respond in JSON format:
{{
    "themes": [
        {{
            "name": "Specific theme name relevant to {query}",
            "description": "How this theme advances {query} research (1-2 sentences)"
        }}
    ]
}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = await llm.complete(
                messages, temperature=0.3, max_tokens=2000, stage="survey.cluster"
            )

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                theme_data = data.get("themes", [])
                themes = [Theme(name=t["name"], description=t["description"]) for t in theme_data]
                logger.info("themes_parsed_successfully", count=len(themes))
                return themes
            logger.warning("no_json_found_in_theme_response")
            return []
        except Exception as e:
            logger.error("theme_identification_failed", error=str(e))
            return []

    async def _assign_papers_to_themes(
        self,
        papers: list[PaperCandidate],
        themes: list[Theme],
        llm: Any,
        progress_cb: Any = None,
    ):
        """Assign papers to themes based on content.

        All papers passed to this method are expected to have abstracts.
        Papers without abstracts are filtered out during candidate conversion.

        Performance: runs the per-paper LLM classification calls **in
        parallel** (concurrency cap of 8) — previously sequential, which
        dominated literature_survey end-to-end latency at ~2s/paper. With
        100 papers that's 25s instead of 3+ minutes. ``progress_cb`` (if
        provided) is awaited as ``(done, total)`` for live SSE progress.
        """
        if not themes:
            logger.warning("no_themes_to_assign_papers")
            return

        theme_names = [t.name for t in themes]

        # Skip off-topic papers during theme assignment. Their concepts didn't
        # shape the theme taxonomy (see _analyze_abstracts_batch pre-filter),
        # so forcing them into a theme just inflates paper counts and dilutes
        # the recommendations downstream. They remain in `papers` for stats.
        on_topic_papers = [
            p for p in papers
            if (p.relevance_score or 0) >= 3
        ]
        # Hard fallback if filter is empty.
        if not on_topic_papers:
            on_topic_papers = papers
            logger.info("theme_assignment_no_on_topic_falling_back", total=len(papers))
        else:
            logger.info(
                "theme_assignment_filtered",
                on_topic=len(on_topic_papers),
                total=len(papers),
                dropped=len(papers) - len(on_topic_papers),
            )
        papers = on_topic_papers

        logger.info("assigning_papers_to_themes", papers_count=len(papers), themes=theme_names)

        # Concurrency cap balances OpenRouter rate limits vs end-to-end
        # latency. 8 keeps us well under provider rate limits for the
        # short prompt + 100-token response shape.
        sem = asyncio.Semaphore(8)
        done_counter = {"n": 0}
        total = len(papers)
        cb_lock = asyncio.Lock()

        async def _classify(paper: PaperCandidate) -> tuple[PaperCandidate, list[str]]:
            prompt = f"""Which theme(s) does this paper belong to?

THEMES: {', '.join(theme_names)}

PAPER: {paper.title}
ABSTRACT: {paper.abstract[:400]}

Respond with theme names separated by commas, or "None" if no match."""
            async with sem:
                try:
                    messages = [{"role": "user", "content": prompt}]
                    response = await llm.complete(
                        messages, temperature=0.2, max_tokens=100, stage="survey.cluster"
                    )
                    if not response:
                        # Empty LLM response — count as no-match rather than
                        # crashing on "None not in NoneType".
                        return paper, []
                    if "None" in response:
                        return paper, []
                    assigned = [
                        t.strip() for t in response.split(",")
                        if t.strip() in theme_names
                    ]
                    return paper, assigned
                except Exception as e:
                    logger.warning(
                        "paper_theme_assignment_failed",
                        paper=paper.title[:50],
                        error=str(e),
                    )
                    return paper, []
                finally:
                    if progress_cb is not None:
                        async with cb_lock:
                            done_counter["n"] += 1
                            try:
                                await progress_cb(done_counter["n"], total)
                            except Exception as _cb_exc:
                                logger.warning(
                                    "theme_assign_progress_cb_failed",
                                    error=str(_cb_exc),
                                )

        results = await asyncio.gather(
            *(_classify(p) for p in papers), return_exceptions=False
        )

        assigned_count = 0
        for paper, assigned in results:
            if assigned:
                paper.themes = assigned
                assigned_count += 1
                for theme_name in assigned:
                    for theme in themes:
                        if theme.name == theme_name:
                            theme.papers.append(paper.__dict__)
                            break

        # If no papers were assigned, assign all to first theme as fallback
        if assigned_count == 0 and themes and papers:
            logger.warning("no_papers_assigned", using_fallback_assignment=True)
            for paper in papers:
                paper.themes = [themes[0].name]
                themes[0].papers.append(paper.__dict__)
            assigned_count = len(papers)

        logger.info("paper_theme_assignment_complete", assigned=assigned_count, total=len(papers))

    async def _generate_recommendations(
        self,
        papers: list[PaperCandidate],
        themes: list[Theme],
        llm: Any
    ):
        """Generate AI recommendations for deep analysis."""
        logger.info("generating_recommendations", total_papers=len(papers))

        # Ensure all papers have at least a minimum relevance score.
        # Using 1.0 (below the relevant_threshold of 3.0) so unscored papers
        # are excluded rather than promoted — only papers the LLM explicitly
        # scored >= 3.0 are eligible for recommendations.
        for p in papers:
            if p.relevance_score < 1.0:  # If no score assigned, give default
                p.relevance_score = 1.0  # Default below relevant threshold

        # Filter to on-topic papers. Bumped from 1.5 → 3.0 to match the
        # stricter clustering pipeline above: only papers the LLM scored as
        # "uses or touches the query" or better are eligible to be
        # recommended for deep reading.
        relevant_threshold = 3.0
        relevant_papers = [p for p in papers if p.relevance_score >= relevant_threshold]

        logger.info("relevant_papers_filtered", count=len(relevant_papers), threshold=relevant_threshold)

        # Graceful relaxation: if stricter threshold yields nothing, fall
        # back to >=2 (tangential ok), then to all papers as last resort.
        if not relevant_papers:
            relevant_papers = [p for p in papers if p.relevance_score >= 2.0]
            logger.warning(
                "relevant_papers_relaxed_threshold",
                count=len(relevant_papers), threshold=2.0,
            )
        if not relevant_papers:
            logger.warning("no_relevant_papers_using_all", total_papers=len(papers))
            relevant_papers = papers

        # Select diverse, high-impact papers
        # Criteria: citation count, theme representation, recency

        recommendations = []

        # 1. Highest cited from each theme (representative)
        for theme in themes:
            theme_papers = [p for p in relevant_papers if theme.name in p.themes]
            if theme_papers:
                top_cited = max(theme_papers, key=lambda p: p.citation_count)
                if top_cited not in recommendations:
                    recommendations.append(top_cited)
                    top_cited.recommended = True
                    top_cited.reason = f"Highly cited in theme: {theme.name}"

        # 2. Recent papers (last 3 years) with good relevance
        recent_papers = [
            p for p in relevant_papers
            if p.year and p.year >= datetime.now().year - 3 and p not in recommendations
        ]
        recent_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        for p in recent_papers[:5]:
            p.recommended = True
            p.reason = "Recent advance in the field"
            recommendations.append(p)

        # 3. Fill remaining slots with high-relevance papers
        remaining = [p for p in relevant_papers if p not in recommendations]
        remaining.sort(key=lambda p: (p.relevance_score, p.citation_count), reverse=True)

        for p in remaining[:self.max_deep_analysis - len(recommendations)]:
            p.recommended = True
            p.reason = "Highly relevant to the topic"
            recommendations.append(p)

        logger.info("recommendations_complete", count=len(recommendations))

    def _generate_interim_summary(
        self, session: SurveySession, known_context: str = ""
    ) -> str:
        """Generate interim summary for user selection."""
        lines = [
            f"# Literature Survey: {session.query}",
            "",
            f"**Found {len(session.papers)} papers** across {len(session.themes)} research themes.",
            "",
            "## Identified Themes",
            "",
        ]

        for theme in session.themes:
            paper_count = len(theme.papers)
            lines.append(f"### {theme.name}")
            lines.append(f"{theme.description}")
            lines.append(f"*{paper_count} papers*")
            lines.append("")

        recommended = [p for p in session.papers if p.recommended]
        lines.extend([
            "## Recommendations",
            "",
            f"**{len(recommended)} papers recommended** for deep analysis (of {self.max_deep_analysis} max).",
            "",
            "The AI has selected papers based on:",
            "- Citation impact (seminal works)",
            "- Theme representation (diverse coverage)",
            "- Recency (recent advances)",
            "- Relevance to your query",
            "",
            "### Next Steps",
            "1. Review the recommended papers below",
            "2. Add/remove papers as needed",
            "3. Click 'Generate Survey' for full analysis",
            "",
            "---",
            "",
            "## Recommended Papers",
            "",
        ])

        for p in recommended[:20]:  # Show top 20
            lines.append(f"- **{p.title}** ({p.year})")
            lines.append(f"  - Authors: {', '.join(p.authors[:3])}")
            lines.append(f"  - Citations: {p.citation_count} | Relevance: {p.relevance_score}/5")
            lines.append(f"  - Why: {p.reason}")
            lines.append("")

        if known_context:
            lines.extend([
                "",
                "---",
                "",
                "## Already in Your Knowledge Base(s)",
                "",
                known_context,
            ])

        return "\n".join(lines)

    def _convert_to_sources(self, papers: list[PaperCandidate]) -> list[SourceReference]:
        """Convert papers to source references.

        SourceReference.relevance_score is constrained to [0.0, 1.0] by its
        pydantic schema, but PaperCandidate.relevance_score can carry the
        raw 0-5 LLM rating (set at literature_survey.py:1090 from
        ``analysis.get("relevance_score", 0)``). Without normalization the
        entire stream errors out with `Input should be less than or equal
        to 1`. Clamp + divide-by-5 when the score is clearly on a 0-5 scale,
        otherwise clamp to [0,1] directly. Bug B-7 in the 2026-05-25
        mode-runner audit.
        """
        def _normalize(raw: float | int | None) -> float:
            if raw is None:
                return 0.0
            r = float(raw)
            if r > 1.0:
                # LLM rating scale (e.g. 0-5) — rescale to [0,1].
                r = r / 5.0
            return max(0.0, min(1.0, r))

        return [
            SourceReference(
                title=p.title,
                authors=", ".join(p.authors[:3]) if p.authors else None,  # type: ignore[arg-type]  # _coerce_authors validator widens authors at runtime
                year=p.year,
                doi=p.doi,
                relevance_score=_normalize(p.relevance_score),
                # Include the KB paper_id (e.g. "scifact:N") so downstream
                # clients (eval harness, UI) can match KB papers that have no
                # DOI.  Web-sourced papers carry a UUID or external ID here;
                # the eval harness's id_mapper handles both gracefully.
                paper_id=p.id if p.id else None,
            )
            for p in papers
        ]

    # Public methods for API/UI integration

    def get_session(self, session_id: str) -> SurveySession | None:
        """Get a survey session by ID."""
        return self.sessions.get(session_id)

    def update_selection(self, session_id: str, selected_paper_ids: list[str]) -> bool:
        """Update user paper selection."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Validate - don't exceed max
        if len(selected_paper_ids) > self.max_deep_analysis:
            selected_paper_ids = selected_paper_ids[:self.max_deep_analysis]

        session.selected_papers = selected_paper_ids
        return True

    async def generate_deep_analysis(
        self,
        session_id: str,
        llm: Any,
    ) -> RAGResponse:
        """
        Generate deep analysis for selected papers.
        
        This is Phase 2 - after user selection.
        """
        session = self.sessions.get(session_id)
        if not session:
            return RAGResponse(
                answer="Session not found.",
                sources=[],
                mode=RAGMode.LITERATURE_SURVEY,
            )

        # Get selected papers
        selected = [p for p in session.papers if p.id in session.selected_papers]

        if not selected:
            return RAGResponse(
                answer="No papers selected for analysis.",
                sources=[],
                mode=RAGMode.LITERATURE_SURVEY,
            )

        logger.info("deep_analysis_start", session_id=session_id, papers=len(selected))

        # TODO: Download full texts and analyze
        # For now, return structured summary

        survey_report = await self._generate_survey_report(session, selected, llm)

        return RAGResponse(
            answer=survey_report,
            sources=self._convert_to_sources(selected),
            mode=RAGMode.LITERATURE_SURVEY,
            metadata={
                "session_id": session_id,
                "phase": "completed",
                "papers_analyzed": len(selected),
                "themes": len(session.themes),
            }
        )

    async def _generate_survey_report(
        self,
        session: SurveySession,
        selected_papers: list[PaperCandidate],
        llm: Any
    ) -> str:
        """Generate final structured survey report."""
        # NOTE (Capsule Cycle B): No multimodal hook here. The survey report is
        # deterministic text aggregation; the LLM calls in this mode
        # (_analyze_batch, _identify_themes, _assign_papers_to_themes) are
        # intermediate paper-metadata processing, not final user-facing synthesis.
        # If/when a final-synthesis LLM call is added, wire via
        # perspicacite.rag.multimodal.wrap_messages_for_chunks here.
        # Build a concise reference block for the LLM to synthesize from.
        papers_text_parts = []
        for i, p in enumerate(selected_papers[:20], 1):
            abstract = (p.abstract or "")[:600]
            authors = ", ".join(p.authors[:3]) if p.authors else "Unknown"
            papers_text_parts.append(
                f"[{i}] {p.title} ({p.year}) — {authors}\n{abstract}"
            )
        papers_text = "\n\n".join(papers_text_parts)

        themes_text = "\n".join(
            f"- {t.name}: {t.description}" for t in session.themes
        ) if session.themes else "(no themes identified)"

        prompt = (
            f'Write a structured literature survey on: "{session.query}"\n\n'
            f"Research themes identified:\n{themes_text}\n\n"
            f"Key papers (cite as [N] — use only information present in the abstracts below):\n"
            f"{papers_text}\n\n"
            "Write a 400-600 word synthesis covering:\n"
            "1. State of the field\n"
            "2. Key themes and methodological patterns across the papers\n"
            "3. Notable contributions and relationships between works\n"
            "4. Open challenges and future directions\n\n"
            "Cite papers by number [N]. Do not invent facts or fabricate citations."
        )

        try:
            return await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
                stage="survey.synthesis",
            )
        except Exception as _exc:
            logger.warning("literature_survey_synthesis_llm_failed", error=str(_exc))
            # Fall back to annotated template
            lines = [
                f"# Literature Survey: {session.query}",
                f"\n## Themes\n{themes_text}",
                "\n## Annotated Bibliography",
            ]
            for i, p in enumerate(selected_papers[:20], 1):
                lines.append(f"\n{i}. **{p.title}** ({p.year})")
                lines.append(f"   {(p.abstract or '')[:300]}...")
            return "\n".join(lines)
