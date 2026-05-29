"""Basic RAG Mode - Exact implementation from release package.

Basic RAG performs simple retrieval and generation:
- Single query (no rephrasing)
- Vector similarity search with optional hybrid retrieval
- Basic document selection
- Direct response generation (no refinement)
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from perspicacite.config.schema import MultimodalMode
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.provenance.context import get_collector
from perspicacite.rag.code_excerpts import collect_code_excerpts
from perspicacite.rag.conversation_helpers import (
    build_user_message_with_history,
    compute_retrieval_query,
    format_conversation_block,
)
from perspicacite.rag.figure_refs import collect_figure_refs
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.multimodal import wrap_messages_for_chunks
from perspicacite.rag.query_scope import resolve_paper_scope_for_query
from perspicacite.rag.telemetry import emit_phase
from perspicacite.rag.utils import (
    flatten_paper_results_to_chunks,
    format_documents_for_prompt,
    format_paper_results_for_prompt,
    format_references,
    get_doc_citation,
    get_system_prompt,
)

logger = get_logger("perspicacite.rag.modes.basic")

# ---------------------------------------------------------------------------
# HyDE model resolution
# ---------------------------------------------------------------------------
# HyDE uses a cheap/fast model for the single hypothetical-abstract call.
# Operators can pin it via ``config.llm.models.hyde`` / providers_per_stage.
# When the stage isn't pinned we fall back to Haiku / deepseek-chat rather
# than the synthesis model (which may be much more expensive).

_HYDE_FALLBACK_PROVIDER = "deepseek"
_HYDE_FALLBACK_MODEL = "deepseek-chat"


def _resolve_hyde_model(config: Any) -> tuple[str, str]:
    """Return ``(provider, model)`` for the HyDE generation call.

    Prefers the ``"hyde"`` stage pin in ``config.llm.models`` /
    ``config.llm.providers_per_stage``; falls back to
    ``(_HYDE_FALLBACK_PROVIDER, _HYDE_FALLBACK_MODEL)`` so operators
    who don't configure the stage still get a cheap model.
    """
    if config is None:
        return _HYDE_FALLBACK_PROVIDER, _HYDE_FALLBACK_MODEL
    llm_cfg = getattr(config, "llm", None)
    if llm_cfg is None:
        return _HYDE_FALLBACK_PROVIDER, _HYDE_FALLBACK_MODEL
    models = getattr(llm_cfg, "models", {}) or {}
    providers = getattr(llm_cfg, "providers_per_stage", {}) or {}
    provider = providers.get("hyde", _HYDE_FALLBACK_PROVIDER)
    model = models.get("hyde", _HYDE_FALLBACK_MODEL)
    return provider, model


async def _apply_hybrid_if_requested(
    request: Any,
    paper_results: list[dict],
    query: str,
    llm: Any,
) -> list[dict]:
    """Re-score paper_results with the hybrid retriever when the caller supplies
    explicit bm25_weight / vector_weight overrides.

    Only fires when at least one weight field is non-None on the request.
    Uses full_text (present in both two-pass and single-pass outputs) for BM25
    scoring.  Does NOT call the LLM — weights are resolved from request fields
    with (0.5, 0.5) as fallback default.

    Returns the same list re-sorted by hybrid score, paper_score updated.
    """
    from types import SimpleNamespace

    from perspicacite.retrieval.hybrid import hybrid_retrieval, resolve_hybrid_weights

    req_bw = getattr(request, "bm25_weight", None)
    req_vw = getattr(request, "vector_weight", None)
    if (req_bw is None and req_vw is None) or not paper_results:
        return paper_results

    final_vw, final_bw = resolve_hybrid_weights(request, default=(0.5, 0.5))
    docs = [
        SimpleNamespace(page_content=p.get("full_text", ""), score=p.get("paper_score", 0.0))
        for p in paper_results
    ]
    vector_scores = [p.get("paper_score", 0.0) for p in paper_results]

    hybrid_ranked = await hybrid_retrieval(
        query=query,
        documents=docs,
        vector_scores=vector_scores,
        vector_weight=final_vw,
        bm25_weight=final_bw,
        use_llm_weights=False,
        llm=llm,
    )

    doc_to_paper = {id(d): p for d, p in zip(docs, paper_results, strict=True)}
    reranked = []
    for doc, hybrid_score in hybrid_ranked:
        paper = dict(doc_to_paper[id(doc)])
        paper["paper_score"] = float(hybrid_score)
        reranked.append(paper)

    logger.info(
        "basic_hybrid_applied",
        vector_weight=final_vw,
        bm25_weight=final_bw,
        n=len(reranked),
    )
    return reranked


def _clean_query_for_keyword_search(q: str) -> str:
    # SciLEx / SemanticScholar wrap the query in quotes for exact-phrase
    # matching. That kills natural-language questions and hyphenated terms
    # (e.g. ``"feature-based molecular networking (FBMN)"`` matches nothing
    # while ``feature based molecular networking FBMN`` finds the canonical
    # papers). Normalize the query for keyword search:
    #   * strip leading interrogatives / imperatives
    #   * drop trailing punctuation
    #   * replace hyphens, parens, and other punctuation with spaces
    import re as _re

    s = (q or "").strip()
    s = _re.sub(
        r"^(what(?:'s| is| are)?|how(?: does| do| can| would)?|why(?: is| are)?|"
        r"when(?: is| are)?|where(?: is| are)?|who(?: is| are)?|"
        r"can you|could you|please|"
        r"summari[sz]e|summary of|describe|explain|tell me (?:about|the)|"
        r"compare|list|find|show me|give me|provide)\b\s*",
        "", s, flags=_re.IGNORECASE,
    )
    # Hyphens and parens defeat exact-phrase match — replace with spaces.
    s = _re.sub(r"[-/(),:;?!*\"'`]+", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s or q


async def _web_fallback_papers(
    *,
    query: str,
    databases: list[str] | None,
    max_docs: int,
    config: Any = None,
    context: str | None = None,
    optimize_query: bool | None = None,
    app_state: Any = None,
    telemetry: list[dict[str, Any]] | None = None,
    min_relevance: float = 0.0,
    reranker_override: bool | None = None,
) -> list[dict[str, Any]]:
    # Live literature search used when the KB returns nothing. Routes
    # through the unified resolve_papers_pipeline (aggregator → Crossref
    # enrich → MiniLM rerank) and converts Paper objects to the dict
    # shape expected by downstream RAG code.
    from perspicacite.rag.resolve_papers import resolve_papers_pipeline

    # Resolve effective app_state: prefer explicit arg, fall back to config
    # wrapper so legacy call sites (config-only) still work.
    effective_app_state = app_state
    if effective_app_state is None and config is not None:
        class _AppStateShim:
            def __init__(self, cfg: Any) -> None:
                self.config = cfg
                self.llm_client = None
        effective_app_state = _AppStateShim(config)

    papers = await resolve_papers_pipeline(
        query=query,
        databases=databases,
        max_docs=max_docs,
        app_state=effective_app_state,
        telemetry=telemetry,
        enrich=True,
        rerank=True,
        reranker_override=reranker_override,
        min_relevance=min_relevance,
        optimize_query=optimize_query,
        context=context,
    )

    candidates: list[dict[str, Any]] = []
    for p in papers:
        if not p.title:
            continue
        # Resolve the originating database(s) from metadata.sources.
        all_sources: list[str] = []
        meta_sources = (getattr(p, "metadata", None) or {}).get("sources")
        if isinstance(meta_sources, list):
            for s in meta_sources:
                _sl = str(s).lower() if s else ""
                if _sl and _sl != "scilex" and _sl not in all_sources:
                    all_sources.append(_sl)
        src_str: str | None = all_sources[0] if all_sources else None
        if not src_str:
            src_obj = getattr(p, "source", None)
            src_str = getattr(src_obj, "value", None) or (
                str(src_obj).replace("PaperSource.", "").lower() if src_obj else None
            )
        best_url = (
            getattr(p, "url", None)
            or getattr(p, "pdf_url", None)
            or (f"https://doi.org/{p.doi}" if p.doi else None)
        )
        abstract = p.abstract or ""
        candidates.append({
            "paper_id": p.id or p.doi or p.title,
            "title": p.title,
            "abstract": abstract,
            "chunk_text": abstract,
            "full_text": abstract,
            "authors": [a.name for a in (p.authors or [])],
            "year": p.year,
            "journal": getattr(p, "journal", None),
            "doi": p.doi,
            "url": best_url,
            "source": src_str,
            "source_apis": None,
            "sources_all": p.discovery_sources or all_sources or None,
            "enrichment_sources": p.enrichment_sources or None,
            "citation_count": getattr(p, "citation_count", None),
            "kb_name": None,
            "paper_score": 0.5,
        })

    logger.info(
        "basic_web_fallback",
        query=query,
        candidates=len(candidates),
        threshold=min_relevance,
    )
    return candidates


async def _apply_copyright_filter(
    *,
    answer: str,
    paper_results: list[dict[str, Any]],
    llm: Any,
    config: Any,
) -> str:
    """Defense-in-depth copyright check on synthesis output.

    Called from each RAG-mode synthesis path. Resolves the
    ``copyright_filter`` section from the runtime Config (when wired
    through ``self.config``), builds source dicts from the paper-result
    list (using ``full_text`` as the comparison body), and runs the
    configured action (log / quote / strip / rewrite).

    No-op when sources are empty or the answer is empty.
    """
    if not answer or not paper_results:
        return answer
    try:
        from perspicacite.rag.copyright_filter import CopyrightFilter
        cf_cfg = getattr(config, "copyright_filter", None)
        if cf_cfg is None or not getattr(cf_cfg, "enabled", True):
            return answer
        sources = [
            {
                "text": p.get("full_text") or "",
                "title": p.get("title"),
            }
            for p in paper_results
        ]
        cf = CopyrightFilter(
            enabled=cf_cfg.enabled,
            action=getattr(cf_cfg, "action", "log"),
            min_ngram=getattr(cf_cfg, "min_ngram", 8),
            llm_client=llm,
            rewrite_model=getattr(cf_cfg, "rewrite_model", "claude-haiku-4-5"),
            rewrite_provider=getattr(cf_cfg, "rewrite_provider", "anthropic"),
        )
        return await cf.apply(answer, sources)
    except Exception as exc:
        # Filter is best-effort; never break the synthesis flow.
        logger.warning("copyright_filter_failed", error=str(exc))
        return answer


class BasicRAGMode(BaseRAGMode):
    """
    Basic RAG Mode - Exact port from release package core/core.py

    Characteristics:
    - Single query retrieval (no query expansion)
    - Vector-based similarity search with optional hybrid retrieval
    - No response refinement
    - Fastest mode, suitable for simple factual queries
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.initial_docs = config.knowledge_base.default_top_k * 3  # 30 default
        self.final_max_docs = 5
        self.max_docs_per_source = 2

        # Enable hybrid retrieval by default for better retrieval quality
        rag_settings = getattr(config.rag_modes, "basic", None)
        if rag_settings is None:
            rag_settings = {}
        elif hasattr(rag_settings, "model_dump"):
            rag_settings = rag_settings.model_dump()
        elif hasattr(rag_settings, "dict"):
            rag_settings = rag_settings.dict()

        self.use_hybrid = rag_settings.get("use_hybrid", True)
        self.use_two_pass = getattr(config.knowledge_base, "use_two_pass", True)

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """
        Execute Basic RAG - single query, direct retrieval, no refinement.

        Ported from: core/core.py::retrieve_documents() and get_response()
        """
        logger.info(
            "basic_rag_start",
            query=request.query,
            use_hybrid=self.use_hybrid,
            use_two_pass=self.use_two_pass,
        )

        dkb = self._build_kb_retriever(request, vector_store, embedding_provider)
        collection = dkb.collection_name
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined  # type: ignore[misc]

        # HyDE: replace the retrieval query with a synthetic abstract.
        if getattr(request, "use_hyde", False):
            from perspicacite.rag.modes.hyde_query import generate_hyde_query
            hyde_provider, hyde_model = _resolve_hyde_model(getattr(self, "config", None))
            retrieval_query = await generate_hyde_query(
                claim=retrieval_query,
                llm_client=llm,
                model=hyde_model,
                provider=hyde_provider,
            )
            logger.info("basic_hyde_applied", query_preview=retrieval_query[:120])

        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = max(1, getattr(request, "max_papers_retrieval", None) or self.final_max_docs)

        if self.use_two_pass:
            # Two-pass retrieval — identify papers, then fetch all their chunks.
            # Pass top_k=initial_docs so pass-1 evaluates a large candidate pool
            # (default 30 chunks → best `cap` papers), giving proper over-retrieve.
            # Previously `top_k=cap` (5) was passed, limiting the candidate pool
            # to only 10 chunks and causing high citation-miss rates.
            paper_results = await dkb.search_two_pass(
                retrieval_query,
                top_k=self.initial_docs,
                paper_scope=scope,
                max_papers_cap=cap,
            )
            logger.info("basic_two_pass", papers=len(paper_results), candidate_pool=self.initial_docs)
        else:
            # Legacy chunk-level retrieval (no two-pass)
            chunk_results = await dkb.search(retrieval_query, top_k=self.initial_docs)
            paper_results = []
            for r in chunk_results:
                meta = r.get("metadata")
                paper_results.append(
                    {
                        "paper_id": getattr(meta, "paper_id", None) if meta else None,
                        "paper_score": r.get("score", 0.0),
                        "title": getattr(meta, "title", None) if meta else None,
                        "authors": getattr(meta, "authors", None) if meta else None,
                        "year": getattr(meta, "year", None) if meta else None,
                        "doi": getattr(meta, "doi", None) if meta else None,
                        "full_text": r.get("text", ""),
                        "kb_name": r.get("kb_name"),
                    }
                )
            logger.info("basic_chunk_retrieval", chunks=len(chunk_results))

        # Hybrid re-scoring: honour bm25_weight / vector_weight when supplied.
        paper_results = await _apply_hybrid_if_requested(
            request, paper_results, retrieval_query, llm
        )

        # Web-search fallback when the KB returned nothing (no KB selected,
        # KB doesn't exist, or KB has no relevant docs). Mirrors the same
        # logic in execute_stream so non-streaming callers behave the same.
        web_fallback_used = False
        if not paper_results:
            paper_results = await _web_fallback_papers(
                query=retrieval_query,
                databases=request.databases,
                max_docs=cap,
                config=getattr(self, "config", None),
                app_state=getattr(request, "app_state", None),
                reranker_override=getattr(request, "use_reranker", None),
                context=getattr(request, "_resolved_context", None),
            )
            web_fallback_used = True

        # Apply optional recency weighting
        if getattr(request, "recency_weight", None):
            from perspicacite.retrieval.recency import apply_recency_weighting

            paper_results = apply_recency_weighting(
                paper_results,
                request.recency_weight,
                getattr(request, "recency_half_life_years", None),
            )

        # Provenance: record retrieval events
        retrieval_source = "web_search" if web_fallback_used else (request.kb_name or "default")
        _c = get_collector()
        if _c is not None:
            _c.add_trace(
                "retrieve",
                detail={
                    "kb_name": retrieval_source,
                    "count": len(paper_results),
                    "web_fallback": web_fallback_used,
                },
            )
            for rank, p in enumerate(paper_results):
                _c.add_retrieval(
                    paper_id=p.get("paper_id"),
                    doi=p.get("doi"),
                    title=p.get("title"),
                    score=float(p.get("paper_score", 0.0) or 0.0),
                    kb_name=p.get("kb_name"),
                    content_type=None,
                    pipeline_step=None,
                    rank=rank,
                    stage_label="basic.retrieve",
                )

        # Build sources from paper results
        sources = []
        for p in paper_results:
            sources.append(
                SourceReference(
                    title=p.get("title") or "Untitled",
                    authors=p.get("authors"),
                    year=p.get("year"),
                    journal=p.get("journal"),
                    doi=p.get("doi"),
                    url=p.get("url"),
                    source=p.get("source"),
                    source_apis=p.get("source_apis"),
                    sources_all=p.get("sources_all"),
                    enrichment_sources=p.get("enrichment_sources"),
                    relevance_score=p.get("paper_score", 0.0),
                    kb_name=p.get("kb_name"),
                    paper_id=p.get("paper_id"),
                    chunk_text=p.get("chunk_text"),
                    abstract=p.get("abstract") or p.get("chunk_text"),
                    metadata=p.get("paper_metadata"),
                )
            )

        # Step 2: Generate response using full paper context
        if paper_results:
            context = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
        else:
            context = "No relevant papers found."

        answer = await self._generate_response_from_context(
            query=request.query,
            context=context,
            llm=llm,
            request=request,
            num_papers=len(paper_results),
            preamble=scope.scope_note,
        )

        # Defense-in-depth copyright filter: detect any verbatim copying
        # from the source chunks into the LLM's answer. Config-driven
        # action (log / quote / strip / rewrite). Sources are the
        # paper-result dicts whose ``full_text`` field carries the
        # exact text the LLM had access to.
        answer = await _apply_copyright_filter(
            answer=answer, paper_results=paper_results, llm=llm,
            config=self.config,
        )

        # Step 6: Append references section to answer using utility function
        if sources:
            references = format_references(sources)
            answer = answer.strip() + "\n\n" + references

        logger.info("basic_rag_complete", sources=len(sources))

        # Sub-project C (2026-05-15): attach code excerpts + figure refs.
        _mm = getattr(self.config, "multimodal", None)
        _show_code = bool(getattr(_mm, "show_code", False)) if _mm else False
        _mode = getattr(_mm, "mode", None) if _mm else None
        _dc_chunks = flatten_paper_results_to_chunks(paper_results)
        _code_excerpts = collect_code_excerpts(_dc_chunks) if _show_code else []
        _figure_refs = (
            collect_figure_refs(_dc_chunks, capsule_root=Path(self.config.capsule.root))
            if _mode is not None and _mode != MultimodalMode.OFF
            else []
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.BASIC,
            iterations=1,
            web_search_used=False,
            code_excerpts=_code_excerpts,
            figures=_figure_refs,
        )

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute Basic RAG with true streaming output."""
        _phase_sink = getattr(request, "telemetry_sink", None)
        emit_phase(_phase_sink, phase="retrieve", state="running")
        yield StreamEvent.status("Basic RAG: Retrieving documents...")

        dkb = self._build_kb_retriever(request, vector_store, embedding_provider)
        collection = dkb.collection_name
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined  # type: ignore[misc]
            yield StreamEvent.status_kind(
                f"Rewrote question using conversation context: '{request.query}' → '{refined}'",
                kind="query_rephrased",
                original=request.query,
                rewritten=refined,
                by="conversation_history",
            )

        # HyDE: replace the retrieval query with a synthetic abstract.
        if getattr(request, "use_hyde", False):
            from perspicacite.rag.modes.hyde_query import generate_hyde_query
            yield StreamEvent.status("HyDE: Generating hypothetical abstract...")
            hyde_provider, hyde_model = _resolve_hyde_model(getattr(self, "config", None))
            hyde_query = await generate_hyde_query(
                claim=retrieval_query,
                llm_client=llm,
                model=hyde_model,
                provider=hyde_provider,
            )
            if hyde_query != retrieval_query:
                yield StreamEvent.status_kind(
                    "HyDE expanded query for retrieval",
                    kind="hyde_expanded",
                    original=retrieval_query,
                    expanded=hyde_query,
                )
            retrieval_query = hyde_query
            logger.info("basic_stream_hyde_applied", query_preview=retrieval_query[:120])

        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = max(1, getattr(request, "max_papers_retrieval", None) or self.final_max_docs)

        if self.use_two_pass:
            paper_results = await dkb.search_two_pass(
                retrieval_query,
                top_k=self.initial_docs,   # over-retrieve then cap — was top_k=cap (bug)
                paper_scope=scope,
                max_papers_cap=cap,
            )
            logger.info("basic_stream_two_pass", papers=len(paper_results), candidate_pool=self.initial_docs)
        else:
            chunk_results = await dkb.search(retrieval_query, top_k=cap)
            paper_results = []
            for r in chunk_results:
                meta = r.get("metadata")
                paper_results.append(
                    {
                        "paper_id": getattr(meta, "paper_id", None) if meta else None,
                        "paper_score": r.get("score", 0.0),
                        "title": getattr(meta, "title", None) if meta else None,
                        "authors": getattr(meta, "authors", None) if meta else None,
                        "year": getattr(meta, "year", None) if meta else None,
                        "doi": getattr(meta, "doi", None) if meta else None,
                        "full_text": r.get("text", ""),
                        "kb_name": r.get("kb_name"),
                    }
                )

        # Hybrid re-scoring: honour bm25_weight / vector_weight when supplied.
        paper_results = await _apply_hybrid_if_requested(
            request, paper_results, retrieval_query, llm
        )

        # Web-search fallback: if the KB query produced no documents (no KB
        # selected, KB doesn't exist, or KB is empty), do a live literature
        # search using the user-selected database providers. This delivers
        # the welcome-screen promise that we "fall back to web literature
        # search when your KB is insufficient" — historically only wired in
        # agentic / literature_survey modes.
        web_fallback_used = False
        if not paper_results:
            _db_pretty = ", ".join(
                d.replace("_", " ").title() for d in (request.databases or [])
            ) or "Semantic Scholar, OpenAlex, PubMed"
            # Announce the fall-back FIRST. The keyword optimizer below
            # is a ~1–5 s LLM call and used to run silently between
            # "Retrieving documents…" and this status, leaving the user
            # staring at a frozen-looking "Retrieving documents…" with
            # no idea why it was slow. Emitting the transition message
            # immediately gives them a visible heartbeat while the
            # optimizer works.
            yield StreamEvent.status(
                f"No KB results — falling back to web literature search across {_db_pretty}…"
            )
            # Tiny optimizer-running status so users know the next ~5 s
            # of silence is the rewrite step, not a hang.
            yield StreamEvent.status("Optimizing search query…")
            # Run the keyword optimizer UPFRONT so the user sees the
            # rewritten query BEFORE the slow aggregator call starts.
            # Previously this fired inside ``_web_fallback_papers`` and its
            # ``query_rephrased`` telemetry was only drained after the full
            # aggregator + Crossref + rerank cycle (10-15s), which made it
            # look like rephrasing wasn't happening for basic/advanced.
            search_query = retrieval_query
            try:
                from perspicacite.search.query_optimizer import optimize_query as _qopt
                _app = getattr(request, "app_state", None)
                opt_res = await _qopt(
                    query=retrieval_query,
                    context=None,
                    app_state=_app,
                    optimize_enabled=True,
                )
                if opt_res.applied and opt_res.searched_query:
                    search_query = opt_res.searched_query
                    yield StreamEvent.status_kind(
                        f"Rewrote search query: '{retrieval_query}' → '{search_query}'",
                        kind="query_rephrased",
                        original=retrieval_query,
                        rewritten=search_query,
                        by="keyword_optimizer",
                    )
            except Exception as _qe:
                logger.debug("basic_upfront_optimizer_failed", error=str(_qe))
            _telemetry = getattr(request, "telemetry_sink", None) or []
            paper_results = await _web_fallback_papers(
                query=search_query,
                databases=request.databases,
                max_docs=cap,
                config=getattr(self, "config", None),
                app_state=getattr(request, "app_state", None),
                reranker_override=getattr(request, "use_reranker", None),
                telemetry=_telemetry,
                # Tell the inner call to skip its own optimizer run; we
                # already did it upfront above.
                optimize_query=False,
            )
            # Drain telemetry into SSE so the UI sees query rewriting +
            # per-provider counts in real time.
            # When _telemetry is a CallbackTelemetrySink (MCP path), events
            # already flowed to ctx.report_progress live — skip the drain.
            if isinstance(_telemetry, list):
                for _ev in _telemetry:
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
                    elif _k == "selection_unavailable":
                        # User picked databases that the server's config didn't
                        # build. Surface a clear status so the UI doesn't show
                        # "got results" when really the picks were silently
                        # dropped (or, with the matching fix in web_search,
                        # so the UI sees WHY zero results came back).
                        yield StreamEvent.status_kind(
                            _ev.get("message", "Selected databases unavailable on this server."),
                            kind="selection_unavailable",
                            requested=_ev.get("requested", []),
                            available=_ev.get("available", []),
                        )
                    elif _k == "provider_progress" and _ev.get("phase") == "done":
                        _bp = _ev.get("by_provider", {}) or {}
                        _msg = (
                            ", ".join(
                                f"{src.replace('_',' ').title()}: {n}"
                                for src, n in sorted(
                                    _bp.items(), key=lambda kv: -kv[1]
                                )
                            )
                            if _bp
                            else f"Total {_ev.get('total', 0)} hits"
                        )
                        yield StreamEvent.status_kind(
                            f"Database results — {_msg}",
                            kind="provider_progress",
                            phase="done",
                            total=_ev.get("total", 0),
                            by_provider=_bp,
                        )
            web_fallback_used = True
            if paper_results:
                # Per-source breakdown so the user sees that multi-DB search
                # actually happened, even when the final top-k is dominated by
                # one provider (the rerank often clusters by source quality).
                from collections import Counter as _Counter
                _src_counts = _Counter(
                    (p.get("source") or "unknown") for p in paper_results
                )
                _src_summary = ", ".join(
                    f"{src.replace('_', ' ').title()}: {n}"
                    for src, n in _src_counts.most_common()
                )
                yield StreamEvent.status(
                    f"Web search returned {len(paper_results)} relevant paper(s) "
                    f"({_src_summary})."
                )
            else:
                yield StreamEvent.status(
                    "Web search returned no relevant papers."
                )

        # Apply optional recency weighting
        if getattr(request, "recency_weight", None):
            from perspicacite.retrieval.recency import apply_recency_weighting

            paper_results = apply_recency_weighting(
                paper_results,
                request.recency_weight,
                getattr(request, "recency_half_life_years", None),
            )

        # Provenance: record retrieval events. Source is web_search when
        # the fallback ran so the UI footer doesn't keep claiming kb=default.
        retrieval_source = "web_search" if web_fallback_used else (request.kb_name or "default")
        _c = get_collector()
        if _c is not None:
            _c.add_trace(
                "retrieve",
                detail={
                    "kb_name": retrieval_source,
                    "count": len(paper_results),
                    "web_fallback": web_fallback_used,
                },
            )
            for rank, p in enumerate(paper_results):
                _c.add_retrieval(
                    paper_id=p.get("paper_id"),
                    doi=p.get("doi"),
                    title=p.get("title"),
                    score=float(p.get("paper_score", 0.0) or 0.0),
                    kb_name=p.get("kb_name") or (
                        "web_search" if web_fallback_used else None
                    ),
                    content_type=None,
                    # Surface the originating provider (e.g. "google_scholar",
                    # "openalex") in the provenance Source column.
                    pipeline_step=p.get("source") or (
                        "web_search" if web_fallback_used else None
                    ),
                    rank=rank,
                    stage_label="basic.web_search" if web_fallback_used else "basic.retrieve",
                )

        # Prepare sources
        sources = []
        for p in paper_results:
            sources.append(
                SourceReference(
                    title=p.get("title") or "Untitled",
                    authors=p.get("authors"),
                    year=p.get("year"),
                    journal=p.get("journal"),
                    doi=p.get("doi"),
                    url=p.get("url"),
                    source=p.get("source"),
                    source_apis=p.get("source_apis"),
                    sources_all=p.get("sources_all"),
                    enrichment_sources=p.get("enrichment_sources"),
                    relevance_score=p.get("paper_score", 0.0),
                    kb_name=p.get("kb_name"),
                    paper_id=p.get("paper_id"),
                    chunk_text=p.get("chunk_text"),
                    abstract=p.get("abstract") or p.get("chunk_text"),
                    metadata=p.get("paper_metadata"),
                )
            )
        for source in sources:
            yield StreamEvent.source(source)

        if not paper_results:
            msg = (
                "No relevant documents found in your KB, and the web "
                "literature search returned no results either."
                if web_fallback_used
                else "No relevant documents found to answer your question."
            )
            yield StreamEvent.content(msg)
            emit_phase(_phase_sink, phase="retrieve", state="done")
            yield StreamEvent.done(
                conversation_id="",
                tokens_used=0,
                mode="basic",
                iterations=1,
            )
            return

        emit_phase(_phase_sink, phase="retrieve", state="done")
        emit_phase(_phase_sink, phase="synthesize", state="running")
        yield StreamEvent.status("Basic RAG: Generating response...")

        context = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        user_body = (
            f"Documents:\n{context}\n\nQuestion: {request.query}\n\n"
            "EVIDENCE EPISTEMICS: Only conclude REFUTED if a retrieved document "
            "ACTIVELY CONTRADICTS the claim with explicit counter-evidence. "
            "If the retrieved papers do not address this specific claim, "
            "state INSUFFICIENT EVIDENCE — do NOT conclude REFUTED."
        )
        if scope.scope_note:
            user_body = f"{scope.scope_note}\n\n{user_body}"
        user_content = build_user_message_with_history(history_block=hist, body=user_body)
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        # Stream the LLM response
        full_response = ""
        try:
            async for chunk in llm.stream(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
                stage="basic.answer",
            ):
                full_response += chunk
                yield StreamEvent.content(chunk)
        except Exception as e:
            logger.error("basic_streaming_error", error=str(e))
            # Fall back to non-streaming
            answer = await self._generate_response_from_context(
                query=request.query,
                context=context,
                llm=llm,
                request=request,
                num_papers=len(paper_results),
                preamble=scope.scope_note,
            )
            yield StreamEvent.content(answer)
            full_response = answer

        # Defense-in-depth copyright filter on the full streamed
        # response. For action="log" we just emit a warning log; for
        # quote/strip/rewrite we emit a "revision" event after the
        # answer with the corrected text — clients may render it or
        # ignore. Does not retract the already-streamed content.
        try:
            revised = await _apply_copyright_filter(
                answer=full_response, paper_results=paper_results, llm=llm,
                config=self.config,
            )
            if revised != full_response:
                yield StreamEvent(
                    event="revision",
                    data=json.dumps({
                        "kind": "copyright_filter",
                        "revised_content": revised,
                    }),
                )
        except Exception as exc:
            logger.warning("copyright_filter_stream_failed", error=str(exc))

        # Append references section after streaming completes
        if sources:
            references = format_references(sources)
            yield StreamEvent.content("\n\n" + references)

        emit_phase(_phase_sink, phase="synthesize", state="done")
        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="basic",
            iterations=1,
        )

    async def _generate_response(
        self,
        query: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
    ) -> str:
        """Generate response without refinement (Basic mode)."""

        if not documents:
            return "No relevant documents found to answer your question."

        # Format context using utility function
        context = format_documents_for_prompt(documents)

        # Build user prompt with context
        template = f"""Based on the following research documents, please answer this question:

Question: {query}

Documents:
{context}

---

Instructions:
- Provide a comprehensive answer with clear sections
- Use markdown formatting (headings ##, bullet points -, minimal bold **)
- Base your answer on the documents provided
- Number of documents: {len(documents)}
- Unique sources: {len(set(get_doc_citation(d) for d in documents))}"""

        try:
            base_messages = [
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": template},
            ]
            messages = wrap_messages_for_chunks(
                base_messages=base_messages,
                chunks=documents,
                model=request.model,
                config=self.config,
            )
            response = await llm.complete(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            )
            return response
        except Exception as e:
            logger.error("basic_response_generation_error", error=str(e))
            return f"Error generating response: {e}"

    async def _generate_response_from_context(
        self,
        query: str,
        context: str,
        llm: Any,
        request: RAGRequest,
        num_papers: int = 0,
        *,
        preamble: str | None = None,
    ) -> str:
        """Generate response from pre-formatted paper context."""
        template = f"""Based on the following research papers, please answer this question:

Question: {query}

{context}

---

Instructions:
- Provide a comprehensive answer with clear sections
- Use markdown formatting (headings ##, bullet points -, minimal bold **)
- Base your answer on the papers provided
- Number of papers: {num_papers}
- IMPORTANT: Extract ALL specific names, tools, methods, chemicals, organisms, or other entities mentioned in the papers that are relevant to the question. Do NOT say "specific names are not listed" if the papers contain them.
- Be specific and concrete — cite specific tools, software, methods, or findings by name rather than giving vague generalizations.
- EVIDENCE EPISTEMICS: Only conclude REFUTED if a retrieved document ACTIVELY CONTRADICTS the claim with explicit counter-evidence. If the retrieved papers do not address this specific claim, state INSUFFICIENT EVIDENCE — do NOT conclude REFUTED."""
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        body = template
        if preamble:
            body = f"{preamble}\n\n{body}"
        user_content = build_user_message_with_history(history_block=hist, body=body)

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": user_content},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.15,
            )
            return response
        except Exception as e:
            logger.error("basic_response_generation_error", error=str(e))
            return f"Error generating response: {e}"
