"""Advanced RAG Mode - Exact implementation from release package.

Advanced RAG adds:
- Query rephrasing/expansion (generate_similar_queries)
- Hybrid retrieval (vector + BM25-inspired scoring)
- WRRF scoring for multi-query fusion
- Optional response refinement
"""

import contextlib
import math
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from perspicacite.config.schema import MultimodalMode
from perspicacite.logging import get_logger
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.provenance.context import get_collector
from perspicacite.rag.code_excerpts import collect_code_excerpts
from perspicacite.rag.conversation_helpers import compute_retrieval_query, format_conversation_block
from perspicacite.rag.figure_refs import collect_figure_refs
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.multimodal import wrap_messages_for_chunks
from perspicacite.rag.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    EVALUATE_RESPONSE_PROMPT,
    FOCUS_INSTRUCTIONS_PROMPT,
    FORMAT_PROMPT,
    MANDATORY_PROMPT,
    REFINE_RESPONSE_HUMAN_PROMPT_SUFFIX,
    REFINE_RESPONSE_SYSTEM_PROMPT,
    get_mandatory_prompt,
)
from perspicacite.rag.query_scope import merge_scope_with_candidates, resolve_paper_scope_for_query
from perspicacite.rag.relevancy import assess_query_complexity, reorder_documents_by_relevance
from perspicacite.rag.telemetry import emit_phase
from perspicacite.rag.utils import (
    flatten_paper_results_to_chunks,
    format_documents_for_prompt,
    format_paper_results_for_prompt,
    format_references,
    get_doc_citation,
    prepare_sources,
)
from perspicacite.rag.wrrf_v1 import doc_page_content, select_wrrf_merged_documents
from perspicacite.retrieval.hybrid import (
    determine_weights_with_llm,
    hybrid_retrieval,
    resolve_hybrid_weights,
)
from perspicacite.retrieval.multi_kb import get_chunks_by_paper_ids_across
from perspicacite.retrieval.recency import apply_recency_weighting_to_papers

logger = get_logger("perspicacite.rag.modes.advanced")


class AdvancedRAGMode(BaseRAGMode):
    """
    Advanced RAG Mode - Exact port from release package core/core.py

    Characteristics:
    - Query rephrasing using LLM (generate_similar_queries)
    - Multiple query execution
    - WRRF (Weighted Reciprocal Rank Fusion) scoring
    - Hybrid retrieval support (when enabled)
    - Optional response refinement (if use_refinement=True)
    """

    def __init__(self, config: Any):
        super().__init__(config)
        rag_settings = getattr(config.rag_modes, "advanced", None)

        # Handle both dict and Pydantic model
        if rag_settings is None:
            rag_settings = {}
        elif hasattr(rag_settings, "model_dump"):
            # Pydantic v2 model
            rag_settings = rag_settings.model_dump()
        elif hasattr(rag_settings, "dict"):
            # Pydantic v1 model
            rag_settings = rag_settings.dict()

        self.initial_docs = 150  # From release package
        self.final_max_docs = 5
        self.max_docs_per_source = 1
        self.rephrases = 3  # Number of additional queries to generate
        self.use_refinement = rag_settings.get("enable_reflection", False)
        self.use_hybrid = rag_settings.get("use_hybrid", True)  # Enable hybrid retrieval by default
        self.use_two_pass = getattr(config.knowledge_base, "use_two_pass", True)

        # WRRF constants from release package
        self.wrrf_k = 60

        # Sigmoid parameters for score normalization
        self.pth = 0.8  # threshold
        self.stp = 30  # steepness
        # v1 get_response: relevancy reorder + focus instructions
        self.use_relevancy_optimization = rag_settings.get("use_relevancy_optimization", True)
        # v1 profonde: refinement_iterations clamped 1–3; core refine_response default 2
        self.refinement_iterations = max(
            1, min(int(rag_settings.get("refinement_iterations", 2)), 3)
        )
        self.evaluator_model = rag_settings.get("evaluator_model")
        self.evaluator_provider = rag_settings.get("evaluator_provider")
        # v1 get_response: max length for mandatory + DEFAULT_SYSTEM_PROMPT before focus instructions
        llm_cfg = getattr(config, "llm", None)
        self.max_context_window = int(getattr(llm_cfg, "max_context_window", None) or 10000)

    def _truncate_combined_prompt_base(self, combined_prompt: str) -> str:
        """core/core.py get_response: truncate mandatory + system only (before focus)."""
        if len(combined_prompt) <= self.max_context_window:
            return combined_prompt
        logger.warning(
            "advanced_prompt_truncated",
            original_len=len(combined_prompt),
            limit=self.max_context_window,
        )
        return combined_prompt[: self.max_context_window]

    @staticmethod
    def _clean_refined_response(text: str) -> str:
        """Approximate v1 LLMWrapper._clean_response when refinement exits early on high score."""
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```json", 1)[1] if t.startswith("```json") else t.split("```", 1)[1]
            if t.rstrip().endswith("```"):
                t = t.rsplit("```", 1)[0]
        return t.strip()

    def _v1_question_line(self, request: RAGRequest) -> str:
        """core/core.py get_response question field."""
        original_query = request.query
        refined = getattr(request, "refined_query", None) or request.query
        return (
            f"User original question: {original_query}\n"
            f"User refined question (based on conversation history): {refined}"
        )

    def _v1_metadata_dict(self, doc: Any) -> dict[str, Any]:
        if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
            m = doc.chunk.metadata
            if hasattr(m, "model_dump"):
                d = m.model_dump()
            elif isinstance(m, dict):
                d = dict(m)
            else:
                d = {
                    "citation": getattr(m, "citation", None) or getattr(m, "title", ""),
                    "url": getattr(m, "url", "") or "",
                    "chunk": str(getattr(m, "year", "") or getattr(m, "chunk", "")),
                }
            d.setdefault("citation", get_doc_citation(doc))
            d.setdefault("url", "")
            d.setdefault("chunk", "")
            return d
        return {"citation": "Unknown", "url": "", "chunk": ""}

    def _v1_context_and_citations(self, documents: list[Any]) -> tuple[str, str]:
        """Build context_with_citations + citation_list like core/core.py get_response."""
        context_with_citations = ""
        source_counter: Counter[tuple[str, str]] = Counter()
        for doc in documents:
            text = doc_page_content(doc)
            md = self._v1_metadata_dict(doc)
            cit = md.get("citation") or "Unknown"
            url = md.get("url") or ""
            context_with_citations += f"{text}, Citation: {cit})\n\n"
            source_counter[(url, cit)] += 1

        citations: list[str] = []
        for (url, citation), count in source_counter.items():
            chunk = "Unknown date"
            for d in documents:
                m = self._v1_metadata_dict(d)
                if m.get("url") == url and m.get("citation") == citation:
                    chunk = str(m.get("chunk", "Unknown date"))
                    break
            citations.append(f"{citation}. Accessed on: {chunk} (url: {url}). [{count} docs]")
        citation_list = "\n".join(citations)
        return context_with_citations, citation_list

    def _build_v1_answer_chat_chunk_docs(
        self,
        request: RAGRequest,
        documents: list[Any],
    ) -> tuple[str, str, float]:
        """
        Same prompt structure as core/core.py get_response for chunk-level docs
        (streaming path; no two-pass).
        """
        kb_title = getattr(request, "kb_name", None) or "Perspicacité"
        kb_scope = getattr(request, "kb_scope", None) or "scientific research and education"
        mandatory = get_mandatory_prompt(kb_title, kb_scope)
        combined_prompt = self._truncate_combined_prompt_base(
            mandatory + "\n" + DEFAULT_SYSTEM_PROMPT
        )
        if self.use_relevancy_optimization:
            combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        context_with_citations, citation_list = self._v1_context_and_citations(documents)
        n_docs = len(documents)
        n_sources = len({get_doc_citation(d) for d in documents})
        question = self._v1_question_line(request)
        user_template = f"""System prompt: {combined_prompt}
Context: {context_with_citations}
Format: {FORMAT_PROMPT}
Question: {question}

Additional information:
- Total documents used: {n_docs}
- Unique sources: {n_sources}
Sources:
{citation_list}

EVIDENCE EPISTEMICS: Only conclude a claim is REFUTED when a retrieved paper EXPLICITLY and DIRECTLY contradicts it. If no retrieved paper addresses a specific claim, state INSUFFICIENT EVIDENCE — do NOT conclude REFUTED merely because supporting evidence was not found.
"""
        if self.use_relevancy_optimization:
            qc = assess_query_complexity(getattr(request, "refined_query", None) or request.query)
            temperature = 0.7 if qc > 0.7 else 0.3
        else:
            temperature = 0.3
        return combined_prompt, user_template, temperature

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """
        Execute Advanced RAG with query rephrasing and WRRF scoring.

        Ported from: core/core.py::retrieve_documents() and get_response()
        """
        logger.info("advanced_rag_start", query=request.query)

        kb_collection = chroma_collection_name_for_kb(request.kb_name)
        kb_names = getattr(request, "kb_names", None) or [request.kb_name]
        collection_names = [chroma_collection_name_for_kb(n) for n in kb_names]
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined
        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            kb_collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = min(50, getattr(request, "max_papers_retrieval", None) or 5)  # was min(5,...)

        # Step 1: Generate similar/rephrased queries
        # This is the key difference from Basic mode
        logger.info("advanced_generate_queries", original=retrieval_query[:100])

        all_queries = await self._generate_similar_queries(
            original_query=retrieval_query, llm=llm, number=self.rephrases
        )

        logger.info("advanced_queries_generated", count=len(all_queries), queries=all_queries)

        # Step 2: Retrieve documents for all queries with WRRF scoring
        # This uses the WRRF (Weighted Reciprocal Rank Fusion) algorithm
        logger.info("advanced_wrrf_retrieval", num_queries=len(all_queries))

        selected_documents = await self._wrrf_retrieval(
            queries=all_queries,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            kb_name=kb_collection,
            collection_names=collection_names,
            llm=llm,
            request=request,
            max_docs=cap,  # respect caller's max_papers_retrieval
        )

        logger.info(
            "advanced_selected_docs", count=len(selected_documents), use_two_pass=self.use_two_pass
        )

        retrieval_docs_for_refine = list(selected_documents)
        if self.use_relevancy_optimization and selected_documents:
            selected_documents = reorder_documents_by_relevance(
                getattr(request, "refined_query", None) or request.query,
                selected_documents,
            )

        # Step 2b: Two-pass — fetch full text for selected papers
        paper_results = []
        if self.use_two_pass:
            from perspicacite.rag.utils import deduplicate_chunk_overlaps

            paper_ids = []
            paper_scores: dict[str, float] = {}
            paper_meta: dict[str, Any] = {}
            paper_kb: dict[str, str | None] = {}
            for doc in selected_documents:
                meta = getattr(doc, "chunk", None)
                if meta and hasattr(meta, "metadata"):
                    pid = getattr(meta.metadata, "paper_id", None)
                    if pid and pid not in paper_ids:
                        paper_ids.append(pid)
                        score = getattr(doc, "wrrf_score", getattr(doc, "score", 0.5))
                        paper_scores[pid] = score
                        paper_meta[pid] = meta.metadata
                        paper_kb[pid] = getattr(doc, "kb_name", None)

            if paper_ids:
                ordered = merge_scope_with_candidates(paper_ids, paper_scores, scope, cap)
                if len(collection_names) == 1:
                    all_chunks = await vector_store.get_chunks_by_paper_ids(
                        collection_names[0], ordered
                    )
                else:
                    all_chunks = await get_chunks_by_paper_ids_across(
                        vector_store,
                        collection_names=collection_names,
                        paper_ids=ordered,
                    )
                deduped = deduplicate_chunk_overlaps(all_chunks)
                # Group by paper
                from collections import OrderedDict

                grouped: OrderedDict[str, list] = OrderedDict()
                for d in deduped:
                    grouped.setdefault(d["paper_id"], []).append(d)
                for pid in ordered:
                    chunks_list = grouped.get(pid, [])
                    full_text = " ".join(c["text"] for c in chunks_list)
                    meta = paper_meta.get(pid)
                    paper_results.append(
                        {
                            "paper_id": pid,
                            "paper_score": paper_scores[pid],
                            "title": getattr(meta, "title", None),
                            "authors": getattr(meta, "authors", None),
                            "year": getattr(meta, "year", None),
                            "doi": getattr(meta, "doi", None),
                            "chunks": chunks_list,
                            "full_text": full_text,
                            "kb_name": paper_kb.get(pid),
                        }
                    )

        # Apply optional recency weighting
        if getattr(request, "recency_weight", None):
            paper_results = apply_recency_weighting_to_papers(
                paper_results,
                recency_weight=getattr(request, "recency_weight", None),
                half_life_years=getattr(request, "recency_half_life_years", None),
            )

        # Provenance
        _c = get_collector()
        if _c is not None:
            _c.add_trace(
                "wrrf_retrieve",
                detail={"papers": len(paper_results), "kb_name": request.kb_name},
            )
            for rank, p in enumerate(paper_results):
                _raw_pid = p.get("paper_id")
                _raw_doi = p.get("doi")
                _raw_title = p.get("title")
                _raw_kb = p.get("kb_name")
                _raw_score = p.get("paper_score") or p.get("score") or 0.0
                _c.add_retrieval(
                    paper_id=str(_raw_pid) if _raw_pid is not None else None,
                    doi=str(_raw_doi) if _raw_doi is not None else None,
                    title=str(_raw_title) if _raw_title is not None else None,
                    score=float(_raw_score),  # type: ignore[arg-type]  # metadata dict value union includes list
                    kb_name=str(_raw_kb) if _raw_kb is not None else None,
                    content_type=None,
                    pipeline_step=None,
                    rank=rank,
                    stage_label="advanced.wrrf_pass2",
                )

        # Step 3: Generate response using full paper context
        if paper_results:
            context = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
        else:
            context = format_documents_for_prompt(selected_documents)

        answer = await self._generate_response_from_context(
            query=request.query,
            context=context,
            llm=llm,
            request=request,
            num_papers=len(paper_results),
            source_documents=retrieval_docs_for_refine if not paper_results else None,
            paper_results=paper_results,
            preamble=scope.scope_note,
        )

        # Step 4: Apply refinement if enabled (Advanced mode feature)
        if self.use_refinement and not self._is_streaming(request):
            logger.info("advanced_apply_refinement")
            answer = await self._refine_response(
                response=answer,
                query=request.query,
                context=context,
                documents=retrieval_docs_for_refine,
                llm=llm,
                request=request,
                max_iterations=self.refinement_iterations,
                eval_model=getattr(request, "evaluator_model", None),
                eval_provider=getattr(request, "evaluator_provider", None),
            )

        # Step 5: Prepare sources
        if paper_results:
            sources = []
            for p in paper_results:
                from perspicacite.models.rag import SourceReference

                _raw_year = p.get("year")
                _raw_score_p = p.get("paper_score")
                _raw_meta_p = p.get("paper_metadata")
                sources.append(
                    SourceReference(
                        title=str(p.get("title") or "Untitled"),
                        authors=p.get("authors"),  # type: ignore[arg-type]  # _coerce_authors validator widens authors at runtime
                        year=int(_raw_year) if _raw_year is not None else None,  # type: ignore[arg-type]  # metadata dict value union
                        doi=str(p["doi"]) if p.get("doi") is not None else None,
                        relevance_score=float(_raw_score_p or 0.0),  # type: ignore[arg-type]  # metadata dict value union
                        kb_name=str(p["kb_name"]) if p.get("kb_name") is not None else request.kb_name,
                        paper_id=str(p["paper_id"]) if p.get("paper_id") is not None else None,
                        metadata=_raw_meta_p,  # type: ignore[arg-type]  # metadata dict value union
                    )
                )
        else:
            sources = prepare_sources(selected_documents, max_docs=cap)  # was self._prepare_sources (always 5)

        # Step 6: Append references section to answer
        if sources:
            references = self._format_references(sources)
            answer = answer.strip() + "\n\n" + references

        logger.info("advanced_rag_complete", sources=len(sources), refined=self.use_refinement)

        # Sub-project C (2026-05-15): attach code excerpts + figure refs.
        _mm = getattr(self.config, "multimodal", None)
        _show_code = bool(getattr(_mm, "show_code", False)) if _mm else False
        _mode = getattr(_mm, "mode", None) if _mm else None
        _all_docs = list(paper_results) if paper_results else list(selected_documents)
        _dc_chunks = flatten_paper_results_to_chunks(_all_docs)
        _code_excerpts = collect_code_excerpts(_dc_chunks) if _show_code else []
        _figure_refs = (
            collect_figure_refs(_dc_chunks, capsule_root=Path(self.config.capsule.root))
            if _mode is not None and _mode != MultimodalMode.OFF
            else []
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.ADVANCED,
            iterations=1,
            web_search_used=False,
            code_excerpts=_code_excerpts,
            figures=_figure_refs,
        )

    def _advanced_paper_stream_messages(
        self,
        request: RAGRequest,
        paper_results: list[dict[str, Any]],
        *,
        preamble: str | None,
    ) -> tuple[list[dict[str, str]], float, bool]:
        """Messages for two-pass paper streaming (aligned with ``_generate_response_from_context``)."""
        kb_title = getattr(request, "kb_name", None) or "Perspicacité"
        kb_scope = getattr(request, "kb_scope", None) or "scientific research and education"
        mandatory = get_mandatory_prompt(kb_title, kb_scope)
        combined_prompt = self._truncate_combined_prompt_base(
            mandatory + "\n" + DEFAULT_SYSTEM_PROMPT
        )
        if self.use_relevancy_optimization:
            combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        context_with_citations = ""
        sc: Counter[tuple[str, str]] = Counter()
        for p in paper_results:
            text = (p.get("full_text") or "")[:12000]
            cit = p.get("title") or p.get("doi") or "Unknown"
            doi = p.get("doi") or ""
            url = f"https://doi.org/{doi}" if doi else ""
            context_with_citations += f"{text}, Citation: {cit})\n\n"
            sc[(url, str(cit))] += 1
        citations: list[str] = []
        for (url, citation), count in sc.items():
            year = ""
            for p in paper_results:
                if (p.get("title") or p.get("doi")) == citation or str(p.get("doi")) == citation:
                    year = str(p.get("year") or "")
                    break
            citations.append(f"{citation}. Accessed on: {year} (url: {url}). [{count} docs]")
        citation_list = "\n".join(citations)
        n_docs = len(paper_results)
        n_sources = len(sc)

        question = self._v1_question_line(request)
        template = f"""System prompt: {combined_prompt}
Context: {context_with_citations}
Format: {FORMAT_PROMPT}
Question: {question}

Additional information:
- Total documents used: {n_docs}
- Unique sources: {n_sources}
Sources:
{citation_list}
"""
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        if hist:
            template = f"{hist}\n\n---\n\n{template}"
        if preamble:
            template = f"{preamble.strip()}\n\n{template}"

        req_model = getattr(request, "model", "") or ""
        is_o_series = req_model.startswith("o") or "gpt-5" in req_model
        if self.use_relevancy_optimization:
            qc = assess_query_complexity(getattr(request, "refined_query", None) or request.query)
            temperature = 0.7 if qc > 0.7 else 0.3
        else:
            temperature = 0.3

        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "user", "content": template},
        ]
        return messages, temperature, is_o_series

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute Advanced RAG with true streaming output."""
        _phase_sink = getattr(request, "telemetry_sink", None)

        emit_phase(_phase_sink, phase="rewrite", state="running")
        yield StreamEvent.status("Advanced RAG: Generating query variations...")

        kb_collection = chroma_collection_name_for_kb(request.kb_name)
        kb_names = getattr(request, "kb_names", None) or [request.kb_name]
        collection_names = [chroma_collection_name_for_kb(n) for n in kb_names]
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined
            yield StreamEvent.status_kind(
                f"Rewrote question using conversation context: '{request.query}' → '{refined}'",
                kind="query_rephrased",
                original=request.query,
                rewritten=refined,
                by="conversation_history",
            )
        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            kb_collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = min(50, getattr(request, "max_papers_retrieval", None) or 5)  # was min(5,...)

        # Step 1: Generate similar/rephrased queries
        all_queries = await self._generate_similar_queries(
            original_query=retrieval_query, llm=llm, number=self.rephrases
        )

        emit_phase(_phase_sink, phase="rewrite", state="done")
        emit_phase(_phase_sink, phase="retrieve_kb", state="running")

        # Surface each variation individually so the trail's
        # QueryVariationsCard can render the actual rewrites instead of
        # only a count. Skip the original (first item in all_queries) —
        # only the rewrites are interesting.
        for _variant in all_queries:
            if not _variant or _variant.strip() == retrieval_query.strip():
                continue
            yield StreamEvent.status_kind(
                f"Variation: '{_variant}'",
                kind="query_rephrased",
                original=retrieval_query,
                rewritten=_variant,
                by="advanced_variations",
            )


        yield StreamEvent.status(
            f"Advanced RAG: Searching with {len(all_queries)} query variations..."
        )

        # Step 2: Retrieve documents using WRRF
        selected_documents = await self._wrrf_retrieval(
            queries=all_queries,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            kb_name=kb_collection,
            collection_names=collection_names,
            llm=llm,
            request=request,
            max_docs=cap,  # respect caller's max_papers_retrieval
        )

        emit_phase(_phase_sink, phase="retrieve_kb", state="done")

        if self.use_relevancy_optimization and selected_documents:
            emit_phase(_phase_sink, phase="screen", state="running")
            selected_documents = reorder_documents_by_relevance(
                getattr(request, "refined_query", None) or request.query,
                selected_documents,
            )
            emit_phase(_phase_sink, phase="screen", state="done")

        # Web-search fallback: if the WRRF retrieval over the KB(s) returned
        # nothing, run a live literature search using the user-selected
        # database providers — mirrors the behaviour of basic mode so the
        # welcome-screen promise ("falls back to web literature search when
        # your KB is insufficient") holds across modes.
        paper_results: list[dict[str, Any]] = []
        if not selected_documents:
            emit_phase(_phase_sink, phase="retrieve_web", state="running")
            from perspicacite.rag.modes.basic import _web_fallback_papers

            _db_pretty = ", ".join(
                d.replace("_", " ").title() for d in (request.databases or [])
            ) or "Semantic Scholar, OpenAlex, PubMed"
            # Run the keyword optimizer UPFRONT so query_rephrased lands
            # in the panel before the slow aggregator call (parity with
            # basic mode — see comment there for rationale).
            search_query = retrieval_query
            try:
                from perspicacite.search.query_optimizer import optimize_query as _qopt
                # request.app_state is auto-attached by RAGEngine (web AppState
                # or MCPState — both duck-type as the protocol). If callers
                # bypass the engine, optimizer skips itself gracefully.
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
                logger.debug("advanced_upfront_optimizer_failed", error=str(_qe))
            yield StreamEvent.status(
                f"No KB results — falling back to web literature search across {_db_pretty}…"
            )
            # Telemetry pattern: if the MCP layer set request.telemetry_sink
            # (Task 2.4), pass it through directly — events flow to
            # ctx.report_progress live and the local drain is a no-op.
            # Otherwise use a fresh list for the SSE drain below.
            _telemetry: Any = getattr(request, "telemetry_sink", None) or []
            paper_results = await _web_fallback_papers(
                query=search_query,
                databases=request.databases,
                max_docs=cap,
                config=getattr(self, "config", None),
                app_state=getattr(request, "app_state", None),
                reranker_override=getattr(request, "use_reranker", None),
                telemetry=_telemetry,
                optimize_query=False,
            )
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
            emit_phase(_phase_sink, phase="retrieve_web", state="done")
            if paper_results:
                yield StreamEvent.status(
                    f"Web search returned {len(paper_results)} relevant paper(s)."
                )
            else:
                yield StreamEvent.status(
                    "Web search returned no relevant papers."
                )


        if self.use_two_pass and selected_documents:
            from perspicacite.rag.utils import deduplicate_chunk_overlaps

            paper_ids: list[str] = []
            paper_scores: dict[str, float] = {}
            paper_meta: dict[str, Any] = {}
            paper_kb: dict[str, str | None] = {}
            for doc in selected_documents:
                meta = getattr(doc, "chunk", None)
                if meta and hasattr(meta, "metadata"):
                    pid = getattr(meta.metadata, "paper_id", None)
                    if pid and pid not in paper_ids:
                        paper_ids.append(pid)
                        score = getattr(doc, "wrrf_score", getattr(doc, "score", 0.5))
                        paper_scores[pid] = score
                        paper_meta[pid] = meta.metadata
                        paper_kb[pid] = getattr(doc, "kb_name", None)
            if paper_ids:
                ordered = merge_scope_with_candidates(paper_ids, paper_scores, scope, cap)
                if len(collection_names) == 1:
                    all_chunks = await vector_store.get_chunks_by_paper_ids(
                        collection_names[0], ordered
                    )
                else:
                    all_chunks = await get_chunks_by_paper_ids_across(
                        vector_store,
                        collection_names=collection_names,
                        paper_ids=ordered,
                    )
                deduped = deduplicate_chunk_overlaps(all_chunks)
                from collections import OrderedDict

                grouped: OrderedDict[str, list] = OrderedDict()
                for d in deduped:
                    grouped.setdefault(d["paper_id"], []).append(d)
                for pid in ordered:
                    chunks_list = grouped.get(pid, [])
                    full_text = " ".join(c["text"] for c in chunks_list)
                    meta = paper_meta.get(pid)
                    paper_results.append(
                        {
                            "paper_id": pid,
                            "paper_score": paper_scores[pid],
                            "title": getattr(meta, "title", None),
                            "authors": getattr(meta, "authors", None),
                            "year": getattr(meta, "year", None),
                            "doi": getattr(meta, "doi", None),
                            "chunks": chunks_list,
                            "full_text": full_text,
                            "kb_name": paper_kb.get(pid),
                        }
                    )

        # Apply optional recency weighting
        if getattr(request, "recency_weight", None):
            paper_results = apply_recency_weighting_to_papers(
                paper_results,
                recency_weight=getattr(request, "recency_weight", None),
                half_life_years=getattr(request, "recency_half_life_years", None),
            )

        # Provenance
        _c = get_collector()
        if _c is not None:
            _c.add_trace(
                "wrrf_retrieve",
                detail={"papers": len(paper_results), "kb_name": request.kb_name},
            )
            for rank, p in enumerate(paper_results):
                _raw_pid = p.get("paper_id")
                _raw_doi = p.get("doi")
                _raw_title = p.get("title")
                _raw_kb = p.get("kb_name")
                _raw_score = p.get("paper_score") or p.get("score") or 0.0
                _c.add_retrieval(
                    paper_id=str(_raw_pid) if _raw_pid is not None else None,
                    doi=str(_raw_doi) if _raw_doi is not None else None,
                    title=str(_raw_title) if _raw_title is not None else None,
                    score=float(_raw_score),
                    kb_name=str(_raw_kb) if _raw_kb is not None else None,
                    content_type=None,
                    pipeline_step=None,
                    rank=rank,
                    stage_label="advanced.wrrf_pass2",
                )

        # Step 3: Prepare sources
        if paper_results:
            sources = []
            for p in paper_results:
                _raw_year_s = p.get("year")
                _raw_score_s = p.get("paper_score")
                _raw_meta_s = p.get("paper_metadata")
                sources.append(
                    SourceReference(
                        title=str(p.get("title") or "Untitled"),
                        authors=p.get("authors"),  # type: ignore[arg-type]  # _coerce_authors validator widens authors at runtime
                        year=int(_raw_year_s) if _raw_year_s is not None else None,
                        doi=str(p["doi"]) if p.get("doi") is not None else None,
                        url=p.get("url") or p.get("pdf_url"),
                        # Propagate provider provenance from the web-fallback
                        # path so the UI shows "europepmc" / "openalex" /
                        # "pubmed" tags instead of "unknown" — these are set
                        # by ``_web_fallback_papers`` in basic.py.
                        source=p.get("source"),
                        source_apis=p.get("source_apis"),
                        sources_all=p.get("sources_all"),
                        enrichment_sources=p.get("enrichment_sources"),
                        relevance_score=float(_raw_score_s or 0.0),
                        kb_name=str(p["kb_name"]) if p.get("kb_name") is not None else request.kb_name,
                        paper_id=str(p["paper_id"]) if p.get("paper_id") is not None else None,
                        metadata=_raw_meta_s,
                    )
                )
        else:
            sources = prepare_sources(selected_documents, max_docs=cap)  # was self._prepare_sources (always 5)
        for source in sources:
            yield StreamEvent.source(source)

        # Step 4: Stream the response generation. Allow proceeding when EITHER
        # KB documents OR web-fallback paper_results are available — the
        # downstream answer path (line ~693) prefers paper_results, then falls
        # back to selected_documents.
        if not selected_documents and not paper_results:
            yield StreamEvent.content("No relevant documents found to answer your question.")
            yield StreamEvent.done(
                conversation_id="",
                tokens_used=0,
                mode="advanced",
                iterations=1,
            )
            return

        emit_phase(_phase_sink, phase="synthesize", state="running")
        yield StreamEvent.status("Advanced RAG: Generating response...")

        if paper_results:
            messages, temperature, is_o_series = self._advanced_paper_stream_messages(
                request, paper_results, preamble=scope.scope_note
            )
        else:
            combined_prompt, user_template, temperature = self._build_v1_answer_chat_chunk_docs(
                request, selected_documents
            )
            messages = [
                {"role": "system", "content": combined_prompt},
                {"role": "user", "content": user_template},
            ]
            model = getattr(request, "model", "") or ""
            is_o_series = model.startswith("o") or "gpt-5" in model

        # Stream the LLM response (v1 get_response structure; no refinement when streaming)
        full_response = ""
        try:
            stream_kwargs: dict[str, Any] = {
                "messages": messages,
                "model": request.model,
                "provider": request.provider,
                "max_tokens": 2000,
                "stage": "advanced.answer",
            }
            if not is_o_series:
                stream_kwargs["temperature"] = temperature
            async for chunk in llm.stream(**stream_kwargs):
                full_response += chunk
                yield StreamEvent.content(chunk)
        except Exception as e:
            logger.error("advanced_streaming_error", error=str(e))
            # Fall back to non-streaming (same v1 template as execute)
            if paper_results:
                ctx = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
                answer = await self._generate_response_from_context(
                    query=request.query,
                    context=ctx,
                    llm=llm,
                    request=request,
                    num_papers=len(paper_results),
                    paper_results=paper_results,
                    preamble=scope.scope_note,
                )
            else:
                answer = await self._generate_response_from_context(
                    query=request.query,
                    context="",
                    llm=llm,
                    request=request,
                    source_documents=selected_documents,
                )
            yield StreamEvent.content(answer)
            full_response = answer

        # Add references using utility function
        if sources:
            references = format_references(sources)
            yield StreamEvent.content("\n\n" + references)

        emit_phase(_phase_sink, phase="synthesize", state="done")
        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="advanced",
            iterations=1,
        )

    async def _generate_similar_queries(
        self,
        original_query: str,
        llm: Any,
        number: int = 3,
    ) -> list[str]:
        """
        Generate similar/rephrased queries using LLM.

        Ported from: core/core.py::generate_similar_queries()

        Returns list including original query + generated variations.
        """
        queries = [original_query]  # Always include original

        if not number or number <= 0:
            return queries

        for i in range(number):
            # Build context with already generated queries
            additional_queries_content = f"Original Query: {original_query}."
            additional_queries_content += "".join(
                [f" Additional Q{j + 1}: {query}" for j, query in enumerate(queries[1:])]
            )

            prompt = """Rephrase slightly the question based on the original query that is not the same as the additional ones. 
Use scientific language. Your answer should be just one phrase. 
Don't deviate the topic of the queries and questions. Do not use bullet points or numbering."""

            try:
                response = await llm.complete(
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": f"Queries already used: {additional_queries_content}",
                        },
                    ],
                    temperature=0.7,
                    max_tokens=100,
                )

                # Clean and add the generated query. Guard against the
                # LLM returning None / empty (some free-tier providers do
                # this intermittently — observed with deepseek-v4-flash).
                # Continue to the next iteration instead of breaking the
                # whole loop, so one transient empty response doesn't
                # collapse "3 variations" down to just the original.
                if not response:
                    logger.warning(
                        "advanced_generated_query_empty_response",
                        attempt=i + 1, of=number,
                    )
                    continue
                new_query = response.strip()
                if new_query and new_query not in queries:
                    queries.append(new_query)
                    logger.debug("advanced_generated_query", query=new_query[:100])

            except Exception as e:
                # Provider/network errors are likely to repeat — break
                # rather than burn quota looping. Empty-response is
                # different (handled above with continue).
                logger.warning(
                    "advanced_query_generation_error",
                    error=str(e), attempt=i + 1, of=number,
                )
                break

        return queries

    async def _wrrf_retrieval(
        self,
        queries: list[str],
        vector_store: Any,
        embedding_provider: Any,
        kb_name: str,
        llm: Any = None,
        request: Any = None,
        collection_names: list[str] | None = None,
        max_docs: int | None = None,
    ) -> list[Any]:
        """
        Retrieve documents using WRRF (Weighted Reciprocal Rank Fusion).

        Ported from: core/core.py::retrieve_documents() - the multi-query branch

        WRRF formula: score = sum(normalized_score / (k + rank))

        If use_hybrid is enabled, also applies BM25 scoring to combine with vector scores.

        When `collection_names` is provided with more than one entry, the search
        is fanned out across all listed collections and each retrieved result is
        tagged with its originating collection name (kb_name). Single-collection
        behaviour is preserved when `collection_names` is None or length 1.
        """
        # Normalize collection list: prefer explicit collection_names, else legacy kb_name.
        effective_collections = list(collection_names) if collection_names else [kb_name]

        rankings: dict[str, dict[int, int]] = {}  # doc_id -> {query_idx: rank}
        scores_per_query: dict[int, dict[str, float]] = {}  # query_idx -> {doc_id: score}
        documents_info: dict[str, Any] = {}  # doc_id -> document

        # Process each query
        for q_idx, query in enumerate(queries):
            logger.debug("advanced_wrrf_processing_query", idx=q_idx, query=query[:100])

            # Get query embedding and search across one or more collections.
            # Use embed_query() so instruct models (stella, gte-Qwen2) apply
            # the task-specific query prompt for correct retrieval alignment.
            query_embedding = await embedding_provider.embed_query([query])
            if len(effective_collections) == 1:
                results = await vector_store.search(
                    collection=effective_collections[0],
                    query_embedding=query_embedding[0],
                    top_k=self.initial_docs,
                )
            else:
                # Fan out across collections and tag each hit with its source kb.
                results = []
                for coll in effective_collections:
                    try:
                        coll_results = await vector_store.search(
                            collection=coll,
                            query_embedding=query_embedding[0],
                            top_k=self.initial_docs,
                        )
                    except Exception as e:
                        logger.warning(
                            "advanced_wrrf_fanout_search_failed",
                            collection=coll,
                            error=str(e),
                        )
                        continue
                    for r in coll_results:
                        # RetrievedChunk allows extra fields; tag with kb_name.
                        with contextlib.suppress(Exception):
                            r.kb_name = coll
                    results.extend(coll_results)
                # Re-sort by score and clip so WRRF rank assignments stay meaningful.
                results.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
                results = results[: self.initial_docs]

            scores_per_query[q_idx] = {}

            # v1: hybrid for every query when advanced_mode + use_hybrid.
            # Per-request override wins over the mode/config default so a strong
            # instruction-tuned embedder can disable hybrid for this call
            # (use_hybrid=False) without reconfiguring the server.
            _req_hybrid = getattr(request, "use_hybrid", None)
            effective_use_hybrid = (
                self.use_hybrid if _req_hybrid is None else _req_hybrid
            )
            if effective_use_hybrid and results and llm is not None:
                try:
                    logger.info("advanced_applying_hybrid", query=query[:100])

                    vector_scores = [getattr(r, "score", 0.5) for r in results]

                    # Determine LLM-based weights first, then let request overrides win
                    llm_vw, llm_bw = await determine_weights_with_llm(query, llm)
                    final_vw, final_bw = resolve_hybrid_weights(request, default=(llm_vw, llm_bw))

                    hybrid_results = await hybrid_retrieval(
                        query=query,
                        documents=results,
                        vector_scores=vector_scores,
                        vector_weight=final_vw,
                        bm25_weight=final_bw,
                        use_llm_weights=False,
                        llm=llm,
                    )

                    results = [doc for doc, _ in hybrid_results]
                    for doc, hybrid_score in hybrid_results:
                        doc.score = hybrid_score

                    logger.info("advanced_hybrid_applied", num_results=len(results))

                except Exception as e:
                    logger.warning("advanced_hybrid_error", error=str(e))

            # Process results for this query
            for rank, doc in enumerate(results, start=1):
                doc_id = doc_page_content(doc)

                # Get relevance score (normalized)
                score = getattr(doc, "score", 0.5)

                # Apply sigmoid normalization (from release package)
                # norm_score = 1 / (1 + exp(-(score - pth) * stp))
                norm_score = 1 / (1 + math.exp(-(score - self.pth) * self.stp))

                if doc_id not in rankings:
                    rankings[doc_id] = {}
                    documents_info[doc_id] = doc

                rankings[doc_id][q_idx] = rank
                scores_per_query[q_idx][doc_id] = norm_score

            logger.debug("advanced_wrrf_query_processed", idx=q_idx, docs=len(results))

        # Calculate WRRF scores
        wrrf_scores = {}
        for doc_id in rankings:
            total_score: float = 0.0
            for q_idx, rank in rankings[doc_id].items():
                norm_score = scores_per_query[q_idx][doc_id]
                # WRRF formula: weighted reciprocal rank fusion
                total_score += norm_score / (self.wrrf_k + rank)
            wrrf_scores[doc_id] = total_score

        sorted_docs = sorted(wrrf_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_docs:
            logger.warning("advanced_wrrf_no_documents")
            return []

        _effective_max_docs = max_docs if max_docs is not None else self.final_max_docs
        selected_documents = select_wrrf_merged_documents(
            sorted_docs,
            documents_info,
            _effective_max_docs,
            self.max_docs_per_source,
        )

        # Original-query protection: ensure the best result from the original
        # (unexpanded) query that WRRF fusion excluded is still included.
        #
        # WRRF multi-query fusion penalizes papers that rank highly for only one
        # specific query variant (the original) but not for expanded variants.
        # Such papers can fall below the max_docs cutoff despite being the most
        # relevant paper for the actual claim vocabulary.
        #
        # Fix: find the highest-ranked chunk in the original query (q_idx=0)
        # whose paper (by citation) is absent from the WRRF selection, and
        # append it as a bonus slot (max_docs + 1 total).
        #
        # The rank cutoff (3 × _effective_max_docs) matches the raw-chunk pool
        # that basic mode uses via search_two_pass (initial_pool = 3 × hard_cap).
        # Any paper reachable by basic mode at the same k is guaranteed to be
        # reachable by this protection as well.
        if len(queries) > 1 and documents_info:
            # Collect all docs seen in the original query, sorted by their rank
            original_q_docs = sorted(
                [(doc_id, qranks[0]) for doc_id, qranks in rankings.items() if 0 in qranks],
                key=lambda x: x[1],
            )
            if original_q_docs:
                selected_citations = {get_doc_citation(d) for d in selected_documents}
                # Protection rank cap: 3× output cap, matching basic mode's raw chunk pool
                rank_cap = 3 * _effective_max_docs

                for original_top_id, orig_rank in original_q_docs:
                    if orig_rank > rank_cap:
                        break  # Beyond protection range; sorted ascending so can stop

                    original_doc = documents_info[original_top_id]
                    original_citation = get_doc_citation(original_doc)

                    if original_citation not in selected_citations:
                        # This paper is in the original query's top-k but was
                        # excluded by WRRF fusion. Add as bonus slot.
                        wrrf_rank = next(
                            (i for i, (d, _) in enumerate(sorted_docs) if d == original_top_id),
                            -1,
                        )
                        logger.info(
                            "advanced_wrrf_original_query_protection",
                            citation=original_citation[:80] if original_citation else "",
                            original_rank=orig_rank,
                            wrrf_rank=wrrf_rank,
                        )
                        selected_documents.append(original_doc)
                        # Only add the single best-ranked missing paper
                        break

        logger.info(
            "advanced_wrrf_selected",
            total_docs=len(sorted_docs),
            selected=len(selected_documents),
            hybrid_used=self.use_hybrid,
        )

        return selected_documents

    def _get_doc_citation(self, doc: Any) -> str:
        """Extract citation from document."""
        # Use utility function
        from perspicacite.rag.utils import get_doc_citation

        return get_doc_citation(doc)

    async def _generate_response(
        self,
        query: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
    ) -> str:
        """Generate response with optional relevancy optimization using v1 prompts."""

        if not documents:
            return "No relevant documents found to answer your question."

        # Format context using utility function
        context = format_documents_for_prompt(documents)

        # Use KB-specific mandatory prompt if KB info available (v1 compatibility)
        kb_title = getattr(request, "kb_name", "Perspicacité")
        kb_scope = getattr(request, "kb_scope", "scientific research and education")

        if kb_title and kb_scope:
            mandatory = get_mandatory_prompt(kb_title, kb_scope)
        else:
            mandatory = MANDATORY_PROMPT

        # Use exact prompts from release package
        combined_prompt = mandatory + "\n" + DEFAULT_SYSTEM_PROMPT

        # Add focus instructions (from relevancy optimization in original)
        combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        template = f"""System prompt: {combined_prompt}
Context: {context}
Format: {FORMAT_PROMPT}
Question: {query}

Additional information:
- Total documents used: {len(documents)}
- Unique sources: {len(set(get_doc_citation(d) for d in documents))}

Provide a comprehensive answer based on the documents above."""

        try:
            base_messages = [
                {"role": "system", "content": combined_prompt},
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
            logger.error("advanced_response_generation_error", error=str(e))
            return f"Error generating response: {e}"

    async def _generate_response_from_context(
        self,
        query: str,
        context: str,
        llm: Any,
        request: RAGRequest,
        num_papers: int = 0,
        source_documents: list[Any] | None = None,
        paper_results: list[dict[str, Any]] | None = None,
        *,
        preamble: str | None = None,
    ) -> str:
        """core/core.py get_response template; two-pass uses same structure with paper dicts."""
        if (
            (not context or context == "No relevant papers found.")
            and not paper_results
            and not source_documents
        ):
            return "No relevant documents found to answer your question."

        kb_title = getattr(request, "kb_name", None) or "Perspicacité"
        kb_scope = getattr(request, "kb_scope", None) or "scientific research and education"
        mandatory = get_mandatory_prompt(kb_title, kb_scope)
        combined_prompt = self._truncate_combined_prompt_base(
            mandatory + "\n" + DEFAULT_SYSTEM_PROMPT
        )
        if self.use_relevancy_optimization:
            combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        if paper_results:
            context_with_citations = ""
            sc: Counter[tuple[str, str]] = Counter()
            for p in paper_results:
                text = (p.get("full_text") or "")[:12000]
                cit = p.get("title") or p.get("doi") or "Unknown"
                doi = p.get("doi") or ""
                url = f"https://doi.org/{doi}" if doi else ""
                context_with_citations += f"{text}, Citation: {cit})\n\n"
                sc[(url, str(cit))] += 1
            citations: list[str] = []
            for (url, citation), count in sc.items():
                year = ""
                for p in paper_results:
                    if (p.get("title") or p.get("doi")) == citation or str(
                        p.get("doi")
                    ) == citation:
                        year = str(p.get("year") or "")
                        break
                citations.append(f"{citation}. Accessed on: {year} (url: {url}). [{count} docs]")
            citation_list = "\n".join(citations)
            n_docs = len(paper_results)
            n_sources = len(sc)
        elif source_documents:
            context_with_citations, citation_list = self._v1_context_and_citations(source_documents)
            n_docs = len(source_documents)
            n_sources = len({get_doc_citation(d) for d in source_documents})
        else:
            context_with_citations = context
            citation_list = ""
            n_docs = num_papers or 1
            n_sources = 1

        question = self._v1_question_line(request)
        template = f"""System prompt: {combined_prompt}
Context: {context_with_citations}
Format: {FORMAT_PROMPT}
Question: {question}

Additional information:
- Total documents used: {n_docs}
- Unique sources: {n_sources}
Sources:
{citation_list}
"""
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        if hist:
            template = f"{hist}\n\n---\n\n{template}"
        if preamble:
            template = f"{preamble.strip()}\n\n{template}"

        model = getattr(request, "model", "") or ""
        is_o_series = model.startswith("o") or "gpt-5" in model
        # v1 get_response: complexity-based temperature only if use_relevancy_optimization
        if self.use_relevancy_optimization:
            qc = assess_query_complexity(getattr(request, "refined_query", None) or request.query)
            temperature = 0.7 if qc > 0.7 else 0.3
        else:
            temperature = 0.3
        chunks_for_mm = source_documents or []
        if is_o_series:
            try:
                base_messages = [
                    {"role": "system", "content": combined_prompt},
                    {"role": "user", "content": template},
                ]
                messages = wrap_messages_for_chunks(
                    base_messages=base_messages,
                    chunks=chunks_for_mm,
                    model=request.model,
                    config=self.config,
                )
                return await llm.complete(
                    messages=messages,
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                )
            except Exception as e:
                logger.error("advanced_response_generation_error", error=str(e))
                return f"Error generating response: {e}"

        try:
            base_messages = [
                {"role": "system", "content": combined_prompt},
                {"role": "user", "content": template},
            ]
            messages = wrap_messages_for_chunks(
                base_messages=base_messages,
                chunks=chunks_for_mm,
                model=request.model,
                config=self.config,
            )
            return await llm.complete(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=temperature,
            )
        except Exception as e:
            logger.error("advanced_response_generation_error", error=str(e))
            return f"Error generating response: {e}"

    async def _refine_response(
        self,
        response: str,
        query: str,
        context: str = "",
        documents: list[Any] | None = None,
        llm: Any = None,
        request: RAGRequest | None = None,
        max_iterations: int | None = None,
        eval_model: str | None = None,
        eval_provider: str | None = None,
    ) -> str:
        """
        Refine response through iterative evaluation.

        Ported from: core/core.py::refine_response()
        """
        mi = max_iterations if max_iterations is not None else self.refinement_iterations
        current_response = response
        _docs: list[Any] = documents if documents is not None else []
        assert request is not None, "_refine_response requires a non-None request"

        for i in range(mi):
            # Evaluate current response
            feedback = await self._evaluate_response(
                response=current_response,
                query=query,
                documents=_docs,
                llm=llm,
                request=request,
                eval_model=eval_model,
                eval_provider=eval_provider,
            )

            # Check if response is good enough
            overall_score = feedback.get("overall_score", 0)
            if overall_score >= 8:
                logger.info("advanced_refinement_complete", score=overall_score, iteration=i + 1)
                return self._clean_refined_response(current_response)

            # Generate improved response
            current_response = await self._improve_response(
                response=current_response,
                query=query,
                documents=_docs,
                feedback=feedback,
                llm=llm,
                request=request,
            )

        logger.info("advanced_refinement_max_iterations", iterations=mi)
        # v1 refine_response: final evaluate_response (result logged; return is last draft)
        try:
            final_fb = await self._evaluate_response(
                response=current_response,
                query=query,
                documents=_docs,
                llm=llm,
                request=request,
                eval_model=eval_model,
                eval_provider=eval_provider,
            )
            logger.info(
                "advanced_refinement_final_eval", overall_score=final_fb.get("overall_score")
            )
        except Exception as e:
            logger.debug("advanced_refinement_final_eval_skipped", error=str(e))
        return current_response

    def _resolve_evaluator_model_provider(
        self,
        request: RAGRequest | None,
        eval_model: str | None = None,
        eval_provider: str | None = None,
    ) -> tuple[str | None, str | None]:
        """v1 evaluator_llm: optional separate model/provider for evaluation calls."""
        m = eval_model or (getattr(request, "evaluator_model", None) if request else None)
        m = m or self.evaluator_model
        p = eval_provider or (getattr(request, "evaluator_provider", None) if request else None)
        p = p or self.evaluator_provider
        return m, p

    async def _evaluate_response(
        self,
        response: str,
        query: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest | None = None,
        eval_model: str | None = None,
        eval_provider: str | None = None,
    ) -> dict:
        """Evaluate response quality using v1 EVALUATE_RESPONSE_PROMPT."""

        # Format documents for context (v1 evaluate_response: all docs)
        doc_texts = []
        for doc in documents or []:
            if isinstance(doc, dict) and "full_text" in doc:
                text = str(doc.get("full_text") or "")
                citation = doc.get("title") or doc.get("doi") or "Unknown"
            elif hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                text = doc.chunk.text
                citation = get_doc_citation(doc)
            else:
                text = str(doc)
                citation = "Unknown"
            doc_texts.append(f"[Citation: {citation}]\n{text}")

        doc_content = "\n\n---\n\n".join(doc_texts) if doc_texts else "No documents"

        user_message = f"""Response to evaluate:
{response}

Original query:
{query}

Source documents:
{doc_content}

Evaluate the response according to the criteria and return a valid JSON."""

        try:
            eval_kw: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": EVALUATE_RESPONSE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.0,
                "max_tokens": 800,
            }
            em, ep = self._resolve_evaluator_model_provider(request, eval_model, eval_provider)
            if em and request is not None:
                eval_kw["model"] = em
                eval_kw["provider"] = ep or request.provider
            elif request is not None:
                eval_kw["model"] = request.model
                eval_kw["provider"] = request.provider
            result = await llm.complete(**eval_kw)

            import json
            import re

            # Extract JSON (handle markdown code blocks)
            result = result.strip()
            if result.startswith("```json"):
                result = result.split("```json")[1]
            if result.endswith("```"):
                result = result.rsplit("```", 1)[0]

            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"overall_score": 5}
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"overall_score": 5}

        except Exception as e:
            logger.error("advanced_evaluation_error", error=str(e))
            return {"overall_score": 5}  # Neutral score on error

    async def _improve_response(
        self,
        response: str,
        query: str,
        documents: list[Any],
        feedback: dict,
        llm: Any,
        request: RAGRequest,
    ) -> str:
        """Generate improved response based on feedback using v1 format."""

        # v1 refine_response: [{citation}]: first 300 chars per document, all docs
        doc_summaries = []
        for j, doc in enumerate(documents or [], 1):
            if isinstance(doc, dict) and "full_text" in doc:
                content = str(doc.get("full_text") or "")
                citation = doc.get("title") or doc.get("doi") or f"Source {j}"
            elif hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                content = doc.chunk.text
                citation = get_doc_citation(doc)
            else:
                content = str(doc)
                citation = f"Source {j}"
            excerpt = content[:300] + ("..." if len(content) > 300 else "")
            doc_summaries.append(f"[{citation}]: {excerpt}")
        doc_context = "\n\n".join(doc_summaries)

        user_message = f"""Original query: {query}

Previous response:
{response}

Feedback from evaluator:
- Overall score: {feedback.get("overall_score", "Not provided")}
- Relevance: {feedback.get("relevance", {}).get("feedback", "Not provided")}
- Accuracy: {feedback.get("accuracy", {}).get("feedback", "Not provided")}
- Completeness: {feedback.get("completeness", {}).get("feedback", "Not provided")}
- Entities Recall: {feedback.get("entities_recall", {}).get("feedback", "Not provided")}
- Faithfulness: {feedback.get("faithfulness", {}).get("feedback", "Not provided")}

Missing key points: {", ".join(feedback.get("completeness", {}).get("missing_key_points", ["None provided"]))}
Missing entities: {", ".join(feedback.get("entities_recall", {}).get("missing_entities", ["None provided"]))}
Unfaithful statements: {", ".join(feedback.get("faithfulness", {}).get("unfaithful_statements", ["None provided"]))}

Source documents contain the following information:
{doc_context}

Provide an improved response that addresses all the feedback points while staying strictly faithful to the source documents."""

        try:
            return await llm.complete(
                messages=[
                    {"role": "system", "content": REFINE_RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message + REFINE_RESPONSE_HUMAN_PROMPT_SUFFIX},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("advanced_improvement_error", error=str(e))
            return response  # Return original on error

    def _is_streaming(self, request: RAGRequest) -> bool:
        """Check if request is for streaming (placeholder)."""
        return False

    def _prepare_sources(self, documents: list[Any]) -> list[SourceReference]:
        """Prepare source references from documents using utility function."""
        # Use utility function with Advanced-specific max_docs
        return prepare_sources(documents, max_docs=self.final_max_docs)

    def _format_references(self, sources: list[SourceReference]) -> str:
        """Format sources as a references section using utility function."""
        return format_references(sources)
