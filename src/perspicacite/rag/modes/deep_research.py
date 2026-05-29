"""Deep Research RAG Mode — formerly ProfoundRAGMode.

Exact implementation from release package v1 (ProfondeChain). Adds:
- Multi-cycle research with planning
- Dynamic plan creation and review
- Web search integration
- Early exit based on confidence
- Reflection and self-evaluation
- Document quality assessment
- WRRF multi-query fusion (vector-only retrieval, matching v1 profonde)
"""

import asyncio
import contextlib
import json
import math
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from perspicacite.config.schema import MultimodalMode
from perspicacite.logging import get_logger
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.provenance.context import get_collector
from perspicacite.rag.code_excerpts import collect_code_excerpts
from perspicacite.rag.figure_refs import collect_figure_refs
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.multimodal import wrap_messages_for_chunks
from perspicacite.rag.paper_metadata_codec import decode_paper_metadata_json
from perspicacite.rag.prompts import (
    ASSESS_DOCUMENT_QUALITY_PROMPT,
    GENERATE_CONTEXTUAL_QUERIES_PROMPT,
    PROFOUND_ADJUST_PLAN_PROMPT,
    PROFOUND_ANALYZE_DOCUMENTS_PROMPT_TEMPLATE,
    PROFOUND_CREATE_PLAN_PROMPT,
    PROFOUND_EVALUATE_PROGRESS_PROMPT,
    PROFOUND_FINAL_ANSWER_IMPROVED_PROMPT,
    PROFOUND_FINAL_ANSWER_ORIGINAL_PROMPT,
    PROFOUND_FORMAT_ANSWER_PROMPT,
    PROFOUND_IS_QUESTION_ANSWERED_PROMPT,
    PROFOUND_ITERATION_SUMMARY_IMPROVED_PROMPT,
    PROFOUND_ITERATION_SUMMARY_ORIGINAL_PROMPT,
    PROFOUND_UNANSWERABLE_QUESTION_PROMPT_TEMPLATE,
    SUMMARIZE_INFORMATION_PROMPT,
)
from perspicacite.rag.relevancy import assess_query_complexity, reorder_documents_by_relevance
from perspicacite.rag.telemetry import emit_phase
from perspicacite.rag.utils import (
    flatten_paper_results_to_chunks,
    format_references,
    get_doc_citation,
)
from perspicacite.rag.wrrf_v1 import doc_page_content, select_wrrf_merged_documents
from perspicacite.retrieval.multi_kb import get_chunks_by_paper_ids_across
from perspicacite.retrieval.recency import apply_recency_weighting_to_papers
from perspicacite.search.screening import (
    screen_papers,
    screen_papers_llm,
    screen_papers_rerank,
)

logger = get_logger("perspicacite.rag.modes.deep_research")


@dataclass
class ResearchStep:
    """A single step in the Profound research process."""

    step_purpose: str
    query: str
    documents: list[Any] = field(default_factory=list)
    analysis: str = ""
    success: bool = False
    key_findings: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    answer_confidence: float = 0.0
    question_answered: bool = False


@dataclass
class PlanStep:
    """A step in the research plan."""

    step_number: int
    purpose: str
    query: str
    expected_outcome: str = ""


class DeepResearchRAGMode(BaseRAGMode):
    """
    Deep Research RAG Mode — formerly ProfoundRAGMode.

    Exact port from release package core/profonde.py with:
    - Multi-cycle research (up to max_cycles)
    - Planning with step-by-step approach
    - Plan review and adjustment
    - Web search fallback
    - Early exit based on confidence threshold
    - Document quality assessment
    - Reflection and iteration

    Characteristics:
    - Most thorough but slowest mode
    - Best for complex research questions
    - Can use external web search
    - Self-evaluates and adjusts strategy
    """

    def __init__(self, config: Any):
        super().__init__(config)
        # Prefer deep_research config; fall back to profound for backward compat
        rag_settings = getattr(config.rag_modes, "deep_research", None) \
                       or getattr(config.rag_modes, "profound", None)

        # Handle both dict and Pydantic model
        if rag_settings is None:
            rag_settings = {}
        elif hasattr(rag_settings, "model_dump"):
            # Pydantic v2 model
            rag_settings = rag_settings.model_dump()
        elif hasattr(rag_settings, "dict"):
            # Pydantic v1 model
            rag_settings = rag_settings.dict()

        # Settings from release package (profonde.py ProfondeChain defaults)
        self.max_cycles = max(1, min(int(rag_settings.get("max_iterations", 1)), 5))
        self.early_exit_confidence = float(rag_settings.get("early_exit_confidence", 0.9))
        self.max_consecutive_failures = max(1, int(rag_settings.get("max_consecutive_failures", 2)))
        # F-17 (audit 2026-05-16): hard wall-clock budget so profound mode
        # can't run away when the LLM is slow (the audit hit a >13min hang
        # on DeepSeek V4 Flash with the default settings).
        self.max_total_seconds = float(rag_settings.get("max_total_seconds", 360.0))
        # Issue 2: independent cap on the final synthesis phase.
        # Cycles are never starved — total wall-clock is at most
        # max_total_seconds + synthesis_timeout_s + overhead.
        self.synthesis_timeout_s = float(rag_settings.get("synthesis_timeout_s", 90.0))
        # Per-cycle LLM call cap — defends against the planning/reflection
        # loop fanning out indefinitely. Default 15 covers a typical cycle
        # (plan + 3-5 steps + reflection + finalize) with headroom.
        self.max_llm_calls = int(rag_settings.get("max_llm_calls", 20))
        # Default ON: Profound is positioned as the "deep research" mode and
        # users expect it to consult the web when the KB falls short. Explicit
        # ``use_websearch: false`` in config still wins.
        self.use_websearch = bool(rag_settings.get("use_websearch", True))
        self.use_relevancy_optimization = bool(rag_settings.get("use_relevancy_optimization", True))
        self.use_refinement = bool(rag_settings.get("enable_reflection", rag_settings.get("use_refinement", True)))
        self.enable_plan_review = bool(rag_settings.get("enable_plan_review", True))
        self.refinement_iterations = max(1, min(int(rag_settings.get("refinement_iterations", 2)), 3))
        self.evaluator_model = rag_settings.get("evaluator_model")
        self.evaluator_provider = rag_settings.get("evaluator_provider")

        # WRRF settings
        self.use_wrrf = rag_settings.get("use_wrrf", True)
        self.wrrf_rephrases = 2
        self.wrrf_k = 60

        # v1: profonde always calls retrieve_documents with advanced_mode=False -> no hybrid
        self.use_hybrid = False

        # Two-pass retrieval (v2 extension; not in v1)
        self.use_two_pass = getattr(config.knowledge_base, "use_two_pass", True)

        # Document retrieval settings
        self.initial_docs = 150 if self.use_wrrf else 5  # More docs for WRRF
        self.final_max_docs = 5 if self.use_wrrf else 2  # More final docs with WRRF
        self.max_docs_per_source = 1

        # Sigmoid parameters for score normalization (from advanced mode)
        self.pth = 0.8
        self.stp = 30

        # State tracking
        self.iterations = 0
        self.consecutive_failures = 0
        self.research_history: list[dict] = []
        self._iteration_summaries: list[dict[str, Any]] = []

    @staticmethod
    def _strip_llm_json_block(response: str) -> str:
        t = response.strip()
        if t.startswith("```json"):
            t = t.split("```json", 1)[1]
        elif t.startswith("```"):
            t = t.split("```", 1)[1]
        if t.rstrip().endswith("```"):
            t = t.rsplit("```", 1)[0]
        t = t.strip()
        if t.startswith("{") and "}" in t:
            t = t[: t.rindex("}") + 1]
        return t

    @staticmethod
    def _renumber_plan_steps(steps: list[PlanStep]) -> None:
        for i, s in enumerate(steps, 1):
            s.step_number = i

    async def _review_and_adjust_plan(
        self,
        request: RAGRequest,
        working_plan: list[PlanStep],
        completed_steps: list[ResearchStep],
        current_step_index: int,
        llm: Any,
    ) -> dict[str, Any]:
        """core/profonde.py::_review_and_adjust_plan (async)."""
        original_question = request.query
        completed_info = "\n\n".join(
            [
                f"Step {i + 1}: {step.step_purpose}\n"
                f"Query: {step.query}\n"
                f"Success: {step.success}\n"
                f"Analysis summary: "
                f"{(step.analysis[:500] + '...') if len(step.analysis) > 500 else step.analysis}"
                for i, step in enumerate(completed_steps)
            ]
        )
        remaining_indices = list(range(current_step_index + 1, len(working_plan)))
        remaining_plan = [working_plan[i].purpose for i in remaining_indices]
        remaining_queries = [working_plan[i].query for i in remaining_indices]
        remaining_info = "\n".join(
            [
                f"{i + 1}. {step} (Query: {query})"
                for i, (step, query) in enumerate(zip(remaining_plan, remaining_queries, strict=True))
            ]
        )

        try:
            eval_response = await llm.complete(
                messages=[
                    {"role": "system", "content": PROFOUND_EVALUATE_PROGRESS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Original Question: {original_question}\n\n"
                            f"Completed Research Steps:\n{completed_info}\n\n"
                            f"Remaining Plan:\n{remaining_info}"
                        ),
                    },
                ],
                model=request.model,
                provider=request.provider,
                temperature=0.2,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            try:
                evaluation = json.loads(self._strip_llm_json_block(eval_response))
            except json.JSONDecodeError:
                logger.warning("profound_plan_eval_json_error", reason="truncated_or_invalid_json")
                return {
                    "reasoning": "Plan evaluation JSON was malformed; continuing with original plan",
                    "plan": remaining_plan,
                    "queries": remaining_queries,
                    "strategy_change": "JSON parse error in evaluation; continuing with original plan",
                }
            logger.info("profound_plan_evaluation", recommendation=evaluation.get("recommendation"))

            recommendation = evaluation.get("recommendation", "modify_plan")
            question_type = evaluation.get("question_type", "answerable")

            if recommendation == "continue_plan":
                return {
                    "reasoning": evaluation.get(
                        "reasoning", "Current plan is appropriate"
                    ),
                    "plan": remaining_plan,
                    "queries": remaining_queries,
                    "strategy_change": "No change needed - continuing with original plan",
                }

            if recommendation == "explain_limitations":
                # Don't trust the LLM's "unanswerable" verdict when no
                # documents have been retrieved yet — the evaluator only
                # sees the question text + step success flags, not the
                # actual KB content. Without evidence, "unanswerable" is
                # just the LLM being cautious about an ambiguous question
                # (e.g., "What sensitivity is achieved?" without naming
                # the device). Force a plan rewrite instead so the next
                # cycle gets a chance to actually retrieve from the KB.
                total_docs = sum(
                    len(getattr(s, "documents", []) or []) for s in completed_steps
                )
                if total_docs == 0:
                    logger.info(
                        "profound_plan_review_override_explain_limitations",
                        reason="no_documents_retrieved_yet",
                        question_type=question_type,
                    )
                    # Fall through to plan-modification path below.
                else:
                    if question_type == "unanswerable":
                        explanation = (
                            "The question appears to be unanswerable with available information. Reason: "
                            f"{evaluation.get('reasoning', 'Insufficient evidence')}"
                        )
                    elif question_type == "false_premise":
                        explanation = (
                            "The question appears to be based on false premises. Reason: "
                            f"{evaluation.get('reasoning', 'Misconception detected')}"
                        )
                    else:
                        explanation = (
                            "The question can only be partially answered. Reason: "
                            f"{evaluation.get('reasoning', 'Limited information available')}"
                        )
                    return {
                        "reasoning": explanation,
                        "plan": [],
                        "queries": [],
                        "strategy_change": f"Research ending due to {question_type} question",
                        "should_complete": True,
                        "question_type": question_type,
                        "completion_explanation": explanation,
                    }

            adjust_response = await llm.complete(
                messages=[
                    {"role": "system", "content": PROFOUND_ADJUST_PLAN_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Original Question: {original_question}\n\n"
                            f"Question Type: {question_type}\n\n"
                            f"Evaluation: {evaluation.get('evaluation', '')}\n\n"
                            f"Completed Research Steps:\n{completed_info}\n\n"
                            f"Remaining Plan:\n{remaining_info}"
                        ),
                    },
                ],
                model=request.model,
                provider=request.provider,
                temperature=0.2,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            result = json.loads(self._strip_llm_json_block(adjust_response))
            result["initial_evaluation"] = evaluation
            return result
        except Exception as e:
            logger.error("profound_plan_review_error", error=str(e))
            return {
                "reasoning": f"Error adjusting plan: {e}",
                "plan": remaining_plan,
                "queries": remaining_queries,
                "strategy_change": "Error occurred, continuing with original plan",
            }

    async def _execute_cycle_steps(
        self,
        request: RAGRequest,
        plan: list[PlanStep],
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
        kb_name: str,
        collection_names: list[str] | None = None,
        telemetry: list[dict[str, Any]] | None = None,
    ) -> tuple[list[ResearchStep], list[Any], str | None, bool]:
        """
        Run one cycle's plan steps with v1 consecutive-failure plan review.

        Returns:
            (cycle_steps, cycle_documents, plan_limit_reason, early_answer_exit)
            plan_limit_reason: unanswerable | false_premise | partially_answerable | limitations_detected
            early_answer_exit: True if question answered with confidence — caller finalizes with question_answered
        """
        working_plan = list(plan)
        cycle_steps: list[ResearchStep] = []
        cycle_documents: list[Any] = []
        cycle_consecutive_failures = 0
        i = 0

        while i < len(working_plan):
            step_info = working_plan[i]
            step = await self._execute_step(
                step_info=step_info,
                query=request.query,
                llm=llm,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                tools=tools,
                kb_name=kb_name,
                model=getattr(request, "model", None),
                collection_names=collection_names,
                telemetry=telemetry,
                databases=getattr(request, "databases", None),
            )
            cycle_steps.append(step)
            cycle_documents.extend(step.documents)

            if step.success:
                cycle_consecutive_failures = 0
            else:
                cycle_consecutive_failures += 1

            if step.success and step.documents:
                qa = step.question_answered
                conf = step.answer_confidence
                if not qa or conf < self.early_exit_confidence:
                    qa, conf = await self._is_question_answered(
                        cycle_steps=cycle_steps,
                        original_query=request.query,
                        llm=llm,
                    )
                if qa and conf >= self.early_exit_confidence:
                    return cycle_steps, cycle_documents, None, True

            if (
                self.enable_plan_review
                and cycle_consecutive_failures >= self.max_consecutive_failures
                and i < len(working_plan) - 1
            ):
                logger.info(
                    "profound_plan_review_trigger",
                    consecutive_failures=cycle_consecutive_failures,
                )
                adjusted = await self._review_and_adjust_plan(
                    request=request,
                    working_plan=working_plan,
                    completed_steps=cycle_steps,
                    current_step_index=i,
                    llm=llm,
                )
                if adjusted.get("should_complete"):
                    qt = adjusted.get("question_type", "limitations_detected")
                    return cycle_steps, cycle_documents, str(qt), False

                new_plan_s = adjusted.get("plan") or []
                new_queries_s = adjusted.get("queries") or []
                if len(new_plan_s) != len(new_queries_s):
                    logger.warning("profound_adjusted_plan_mismatch")
                else:
                    prefix = working_plan[: i + 1]
                    new_steps = [
                        PlanStep(step_number=0, purpose=str(p), query=str(q))
                        for p, q in zip(new_plan_s, new_queries_s, strict=True)
                    ]
                    working_plan[:] = prefix + new_steps
                    self._renumber_plan_steps(working_plan)
                    cycle_consecutive_failures = 0

            i += 1

        return cycle_steps, cycle_documents, None, False

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """
        Execute Profound RAG with multi-cycle planning and reflection.

        Ported from: core/profonde.py::ProfondeChain.process()
        """
        # Per-request override: ChatRequest may set max_iterations.
        _req_max = getattr(request, "max_iterations", None)
        if _req_max is not None:
            try:
                max_cycles = max(1, min(int(_req_max), 5))
            except (TypeError, ValueError):
                max_cycles = self.max_cycles
        else:
            max_cycles = self.max_cycles

        logger.info("profound_rag_start", query=request.query, max_cycles=max_cycles)

        # Reset state
        self.iterations = 0
        self.consecutive_failures = 0
        self.research_history = []
        self._iteration_summaries = []

        all_steps: list[ResearchStep] = []
        all_documents: list[Any] = []
        completion_reason: str | None = None
        kb_name = chroma_collection_name_for_kb(request.kb_name)
        kb_names = getattr(request, "kb_names", None) or [request.kb_name]
        collection_names = [chroma_collection_name_for_kb(n) for n in kb_names]

        import time as _time
        _start_time = _time.monotonic()
        _cancelled_tid: str | None = None

        # Per-call budget override: request.max_total_seconds takes
        # precedence over the config-file default stored on self.
        max_total_seconds = (
            getattr(request, "max_total_seconds", None)
            or self.max_total_seconds
        )

        # Main research loop
        for cycle in range(max_cycles):
            # Cancellation check — respect MCP cancel_task requests
            from perspicacite.rag.cancellation import is_cancelled
            _tid = getattr(request, "task_id", None)
            if _tid and is_cancelled(_tid):
                logger.info("profound_cancelled", task_id=_tid, cycle=cycle)
                _cancelled_tid = _tid
                break

            # F-17: hard wall-clock budget. Break before starting a new
            # cycle if we've already burned the budget — the answer so
            # far is still finalized below.
            elapsed = _time.monotonic() - _start_time
            if elapsed > max_total_seconds:
                logger.warning(
                    "profound_time_budget_exceeded",
                    cycle=cycle,
                    elapsed_s=round(elapsed, 1),
                    budget_s=max_total_seconds,
                )
                completion_reason = "time_budget_exceeded"
                break
            self.iterations = cycle + 1
            logger.info("profound_cycle_start", cycle=self.iterations)

            plan = await self._create_plan(query=request.query, llm=llm)
            logger.info(
                "profound_plan_created", steps=len(plan), purposes=[s.purpose for s in plan]
            )
            _c_plan = get_collector()
            if _c_plan is not None:
                _c_plan.add_trace("plan", detail={"cycle": self.iterations, "steps": len(plan)})

            cycle_steps, cycle_documents, plan_limit_reason, early_exit = (
                await self._execute_cycle_steps(
                    request=request,
                    plan=plan,
                    llm=llm,
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    tools=tools,
                    kb_name=kb_name,
                    collection_names=collection_names,
                )
            )

            # Apply recency weighting to paper-dict results for this cycle
            cycle_paper_docs = [
                d for d in cycle_documents if isinstance(d, dict) and "paper_id" in d
            ]
            if getattr(request, "recency_weight", None) and cycle_paper_docs:
                apply_recency_weighting_to_papers(
                    cycle_paper_docs,
                    recency_weight=getattr(request, "recency_weight", None),
                    half_life_years=getattr(request, "recency_half_life_years", None),
                )

            # Provenance: record cycle trace + per-paper retrieval events
            _c = get_collector()
            if _c is not None:
                _c.add_trace(
                    "cycle",
                    detail={
                        "n": self.iterations,
                        "papers": len(cycle_paper_docs),
                        "kb_name": request.kb_name,
                    },
                )
                for rank, p in enumerate(cycle_paper_docs):
                    _c.add_retrieval(
                        paper_id=p.get("paper_id"),
                        doi=p.get("doi"),
                        title=p.get("title"),
                        score=float(p.get("paper_score", p.get("score", 0.0)) or 0.0),
                        kb_name=p.get("kb_name"),
                        content_type=None,
                        pipeline_step=None,
                        rank=rank,
                        stage_label=f"profound.cycle{self.iterations}.retrieve",
                    )

            all_steps.extend(cycle_steps)
            all_documents.extend(cycle_documents)

            if early_exit:
                logger.info("profound_early_exit", cycle=self.iterations)
                try:
                    async with asyncio.timeout(self.synthesis_timeout_s):
                        return await self._finalize_response(
                            query=request.query,
                            steps=all_steps,
                            documents=all_documents,
                            llm=llm,
                            request=request,
                            exited_early=True,
                            completion_reason="question_answered",
                            cancelled_tid=_cancelled_tid,
                        )
                except TimeoutError:
                    logger.warning(
                        "profound_synthesis_timeout_execute",
                        synthesis_timeout_s=self.synthesis_timeout_s,
                    )
                    sources = self._prepare_sources(all_documents)
                    return RAGResponse(
                        answer=(
                            "Research synthesis exceeded the time budget. "
                            "Partial research was completed but the final answer could not be generated."
                        ),
                        sources=sources,
                        mode=RAGMode.PROFOUND,
                        iterations=self.iterations,
                        web_search_used=any(
                            isinstance(d, dict) and d.get("source") == "web_search"
                            for d in all_documents
                        ),
                        metadata={"completion_reason": "synthesis_timeout"},
                    )

            if plan_limit_reason:
                completion_reason = plan_limit_reason
                logger.info("profound_plan_limit_exit", reason=plan_limit_reason)
                if plan_limit_reason in (
                    "unanswerable",
                    "false_premise",
                    "partially_answerable",
                    "limitations_detected",
                ):
                    self._iteration_summaries.append(
                        {
                            "findings": (
                                f"Research completed because the question was determined "
                                f"to be {plan_limit_reason}."
                            ),
                            "missing": [],
                            "should_continue": False,
                        }
                    )
                break

            summary = await self._create_iteration_summary(request.query, cycle_steps, llm)
            self._iteration_summaries.append(summary)
            _c_refl = get_collector()
            if _c_refl is not None:
                _c_refl.add_trace(
                    "reflection",
                    detail={
                        "cycle": self.iterations,
                        "should_continue": bool(summary.get("should_continue", False)),
                    },
                )
            self.research_history.append(
                {
                    "cycle": self.iterations,
                    "steps": [
                        {
                            "purpose": s.step_purpose,
                            "success": s.success,
                            "findings": s.key_findings,
                            "missing": s.missing_info,
                        }
                        for s in cycle_steps
                    ],
                }
            )

            should_continue = (
                bool(summary.get("should_continue", False)) and cycle < max_cycles - 1
            )
            if not should_continue:
                logger.info("profound_iteration_summary_stop", cycle=self.iterations)
                break

            cycle_successes = sum(1 for s in cycle_steps if s.success)
            if cycle_successes == 0:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.warning("profound_max_failures_reached")
                    break
            else:
                self.consecutive_failures = 0

        try:
            async with asyncio.timeout(self.synthesis_timeout_s):
                return await self._finalize_response(
                    query=request.query,
                    steps=all_steps,
                    documents=all_documents,
                    llm=llm,
                    request=request,
                    exited_early=False,
                    completion_reason=completion_reason,
                    cancelled_tid=_cancelled_tid,
                )
        except TimeoutError:
            logger.warning(
                "profound_synthesis_timeout_execute",
                synthesis_timeout_s=self.synthesis_timeout_s,
            )
            sources = self._prepare_sources(all_documents)
            return RAGResponse(
                answer=(
                    "Research synthesis exceeded the time budget. "
                    "Partial research was completed but the final answer could not be generated."
                ),
                sources=sources,
                mode=RAGMode.PROFOUND,
                iterations=self.iterations,
                web_search_used=any(
                    isinstance(d, dict) and d.get("source") == "web_search"
                    for d in all_documents
                ),
                metadata={"completion_reason": "synthesis_timeout"},
            )

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute Profound RAG with streaming output."""

        _phase_sink = getattr(request, "telemetry_sink", None)
        emit_phase(_phase_sink, phase="rewrite", state="running")

        # Per-request override: ChatRequest may set max_iterations to let
        # the user pick a number of cycles from the composer. Bound to
        # the same safe range used at __init__ (1-5).
        _req_max = getattr(request, "max_iterations", None)
        if _req_max is not None:
            try:
                max_cycles = max(1, min(int(_req_max), 5))
            except (TypeError, ValueError):
                max_cycles = self.max_cycles
        else:
            max_cycles = self.max_cycles

        yield StreamEvent.status(
            f"Deep Research: Initializing deep research ({max_cycles} cycle{'s' if max_cycles != 1 else ''})..."
        )

        # Reset state
        self.iterations = 0
        self.consecutive_failures = 0
        self.research_history = []
        self._iteration_summaries = []

        _diag: dict = {
            "cycles_completed": 0,
            "papers_retrieved": 0,
        }

        # Live token counters. Each LLM call this Profond run makes goes
        # through the wrapper below, which accumulates into these dicts;
        # we yield a `usage` status frame between research steps so the
        # status bar's tokens-in / tokens-out counters tick up live
        # instead of jumping from 0 to a big number only after the final
        # synthesis streams.
        _tok = {"in": 0, "out": 0}

        class _CountingLLM:
            __slots__ = ("_acc", "_real")

            def __init__(self, real: Any, acc: dict[str, int]) -> None:
                object.__setattr__(self, "_real", real)
                object.__setattr__(self, "_acc", acc)

            async def complete(self, *args: Any, **kwargs: Any) -> Any:
                resp = await self._real.complete(*args, **kwargs)
                try:
                    self._acc["in"] += int(getattr(resp, "input_tokens", 0) or 0)
                    self._acc["out"] += int(getattr(resp, "output_tokens", 0) or 0)
                except (AttributeError, TypeError, ValueError):
                    pass
                return resp

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

        llm = _CountingLLM(llm, _tok)

        all_steps: list[ResearchStep] = []
        all_documents: list[Any] = []
        completion_reason: str | None = None
        kb_name = chroma_collection_name_for_kb(request.kb_name)
        kb_names = getattr(request, "kb_names", None) or [request.kb_name]
        collection_names = [chroma_collection_name_for_kb(n) for n in kb_names]

        import time as _time
        _start_time = _time.monotonic()

        # Per-call budget override: request.max_total_seconds takes
        # precedence over the config-file default stored on self.
        max_total_seconds = (
            getattr(request, "max_total_seconds", None)
            or self.max_total_seconds
        )

        # Run the keyword optimizer UPFRONT so the rephrase event lands
        # immediately. Without this, the rephrase only fires when the
        # per-step web search runs the optimizer internally — visible
        # mid-cycle (after 1-3 min) instead of at the very start.
        try:
            from perspicacite.search.query_optimizer import optimize_query as _qopt
            _app = getattr(request, "app_state", None)
            opt_res = await _qopt(
                query=request.query,
                context=None,
                app_state=_app,
                optimize_enabled=True,
            )
            if opt_res.applied and opt_res.searched_query:
                yield StreamEvent.status_kind(
                    f"Rewrote search query: '{request.query}' → '{opt_res.searched_query}'",
                    kind="query_rephrased",
                    original=request.query,
                    rewritten=opt_res.searched_query,
                    by="keyword_optimizer",
                )
        except Exception as _qe:
            logger.debug("profound_upfront_optimizer_failed", error=str(_qe))

        emit_phase(_phase_sink, phase="rewrite", state="done")
        emit_phase(_phase_sink, phase="retrieve", state="running")
        for cycle in range(max_cycles):
            from perspicacite.rag.cancellation import is_cancelled
            _tid = getattr(request, "task_id", None)
            if _tid and is_cancelled(_tid):
                logger.info("profound_cancelled", task_id=_tid, cycle=cycle)
                yield StreamEvent.diagnostic(**_diag, cancellation_reason="cancelled")
                yield StreamEvent.metadata(
                    iteration_count=self.iterations,
                    completion_reason="cancelled",
                )
                yield StreamEvent(event="error", data=json.dumps({"reason": "cancelled", "task_id": _tid}))
                return

            # F-17: hard wall-clock budget. Break before starting a new
            # cycle if we've already burned the budget — the answer so
            # far is still finalized below.
            elapsed = _time.monotonic() - _start_time
            if elapsed > max_total_seconds:
                logger.warning(
                    "profound_time_budget_exceeded",
                    cycle=cycle,
                    elapsed_s=round(elapsed, 1),
                    budget_s=max_total_seconds,
                )
                completion_reason = "time_budget_exceeded"
                break

            self.iterations = cycle + 1
            yield StreamEvent.status(
                f"Deep Research: Research cycle {self.iterations}/{max_cycles}..."
            )

            plan = await self._create_plan(query=request.query, llm=llm)
            yield StreamEvent.status_kind(
                "tokens", kind="usage", tokens_in=_tok["in"], tokens_out=_tok["out"],
            )
            yield StreamEvent.status(f"Deep Research: Executing {len(plan)} research steps...")
            if self.use_websearch and "web_search" in tools.list_tools():
                yield StreamEvent.status(
                    "Deep Research: Web search is available — will consult live "
                    "databases when KB coverage is insufficient."
                )
            _c_plan_s = get_collector()
            if _c_plan_s is not None:
                _c_plan_s.add_trace("plan", detail={"cycle": self.iterations, "steps": len(plan)})

            _cycle_telemetry = getattr(request, "telemetry_sink", None) or []
            # Run _execute_cycle_steps (non-generator) while yielding SSE heartbeats
            # every 30 s so streaming clients don't time out during long LLM calls.
            _cycle_task = asyncio.create_task(
                self._execute_cycle_steps(
                    request=request,
                    plan=plan,
                    llm=llm,
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    tools=tools,
                    kb_name=kb_name,
                    collection_names=collection_names,
                    telemetry=_cycle_telemetry,
                )
            )
            _heartbeat_n = 0
            while not _cycle_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(_cycle_task), timeout=30.0)
                except TimeoutError:
                    _heartbeat_n += 1
                    yield StreamEvent.status(
                        f"Deep Research: Cycle {self.iterations} — working… "
                        f"({_heartbeat_n * 30}s elapsed)"
                    )
            cycle_steps, cycle_documents, plan_limit_reason, early_exit = _cycle_task.result()
            # Drain web-search telemetry into the SSE stream so the user
            # sees per-DB activity for this cycle. Without this, profound
            # was a "black box" for 5+ min per cycle.
            # When _cycle_telemetry is a CallbackTelemetrySink (MCP path),
            # events already flowed to ctx.report_progress live — skip drain.
            if isinstance(_cycle_telemetry, list):
                for _ev in _cycle_telemetry:
                    _k = _ev.get("kind")
                    if _k == "query_rephrased":
                        yield StreamEvent.status_kind(
                            f"Cycle {self.iterations}: rewrote search query — "
                            f"'{_ev.get('original','')}' → '{_ev.get('rewritten','')}'",
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
                            f"Cycle {self.iterations}: querying databases — {_provs} (keywords: '{_sq}')"
                            if _sq
                            else f"Cycle {self.iterations}: querying databases — {_provs}…"
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
                            f"Cycle {self.iterations}: database results — {_msg}",
                            kind="provider_progress",
                            phase="done",
                            total=_ev.get("total", 0),
                            by_provider=_bp,
                        )

            # Surface web search activity post-hoc — _execute_cycle_steps is
            # not an async generator, so we can't yield from inside stage_3.
            # Counting ``source == "web_search"`` documents tells the user
            # whether the live search actually ran this cycle.
            _web_docs = [
                d for d in cycle_documents
                if isinstance(d, dict) and d.get("source") == "web_search"
            ]
            if _web_docs:
                yield StreamEvent.status(
                    f"Deep Research: Cycle {self.iterations} consulted the web "
                    f"({len(_web_docs)} document{'s' if len(_web_docs) != 1 else ''} from live search)."
                )

            # Apply recency weighting to paper-dict results for this cycle
            cycle_paper_docs = [
                d for d in cycle_documents if isinstance(d, dict) and "paper_id" in d
            ]
            if getattr(request, "recency_weight", None) and cycle_paper_docs:
                apply_recency_weighting_to_papers(
                    cycle_paper_docs,
                    recency_weight=getattr(request, "recency_weight", None),
                    half_life_years=getattr(request, "recency_half_life_years", None),
                )

            # Provenance: record cycle trace + per-paper retrieval events
            _c = get_collector()
            if _c is not None:
                _c.add_trace(
                    "cycle",
                    detail={
                        "n": self.iterations,
                        "papers": len(cycle_paper_docs),
                        "kb_name": request.kb_name,
                    },
                )
                for rank, p in enumerate(cycle_paper_docs):
                    _c.add_retrieval(
                        paper_id=p.get("paper_id"),
                        doi=p.get("doi"),
                        title=p.get("title"),
                        score=float(p.get("paper_score", p.get("score", 0.0)) or 0.0),
                        kb_name=p.get("kb_name"),
                        content_type=None,
                        pipeline_step=None,
                        rank=rank,
                        stage_label=f"profound.cycle{self.iterations}.retrieve",
                    )

            all_steps.extend(cycle_steps)
            all_documents.extend(cycle_documents)

            _diag["cycles_completed"] = self.iterations
            _diag["papers_retrieved"] = max(
                _diag["papers_retrieved"], len(cycle_documents)
            )

            # Surface cumulative LLM token usage for this cycle so the
            # status bar updates live instead of waiting for the final
            # synthesis to stream.
            yield StreamEvent.status_kind(
                "tokens", kind="usage", tokens_in=_tok["in"], tokens_out=_tok["out"],
            )

            if early_exit:
                _diag["cycles_completed"] = self.iterations
                yield StreamEvent.diagnostic(**_diag, cancellation_reason="early_exit_confidence")
                emit_phase(_phase_sink, phase="retrieve", state="done")
                emit_phase(_phase_sink, phase="reason", state="done")
                emit_phase(_phase_sink, phase="synthesize", state="running")
                yield StreamEvent.status("Deep Research: Early exit — synthesizing final answer...")
                try:
                    async with asyncio.timeout(self.synthesis_timeout_s):
                        async for event in self._stream_final_response(
                            query=request.query,
                            steps=all_steps,
                            documents=all_documents,
                            llm=llm,
                            request=request,
                            exited_early=True,
                            completion_reason="question_answered",
                        ):
                            yield event
                except TimeoutError:
                    logger.warning(
                        "profound_synthesis_timeout_early_exit",
                        synthesis_timeout_s=self.synthesis_timeout_s,
                    )
                    completion_reason = "synthesis_timeout"
                    yield StreamEvent.status(
                        "Deep research: synthesis time budget reached — returning partial answer."
                    )
                    _fallback = self._build_fallback_report(
                        query=request.query,
                        steps=all_steps,
                        documents=all_documents,
                        reason="synthesis_timeout",
                    )
                    if _fallback.strip():
                        yield StreamEvent.content(_fallback)
                yield StreamEvent.metadata(
                    iteration_count=self.iterations,
                    completion_reason=completion_reason or "complete",
                )
                emit_phase(_phase_sink, phase="synthesize", state="done")
                return

            if plan_limit_reason:
                completion_reason = plan_limit_reason
                yield StreamEvent.status(
                    f"Deep Research: Plan review ended research ({plan_limit_reason})"
                )
                if plan_limit_reason in (
                    "unanswerable",
                    "false_premise",
                    "partially_answerable",
                    "limitations_detected",
                ):
                    self._iteration_summaries.append(
                        {
                            "findings": (
                                f"Research completed because the question was determined "
                                f"to be {plan_limit_reason}."
                            ),
                            "missing": [],
                            "should_continue": False,
                        }
                    )
                break

            emit_phase(_phase_sink, phase="reflect", state="running", cycle=cycle)
            summary = await self._create_iteration_summary(request.query, cycle_steps, llm)
            emit_phase(_phase_sink, phase="reflect", state="done", cycle=cycle)
            self._iteration_summaries.append(summary)
            _c_refl_s = get_collector()
            if _c_refl_s is not None:
                _c_refl_s.add_trace(
                    "reflection",
                    detail={
                        "cycle": self.iterations,
                        "should_continue": bool(summary.get("should_continue", False)),
                    },
                )
            self.research_history.append(
                {
                    "cycle": self.iterations,
                    "steps": [
                        {
                            "purpose": s.step_purpose,
                            "success": s.success,
                            "findings": s.key_findings,
                            "missing": s.missing_info,
                        }
                        for s in cycle_steps
                    ],
                }
            )

            should_continue = (
                bool(summary.get("should_continue", False)) and cycle < max_cycles - 1
            )
            if not should_continue:
                yield StreamEvent.status("Deep Research: Research complete based on iteration summary")
                break

            cycle_successes = sum(1 for s in cycle_steps if s.success)
            if cycle_successes == 0:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    yield StreamEvent.status("Deep Research: Max consecutive failures reached")
                    break
            else:
                self.consecutive_failures = 0

        emit_phase(_phase_sink, phase="retrieve", state="done")
        emit_phase(_phase_sink, phase="reason", state="done")
        emit_phase(_phase_sink, phase="synthesize", state="running")
        yield StreamEvent.status_kind(
            "tokens", kind="usage", tokens_in=_tok["in"], tokens_out=_tok["out"],
        )
        yield StreamEvent.status("Deep Research: Synthesizing final answer...")
        try:
            async with asyncio.timeout(self.synthesis_timeout_s):
                async for event in self._stream_final_response(
                    query=request.query,
                    steps=all_steps,
                    documents=all_documents,
                    llm=llm,
                    request=request,
                    exited_early=False,
                    completion_reason=completion_reason,
                ):
                    yield event
        except TimeoutError:
            logger.warning(
                "profound_synthesis_timeout_normal",
                synthesis_timeout_s=self.synthesis_timeout_s,
            )
            yield StreamEvent.status(
                "Deep research: synthesis time budget reached — returning partial answer."
            )
            _fallback = self._build_fallback_report(
                query=request.query,
                steps=all_steps,
                documents=all_documents,
                reason="synthesis_timeout",
            )
            if _fallback.strip():
                yield StreamEvent.content(_fallback)
            completion_reason = "synthesis_timeout"
        yield StreamEvent.diagnostic(**_diag)
        yield StreamEvent.metadata(
            iteration_count=self.iterations,
            completion_reason=completion_reason or "complete",
        )
        emit_phase(_phase_sink, phase="synthesize", state="done")

    def _build_fallback_report(
        self,
        query: str,
        steps: list,
        documents: list,
        reason: str = "synthesis_timeout",
    ) -> str:
        """Build a minimal report from research step summaries when LLM synthesis times out."""
        lines: list[str] = [
            f"**Research Summary** *(synthesis {reason} — partial results)*\n",
            f"**Query:** {query}\n",
        ]
        # Step analyses
        if steps:
            lines.append(f"**Research steps completed ({self.iterations} cycle(s)):**\n")
            for i, step in enumerate(steps, 1):
                analysis = getattr(step, "analysis", "") or ""
                if analysis.strip():
                    lines.append(f"{i}. {analysis[:600]}{'…' if len(analysis) > 600 else ''}\n")
        # Key documents found
        docs_shown = 0
        if documents:
            lines.append("\n**Sources retrieved:**\n")
            for d in documents[:5]:
                if isinstance(d, dict):
                    title = (d.get("title") or "Untitled").strip()
                    year = d.get("year") or ""
                    authors = d.get("authors") or []
                    author_str = ", ".join(str(a) for a in authors[:2]) if authors else ""
                    lines.append(
                        f"- {title}"
                        + (f" ({author_str}, {year})" if (author_str or year) else "")
                        + "\n"
                    )
                    docs_shown += 1
                elif hasattr(d, "chunk") and hasattr(d.chunk, "metadata"):
                    meta = d.chunk.metadata
                    title = (getattr(meta, "title", None) or "Untitled").strip()
                    year = getattr(meta, "year", None) or ""
                    lines.append(f"- {title}" + (f" ({year})" if year else "") + "\n")
                    docs_shown += 1
        if docs_shown == 0:
            lines.append("*(No documents retrieved)*\n")
        return "".join(lines)

    async def _create_plan(
        self,
        query: str,
        llm: Any,
    ) -> list[PlanStep]:
        """
        Ported from: core/profonde.py::_create_research_plan (v1 JSON plan + queries).
        """
        prev_findings = [
            {"summary": s.get("findings", ""), "missing_info": s.get("missing", [])}
            for s in self._iteration_summaries
        ]
        context = {"question": query, "previous_findings": prev_findings}

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": PROFOUND_CREATE_PLAN_PROMPT},
                    {"role": "user", "content": f"Context: {json.dumps(context)}"},
                ],
                temperature=0.3,
                # 800 was tight — the prompt asks for plan + parallel queries
                # in JSON and was getting truncated mid-string ("Unterminated
                # string starting at: line 2 column 18"). Bumped to 1500.
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            if not response:
                logger.warning(
                    "profound_plan_creation_error",
                    error="empty_response_from_llm",
                )
                return [
                    PlanStep(1, "Search for general information", query),
                    PlanStep(2, "Search for specific details", f"{query} methodology"),
                ]
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json", 1)[1]
            elif response.startswith("```"):
                response = response.split("```", 1)[1]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            response = response.strip()
            if response.startswith("{") and "}" in response:
                response = response[: response.rindex("}") + 1]

            from perspicacite.rag.utils.json_salvage import (
                clean_control_chars,
                salvage_truncated_array,
            )
            response = clean_control_chars(response)
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                salvaged_plan = salvage_truncated_array(response, "plan")
                if salvaged_plan is not None:
                    logger.info("profound_plan_json_salvaged", recovered=len(salvaged_plan))
                    result = {"plan": salvaged_plan, "queries": [s.get("query", "") for s in salvaged_plan if isinstance(s, dict)]}
                else:
                    raise
            plan_s = result.get("plan", [])
            queries_s = result.get("queries", [])
            if not plan_s or not queries_s:
                return [PlanStep(1, "Search for general information", query)]

            steps: list[PlanStep] = []
            # strict=False: plan_s and queries_s come from independent LLM JSON
            # fields; a length mismatch is malformed model output, so we pair
            # the common prefix rather than raising.
            for i, (purp, q) in enumerate(zip(plan_s, queries_s, strict=False), 1):
                steps.append(PlanStep(step_number=i, purpose=str(purp), query=str(q)))

            return steps

        except Exception as e:
            logger.error("profound_plan_creation_error", error=str(e))
            return [
                PlanStep(1, "Search for general information", query),
                PlanStep(2, "Search for specific details", f"{query} methodology"),
            ]

    async def _create_iteration_summary(
        self, question: str, steps: list[ResearchStep], llm: Any
    ) -> dict[str, Any]:
        """Ported from: core/profonde.py::_create_iteration_summary."""
        prompt = (
            PROFOUND_ITERATION_SUMMARY_IMPROVED_PROMPT
            if self.use_relevancy_optimization
            else PROFOUND_ITERATION_SUMMARY_ORIGINAL_PROMPT
        )
        steps_summary = "\n\n".join(
            [
                f"Step: {s.query}\nPurpose: {s.step_purpose}\nAnalysis: {s.analysis}"
                for s in steps
            ]
        )
        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"Original Question: {question}\n\nResearch Steps:\n{steps_summary}",
                    },
                ],
                temperature=0.3,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json", 1)[1]
            elif response.startswith("```"):
                response = response.split("```", 1)[1]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            response = response.strip()
            if response.startswith("{") and "}" in response:
                response = response[: response.rindex("}") + 1]
            return json.loads(response)
        except Exception as e:
            logger.error("profound_iteration_summary_error", error=str(e))
            return {
                "findings": "Error summarizing findings",
                "missing": [],
                "should_continue": False,
                "reasoning": str(e),
            }

    async def _generate_similar_queries(
        self, original_query: str, llm: Any, number: int = 2
    ) -> list[str]:
        """Generate similar query variations for WRRF."""
        queries = [original_query]
        if not number or number <= 0:
            return queries
        for _i in range(number):
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
                new_query = response.strip()
                if new_query and new_query not in queries:
                    queries.append(new_query)
            except Exception as e:
                logger.warning("profound_query_generation_error", error=str(e))
                break
        return queries

    async def _wrrf_retrieval(
        self,
        queries: list[str],
        vector_store: Any,
        embedding_provider: Any,
        kb_name: str,
        llm: Any = None,
        collection_names: list[str] | None = None,
    ) -> list[Any]:
        """
        core/core.py retrieve_documents multi-query branch: vector only (v1 profonde
        uses advanced_mode=False so hybrid is never applied).

        When `collection_names` has more than one entry, the per-query search
        is fanned out across all listed collections and each result is tagged
        with its originating collection name (kb_name). Single-collection
        behaviour is preserved when `collection_names` is None or length 1.
        """
        effective_collections = list(collection_names) if collection_names else [kb_name]

        rankings: dict[Any, dict[int, int]] = {}
        scores_per_query: dict[int, dict[Any, float]] = {}
        documents_info: dict[Any, Any] = {}

        for q_idx, query in enumerate(queries):
            query_embedding = await embedding_provider.embed_query([query])
            if len(effective_collections) == 1:
                results = await vector_store.search(
                    collection=effective_collections[0],
                    query_embedding=query_embedding[0],
                    top_k=self.initial_docs,
                )
            else:
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
                            "profound_wrrf_fanout_search_failed",
                            collection=coll,
                            error=str(e),
                        )
                        continue
                    for r in coll_results:
                        with contextlib.suppress(Exception):
                            r.kb_name = coll
                    results.extend(coll_results)
                results.sort(key=lambda r: getattr(r, "score", 0.0), reverse=True)
                results = results[: self.initial_docs]
            scores_per_query[q_idx] = {}
            for rank, doc in enumerate(results, start=1):
                doc_id = doc_page_content(doc)
                score = getattr(doc, "score", 0.5)
                norm_score = 1 / (1 + math.exp(-(score - self.pth) * self.stp))
                if doc_id not in rankings:
                    rankings[doc_id] = {}
                    documents_info[doc_id] = doc
                rankings[doc_id][q_idx] = rank
                scores_per_query[q_idx][doc_id] = norm_score

        wrrf_scores: dict[Any, float] = {}
        for doc_id in rankings:
            total_score = 0.0
            for q_idx, rank in rankings[doc_id].items():
                norm_score = scores_per_query[q_idx][doc_id]
                total_score += norm_score / (self.wrrf_k + rank)
            wrrf_scores[doc_id] = total_score

        sorted_docs = sorted(wrrf_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_docs:
            return []
        return select_wrrf_merged_documents(
            sorted_docs,
            documents_info,
            self.final_max_docs,
            self.max_docs_per_source,
        )

    async def _basic_vector_retrieve(
        self,
        query: str,
        vector_store: Any,
        embedding_provider: Any,
        kb_name: str,
        collection_names: list[str] | None = None,
    ) -> list[Any]:
        """core/core.py single-query basic path: vector search + per-source cap.

        When `collection_names` is provided with more than one entry, the
        search is fanned out across all listed collections and each retrieved
        result is tagged with its originating collection name (kb_name).
        Single-collection behaviour is preserved when `collection_names` is
        None or has a single entry.
        """
        effective_collections = list(collection_names) if collection_names else [kb_name]

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
                        "profound_basic_fanout_search_failed",
                        collection=coll,
                        error=str(e),
                    )
                    continue
                for r in coll_results:
                    with contextlib.suppress(Exception):
                        r.kb_name = coll
                results.extend(coll_results)
        results = sorted(results, key=lambda r: getattr(r, "score", 0.0), reverse=True)
        selected: list[Any] = []
        source_counter: Counter[str] = Counter()
        for doc in results:
            source = get_doc_citation(doc)
            if source_counter[source] >= self.max_docs_per_source:
                continue
            selected.append(doc)
            source_counter[source] += 1
            if len(selected) >= self.final_max_docs:
                break
        return selected

    async def _enrich_with_full_text(
        self,
        results: list[Any],
        kb_name: str,
        vector_store: Any,
        collection_names: list[str] | None = None,
    ) -> list[Any]:
        """Two-pass enrichment: given chunk-level results, fetch full paper text.

        When `collection_names` has more than one entry, fans the
        `get_chunks_by_paper_ids` call across collections via
        `get_chunks_by_paper_ids_across`. Each resulting paper-level dict is
        tagged with the originating collection's kb_name (derived from the
        first-pass `r.kb_name` attribute set by the fan-out retriever).
        Single-collection behaviour is preserved when `collection_names` is
        None or has a single entry.
        """
        from perspicacite.rag.utils import deduplicate_chunk_overlaps

        effective_collections = list(collection_names) if collection_names else [kb_name]

        paper_ids = []
        paper_scores: dict[str, float] = {}
        paper_kb: dict[str, str | None] = {}
        for r in results:
            meta = getattr(r, "chunk", None)
            if meta and hasattr(meta, "metadata"):
                pid = getattr(meta.metadata, "paper_id", None)
            elif isinstance(r, dict):
                pid = r.get("paper_id")
            else:
                pid = None
            if pid and pid not in paper_ids:
                paper_ids.append(pid)
                score = getattr(r, "score", getattr(r, "wrrf_score", 0.5))
                paper_scores[pid] = score
                # First-pass kb tag (set by fan-out search). May be None for
                # single-KB callers; we fall back to kb_name in that case.
                paper_kb[pid] = getattr(r, "kb_name", None) if not isinstance(r, dict) else r.get("kb_name")

        if not paper_ids:
            return results

        if len(effective_collections) == 1:
            all_chunks = await vector_store.get_chunks_by_paper_ids(
                effective_collections[0], paper_ids
            )
        else:
            all_chunks = await get_chunks_by_paper_ids_across(
                vector_store,
                collection_names=effective_collections,
                paper_ids=paper_ids,
            )
        deduped = deduplicate_chunk_overlaps(all_chunks)

        # Group by paper_id
        from collections import OrderedDict
        grouped: OrderedDict[str, list] = OrderedDict()
        for d in deduped:
            grouped.setdefault(d["paper_id"], []).append(d)

        # Return paper-level dicts
        paper_results = []
        for pid in paper_ids:
            chunks_list = grouped.get(pid, [])
            full_text = " ".join(c["text"] for c in chunks_list)
            # Get metadata from first chunk
            meta = chunks_list[0]["metadata"] if chunks_list else None
            paper_results.append({
                "paper_id": pid,
                "paper_score": paper_scores.get(pid, 0.5),
                "title": getattr(meta, "title", None) if meta else None,
                "authors": getattr(meta, "authors", None) if meta else None,
                "year": getattr(meta, "year", None) if meta else None,
                "doi": getattr(meta, "doi", None) if meta else None,
                "chunks": chunks_list,
                "full_text": full_text,
                "source": "kb",
                "kb_name": paper_kb.get(pid),
            })

        return paper_results

    def _apply_step_analysis(self, step: ResearchStep, analysis: dict[str, Any]) -> None:
        """Map profonde.py _analyze_documents JSON onto ResearchStep."""
        step.analysis = str(analysis.get("analysis", ""))
        step.success = bool(analysis.get("success")) and bool(analysis.get("purpose_fulfilled"))
        step.question_answered = bool(analysis.get("question_answered"))
        step.answer_confidence = float(analysis.get("answer_confidence", 0.0))
        step.key_findings = [str(x) for x in analysis.get("key_points", [])]
        step.missing_info = [str(x) for x in analysis.get("missing_aspects", [])]

    def _parse_web_tool_results(self, web_result: Any) -> list[dict[str, str]]:
        """Normalize rewired web_search.execute() output into v1-style {content, citation, url} dicts."""
        if web_result is None:
            return []
        if isinstance(web_result, list):
            out: list[dict[str, str]] = []
            for item in web_result:
                if not isinstance(item, dict):
                    continue
                content = (
                    item.get("content")
                    or item.get("text")
                    or item.get("snippet")
                    or item.get("body")
                    or ""
                )
                if not content:
                    continue
                out.append(
                    {
                        "content": str(content),
                        "citation": str(item.get("citation") or item.get("title") or "Web search"),
                        "url": str(item.get("url") or ""),
                        # Carry through bibliographic metadata so the
                        # source card / references list don't render as
                        # "Unknown, (n.d.)". These come from
                        # WebSearchTool's JSON output.
                        "title": str(item.get("title") or ""),
                        "authors": item.get("authors") or "",
                        "year": str(item.get("year") or ""),
                        "doi": str(item.get("doi") or ""),
                        "abstract": str(item.get("abstract") or item.get("snippet") or ""),
                        "source_provider": str(item.get("source") or ""),
                    }
                )
            return out
        if isinstance(web_result, str):
            s = web_result.strip()
            if not s:
                return []
            if s.startswith("[") or s.startswith("{"):
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return self._parse_web_tool_results(parsed)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("results"), list):
                        return self._parse_web_tool_results(parsed["results"])
                    return self._parse_web_tool_results([parsed])
            return [{"content": s, "citation": "Web search", "url": ""}]
        return []

    def _web_results_to_document_dicts(
        self, results: list[dict[str, str]], query: str
    ) -> list[dict[str, Any]]:
        """Shapes for _analyze_documents_json (full_text) and _prepare_sources.

        Carries full bibliographic metadata (title, authors, year, doi,
        abstract, source) through to ``_prepare_sources`` so the final
        citation list renders proper "Authors, Title, Year" entries
        instead of "Unknown, (n.d.)".

        Also sets ``paper_id`` (DOI when available, else SHA-256 of the
        title) so the Retrieval panel in the GUI can register these as
        proper retrieval events. Without paper_id the ``cycle_paper_docs``
        filter in execute_stream drops them and the Retrieval table
        shows up empty for any KB-less profound run.
        """
        import hashlib
        out: list[dict[str, Any]] = []
        for r in results:
            # Prefer explicit fields when present (new parser path), fall
            # back to the legacy ``citation`` field as title for older
            # tool outputs.
            title = r.get("title") or r.get("citation") or "Web search"
            authors = r.get("authors") or ""
            # authors can be a string ("A, B, C") or list — normalize to
            # list of strings for downstream SourceReference.
            if isinstance(authors, str):
                authors_list = [a.strip() for a in authors.split(",") if a.strip()]
            elif isinstance(authors, list):
                authors_list = [str(a).strip() for a in authors if str(a).strip()]
            else:
                authors_list = []
            year_raw = r.get("year") or ""
            try:
                year_val: int | None = int(str(year_raw)[:4]) if year_raw else None
            except ValueError:
                year_val = None
            _doi_val = r.get("doi") or ""
            if _doi_val:
                paper_id = f"doi:{_doi_val}"
            else:
                paper_id = (
                    "web:" + hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
                )
            out.append({
                "paper_id": paper_id,
                "source": "web_search",
                "full_text": r.get("content", ""),
                "abstract": r.get("abstract") or r.get("content", ""),
                "title": title,
                "authors": authors_list,
                "year": year_val,
                "doi": _doi_val,
                "url": r.get("url") or "",
                "source_provider": r.get("source_provider") or "",
                "paper_score": 0.5,  # neutral default — rerank updates this later
                "query": query,
            })
        return out

    async def _execute_step(
        self,
        step_info: PlanStep,
        query: str,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
        kb_name: str,
        model: str | None = None,
        collection_names: list[str] | None = None,
        telemetry: list[dict[str, Any]] | None = None,
        databases: list[str] | None = None,
    ) -> ResearchStep:
        """
        Execute a single research step with v1's 3-stage fallback:
        Stage 1: Basic RAG (single query retrieval)
        Stage 2: Advanced RAG with contextual queries (WRRF with rephrased queries)
        Stage 3: Web search (if enabled)

        Ported from: core/core.py + core/profonde.py
        """
        step = ResearchStep(
            step_purpose=step_info.purpose,
            query=step_info.query,
        )

        logger.debug(
            "profound_execute_step",
            step=step_info.step_number,
            purpose=step_info.purpose,
            query=step_info.query[:100],
        )

        # ========================================================================
        # STAGE 1: Basic RAG (single query retrieval with hybrid option)
        # ========================================================================
        logger.debug("profound_stage_1_basic_rag")
        kb_after_s1: list[Any] = []
        try:
            basic_results = await self._basic_vector_retrieve(
                step_info.query,
                vector_store,
                embedding_provider,
                kb_name,
                collection_names=collection_names,
            )

            # Two-pass enrichment: fetch full paper text (if enabled)
            if self.use_two_pass:
                enriched = await self._enrich_with_full_text(
                    basic_results[: self.final_max_docs],
                    kb_name,
                    vector_store,
                    collection_names=collection_names,
                )
            else:
                enriched = []

            docs_for_assess = enriched if enriched else basic_results
            kb_after_s1 = (
                list(docs_for_assess)
                if docs_for_assess
                else (list(basic_results) if basic_results else [])
            )
            is_sufficient, missing_aspects = await self._assess_documents_quality(
                query=step_info.query,
                documents=docs_for_assess,
                llm=llm,
                step_purpose=step_info.purpose,
            )

            if is_sufficient and docs_for_assess:
                if self.use_relevancy_optimization:
                    docs_for_assess = reorder_documents_by_relevance(
                        step_info.query, list(docs_for_assess)
                    )
                step.documents = docs_for_assess
                analysis = await self._analyze_documents_json(
                    step_info=step_info,
                    documents=step.documents,
                    original_question=query,
                    llm=llm,
                    model=model,
                )
                self._apply_step_analysis(step, analysis)
                logger.debug("profound_stage_1_success", docs=len(step.documents))
                return step

            logger.debug("profound_stage_1_insufficient", missing=missing_aspects)

        except Exception as e:
            logger.warning("profound_stage_1_error", error=str(e))
            basic_results = []
            missing_aspects = []
            kb_after_s1 = []

        latest_kb_docs: list[Any] = list(kb_after_s1)

        # ========================================================================
        # STAGE 2: Advanced RAG with contextual queries (WRRF)
        # ========================================================================
        logger.debug("profound_stage_2_advanced_rag")
        try:
            contextual_queries = await self._generate_contextual_queries(
                original_query=step_info.query,
                initial_documents=basic_results,
                missing_aspects=missing_aspects,
                llm=llm,
            )
            if not contextual_queries:
                contextual_queries = [step_info.query]

            if len(contextual_queries) == 1:
                stage2_docs = await self._basic_vector_retrieve(
                    contextual_queries[0],
                    vector_store,
                    embedding_provider,
                    kb_name,
                    collection_names=collection_names,
                )
            else:
                stage2_docs = await self._wrrf_retrieval(
                    contextual_queries,
                    vector_store,
                    embedding_provider,
                    kb_name,
                    llm=None,
                    collection_names=collection_names,
                )

            if self.use_two_pass and stage2_docs:
                enriched_wrrf = await self._enrich_with_full_text(
                    stage2_docs,
                    kb_name,
                    vector_store,
                    collection_names=collection_names,
                )
            else:
                enriched_wrrf = []

            docs_assess = enriched_wrrf if enriched_wrrf else stage2_docs
            latest_kb_docs = (
                list(docs_assess) if docs_assess else list(kb_after_s1)
            )
            is_sufficient, missing_aspects_2 = await self._assess_documents_quality(
                query=step_info.query,
                documents=docs_assess,
                llm=llm,
                step_purpose=step_info.purpose,
            )

            if is_sufficient and docs_assess:
                if self.use_relevancy_optimization:
                    docs_assess = reorder_documents_by_relevance(
                        step_info.query, list(docs_assess)
                    )
                step.documents = docs_assess
                analysis = await self._analyze_documents_json(
                    step_info=step_info,
                    documents=step.documents,
                    original_question=query,
                    llm=llm,
                    model=model,
                )
                self._apply_step_analysis(step, analysis)
                step.missing_info = missing_aspects_2
                logger.debug("profound_stage_2_success", docs=len(step.documents))
                return step

            logger.debug("profound_stage_2_insufficient", missing=missing_aspects_2)

        except Exception as e:
            logger.warning("profound_stage_2_error", error=str(e))

        # ========================================================================
        # STAGE 3: Web search (if enabled and KB results insufficient)
        # v1 profonde: append web docs, then _analyze_documents on KB + web combined.
        # ========================================================================
        if self.use_websearch:
            logger.debug("profound_stage_3_web_search")
            try:
                import hashlib as _hashlib

                from perspicacite.rag.resolve_papers import resolve_papers_pipeline
                _app_state = getattr(self, "app_state", None)
                web_papers = await resolve_papers_pipeline(
                    query=step_info.query,
                    databases=databases,
                    max_docs=5,
                    app_state=_app_state,
                    telemetry=telemetry,
                    enrich=True,
                    rerank=True,
                )

                if web_papers:
                    web_docs: list[dict[str, Any]] = []
                    for p in web_papers:
                        _doi = p.doi or ""
                        _pid = f"doi:{_doi}" if _doi else (
                            "web:" + _hashlib.sha256((p.title or "").encode()).hexdigest()[:12]
                        )
                        web_docs.append({
                            "paper_id": _pid,
                            "source": "web_search",
                            "full_text": p.abstract or "",
                            "abstract": p.abstract or "",
                            "title": p.title or "",
                            "authors": [a.name for a in (p.authors or [])],
                            "year": p.year,
                            "doi": _doi,
                            "url": p.url or "",
                            "source_provider": (p.discovery_sources[0] if p.discovery_sources else ""),
                            "paper_score": 0.5,
                            "query": step_info.query,
                        })
                    combined_docs = list(latest_kb_docs) + web_docs
                    step.documents = combined_docs
                    analysis = await self._analyze_documents_json(
                        step_info=step_info,
                        documents=combined_docs,
                        original_question=query,
                        llm=llm,
                        model=model,
                    )
                    self._apply_step_analysis(step, analysis)
                    logger.debug(
                        "profound_stage_3_success",
                        docs=len(step.documents),
                        success=step.success,
                    )
                    return step

            except Exception as e:
                logger.warning("profound_stage_3_error", error=str(e))

        # All stages failed
        step.success = False
        step.missing_info = [f"Could not find sufficient information for: {step_info.purpose}"]
        logger.debug("profound_all_stages_failed", purpose=step_info.purpose)
        return step

    async def _summarize_snippet(self, text: str, llm: Any) -> str:
        """core/core.py summarize_information for contextual query prompts."""
        if not text:
            return ""
        try:
            t = await llm.complete(
                messages=[
                    {"role": "system", "content": SUMMARIZE_INFORMATION_PROMPT},
                    {"role": "user", "content": text[:2000]},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return t.strip()
        except Exception:
            return text[:500]

    async def _generate_contextual_queries(
        self,
        original_query: str,
        initial_documents: list[Any],
        missing_aspects: list[str],
        llm: Any,
    ) -> list[str]:
        """core/core.py::generate_contextual_queries (v1)."""
        if not missing_aspects:
            return await self._generate_similar_queries(original_query, llm, self.wrrf_rephrases)

        doc_summaries: list[str] = []
        for doc in initial_documents[:3]:
            if hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                raw = doc.chunk.text
                citation = get_doc_citation(doc)
            elif isinstance(doc, dict) and doc.get("full_text"):
                raw = doc["full_text"][:2000]
                citation = doc.get("title") or doc.get("doi") or "Unknown"
            else:
                raw = str(doc)[:2000]
                citation = "Unknown"
            summarized = await self._summarize_snippet(raw, llm)
            doc_summaries.append(f"Source: {citation}\nKey points: {summarized}")

        context = {
            "original_query": original_query,
            "document_summaries": doc_summaries,
            "missing_aspects": missing_aspects,
        }

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": GENERATE_CONTEXTUAL_QUERIES_PROMPT},
                    {"role": "user", "content": f"Context: {json.dumps(context)}"},
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            if not response:
                logger.warning(
                    "profound_contextual_queries_error",
                    error="empty_response_from_llm",
                )
                return [original_query]
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json", 1)[1]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            result = json.loads(response.strip())
            queries = result.get("queries", [])
            if not isinstance(queries, list) or not queries:
                return [original_query]
            return queries[:4]
        except Exception as e:
            logger.warning("profound_contextual_queries_error", error=str(e))
            return [original_query]

    async def _assess_documents_quality(
        self,
        query: str,
        documents: list[Any],
        llm: Any,
        step_purpose: str,
    ) -> tuple[bool, list[str]]:
        """core/core.py::assess_document_quality — JSON is_sufficient + missing_aspects."""
        if not documents:
            return False, ["No documents retrieved"]

        doc_contents: list[str] = []
        for doc in documents[:5]:
            if isinstance(doc, dict) and "full_text" in doc:
                text = doc["full_text"][:1200]
                cit = doc.get("title") or doc.get("doi") or "Unknown"
            elif hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                text = doc.chunk.text[:1200]
                cit = get_doc_citation(doc)
            else:
                text = str(doc)[:1200]
                cit = "Unknown"
            doc_contents.append(f"Source: {cit}\nContent: {text}")

        context = {"query": query, "step_purpose": step_purpose, "documents": doc_contents}

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": ASSESS_DOCUMENT_QUALITY_PROMPT},
                    {"role": "user", "content": f"Context: {json.dumps(context)}"},
                ],
                temperature=0.0,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json", 1)[1]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            result = json.loads(response.strip())
            missing = result.get("missing_aspects", ["No specific aspects identified"])
            if not isinstance(missing, list):
                missing = [str(missing)]
            return bool(result.get("is_sufficient", False)), missing
        except Exception as e:
            logger.error("profound_assessment_error", error=str(e))
            return False, ["Error in quality assessment"]

    async def _analyze_documents_json(
        self,
        step_info: PlanStep,
        documents: list[Any],
        original_question: str,
        llm: Any,
        model: str | None = None,
    ) -> dict[str, Any]:
        """core/profonde.py::_analyze_documents JSON output (temperatures match v1)."""
        formatted_docs: list[str] = []
        for doc in documents:
            if isinstance(doc, dict) and "full_text" in doc:
                content = doc["full_text"][:4000]
                md = doc
                citation = md.get("title") or md.get("doi") or "Unknown"
            elif hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                content = doc.chunk.text[:4000]
                citation = get_doc_citation(doc)
            else:
                content = str(doc)[:4000]
                citation = "Unknown"
            formatted_docs.append(f"[Citation: {citation}]\n{content}")

        doc_content = "\n\n---\n\n".join(formatted_docs)
        system_prompt = PROFOUND_ANALYZE_DOCUMENTS_PROMPT_TEMPLATE.format(
            step_purpose=step_info.purpose,
            original_question=original_question,
        )

        m = model or ""
        is_o_series = m.startswith("o") or "gpt-5" in m
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {step_info.query}\n\nDocuments:\n{doc_content}"},
        ]

        try:
            if is_o_series:
                response = await llm.complete(messages=messages, max_tokens=1200)
            elif self.use_relevancy_optimization:
                qc = assess_query_complexity(step_info.query)
                response = await llm.complete(
                    messages=messages,
                    max_tokens=1200,
                    temperature=(0.3 if qc < 0.7 else 0.5),
                    response_format={"type": "json_object"},
                )
            else:
                response = await llm.complete(
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json", 1)[1]
            elif response.startswith("```"):
                response = response.split("```", 1)[1]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            response = response.strip()
            if response.startswith("{") and "}" in response:
                response = response[: response.rindex("}") + 1]
            from perspicacite.rag.utils.json_salvage import (
                clean_control_chars,
                salvage_truncated_array,
            )
            cleaned = clean_control_chars(response)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as _de:
                salvaged = salvage_truncated_array(cleaned, "analyses")
                if salvaged is not None:
                    logger.info("profound_analyze_json_salvaged", recovered=len(salvaged))
                    return {"analyses": salvaged}
                raise
        except Exception as e:
            logger.error("profound_analyze_error", error=str(e))
            return {
                "analysis": f"Error analyzing documents: {e}",
                "success": False,
                "key_points": [],
                "missing_aspects": [],
                "purpose_fulfilled": False,
                "question_answered": False,
                "answer_confidence": 0.0,
            }

    async def _is_question_answered(
        self,
        cycle_steps: list[ResearchStep],
        original_query: str,
        llm: Any,
    ) -> tuple[bool, float]:
        """core/profonde.py::_is_question_answered when step-level confidence is low."""
        if not cycle_steps:
            return False, 0.0

        # v1: Step purpose / Success / Analysis (ResearchStep.purpose -> step_purpose here)
        steps_summary = "\n\n".join(
            [
                f"Step purpose: {s.step_purpose}\nSuccess: {s.success}\nAnalysis: {s.analysis}"
                for s in cycle_steps
            ]
        )

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": PROFOUND_IS_QUESTION_ANSWERED_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Original Question: {original_query}\n\n"
                            f"Research completed so far:\n{steps_summary}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json", 1)[1]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            result = json.loads(response.strip())
            return (
                bool(result.get("question_answered", False)),
                float(result.get("confidence", 0.0)),
            )
        except Exception as e:
            logger.error("profound_is_answered_error", error=str(e))
            return False, 0.0

    def _format_research_context(self, query: str, steps: list[ResearchStep]) -> str:
        """Format the per-step research summary for the draft prompt.

        Previously each step's analysis was hard-capped at 300 chars and
        only the first 3 key findings were included — about 400 chars per
        step. With 3-4 steps that's ~1.5 KB of context, leaving the draft
        LLM with almost nothing to write from. The format pass downstream
        could only re-shuffle that minimal text, producing a "few lines"
        final report. We now include the full analysis (up to 2000 chars)
        and all key findings, so the LLM has material to synthesize a
        substantive deep-research report.
        """
        research_summary = []
        for step in steps:
            findings_block = (
                "\n".join(f"  - {f}" for f in step.key_findings)
                if step.key_findings else "  (no findings recorded)"
            )
            analysis = (step.analysis or "").strip()
            if len(analysis) > 2000:
                analysis = analysis[:2000] + "…"
            research_summary.append(
                f"## Step: {step.step_purpose}\n"
                f"Query: {step.query}\n"
                f"Success: {step.success}\n"
                f"Key Findings:\n{findings_block}\n"
                f"Analysis:\n{analysis}"
            )
        research_text = "\n\n---\n\n".join(research_summary)
        for s in self._iteration_summaries:
            research_text += (
                f"\n\n### Iteration summary\n"
                f"Findings: {s.get('findings', '')}\n"
                f"Missing: {s.get('missing', [])}\n"
            )
        return research_text

    async def _profound_final_draft_answer(
        self,
        query: str,
        research_text: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
        completion_reason: str | None,
    ) -> str:
        """core/profonde.py::_generate_final_answer first LLM stage (before refine/format)."""
        limitations = completion_reason in (
            "unanswerable",
            "false_premise",
            "partially_answerable",
            "limitations_detected",
        )
        if limitations:
            cr_display = (
                (completion_reason or "").replace("_", " ")
                if completion_reason != "limitations_detected"
                else "limited by available information"
            )
            system_prompt = PROFOUND_UNANSWERABLE_QUESTION_PROMPT_TEMPLATE.format(
                completion_reason=cr_display
            )
        elif self.use_relevancy_optimization:
            system_prompt = PROFOUND_FINAL_ANSWER_IMPROVED_PROMPT
        else:
            system_prompt = PROFOUND_FINAL_ANSWER_ORIGINAL_PROMPT

        # Include the actual source documents (title + abstract/full_text)
        # in the prompt. Without this, the draft LLM only sees per-step
        # `step.analysis[:2000]` summaries — typically ~1.5 KB of meta-text
        # with no primary content, producing a "few lines" final report
        # even with max_tokens=2000. The format pass downstream can only
        # re-shape whatever the draft produces, so source text MUST land
        # in the draft prompt.
        #
        # Cap each doc at 1500 chars and the whole block at ~12 KB so we
        # leave headroom under the 16-32k typical context window.
        sources_block_lines: list[str] = []
        _docs_total_chars = 0
        _DOCS_BUDGET = 12000
        _PER_DOC = 1500
        for i, d in enumerate(documents or [], 1):
            if isinstance(d, dict):
                title = (d.get("title") or d.get("citation") or "Untitled").strip()
                body = (
                    d.get("full_text")
                    or d.get("abstract")
                    or d.get("chunk_text")
                    or ""
                )
            elif hasattr(d, "chunk") and hasattr(d.chunk, "text"):
                meta = getattr(d.chunk, "metadata", None)
                title = (getattr(meta, "title", None) or "Untitled").strip()
                body = d.chunk.text or ""
            else:
                continue
            body = str(body).strip()
            if not body:
                continue
            if len(body) > _PER_DOC:
                body = body[:_PER_DOC] + "…"
            entry = f"[{i}] {title}\n{body}"
            if _docs_total_chars + len(entry) > _DOCS_BUDGET:
                break
            sources_block_lines.append(entry)
            _docs_total_chars += len(entry)

        sources_block = (
            "\n\n".join(sources_block_lines)
            if sources_block_lines else "(no documents retrieved)"
        )

        # Detect claim-verification queries so the VERDICT instruction gets
        # reinforced in the user turn (the system prompt already asks for it,
        # but an explicit user-turn reminder prevents the "write a report"
        # framing from overriding the VERDICT format).
        _is_claim_task = (
            "claim verification" in query.lower()
            or query.strip().startswith("VERDICT")
            or "state your verdict" in query.lower()
        )
        _claim_reminder = (
            "\n\nIMPORTANT: This is a claim verification task. "
            "Your VERY FIRST LINE must be exactly one of:\n"
            "  VERDICT: SUPPORTED\n  VERDICT: REFUTED\n  VERDICT: INSUFFICIENT EVIDENCE\n"
            "Then explain your reasoning in 2–4 sentences.\n\n"
            "CRITICAL RULES:\n"
            "• VERDICT: REFUTED — ONLY if at least one retrieved source explicitly states "
            "the OPPOSITE of the claim (e.g., paper says 'X does NOT occur' when claim says "
            "'X occurs'). A paper being silent, off-topic, or failing to mention the claim is "
            "NOT refutation.\n"
            "• VERDICT: INSUFFICIENT EVIDENCE — if retrieved papers do not directly address "
            "the claim (they may discuss related topics but not confirm or deny the specific "
            "claim). 'Not directly supported' = INSUFFICIENT EVIDENCE, NOT REFUTED.\n"
            "• VERDICT: SUPPORTED — if retrieved sources contain statements that support the claim.\n"
            "COMMON ERROR TO AVOID: Do NOT choose REFUTED because 'the evidence is absent' "
            "or 'no paper directly supports the claim.' Absence of supporting evidence means "
            "INSUFFICIENT EVIDENCE."
            if _is_claim_task else ""
        )

        user_content = f"""Original question: {query}

Research conducted ({self.iterations} cycles):
{research_text}

Source documents (full abstracts / extracted text):
{sources_block}

Generate a final answer. Synthesize across the source documents above —
quote sparingly, cite by author/year inline (per the formatting rules),
and produce a substantive multi-paragraph report. Cover background,
key findings, mechanism / methodology, and open questions when the
material supports it.{_claim_reminder}"""

        if limitations:
            user_content = f"""Original Question: {query}

Research History:
{research_text}

Follow the system instructions for this situation."""

        model = getattr(request, "model", "") or ""
        is_o_series = model.startswith("o") or "gpt-5" in model
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        messages = wrap_messages_for_chunks(
            base_messages=base_messages,
            chunks=documents,
            model=request.model,
            config=self.config,
        )

        try:
            if limitations:
                if is_o_series:
                    return await llm.complete(
                        messages=messages,
                        model=request.model,
                        provider=request.provider,
                        max_tokens=2000,
                    ) or ""
                return await llm.complete(
                    messages=messages,
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                    temperature=0.3,
                ) or ""
            if is_o_series:
                return await llm.complete(
                    messages=messages,
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                ) or ""
            if self.use_relevancy_optimization:
                qc = assess_query_complexity(query)
                return await llm.complete(
                    messages=messages,
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                    temperature=(0.3 if qc < 0.7 else 0.5),
                    stage="profound.draft",
                ) or ""
            return await llm.complete(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
                stage="profound.draft",
            ) or ""
        except Exception as e:
            logger.error("profound_final_answer_error", error=str(e))
            return f"Error generating response: {e}"

    async def _finalize_response(
        self,
        query: str,
        steps: list[ResearchStep],
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
        exited_early: bool,
        completion_reason: str | None = None,
        cancelled_tid: str | None = None,
    ) -> RAGResponse:
        """Generate final response based on all research using v1 prompts."""

        research_text = self._format_research_context(query, steps)
        answer = await self._profound_final_draft_answer(
            query, research_text, documents, llm, request, completion_reason
        )

        # v1 refine_response on draft (not for unanswerable / false_premise)
        if (
            self.use_refinement
            and completion_reason not in ("unanswerable", "false_premise")
            and not str(answer).startswith("Error generating response:")
        ):
            from perspicacite.rag.modes.advanced import AdvancedRAGMode

            adv = AdvancedRAGMode(self.config)
            em = self.evaluator_model or getattr(request, "evaluator_model", None)
            ep = self.evaluator_provider or getattr(request, "evaluator_provider", None)
            answer = await adv._refine_response(
                response=answer,
                query=query,
                documents=documents,
                llm=llm,
                request=request,
                max_iterations=self.refinement_iterations,
                eval_model=em,
                eval_provider=ep,
            )

        # Stage 2: Format the answer using v1 format prompt
        model = getattr(request, "model", "") or ""
        is_o_series = model.startswith("o") or "gpt-5" in model
        try:
            fmt_kw: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": PROFOUND_FORMAT_ANSWER_PROMPT},
                    {"role": "user", "content": f"Format this research answer:\n\n{answer}"},
                ],
                "model": request.model,
                "provider": request.provider,
                # Format pass needs room to keep the entire substantive
                # draft + inline citation footnotes. 2500 capped the
                # output mid-section; 4500 gives ~3.5 KB of headroom over
                # the typical draft size so nothing is truncated.
                "max_tokens": 4500,
                "stage": "profound.answer",
            }
            if not is_o_series:
                fmt_kw["temperature"] = 0.2
            formatted_answer = await llm.complete(**fmt_kw)
            answer = formatted_answer
        except Exception as e:
            logger.warning("profound_format_error", error=str(e))
            # Continue with unformatted answer

        # Prepare sources
        sources = self._prepare_sources(documents)

        # Append references section to answer (if not already included by formatter)
        if sources and "### ✨ Perspicacite Profonde findings" not in answer:
            references = self._format_references(sources)
            answer = answer.strip() + "\n\n" + references

        # Sub-project C (2026-05-15): attach code excerpts + figure refs.
        _mm = getattr(self.config, "multimodal", None)
        _show_code = bool(getattr(_mm, "show_code", False)) if _mm else False
        _mode = getattr(_mm, "mode", None) if _mm else None
        _dc_chunks = flatten_paper_results_to_chunks(documents)
        _code_excerpts = collect_code_excerpts(_dc_chunks) if _show_code else []
        _figure_refs = (
            collect_figure_refs(_dc_chunks, capsule_root=Path(self.config.capsule.root))
            if _mode is not None and _mode != MultimodalMode.OFF
            else []
        )

        _resp_metadata: dict = {}
        if cancelled_tid:
            _resp_metadata["cancelled"] = True
            _resp_metadata["task_id"] = cancelled_tid

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.PROFOUND,
            iterations=self.iterations,
            web_search_used=any(
                isinstance(d, dict) and d.get("source") == "web_search" for d in documents
            ),
            code_excerpts=_code_excerpts,
            figures=_figure_refs,
            metadata=_resp_metadata,
        )

    async def _stream_final_response(
        self,
        query: str,
        steps: list[ResearchStep],
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
        exited_early: bool,
        completion_reason: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream format pass after v1-aligned draft (+ optional refine), matching non-UI v1 pipeline."""

        research_text = self._format_research_context(query, steps)
        # Filter to relevant documents before citing. Without this, every
        # paper the cycles touched was emitted as a citation — many
        # tangentially related at best. We rerank by MiniLM (same model
        # used elsewhere) and keep the top 10. Falls back to all docs if
        # the rerank fails (offline / model load issue).
        filtered_docs = await self._filter_documents_by_relevance(
            documents=documents,
            query=query,
            request=request,
            llm=llm,
            max_keep=10,
        )
        sources = self._prepare_sources(filtered_docs)
        for source in sources:
            yield StreamEvent.source(source)

        draft = await self._profound_final_draft_answer(
            query, research_text, documents, llm, request, completion_reason
        )

        if (
            self.use_refinement
            and completion_reason not in ("unanswerable", "false_premise")
            and not str(draft).startswith("Error generating response:")
        ):
            from perspicacite.rag.modes.advanced import AdvancedRAGMode

            adv = AdvancedRAGMode(self.config)
            em = self.evaluator_model or getattr(request, "evaluator_model", None)
            ep = self.evaluator_provider or getattr(request, "evaluator_provider", None)
            draft = await adv._refine_response(
                response=draft,
                query=query,
                documents=documents,
                llm=llm,
                request=request,
                max_iterations=self.refinement_iterations,
                eval_model=em,
                eval_provider=ep,
            )

        model = getattr(request, "model", "") or ""
        is_o_series = model.startswith("o") or "gpt-5" in model
        fmt_kw: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": PROFOUND_FORMAT_ANSWER_PROMPT},
                {"role": "user", "content": f"Format this research answer:\n\n{draft}"},
            ],
            "model": request.model,
            "provider": request.provider,
            "max_tokens": 2500,
            "stage": "profound.answer",
        }
        if not is_o_series:
            fmt_kw["temperature"] = 0.2

        try:
            async for chunk in llm.stream(**fmt_kw):
                yield StreamEvent.content(chunk)
        except Exception as e:
            logger.error("profound_streaming_error", error=str(e))
            if is_o_series:
                formatted = await llm.complete(
                    messages=fmt_kw["messages"],
                    model=request.model,
                    provider=request.provider,
                    max_tokens=4500,
                )
            else:
                formatted = await llm.complete(
                    messages=fmt_kw["messages"],
                    model=request.model,
                    provider=request.provider,
                    max_tokens=4500,
                    temperature=0.2,
                )
            yield StreamEvent.content(formatted)

        if sources:
            references = self._format_references(sources)
            yield StreamEvent.content("\n\n" + references)

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="deep_research",
            iterations=self.iterations,
        )

    async def _filter_documents_by_relevance(
        self,
        documents: list[Any],
        query: str,
        request: RAGRequest | None = None,
        llm: Any = None,
        max_keep: int = 10,
    ) -> list[Any]:
        """Drop low-relevance docs before citation emission.

        Profound accumulates documents across cycles + KB + web search;
        without filtering, every doc that was *retrieved* becomes a cited
        source even when only a few actually contributed to the answer.

        Strategy:
          1. Extract (title + abstract or content snippet) per doc.
          2. Score with the shared MiniLM cross-encoder against the user query.
          3. Keep top-N by score where score > 0.0 (cross-encoder positive).
          4. Always preserve KB-sourced docs (they're pre-trusted).
          5. Fall back to ALL docs on any rerank failure.
        """
        if not documents or len(documents) <= max_keep:
            return documents

        # Configurable screening knobs (RAGRequest.screen_method /
        # screen_threshold). Default behavior is rerank @ 0.0 (legacy).
        screen_method = getattr(request, "screen_method", None) if request else None
        screen_threshold = (
            getattr(request, "screen_threshold", None) if request else None
        )
        effective_threshold = (
            screen_threshold if screen_threshold is not None else 0.0
        )

        def _extract_text(d: Any) -> str:
            if hasattr(d, "chunk") and hasattr(d.chunk, "text"):
                title = getattr(d.chunk.metadata, "title", "") or ""
                return f"{title} {d.chunk.text[:600]}".strip()
            if isinstance(d, dict):
                title = d.get("title") or ""
                body = (
                    d.get("abstract")
                    or d.get("content")
                    or d.get("full_text")
                    or ""
                )
                return f"{title} {str(body)[:600]}".strip()
            return str(d)[:600]

        # Always keep KB-sourced docs unconditionally.
        kb_docs: list[Any] = []
        rest: list[Any] = []
        for d in documents:
            is_kb = (
                isinstance(d, dict)
                and d.get("source")
                and d.get("source") != "web_search"
                and not str(d.get("source", "")).startswith("PaperSource.")
            ) or hasattr(d, "chunk")  # KB chunks
            if is_kb:
                kb_docs.append(d)
            else:
                rest.append(d)

        # If KB already saturates max_keep, no filtering needed.
        if len(kb_docs) >= max_keep:
            return kb_docs[:max_keep]

        # Screen the non-KB tail with the configured method, falling back to
        # rerank for unknown methods (and for "llm" when no llm is in scope).
        slots = max_keep - len(kb_docs)
        try:
            # screen_papers (bm25) reads title/abstract directly; the
            # rerank/llm paths use the pre-extracted text field.
            items = [
                {"_doc": d, "text": _extract_text(d), "title": _extract_text(d)}
                for d in rest
            ]
            if screen_method == "bm25":
                results = screen_papers(
                    items,
                    reference=query,
                    method="bm25",
                    threshold=effective_threshold,
                )
                used_method = "bm25"
            elif screen_method == "llm" and llm is not None:
                results = await screen_papers_llm(
                    items,
                    query=query,
                    llm=llm,
                    threshold=effective_threshold,
                )
                used_method = "llm"
            else:
                # None / "rerank" / unknown / "llm" without llm -> rerank.
                results = await screen_papers_rerank(
                    items, query=query, threshold=effective_threshold,
                )
                used_method = "rerank"
            # results are ScreenResult with .item and .score
            scored = [(r.score, r.item) for r in results]
            scored.sort(key=lambda kv: kv[0], reverse=True)
            kept_rest = [item["_doc"] for _, item in scored[:slots]]
            logger.info(
                "profound_citation_rerank",
                in_=len(rest), kept=len(kept_rest),
                kb_kept=len(kb_docs), max_keep=max_keep,
                screen_method=used_method, threshold=effective_threshold,
            )
            return kb_docs + kept_rest
        except Exception as e:
            logger.warning("profound_citation_rerank_failed", error=str(e))
            return documents

    def _prepare_sources(self, documents: list[Any]) -> list[SourceReference]:
        """Prepare source references from documents with web search handling."""
        seen = set()
        sources = []

        for doc in documents:
            # Handle paper-level dicts (from two-pass enrichment)
            if isinstance(doc, dict) and "paper_id" in doc:
                title = doc.get("title") or "Untitled"
                if title in seen:
                    continue
                seen.add(title)
                sources.append(
                    SourceReference(
                        title=title,
                        authors=doc.get("authors"),  # type: ignore[arg-type]  # _coerce_authors validator widens authors at runtime
                        year=doc.get("year"),
                        doi=doc.get("doi"),
                        relevance_score=doc.get("paper_score", 0.5),
                        kb_name=doc.get("kb_name"),
                        paper_id=doc.get("paper_id"),
                        metadata=doc.get("paper_metadata"),
                    )
                )
                continue

            # Web search chunks (v1 citation / v2 tool output).
            # Carry through bibliographic metadata so references render
            # as "Authors. Title (Year). DOI." instead of "Unknown, (n.d.)".
            if isinstance(doc, dict) and doc.get("source") == "web_search":
                title = (
                    doc.get("title")
                    or doc.get("citation")
                    or f"Web search: {doc.get('query', 'Unknown')}"
                )
                if title in seen:
                    continue
                seen.add(title)
                authors_raw = doc.get("authors")
                if isinstance(authors_raw, list):
                    authors_list = [str(a) for a in authors_raw if a]
                elif isinstance(authors_raw, str) and authors_raw:
                    authors_list = [a.strip() for a in authors_raw.split(",") if a.strip()]
                else:
                    authors_list = []
                year_val = doc.get("year")
                if isinstance(year_val, str) and year_val.isdigit():
                    year_val = int(year_val[:4])
                elif not isinstance(year_val, int):
                    year_val = None
                sources.append(
                    SourceReference(
                        title=str(title),
                        authors=authors_list,
                        year=year_val,
                        doi=doc.get("doi") or None,
                        url=doc.get("url") or None,
                        source=doc.get("source_provider") or None,
                        relevance_score=0.5,
                        metadata=None,
                    )
                )
                continue

            # Handle vector store results
            if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
                meta = doc.chunk.metadata
                title = getattr(meta, "title", "Untitled")
                authors = getattr(meta, "authors", [])
                year = getattr(meta, "year", None)
                doi = getattr(meta, "doi", None)
                # Decode the ASB ``paper_metadata_json`` blob if present.
                _pm_dict = decode_paper_metadata_json(meta)
            else:
                continue

            # Deduplicate
            if title in seen:
                continue
            seen.add(title)

            # Format authors
            authors_str = None
            if authors:
                if isinstance(authors, list):
                    authors_str = ", ".join(str(a) for a in authors[:3])
                    if len(authors) > 3:
                        authors_str += " et al."
                else:
                    authors_str = str(authors)

            sources.append(
                SourceReference(
                    title=title,
                    authors=authors_str,  # type: ignore[arg-type]  # _coerce_authors validator widens authors at runtime
                    year=year,
                    doi=doi,
                    relevance_score=getattr(doc, "score", 0.0),
                    kb_name=getattr(doc, "kb_name", None),
                    paper_id=getattr(meta, "paper_id", None),
                    metadata=_pm_dict,
                )
            )

        return sources

    def _format_references(self, sources: list[SourceReference]) -> str:
        """Format sources as a references section using shared utility."""
        return format_references(sources)


# Backward-compatible alias — code that imports ProfoundRAGMode still works.
ProfoundRAGMode = DeepResearchRAGMode
