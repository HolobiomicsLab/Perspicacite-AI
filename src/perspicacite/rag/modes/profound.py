"""Profound RAG Mode - Exact implementation from release package v1.

Profound RAG (ProfondeChain) adds:
- Multi-cycle research with planning
- Dynamic plan creation and review
- Web search integration
- Early exit based on confidence
- Reflection and self-evaluation
- Document quality assessment
- WRRF multi-query fusion (vector-only retrieval, matching v1 profonde)
"""

import json
import math
import re
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.prompts import (
    ASSESS_DOCUMENT_QUALITY_PROMPT,
    GENERATE_CONTEXTUAL_QUERIES_PROMPT,
    GENERATE_SIMILAR_QUERIES_PROMPT,
    PROFOUND_ADJUST_PLAN_PROMPT,
    PROFOUND_ANALYZE_DOCUMENTS_PROMPT_TEMPLATE,
    PROFOUND_CREATE_PLAN_PROMPT,
    PROFOUND_EVALUATE_PROGRESS_PROMPT,
    PROFOUND_ITERATION_SUMMARY_ORIGINAL_PROMPT,
    PROFOUND_ITERATION_SUMMARY_IMPROVED_PROMPT,
    PROFOUND_FINAL_ANSWER_ORIGINAL_PROMPT,
    PROFOUND_FINAL_ANSWER_IMPROVED_PROMPT,
    PROFOUND_FORMAT_ANSWER_PROMPT,
    PROFOUND_IS_QUESTION_ANSWERED_PROMPT,
    PROFOUND_UNANSWERABLE_QUESTION_PROMPT_TEMPLATE,
    SUMMARIZE_INFORMATION_PROMPT,
)
from perspicacite.rag.relevancy import assess_query_complexity, reorder_documents_by_relevance
from perspicacite.rag.wrrf_v1 import doc_page_content, select_wrrf_merged_documents
from perspicacite.rag.utils import (
    format_references,
    prepare_sources,
    get_doc_citation,
    format_documents_for_prompt,
    get_system_prompt,
)

logger = get_logger("perspicacite.rag.modes.profound")


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


class ProfoundRAGMode(BaseRAGMode):
    """
    Profound RAG Mode - Exact port from release package core/profonde.py

    This is the original "Profound" mode from Perspicacité v1 with:
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
        rag_settings = getattr(config.rag_modes, "profound", None)

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
        self.use_websearch = bool(rag_settings.get("use_websearch", False))
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
                for i, (step, query) in enumerate(zip(remaining_plan, remaining_queries))
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
            )
            evaluation = json.loads(self._strip_llm_json_block(eval_response))
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
                        for p, q in zip(new_plan_s, new_queries_s)
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
        logger.info("profound_rag_start", query=request.query, max_cycles=self.max_cycles)

        # Reset state
        self.iterations = 0
        self.consecutive_failures = 0
        self.research_history = []
        self._iteration_summaries = []

        all_steps: list[ResearchStep] = []
        all_documents: list[Any] = []
        completion_reason: str | None = None
        kb_name = chroma_collection_name_for_kb(request.kb_name)

        # Main research loop
        for cycle in range(self.max_cycles):
            self.iterations = cycle + 1
            logger.info("profound_cycle_start", cycle=self.iterations)

            plan = await self._create_plan(query=request.query, llm=llm)
            logger.info(
                "profound_plan_created", steps=len(plan), purposes=[s.purpose for s in plan]
            )

            cycle_steps, cycle_documents, plan_limit_reason, early_exit = (
                await self._execute_cycle_steps(
                    request=request,
                    plan=plan,
                    llm=llm,
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    tools=tools,
                    kb_name=kb_name,
                )
            )
            all_steps.extend(cycle_steps)
            all_documents.extend(cycle_documents)

            if early_exit:
                logger.info("profound_early_exit", cycle=self.iterations)
                return await self._finalize_response(
                    query=request.query,
                    steps=all_steps,
                    documents=all_documents,
                    llm=llm,
                    request=request,
                    exited_early=True,
                    completion_reason="question_answered",
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

            should_continue = bool(summary.get("should_continue", False)) and cycle < self.max_cycles - 1
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

        return await self._finalize_response(
            query=request.query,
            steps=all_steps,
            documents=all_documents,
            llm=llm,
            request=request,
            exited_early=False,
            completion_reason=completion_reason,
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
        import json

        yield StreamEvent.status("Profound RAG: Initializing deep research...")

        # Reset state
        self.iterations = 0
        self.consecutive_failures = 0
        self.research_history = []
        self._iteration_summaries = []

        all_steps: list[ResearchStep] = []
        all_documents: list[Any] = []
        completion_reason: str | None = None
        kb_name = chroma_collection_name_for_kb(request.kb_name)

        for cycle in range(self.max_cycles):
            self.iterations = cycle + 1
            yield StreamEvent.status(
                f"Profound RAG: Research cycle {self.iterations}/{self.max_cycles}..."
            )

            plan = await self._create_plan(query=request.query, llm=llm)
            yield StreamEvent.status(f"Profound RAG: Executing {len(plan)} research steps...")

            cycle_steps, cycle_documents, plan_limit_reason, early_exit = (
                await self._execute_cycle_steps(
                    request=request,
                    plan=plan,
                    llm=llm,
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    tools=tools,
                    kb_name=kb_name,
                )
            )
            all_steps.extend(cycle_steps)
            all_documents.extend(cycle_documents)

            if early_exit:
                yield StreamEvent.status("Profound RAG: Early exit — synthesizing final answer...")
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
                return

            if plan_limit_reason:
                completion_reason = plan_limit_reason
                yield StreamEvent.status(
                    f"Profound RAG: Plan review ended research ({plan_limit_reason})"
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

            summary = await self._create_iteration_summary(request.query, cycle_steps, llm)
            self._iteration_summaries.append(summary)
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

            should_continue = bool(summary.get("should_continue", False)) and cycle < self.max_cycles - 1
            if not should_continue:
                yield StreamEvent.status("Profound RAG: Research complete based on iteration summary")
                break

            cycle_successes = sum(1 for s in cycle_steps if s.success)
            if cycle_successes == 0:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    yield StreamEvent.status("Profound RAG: Max consecutive failures reached")
                    break
            else:
                self.consecutive_failures = 0

        yield StreamEvent.status("Profound RAG: Synthesizing final answer...")
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
                max_tokens=800,
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

            result = json.loads(response)
            plan_s = result.get("plan", [])
            queries_s = result.get("queries", [])
            if not plan_s or not queries_s:
                return [PlanStep(1, "Search for general information", query)]

            steps: list[PlanStep] = []
            for i, (purp, q) in enumerate(zip(plan_s, queries_s), 1):
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
        for i in range(number):
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
    ) -> list[Any]:
        """
        core/core.py retrieve_documents multi-query branch: vector only (v1 profonde
        uses advanced_mode=False so hybrid is never applied).
        """
        rankings: dict[Any, dict[int, int]] = {}
        scores_per_query: dict[int, dict[Any, float]] = {}
        documents_info: dict[Any, Any] = {}

        for q_idx, query in enumerate(queries):
            query_embedding = await embedding_provider.embed([query])
            results = await vector_store.search(
                collection=kb_name,
                query_embedding=query_embedding[0],
                top_k=self.initial_docs,
            )
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
    ) -> list[Any]:
        """core/core.py single-query basic path: vector search + per-source cap."""
        query_embedding = await embedding_provider.embed([query])
        results = await vector_store.search(
            collection=kb_name,
            query_embedding=query_embedding[0],
            top_k=self.initial_docs,
        )
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
    ) -> list[Any]:
        """Two-pass enrichment: given chunk-level results, fetch full paper text."""
        from perspicacite.rag.utils import deduplicate_chunk_overlaps

        paper_ids = []
        paper_scores: dict[str, float] = {}
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

        if not paper_ids:
            return results

        all_chunks = await vector_store.get_chunks_by_paper_ids(kb_name, paper_ids)
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
        """Shapes for _analyze_documents_json (full_text) and _prepare_sources."""
        return [
            {
                "source": "web_search",
                "full_text": r["content"],
                "title": r["citation"],
                "url": r["url"],
                "doi": "",
                "query": query,
            }
            for r in results
        ]

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
                step_info.query, vector_store, embedding_provider, kb_name
            )

            # Two-pass enrichment: fetch full paper text (if enabled)
            if self.use_two_pass:
                enriched = await self._enrich_with_full_text(basic_results[:self.final_max_docs], kb_name, vector_store)
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
                )
            else:
                stage2_docs = await self._wrrf_retrieval(
                    contextual_queries,
                    vector_store,
                    embedding_provider,
                    kb_name,
                    llm=None,
                )

            if self.use_two_pass and stage2_docs:
                enriched_wrrf = await self._enrich_with_full_text(stage2_docs, kb_name, vector_store)
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
        if self.use_websearch and "web_search" in tools.list_tools():
            logger.debug("profound_stage_3_web_search")
            try:
                web_tool = tools.get("web_search")
                web_raw = await web_tool.execute(query=step_info.query, max_results=3)
                web_results = self._parse_web_tool_results(web_raw)

                if web_results:
                    web_docs = self._web_results_to_document_dicts(
                        web_results, step_info.query
                    )
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
                max_tokens=500,
            )
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
                )
            else:
                response = await llm.complete(
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.3,
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
        research_summary = []
        for step in steps:
            research_summary.append(
                f"Step: {step.step_purpose}\n"
                f"Query: {step.query}\n"
                f"Success: {step.success}\n"
                f"Key Findings: {', '.join(step.key_findings[:3])}\n"
                f"Analysis: {step.analysis[:300]}..."
            )
        research_text = "\n\n---\n\n".join(research_summary)
        for s in self._iteration_summaries:
            research_text += (
                f"\n\nIteration summary findings: {s.get('findings', '')}\n"
                f"Missing: {s.get('missing', [])}\n"
            )
        return research_text

    async def _profound_final_draft_answer(
        self,
        query: str,
        research_text: str,
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
                completion_reason.replace("_", " ")
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

        user_content = f"""Original question: {query}

Research conducted ({self.iterations} cycles):
{research_text}

Generate a final answer."""

        if limitations:
            user_content = f"""Original Question: {query}

Research History:
{research_text}

Follow the system instructions for this situation."""

        model = getattr(request, "model", "") or ""
        is_o_series = model.startswith("o") or "gpt-5" in model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            if limitations:
                if is_o_series:
                    return await llm.complete(
                        messages=messages,
                        model=request.model,
                        provider=request.provider,
                        max_tokens=2000,
                    )
                return await llm.complete(
                    messages=messages,
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                    temperature=0.3,
                )
            if is_o_series:
                return await llm.complete(
                    messages=messages,
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                )
            if self.use_relevancy_optimization:
                qc = assess_query_complexity(query)
                return await llm.complete(
                    messages=messages,
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                    temperature=(0.3 if qc < 0.7 else 0.5),
                )
            return await llm.complete(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            )
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
    ) -> RAGResponse:
        """Generate final response based on all research using v1 prompts."""

        research_text = self._format_research_context(query, steps)
        answer = await self._profound_final_draft_answer(
            query, research_text, llm, request, completion_reason
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
                "max_tokens": 2500,
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

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.PROFOUND,
            iterations=self.iterations,
            web_search_used=any(
                isinstance(d, dict) and d.get("source") == "web_search" for d in documents
            ),
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
        sources = self._prepare_sources(documents)
        for source in sources:
            yield StreamEvent.source(source)

        draft = await self._profound_final_draft_answer(
            query, research_text, llm, request, completion_reason
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
                    max_tokens=2500,
                )
            else:
                formatted = await llm.complete(
                    messages=fmt_kw["messages"],
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2500,
                    temperature=0.2,
                )
            yield StreamEvent.content(formatted)

        if sources:
            references = self._format_references(sources)
            yield StreamEvent.content("\n\n" + references)

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="profound",
            iterations=self.iterations,
        )

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
                        authors=doc.get("authors"),
                        year=doc.get("year"),
                        doi=doc.get("doi"),
                        relevance_score=doc.get("paper_score", 0.5),
                    )
                )
                continue

            # Web search chunks (v1 citation / v2 tool output)
            if isinstance(doc, dict) and doc.get("source") == "web_search":
                title = (
                    doc.get("title")
                    or doc.get("citation")
                    or f"Web search: {doc.get('query', 'Unknown')}"
                )
                sources.append(
                    SourceReference(
                        title=str(title),
                        url=doc.get("url") or None,
                        relevance_score=0.5,
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
                    authors=authors_str,
                    year=year,
                    doi=doi,
                    relevance_score=getattr(doc, "score", 0.0),
                )
            )

        return sources

    def _format_references(self, sources: list[SourceReference]) -> str:
        """Format sources as a references section using shared utility."""
        return format_references(sources)
