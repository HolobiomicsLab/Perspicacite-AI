"""Agentic RAG - True agent-based research with self-reflection and tool use.

This implementation ports and enhances the v1 Profound mode capabilities:
- Document quality assessment
- Early exit based on confidence
- Dynamic plan adjustment
- Web search fallback
- Tool selection and execution
- Self-evaluation and refinement
- Hybrid retrieval support
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.rag.tools import Tool, ToolRegistry
from perspicacite.retrieval.hybrid import hybrid_retrieval

logger = get_logger("perspicacite.rag.modes.agentic")


@dataclass
class AgentStep:
    """A single step in the agent's research process."""

    query: str
    purpose: str
    tool_used: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""
    documents: list[Any] = field(default_factory=list)
    analysis: str = ""
    confidence: float = 0.0
    success: bool = False
    key_points: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)


@dataclass
class ResearchIteration:
    """One complete iteration of research."""

    iteration_num: int
    plan: list[str]
    steps: list[AgentStep]
    summary: str = ""
    missing_info: list[str] = field(default_factory=list)
    should_continue: bool = True
    question_answered: bool = False


class DocumentQualityAssessor:
    """Assess if retrieved documents are sufficient to answer a query."""

    def __init__(self, llm: Any):
        self.llm = llm

    async def assess(
        self,
        query: str,
        documents: list[Any],
        step_purpose: str = "",
    ) -> tuple[bool, list[str], float]:
        """
        Assess document quality and sufficiency.

        Returns:
            Tuple of (is_sufficient, missing_aspects, confidence_score)
        """
        if not documents:
            return False, ["No documents retrieved"], 0.0

        # Format documents for assessment
        doc_texts = []
        for i, doc in enumerate(documents[:5]):  # Limit to top 5
            if hasattr(doc, "chunk"):
                text = doc.chunk.text[:500] if hasattr(doc.chunk, "text") else str(doc.chunk)[:500]
            else:
                text = str(doc)[:500]
            doc_texts.append(f"Document {i + 1}:\n{text}")

        doc_content = "\n\n---\n\n".join(doc_texts)

        system_prompt = f"""You are a research quality assessor. Evaluate if the provided documents are sufficient to answer the query.

Purpose: {step_purpose or "Answer the research question"}

Respond in JSON format:
{{
    "is_sufficient": true/false,
    "missing_aspects": ["aspect1", "aspect2"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Guidelines:
- is_sufficient: Do documents contain enough relevant information?
- missing_aspects: What key information is still needed?
- confidence: How confident are you in this assessment?"""

        try:
            response = await self.llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{doc_content}"},
                ],
                temperature=0.0,
                max_tokens=300,
            )

            # Parse JSON response
            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            return (
                result.get("is_sufficient", False),
                result.get("missing_aspects", []),
                result.get("confidence", 0.5),
            )

        except Exception as e:
            logger.error("quality_assessment_error", error=str(e))
            # Conservative default - assume insufficient
            return False, ["Assessment error"], 0.0


class AgenticRAGMode(BaseRAGMode):
    """
    Agentic RAG - True agent-based research.

    Key capabilities ported from v1:
    1. Document quality assessment
    2. Early exit when question answered
    3. Dynamic plan adjustment
    4. Web search fallback
    5. Tool selection and execution
    6. Self-evaluation

    Unlike standard RAG modes, this acts as an autonomous agent that:
    - Decides which tools to use
    - Evaluates if information is sufficient
    - Adjusts strategy based on findings
    - Knows when to stop (early exit)
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.quality_assessor: DocumentQualityAssessor | None = None
        self.early_exit_confidence = getattr(
            getattr(config.rag_modes, "agentic", None), "early_exit_confidence", 0.85
        )
        self.max_consecutive_failures = 2

        # Hybrid retrieval settings
        rag_settings = getattr(config.rag_modes, "agentic", None)
        if rag_settings is None:
            rag_settings = {}
        elif hasattr(rag_settings, "model_dump"):
            rag_settings = rag_settings.model_dump()
        elif hasattr(rag_settings, "dict"):
            rag_settings = rag_settings.dict()

        self.use_hybrid = rag_settings.get("use_hybrid", True)
        self.initial_docs = 30
        self.final_max_docs = 5

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: ToolRegistry,
    ) -> RAGResponse:
        """
        Execute agentic RAG with full tool use and self-reflection.
        """
        self.quality_assessor = DocumentQualityAssessor(llm)

        max_iterations = request.max_iterations or getattr(
            getattr(self.config.rag_modes, "agentic", None), "max_iterations", 3
        )

        logger.info("agentic_rag_start", query=request.query, max_iterations=max_iterations)

        iterations: list[ResearchIteration] = []
        all_documents: list[Any] = []

        # Main research loop
        for cycle in range(max_iterations):
            logger.info("research_cycle_start", cycle=cycle + 1)

            # Create research plan (possibly incorporating previous findings)
            prev_findings = [
                {"summary": it.summary, "missing": it.missing_info} for it in iterations
            ]

            plan_result = await self._create_research_plan(
                request.query, llm, request, prev_findings
            )

            # Initialize iteration
            iteration = ResearchIteration(
                iteration_num=cycle + 1,
                plan=plan_result["plan"],
                steps=[],
            )

            # Execute each step in the plan
            for step_idx, (step_desc, step_query) in enumerate(
                zip(plan_result["plan"], plan_result["queries"])
            ):
                step = await self._execute_step(
                    query=step_query,
                    purpose=step_desc,
                    original_query=request.query,
                    llm=llm,
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    tools=tools,
                    request=request,
                )

                iteration.steps.append(step)
                all_documents.extend(step.documents)

                # Check for early exit after each step
                is_answered, confidence = await self._is_question_answered(
                    iteration.steps, request.query, llm
                )

                if is_answered and confidence >= self.early_exit_confidence:
                    logger.info("early_exit_triggered", confidence=confidence)
                    iteration.question_answered = True
                    iteration.should_continue = False
                    break

            # Create iteration summary
            summary = await self._create_iteration_summary(request.query, iteration.steps, llm)
            iteration.summary = summary.get("findings", "")
            iteration.missing_info = summary.get("missing", [])
            iteration.should_continue = summary.get("should_continue", False)

            iterations.append(iteration)

            # Decide whether to continue
            if iteration.question_answered or not iteration.should_continue:
                break

            # Review and adjust plan for next iteration
            if cycle < max_iterations - 1:
                adjusted = await self._review_and_adjust_plan(
                    request.query,
                    plan_result["plan"],
                    plan_result["queries"],
                    iteration.steps,
                    llm,
                )

                # If evaluation says to stop, break
                if adjusted.get("should_complete"):
                    break

        # Generate final answer
        answer = await self._generate_final_answer(request.query, iterations, llm, request)

        # Deduplicate and prepare sources
        sources = self._prepare_sources(all_documents)

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.AGENTIC,
            iterations=len(iterations),
            research_plan=[s for it in iterations for s in it.plan],
        )

    async def _execute_step(
        self,
        query: str,
        purpose: str,
        original_query: str,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: ToolRegistry,
        request: RAGRequest,
    ) -> AgentStep:
        """
        Execute a single research step with tool selection and quality assessment.
        """
        step = AgentStep(query=query, purpose=purpose)

        # Stage 1: Try KB search first
        logger.debug("step_kb_search", query=query, use_hybrid=self.use_hybrid)

        try:
            query_embedding = await embedding_provider.embed([query])
            kb_results = await vector_store.search(
                collection=chroma_collection_name_for_kb(request.kb_name),
                query_embedding=query_embedding[0],
                top_k=self.initial_docs,
            )

            # Apply hybrid retrieval if enabled
            if self.use_hybrid and kb_results and llm is not None:
                try:
                    logger.debug("step_applying_hybrid", query=query[:100])
                    vector_scores = [getattr(r, "score", 0.5) for r in kb_results]
                    hybrid_results = await hybrid_retrieval(
                        query=query,
                        documents=kb_results,
                        vector_scores=vector_scores,
                        use_llm_weights=True,
                        llm=llm,
                    )
                    kb_results = [doc for doc, _ in hybrid_results]
                    logger.debug("step_hybrid_applied", num_results=len(kb_results))
                except Exception as e:
                    logger.warning("step_hybrid_error", error=str(e))

            # Assess document quality
            is_sufficient, missing, confidence = await self.quality_assessor.assess(
                query, kb_results, purpose
            )

            if is_sufficient and confidence >= 0.7:
                step.documents = kb_results
                step.success = True
                step.confidence = confidence
                step.tool_used = "kb_search"

                # Analyze documents
                analysis = await self._analyze_documents(
                    query, kb_results, purpose, original_query, llm
                )
                step.analysis = analysis.get("analysis", "")
                step.key_points = analysis.get("key_points", [])
                step.missing_aspects = analysis.get("missing_aspects", [])

                return step

        except Exception as e:
            logger.warning("kb_search_error", error=str(e))

        # Stage 2: Use web search if KB insufficient and tool available
        if "web_search" in tools.list_tools():
            logger.debug("step_web_search", query=query)

            try:
                web_tool = tools.get("web_search")
                web_result = await web_tool.execute(query=query, max_results=5)

                step.tool_used = "web_search"
                step.tool_output = web_result

                # For web search, we need to fetch PDFs if available
                if "fetch_pdf" in tools.list_tools():
                    # Try to extract URLs and fetch PDFs
                    import re

                    urls = re.findall(r'https?://[^\s<>"{}|\\^`[\]]+', web_result)

                    for url in urls[:2]:  # Limit to first 2
                        try:
                            pdf_tool = tools.get("fetch_pdf")
                            pdf_content = await pdf_tool.execute(url=url)
                            step.documents.append(
                                {
                                    "source": "web_pdf",
                                    "url": url,
                                    "content": pdf_content,
                                }
                            )
                        except Exception:
                            pass

                if step.documents:
                    step.success = True
                    step.confidence = 0.6  # Lower confidence for web sources

            except Exception as e:
                logger.warning("web_search_error", error=str(e))

        return step

    async def _analyze_documents(
        self,
        query: str,
        documents: list[Any],
        step_purpose: str,
        original_question: str,
        llm: Any,
    ) -> dict:
        """Analyze documents for relevance and extract key information."""

        if not documents:
            return {
                "analysis": "No documents to analyze",
                "success": False,
                "key_points": [],
                "missing_aspects": [],
                "purpose_fulfilled": False,
                "question_answered": False,
                "answer_confidence": 0.0,
            }

        # Format documents
        doc_texts = []
        for i, doc in enumerate(documents[:5]):
            if hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                text = doc.chunk.text[:800]
            elif hasattr(doc, "content"):
                text = str(doc.content)[:800]
            else:
                text = str(doc)[:800]
            doc_texts.append(f"[Document {i + 1}]\n{text}")

        doc_content = "\n\n---\n\n".join(doc_texts)

        system_prompt = f"""Analyze the provided documents in relation to the research query.

Purpose: {step_purpose}
Original Question: {original_question}

Respond in JSON format:
{{
    "analysis": "detailed analysis with citations",
    "success": true/false,
    "key_points": ["point1", "point2"],
    "missing_aspects": ["aspect1"],
    "purpose_fulfilled": true/false,
    "question_answered": true/false,
    "answer_confidence": 0.0-1.0
}}"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{doc_content}"},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)

        except Exception as e:
            logger.error("document_analysis_error", error=str(e))
            return {
                "analysis": f"Error analyzing documents: {e}",
                "success": False,
                "key_points": [],
                "missing_aspects": ["Analysis failed"],
                "purpose_fulfilled": False,
                "question_answered": False,
                "answer_confidence": 0.0,
            }

    async def _is_question_answered(
        self,
        steps: list[AgentStep],
        original_question: str,
        llm: Any,
    ) -> tuple[bool, float]:
        """
        Evaluate if the original question is sufficiently answered.

        Returns:
            Tuple of (is_answered, confidence)
        """
        if not steps:
            return False, 0.0

        # Format step information
        steps_info = "\n\n".join(
            [
                f"Step: {step.purpose}\nSuccess: {step.success}\nAnalysis: {step.analysis[:300]}..."
                for step in steps
            ]
        )

        system_prompt = """Evaluate whether the completed research steps collectively answer the original question.

Only consider a question "answered" if we have found affirmative information that directly supports what was asked.

Respond in JSON format:
{
    "question_answered": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "remaining_gaps": ["gap1", "gap2"]
}"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Question: {original_question}\n\nResearch:\n{steps_info}",
                    },
                ],
                temperature=0.0,
                max_tokens=300,
            )

            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            return (
                result.get("question_answered", False),
                result.get("confidence", 0.0),
            )

        except Exception as e:
            logger.error("question_answered_eval_error", error=str(e))
            return False, 0.0

    async def _create_research_plan(
        self,
        question: str,
        llm: Any,
        request: RAGRequest,
        previous_findings: list[dict] | None = None,
    ) -> dict:
        """Create a research plan with specific steps and queries."""

        context = {
            "question": question,
            "previous_findings": previous_findings or [],
        }

        system_prompt = """Create a detailed research plan for answering the question.
Consider any previous findings to avoid redundancy.

Break down the research into 2-4 specific steps targeting:
1. Core concepts and definitions
2. Technical details and methodologies
3. Evidence and findings
4. Comparisons or alternatives (if applicable)

Respond in JSON format:
{
    "reasoning": "explanation of research approach",
    "plan": ["step1 description", "step2 description", ...],
    "queries": ["specific search query for step1", "specific search query for step2", ...]
}

Guidelines:
- Use technical/field-specific terminology in queries
- Make queries specific and keyword-focused
- Ensure steps build on each other logically"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {context}"},
                ],
                temperature=0.3,
                max_tokens=600,
            )

            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            # Validate structure
            if "plan" not in result or "queries" not in result:
                raise ValueError("Invalid plan structure")

            return result

        except Exception as e:
            logger.error("plan_creation_error", error=str(e))
            # Fallback plan
            return {
                "reasoning": "Error in plan creation, using basic search",
                "plan": ["Search for general information"],
                "queries": [question],
            }

    async def _create_iteration_summary(
        self,
        question: str,
        steps: list[AgentStep],
        llm: Any,
    ) -> dict:
        """Create summary of current research iteration."""

        steps_summary = "\n\n".join(
            [
                f"Step: {step.query}\nPurpose: {step.purpose}\nAnalysis: {step.analysis}"
                for step in steps
            ]
        )

        system_prompt = """Review the research steps and their findings to determine:
1. What we've learned that directly addresses the original question
2. What information is still missing
3. Whether we need another research iteration

Respond in JSON format:
{
    "findings": "summary of what we've learned with citations",
    "missing": ["missing piece1", "missing piece2"],
    "should_continue": true/false,
    "reasoning": "explanation of decision"
}

Be selective - include only high-quality, directly relevant findings."""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}\n\nSteps:\n{steps_summary}"},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)

        except Exception as e:
            logger.error("iteration_summary_error", error=str(e))
            return {
                "findings": "Error summarizing findings",
                "missing": [],
                "should_continue": False,
                "reasoning": f"Error: {e}",
            }

    async def _review_and_adjust_plan(
        self,
        original_question: str,
        current_plan: list[str],
        current_queries: list[str],
        completed_steps: list[AgentStep],
        llm: Any,
    ) -> dict:
        """Review and adjust the research plan based on results."""

        completed_info = "\n\n".join(
            [
                f"Step: {step.purpose}\n"
                f"Query: {step.query}\n"
                f"Success: {step.success}\n"
                f"Analysis: {step.analysis[:400]}"
                for step in completed_steps
            ]
        )

        system_prompt = """Evaluate the research progress and determine the best course of action.

Consider:
1. The question may be unanswerable with available information
2. The question may be based on false premises
3. The current plan is appropriate but needs refinement
4. The research plan needs significant changes

Respond in JSON format:
{
    "evaluation": "detailed evaluation of progress",
    "question_type": "answerable" | "partially_answerable" | "unanswerable" | "false_premise",
    "recommendation": "continue_plan" | "modify_plan" | "explain_limitations",
    "reasoning": "explanation",
    "should_complete": true/false
}"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Question: {original_question}\n\nCompleted:\n{completed_info}",
                    },
                ],
                temperature=0.3,
                max_tokens=400,
            )

            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)

            # If we should explain limitations, signal completion
            if result.get("recommendation") == "explain_limitations":
                result["should_complete"] = True

            return result

        except Exception as e:
            logger.error("plan_review_error", error=str(e))
            return {
                "reasoning": f"Error: {e}",
                "should_complete": False,
            }

    async def _generate_final_answer(
        self,
        question: str,
        iterations: list[ResearchIteration],
        llm: Any,
        request: RAGRequest,
    ) -> str:
        """Generate final answer based on all iterations."""

        iterations_summary = "\n\n".join(
            [
                f"Iteration {it.iteration_num}:\n"
                f"Findings: {it.summary}\n"
                f"Missing: {', '.join(it.missing_info)}"
                for it in iterations
            ]
        )

        system_prompt = """Review all research iterations and generate a comprehensive final answer.

Your answer should:
1. Directly address the original question
2. Prioritize the most relevant findings
3. Maintain scientific precision and accuracy
4. Be clear and direct
5. Acknowledge limitations rather than speculating

If the research does not provide enough information, clearly state this."""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nResearch History:\n{iterations_summary}",
                    },
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
            )
            return response

        except Exception as e:
            logger.error("final_answer_error", error=str(e))
            return f"Error generating answer: {e}"

    def _prepare_sources(self, documents: list[Any]) -> list[SourceReference]:
        """Deduplicate and prepare sources from documents."""
        seen = set()
        sources = []

        for doc in documents:
            # Extract metadata
            if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
                meta = doc.chunk.metadata
                title = getattr(meta, "title", "Untitled")
                authors = getattr(meta, "authors", [])
                year = getattr(meta, "year", None)
            elif hasattr(doc, "content") and isinstance(doc, dict):
                title = doc.get("source", "Web source")
                authors = []
                year = None
            else:
                continue

            # Deduplicate by title
            if title in seen:
                continue
            seen.add(title)

            sources.append(
                SourceReference(
                    title=title,
                    authors=authors,
                    year=year,
                    relevance_score=getattr(doc, "score", 0.0),
                )
            )

        # Sort by relevance
        sources.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        return sources[:10]  # Limit to top 10

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute agentic RAG with streaming output."""
        import json

        yield StreamEvent.status("Initializing agentic research...")

        self.quality_assessor = DocumentQualityAssessor(llm)

        max_iterations = request.max_iterations or getattr(
            getattr(self.config.rag_modes, "agentic", None), "max_iterations", 3
        )

        iterations: list[ResearchIteration] = []
        all_documents: list[Any] = []

        # Main research loop
        for cycle in range(max_iterations):
            yield StreamEvent.status(f"Agentic RAG: Research cycle {cycle + 1}/{max_iterations}...")

            # Create research plan
            prev_findings = [
                {"summary": it.summary, "missing": it.missing_info} for it in iterations
            ]

            plan_result = await self._create_research_plan(
                request.query, llm, request, prev_findings
            )

            iteration = ResearchIteration(
                iteration_num=cycle + 1,
                plan=plan_result["plan"],
                steps=[],
            )

            # Execute each step
            for step_idx, (step_desc, step_query) in enumerate(
                zip(plan_result["plan"], plan_result["queries"])
            ):
                yield StreamEvent.status(
                    f"Agentic RAG: Executing step {step_idx + 1}: {step_desc[:50]}..."
                )

                step = await self._execute_step(
                    query=step_query,
                    purpose=step_desc,
                    original_query=request.query,
                    llm=llm,
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    tools=tools,
                    request=request,
                )

                iteration.steps.append(step)
                all_documents.extend(step.documents)

                if step.success:
                    yield StreamEvent.status(
                        f"Agentic RAG: Step {step_idx + 1} complete - found {len(step.documents)} documents"
                    )

                # Check for early exit
                is_answered, confidence = await self._is_question_answered(
                    iteration.steps, request.query, llm
                )

                if is_answered and confidence >= self.early_exit_confidence:
                    yield StreamEvent.status(
                        f"Agentic RAG: Early exit triggered (confidence: {confidence:.2f})"
                    )
                    iteration.question_answered = True
                    iteration.should_continue = False
                    break

            # Create iteration summary
            summary = await self._create_iteration_summary(request.query, iteration.steps, llm)
            iteration.summary = summary.get("findings", "")
            iteration.missing_info = summary.get("missing", [])
            iterations.append(iteration)

            if not iteration.should_continue:
                break

            # Review and adjust plan
            if cycle < max_iterations - 1:
                adjusted = await self._review_and_adjust_plan(
                    request.query,
                    plan_result["plan"],
                    plan_result["queries"],
                    iteration.steps,
                    llm,
                )

                if adjusted.get("should_complete"):
                    yield StreamEvent.status("Agentic RAG: Research complete based on review")
                    break

        # Stream final answer generation
        yield StreamEvent.status("Agentic RAG: Synthesizing final answer...")

        # Prepare sources
        sources = self._prepare_sources(all_documents)
        for source in sources:
            yield StreamEvent.source(source)

        # Build context for final answer
        iterations_summary = "\n\n".join(
            [
                f"Iteration {it.iteration_num}:\n"
                f"Findings: {it.summary}\n"
                f"Missing: {', '.join(it.missing_info)}"
                for it in iterations
            ]
        )

        system_prompt = """Review all research iterations and generate a comprehensive final answer.

Your answer should:
1. Directly address the original question
2. Prioritize the most relevant findings
3. Maintain scientific precision and accuracy
4. Be clear and direct
5. Acknowledge limitations rather than speculating

If the research does not provide enough information, clearly state this."""

        # Stream the LLM response
        try:
            async for chunk in llm.stream(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Question: {request.query}\n\nResearch History:\n{iterations_summary}",
                    },
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            ):
                yield StreamEvent.content(chunk)
        except Exception as e:
            logger.error("agentic_streaming_error", error=str(e))
            # Fall back to non-streaming
            answer = await self._generate_final_answer(request.query, iterations, llm, request)
            yield StreamEvent.content(answer)

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="agentic",
            iterations=len(iterations),
        )
