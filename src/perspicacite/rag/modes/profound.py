"""Profound RAG Mode - Exact implementation from release package v1.

Profound RAG (ProfondeChain) adds:
- Multi-cycle research with planning
- Dynamic plan creation and review
- Web search integration
- Early exit based on confidence
- Reflection and self-evaluation
- Document quality assessment
"""

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

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
        rag_settings = getattr(config.rag_modes, 'profound', {})
        
        # Settings from release package
        self.max_cycles = rag_settings.get('max_iterations', 3)
        self.early_exit_confidence = 0.85
        self.max_consecutive_failures = 2
        self.use_websearch = True
        
        # Document retrieval settings
        self.initial_docs = 5
        self.final_max_docs = 2
        self.max_docs_per_source = 1
        
        # State tracking
        self.iterations = 0
        self.consecutive_failures = 0
        self.research_history: list[dict] = field(default_factory=list)

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
        logger.info("profound_rag_start", 
                   query=request.query, 
                   max_cycles=self.max_cycles)
        
        # Reset state
        self.iterations = 0
        self.consecutive_failures = 0
        self.research_history = []
        
        all_steps: list[ResearchStep] = []
        all_documents: list[Any] = []

        # Main research loop
        for cycle in range(self.max_cycles):
            self.iterations = cycle + 1
            logger.info("profound_cycle_start", cycle=self.iterations)
            
            # Step 1: Create or adjust research plan
            plan = await self._create_plan(
                query=request.query,
                llm=llm,
                history=self.research_history,
            )
            
            logger.info("profound_plan_created", 
                       steps=len(plan),
                       purposes=[s.purpose for s in plan])

            # Step 2: Execute each step in the plan
            cycle_steps = []
            cycle_documents = []
            
            for step_info in plan:
                step = await self._execute_step(
                    step_info=step_info,
                    query=request.query,
                    llm=llm,
                    vector_store=vector_store,
                    embedding_provider=embedding_provider,
                    tools=tools,
                    kb_name=request.kb_name,
                )
                
                cycle_steps.append(step)
                cycle_documents.extend(step.documents)
                all_steps.append(step)
                all_documents.extend(step.documents)

                # Check for early exit after each step
                if step.success and step.documents:
                    is_answered, confidence = await self._evaluate_if_answered(
                        steps=cycle_steps,
                        original_query=request.query,
                        llm=llm,
                    )
                    
                    if is_answered and confidence >= self.early_exit_confidence:
                        logger.info("profound_early_exit", 
                                   cycle=self.iterations,
                                   confidence=confidence)
                        return await self._finalize_response(
                            query=request.query,
                            steps=all_steps,
                            documents=all_documents,
                            llm=llm,
                            request=request,
                            exited_early=True,
                        )

            # Step 3: Review progress and decide whether to continue
            self.research_history.append({
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
            })
            
            should_continue = await self._review_progress(
                query=request.query,
                history=self.research_history,
                llm=llm,
            )
            
            if not should_continue:
                logger.info("profound_review_says_complete", cycle=self.iterations)
                break

            # Check consecutive failures
            cycle_successes = sum(1 for s in cycle_steps if s.success)
            if cycle_successes == 0:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.warning("profound_max_failures_reached")
                    break
            else:
                self.consecutive_failures = 0

        # Finalize response
        return await self._finalize_response(
            query=request.query,
            steps=all_steps,
            documents=all_documents,
            llm=llm,
            request=request,
            exited_early=False,
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
        yield StreamEvent.status("Profound RAG: Initializing deep research...")
        
        # Delegate to non-streaming for core logic
        response = await self.execute(
            request, llm, vector_store, embedding_provider, tools
        )
        
        yield StreamEvent.status("Profound RAG: Generating final answer...")
        
        # Stream the answer word by word
        words = response.answer.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield StreamEvent.content(chunk)
        
        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="profound",
            iterations=getattr(response, 'iterations', self.iterations),
        )

    async def _create_plan(
        self,
        query: str,
        llm: Any,
        history: list[dict],
    ) -> list[PlanStep]:
        """
        Create a research plan with specific steps.
        
        Ported from: core/profonde.py - planning logic
        """
        # Build context from history
        history_context = ""
        if history:
            history_context = "Previous research cycles:\n"
            for h in history:
                history_context += f"\nCycle {h['cycle']}:\n"
                for step in h['steps']:
                    history_context += f"  - {step['purpose']}: {'✓' if step['success'] else '✗'}\n"
                    if step['missing']:
                        history_context += f"    Missing: {', '.join(step['missing'])}\n"

        system_prompt = """You are a research planner. Create a step-by-step plan to answer the research question.

Consider:
1. What specific information needs to be found
2. What are the key concepts to investigate
3. What order makes sense for the research steps

Respond in JSON format:
{
    "reasoning": "explanation of approach",
    "steps": [
        {
            "purpose": "what this step aims to achieve",
            "query": "specific search query for this step",
            "expected_outcome": "what we expect to find"
        }
    ]
}

Guidelines:
- Create 2-4 steps
- Make queries specific and technical
- Each step should build on previous steps
- Focus on the most important aspects first"""

        user_message = f"""Research question: {query}

{history_context}

Create a research plan to answer this question."""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            
            # Parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            steps_data = result.get("steps", [])
            
            # Create PlanStep objects
            plan = []
            for i, step_data in enumerate(steps_data, 1):
                plan.append(PlanStep(
                    step_number=i,
                    purpose=step_data.get("purpose", f"Step {i}"),
                    query=step_data.get("query", query),
                    expected_outcome=step_data.get("expected_outcome", ""),
                ))
            
            return plan if plan else [PlanStep(1, "Search for information", query)]
            
        except Exception as e:
            logger.error("profound_plan_creation_error", error=str(e))
            # Fallback plan
            return [
                PlanStep(1, "Search for general information", query),
                PlanStep(2, "Search for specific details", f"{query} details methodology"),
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
    ) -> ResearchStep:
        """
        Execute a single research step.
        
        Ported from: core/profonde.py - step execution logic
        """
        step = ResearchStep(
            step_purpose=step_info.purpose,
            query=step_info.query,
        )
        
        logger.debug("profound_execute_step", 
                    step=step_info.step_number,
                    purpose=step_info.purpose,
                    query=step_info.query[:100])

        # Try KB search first
        try:
            query_embedding = await embedding_provider.embed([step_info.query])
            kb_results = await vector_store.search(
                collection=kb_name,
                query_embedding=query_embedding[0],
                top_k=self.initial_docs,
            )
            
            # Assess document quality
            is_sufficient, missing_aspects, confidence = await self._assess_documents(
                query=step_info.query,
                documents=kb_results,
                purpose=step_info.purpose,
                llm=llm,
            )
            
            if is_sufficient and kb_results:
                step.documents = kb_results[:self.final_max_docs]
                step.success = True
                step.key_findings = [f"Found {len(kb_results)} relevant documents"]
                step.missing_info = missing_aspects
                
                # Analyze documents
                analysis = await self._analyze_step_documents(
                    step_info=step_info,
                    documents=kb_results,
                    llm=llm,
                )
                step.analysis = analysis
                
                logger.debug("profound_step_kb_success", 
                            docs=len(step.documents),
                            confidence=confidence)
                return step
                
        except Exception as e:
            logger.warning("profound_kb_search_error", error=str(e))

        # Try web search if KB insufficient and enabled
        if self.use_websearch and "web_search" in tools.list_tools():
            try:
                web_tool = tools.get("web_search")
                web_result = await web_tool.execute(
                    query=step_info.query, 
                    max_results=3
                )
                
                # Create document from web result
                if web_result:
                    step.documents.append({
                        "source": "web_search",
                        "content": web_result,
                        "query": step_info.query,
                    })
                    step.success = True
                    step.key_findings = ["Information from web search"]
                    step.analysis = web_result[:500]
                    
                    logger.debug("profound_step_web_success")
                    return step
                    
            except Exception as e:
                logger.warning("profound_web_search_error", error=str(e))

        # Step failed to find sufficient information
        step.success = False
        step.missing_info = [f"Could not find sufficient information for: {step_info.purpose}"]
        
        logger.debug("profound_step_failed", purpose=step_info.purpose)
        return step

    async def _assess_documents(
        self,
        query: str,
        documents: list[Any],
        purpose: str,
        llm: Any,
    ) -> tuple[bool, list[str], float]:
        """
        Assess if documents are sufficient for the query.
        
        Ported from: core/core.py::assess_document_quality()
        """
        if not documents:
            return False, ["No documents retrieved"], 0.0
        
        # Format documents for assessment
        doc_texts = []
        for i, doc in enumerate(documents[:3]):
            if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'text'):
                text = doc.chunk.text[:400]
            else:
                text = str(doc)[:400]
            doc_texts.append(f"Doc {i+1}: {text}")
        
        doc_content = "\n".join(doc_texts)
        
        system_prompt = f"""Assess if the documents adequately address the query.

Purpose: {purpose}

Respond in JSON format:
{{
    "is_sufficient": true/false,
    "missing_aspects": ["aspect1", "aspect2"],
    "confidence": 0.0-1.0,
    "analysis": "brief analysis"
}}"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nDocuments:\n{doc_content}"},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
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
            logger.error("profound_assessment_error", error=str(e))
            return False, ["Assessment error"], 0.0

    async def _analyze_step_documents(
        self,
        step_info: PlanStep,
        documents: list[Any],
        llm: Any,
    ) -> str:
        """Analyze documents for a research step."""
        if not documents:
            return "No documents to analyze"
        
        doc_texts = []
        for i, doc in enumerate(documents[:2]):
            if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'text'):
                text = doc.chunk.text[:500]
            else:
                text = str(doc)[:500]
            doc_texts.append(f"[{i+1}] {text}")
        
        doc_content = "\n\n".join(doc_texts)
        
        prompt = f"""Analyze these documents for the step: {step_info.purpose}

Documents:
{doc_content}

Provide a brief analysis of what was found (2-3 sentences)."""

        try:
            return await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )
        except Exception:
            return "Analysis not available"

    async def _evaluate_if_answered(
        self,
        steps: list[ResearchStep],
        original_query: str,
        llm: Any,
    ) -> tuple[bool, float]:
        """
        Evaluate if the question is sufficiently answered.
        
        Ported from: core/profonde.py - evaluation logic
        """
        if not steps:
            return False, 0.0
        
        # Format step information
        steps_info = "\n\n".join([
            f"Step: {s.step_purpose}\n"
            f"Success: {s.success}\n"
            f"Findings: {', '.join(s.key_findings[:2])}\n"
            f"Missing: {', '.join(s.missing_info[:2])}"
            for s in steps
        ])
        
        system_prompt = """Evaluate whether the research has sufficiently answered the original question.

Respond in JSON format:
{
    "question_answered": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "remaining_gaps": ["gap1", "gap2"]
}

Guidelines:
- question_answered: true only if we have found affirmative information
- confidence: how certain are you that the answer is complete
- Be conservative - if important information is missing, mark as not answered"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {original_query}\n\nResearch:\n{steps_info}"},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            return (
                result.get("question_answered", False),
                result.get("confidence", 0.0),
            )
            
        except Exception as e:
            logger.error("profound_evaluation_error", error=str(e))
            return False, 0.0

    async def _review_progress(
        self,
        query: str,
        history: list[dict],
        llm: Any,
    ) -> bool:
        """
        Review research progress and decide whether to continue.
        
        Ported from: core/profonde.py - plan review logic
        """
        if len(history) >= self.max_cycles:
            return False
        
        # Summarize current progress
        history_summary = "\n".join([
            f"Cycle {h['cycle']}: " + 
            ", ".join([f"{s['purpose']}({'✓' if s['success'] else '✗'})" for s in h['steps']])
            for h in history
        ])
        
        system_prompt = """Review the research progress and decide if we should continue.

Respond in JSON format:
{
    "should_continue": true/false,
    "reasoning": "explanation",
    "suggestion": "what to do next if continuing"
}

Guidelines:
- should_continue: true if we haven't found a complete answer yet
- Consider: information gaps, failed steps, unanswerable questions
- Don't continue if we've had multiple failed cycles"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}\n\nProgress:\n{history_summary}"},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            return result.get("should_continue", False)
            
        except Exception as e:
            logger.error("profound_review_error", error=str(e))
            return False

    async def _finalize_response(
        self,
        query: str,
        steps: list[ResearchStep],
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
        exited_early: bool,
    ) -> RAGResponse:
        """Generate final response based on all research."""
        
        # Format research summary
        research_summary = []
        for step in steps:
            research_summary.append(
                f"Step: {step.step_purpose}\n"
                f"Query: {step.query}\n"
                f"Success: {step.success}\n"
                f"Analysis: {step.analysis[:300]}..."
            )
        
        research_text = "\n\n---\n\n".join(research_summary)
        
        system_prompt = """Generate a comprehensive answer based on the research conducted.

Your answer should:
1. Directly address the original question
2. Synthesize findings from all research steps
3. Maintain scientific accuracy
4. Acknowledge any information gaps
5. Be clear and well-structured

If the research did not find sufficient information, clearly state this."""

        user_message = f"""Original question: {query}

Research conducted ({self.iterations} cycles):
{research_text}

Generate a final answer."""

        try:
            answer = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("profound_final_answer_error", error=str(e))
            answer = f"Error generating response: {e}"

        # Prepare sources
        sources = self._prepare_sources(documents)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.PROFOUND,
            iterations=self.iterations,
            web_search_used=any(
                isinstance(d, dict) and d.get("source") == "web_search"
                for d in documents
            ),
        )

    def _prepare_sources(self, documents: list[Any]) -> list[SourceReference]:
        """Prepare source references from documents."""
        seen = set()
        sources = []
        
        for doc in documents:
            # Handle web search results
            if isinstance(doc, dict):
                title = doc.get("source", "Web source")
                if title == "web_search":
                    title = f"Web search: {doc.get('query', 'Unknown')}"
                sources.append(SourceReference(
                    title=title,
                    relevance_score=0.5,
                ))
                continue
            
            # Handle vector store results
            if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'metadata'):
                meta = doc.chunk.metadata
                title = getattr(meta, 'title', 'Untitled')
                authors = getattr(meta, 'authors', [])
                year = getattr(meta, 'year', None)
                doi = getattr(meta, 'doi', None)
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

            sources.append(SourceReference(
                title=title,
                authors=authors_str,
                year=year,
                doi=doi,
                relevance_score=getattr(doc, 'score', 0.0),
            ))

        return sources
