"""Main agentic orchestrator with session management."""

import json
import re
import uuid
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from .intent import IntentClassifier, Intent
from .planner import ResearchPlanner, Step, StepType, Plan, _log_steps_detail
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.retrieval.hybrid import hybrid_retrieval
from perspicacite.rag.utils import format_references_academic

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A conversation message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSession:
    """Persistent session for agent conversations."""

    session_id: str
    messages: List[Message] = field(default_factory=list)
    knowledge_base: Optional[DynamicKnowledgeBase] = None
    research_findings: List[Dict[str, Any]] = field(default_factory=list)
    kb_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Add a message to the session."""
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
        self.last_active = datetime.now()

    def get_conversation_history(self, limit: int = 10) -> List[dict]:
        """Get conversation history as list of dicts."""
        return [{"role": m.role, "content": m.content} for m in self.messages[-limit:]]

    def get_context_string(self) -> str:
        """Get recent conversation context as string."""
        context = []
        for msg in self.messages[-4:]:
            context.append(f"{msg.role}: {msg.content[:300]}")
        return "\n".join(context)


class DocumentQualityAssessor:
    """Assess if retrieved documents are sufficient to answer a query.
    
    Ported from AgenticRAGMode to enable early exit and quality-aware retrieval.
    """

    def __init__(self, llm: Any):
        self.llm = llm

    async def assess(
        self,
        query: str,
        documents: List[Any],
        step_purpose: str = "",
    ) -> tuple[bool, List[str], float]:
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
            elif isinstance(doc, dict):
                text = doc.get("text", doc.get("content", str(doc)))[:500]
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
            # Build prompt for LLMAdapter interface (simple prompt string, not messages)
            prompt = f"{system_prompt}\n\nQuery: {query}\n\nDocuments:\n{doc_content}"
            response = await self.llm.complete(
                prompt=prompt,
                temperature=0.0,
                max_tokens=300,
            )

            # Parse JSON response
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
            logger.warning(f"Quality assessment error: {e}")
            # Conservative default - assume insufficient
            return False, ["Assessment error"], 0.0


class AgenticOrchestrator:
    """
    True agentic orchestrator with LLM-driven planning and execution.
    
    Unified implementation consolidating:
    - Intent classification and dynamic planning
    - Document quality assessment and early exit
    - Session management and streaming
    - Multi-source tool execution
    """

    def __init__(
        self,
        llm_client,
        tool_registry,
        embedding_provider,
        vector_store,
        max_iterations: int = 5,
        use_hybrid: bool = True,
        early_exit_confidence: float = 0.85,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.embeddings = embedding_provider
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        self.use_hybrid = use_hybrid
        self.early_exit_confidence = early_exit_confidence

        self.intent_classifier = IntentClassifier(llm_client)
        self.planner = ResearchPlanner(llm_client)
        self.quality_assessor = DocumentQualityAssessor(llm_client)

        # Session management
        self.sessions: Dict[str, AgentSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> AgentSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]

        new_session_id = session_id or str(uuid.uuid4())
        session = AgentSession(session_id=new_session_id)

        # Create persistent KB for this session
        session.knowledge_base = DynamicKnowledgeBase(
            vector_store=self.vector_store,
            embedding_service=self.embeddings,
        )

        self.sessions[new_session_id] = session
        return session

    async def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        kb_name: Optional[str] = None,
        stream: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main chat entry point with true agentic behavior.

        Yields:
            Dict with type: "thinking", "tool_call", "tool_result", "answer", "papers_found"
        """
        logger.info("=" * 80)
        logger.info("NEW CHAT REQUEST")
        logger.info(f"Query: {query}")
        logger.info(f"Session ID (client): {session_id!r}")
        logger.info(f"KB: {kb_name or 'none'}")

        session = self.get_or_create_session(session_id)
        logger.info(f"Resolved session_id: {session.session_id}")
        session.add_message("user", query)
        session.kb_name = kb_name
        logger.info(f"Session messages count: {len(session.messages)}")

        # Clear accumulated papers from previous requests
        self._found_papers = []

        # Step 1: Classify intent
        yield {"type": "thinking", "message": "Analyzing your query..."}

        intent_result = await self.intent_classifier.classify(
            query=query,
            conversation_history=session.get_conversation_history(),
            active_kb_name=kb_name,
        )
        logger.info(f"Intent classified: {intent_result.intent.name}")
        logger.info(f"Confidence: {intent_result.confidence}")
        logger.info(f"Suggested tools: {intent_result.suggested_tools}")

        yield {
            "type": "thinking",
            "message": f"Intent: {intent_result.intent.name.replace('_', ' ').title()}",
            "details": intent_result.reasoning,
        }

        # Step 2: Create dynamic plan
        yield {"type": "thinking", "message": "Creating research plan..."}

        # Available tools: registered tools (excluding deactivated ones) + built-in
        available_tools = [t for t in self.tools.list_tools() if t != "lotus_search"] + [
            "literature_search",
            "kb_search",
        ]
        logger.info(f"Available tools: {available_tools}")
        previous_findings = self._summarize_findings(session.research_findings)

        plan = await self.planner.create_plan(
            query=query,
            intent_result=intent_result,
            available_tools=available_tools,
            conversation_history=session.get_conversation_history(),
            previous_findings=previous_findings,
            active_kb_name=kb_name,
        )

        # If a KB is selected, always search it first (don't rely on the LLM planner)
        if kb_name:
            has_kb_step = any(s.type == StepType.KB_SEARCH for s in plan.steps)
            if not has_kb_step:
                from perspicacite.rag.agentic.planner import ResearchPlanner

                clean_query = ResearchPlanner._clean_query_for_search(query)
                kb_step = Step(
                    id="step1",
                    type=StepType.KB_SEARCH,
                    description=f"Search knowledge base '{kb_name}'",
                    tool="kb_search",
                    tool_input={"query": clean_query},
                )
                plan.steps.insert(0, kb_step)
                plan.estimated_steps = len(plan.steps)
                logger.info(
                    f"Injected kb_search as step1 for KB {kb_name!r} tool_input.query={clean_query!r}"
                )

        logger.info(f"Orchestrator plan reasoning ({len(plan.reasoning)} chars): {plan.reasoning}")
        _log_steps_detail(plan.steps, "Orchestrator plan (final, after KB inject if any)")

        if plan.can_answer_from_history:
            yield {"type": "thinking", "message": "I can answer from our conversation history..."}

        # Step 3: Execute plan iteratively
        step_results: Dict[str, Any] = {}
        completed_steps: List[Step] = []
        self._found_papers: List[Dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Find next executable step
            next_step = self._get_next_step(plan, completed_steps, step_results)

            if not next_step:
                logger.info("No more steps to execute")
                break

            logger.info(
                f"Next step: {next_step.id} ({next_step.type.value}) - {next_step.description}"
            )

            # Check condition
            if next_step.condition and not self._evaluate_condition(
                next_step.condition, step_results
            ):
                logger.info(f"Step {next_step.id} condition not met, skipping")
                completed_steps.append(next_step)
                continue

            # Execute step
            yield {
                "type": "tool_call",
                "step": next_step.id,
                "tool": next_step.tool or next_step.type.value,
                "description": next_step.description,
                "query": next_step.tool_input.get("query", ""),
            }

            import time

            step_start_time = time.time()
            logger.info(f"Executing step {next_step.id}...")
            result = await self._execute_step(next_step, query, step_results, session)
            step_duration = time.time() - step_start_time

            # Log the result
            result_str = str(result)
            logger.info(f"Step {next_step.id} completed in {step_duration:.2f}s")
            logger.info(f"Result length: {len(result_str)} chars")
            # Show first 2000 chars to see actual content (not just titles)
            preview_len = min(2000, len(result_str))
            logger.info(
                f"Result preview: {result_str[:preview_len]}{'...[truncated]' if len(result_str) > preview_len else ''}"
            )

            step_results[next_step.id] = result
            completed_steps.append(next_step)

            yield {
                "type": "tool_result",
                "step": next_step.id,
                "result_summary": self._summarize_result(result),
            }

            # Evaluate if we need to replan
            if next_step.type in (
                StepType.LOTUS_SEARCH,
                StepType.LITERATURE_SEARCH,
                StepType.KB_SEARCH,
            ):
                should_continue = await self._evaluate_progress(
                    query, plan, completed_steps, step_results
                )
                logger.info(f"Progress evaluation: {should_continue}")

                if should_continue == "replan":
                    evaluation = "Need more specific search"
                    plan = await self.planner.replan(
                        query, plan, completed_steps, step_results, evaluation
                    )
                    yield {"type": "thinking", "message": "Adjusting research plan..."}
                elif should_continue == "answer":
                    logger.info("Sufficient results, moving to answer")
                    break

        logger.info(f"\n=== Execution complete ===")
        logger.info(f"Completed {len(completed_steps)} steps")
        logger.info(f"Step results keys: {list(step_results.keys())}")

        # Step 4: Extract and process papers with progress updates
        yield {"type": "thinking", "message": "Extracting papers from search results..."}
        papers = self._extract_papers_from_results(step_results)
        logger.info(f"Extracted {len(papers)} papers from search results")
        
        # Score papers for relevance
        yield {"type": "thinking", "message": f"Scoring {len(papers)} papers for relevance..."}
        papers = await self._score_papers_for_relevance(query, papers, min_score=3)
        included_count = len([p for p in papers if p.get("relevance_score", 3) >= 3])
        yield {"type": "thinking", "message": f"Relevance filtering: {included_count}/{len(papers)} papers included"}
        
        # Download full text for top relevant papers (up to 5)
        max_download = 5
        download_candidates = [p for p in papers if p.get("relevance_score", 3) >= 3][:max_download]
        if download_candidates:
            yield {"type": "thinking", "message": f"Attempting to download {len(download_candidates)} papers for full text..."}
            
            downloaded_count = 0
            for i, paper in enumerate(download_candidates, 1):
                title = paper.get('title', 'Unknown')[:50]
                yield {"type": "thinking", "message": f"Downloading paper {i}/{len(download_candidates)}: {title}..."}
                
                enriched = await self._download_single_paper(paper)
                if enriched.get("pdf_downloaded"):
                    downloaded_count += 1
                    yield {"type": "thinking", "message": f"✓ Downloaded paper {i}: {title}"}
                else:
                    yield {"type": "thinking", "message": f"✗ Paper {i} not available: {title}"}
            
            yield {"type": "thinking", "message": f"Downloaded {downloaded_count}/{len(download_candidates)} papers successfully"}
        
        # Generate final answer
        yield {"type": "thinking", "message": "Synthesizing answer..."}
        answer = await self._generate_answer(
            query=query, plan=plan, step_results=step_results, session=session, papers=papers
        )

        session.add_message(
            "assistant",
            answer,
            {
                "intent": intent_result.intent.name,
                "steps_completed": len(completed_steps),
                "tools_used": [s.tool for s in completed_steps if s.tool],
            },
        )

        yield {"type": "answer", "content": answer, "session_id": session.session_id}

        # Yield found papers so the UI can offer "Add to KB"
        # Only include papers that:
        # 1. Passed relevance filtering (score >= 3)
        # 2. Are NOT already in the KB (source != "kb_search")
        relevant_papers = [
            p for p in papers 
            if p.get("relevance_score", 3) >= 3 and p.get("source") != "kb_search"
        ]
        if relevant_papers:
            yield {"type": "papers_found", "papers": relevant_papers}

    def _get_next_step(
        self, plan: Plan, completed: List[Step], results: Dict[str, Any]
    ) -> Optional[Step]:
        """Get the next executable step based on dependencies."""
        completed_ids = {s.id for s in completed}

        for step in plan.steps:
            if step.id in completed_ids:
                continue

            # Check if dependencies are satisfied
            if all(dep in completed_ids for dep in step.depends_on):
                return step

        return None

    def _evaluate_condition(self, condition: str, results: Dict[str, Any]) -> bool:
        """Evaluate a step condition."""
        # Simple condition evaluation
        condition_lower = condition.lower()

        if "found" in condition_lower or "results" in condition_lower:
            # Check if any previous step had results
            for result in results.values():
                if result and str(result) not in ["", "None", "[]", "{}"]:
                    if "not found" not in str(result).lower() and "no " not in str(result).lower():
                        return True
            return False

        return True  # Default to executing

    async def _execute_step(
        self, step: Step, original_query: str, step_results: Dict[str, Any], session: AgentSession
    ) -> Any:
        """Execute a single step."""

        if step.type == StepType.LOTUS_SEARCH:
            logger.info("LOTUS_SEARCH: skipped (deactivated)")
            return "LOTUS search is currently deactivated."

        elif step.type == StepType.LITERATURE_SEARCH:
            query = step.tool_input.get("query", original_query)
            logger.info(f"LITERATURE_SEARCH: query='{query}'")
            return await self._fallback_openalex_search(query)

        elif step.type == StepType.DOWNLOAD_PAPERS:
            # Download papers from OpenAlex results
            openalex_result = step_results.get(step.depends_on[0]) if step.depends_on else None
            if openalex_result and isinstance(openalex_result, list):
                downloaded = []
                for paper in openalex_result[:3]:  # Max 3 papers
                    if isinstance(paper, dict) and "id" in paper:
                        # Download logic here
                        downloaded.append(paper)
                return downloaded
            return []

        elif step.type == StepType.KB_SEARCH:
            if session.kb_name:
                try:
                    collection_name = chroma_collection_name_for_kb(session.kb_name)
                    kb_query = step.tool_input.get("query", original_query)

                    logger.info("========== KB_SEARCH ==========")
                    logger.info(
                        f"KB_SEARCH: kb_name={session.kb_name!r} collection={collection_name!r} "
                        f"step_id={step.id!r}"
                    )
                    logger.info(f"KB_SEARCH: search_query ({len(kb_query)} chars)={kb_query!r}")

                    dkb = DynamicKnowledgeBase(
                        vector_store=self.vector_store,
                        embedding_service=self.embeddings,
                    )
                    dkb.collection_name = collection_name
                    dkb._initialized = True
                    top_k = dkb.config.top_k
                    logger.info(
                        f"KB_SEARCH: top_k={top_k} min_relevance_score={dkb.config.min_relevance_score} "
                        f"embedding_model={getattr(self.embeddings, 'model_name', '?')!r}"
                    )

                    results = await dkb.search(kb_query, top_k=top_k)
                    logger.info(
                        f"KB_SEARCH: vector hits (after dedupe/score filter)={len(results)}"
                    )

                    # Apply hybrid retrieval if enabled
                    if self.use_hybrid and results:
                        try:
                            logger.info("KB_SEARCH: applying hybrid retrieval (BM25 + vector)")
                            # Convert results to format expected by hybrid_retrieval
                            vector_scores = [r.get("score", 0.5) for r in results]

                            # Create document objects with proper attributes
                            doc_objects = []
                            for r in results:
                                doc = type("Doc", (), {})()
                                doc.metadata = r.get("metadata")
                                doc.page_content = r.get("text", "")
                                doc.score = r.get("score", 0.5)
                                doc_objects.append(doc)

                            hybrid_results = await hybrid_retrieval(
                                query=kb_query,
                                documents=doc_objects,
                                vector_scores=vector_scores,
                                use_llm_weights=True,
                                llm=self.llm,
                            )

                            # Update results with hybrid scores
                            results = []
                            for doc, hybrid_score in hybrid_results:
                                results.append(
                                    {
                                        "text": doc.page_content,
                                        "metadata": doc.metadata,
                                        "score": hybrid_score,
                                    }
                                )

                            logger.info(
                                f"KB_SEARCH: hybrid retrieval complete, {len(results)} results"
                            )
                        except Exception as e:
                            logger.warning(
                                f"KB_SEARCH: hybrid retrieval failed: {e}", exc_info=True
                            )
                    
                    # Filter results by minimum relevance score (0.5 = medium relevance)
                    min_relevance_threshold = 0.5
                    filtered_results = [r for r in results if r.get("score", 0) >= min_relevance_threshold]
                    if len(filtered_results) < len(results):
                        logger.info(
                            f"KB_SEARCH: filtered {len(results) - len(filtered_results)} low-relevance results "
                            f"(score < {min_relevance_threshold}), kept {len(filtered_results)}"
                        )
                        results = filtered_results
                    for j, r in enumerate(results, 1):
                        meta = r.get("metadata")
                        pid = (
                            getattr(meta, "paper_id", None)
                            if meta is not None
                            else r.get("paper_id")
                        )
                        title = (
                            getattr(meta, "title", None) or "Unknown"
                            if meta is not None
                            else "Unknown"
                        )
                        txt = r.get("text") or ""

                        # Warn if text is empty - this indicates a data quality issue
                        if not txt.strip():
                            logger.warning(
                                f"KB_SEARCH hit {j}: EMPTY TEXT CONTENT for paper_id={pid!r} title={title!r}"
                            )

                        logger.info(
                            f"KB_SEARCH hit {j}/{len(results)}: paper_id={pid!r} "
                            f"score={r.get('score', 0):.4f} title={title!r} text_len={len(txt)}"
                        )
                        preview = txt[:280].replace("\n", " ")
                        if preview.strip():
                            logger.info(
                                f"KB_SEARCH hit {j} text_preview: {preview}{'…' if len(txt) > 280 else ''}"
                            )

                    if results:
                        formatted_parts = [
                            f"Found {len(results)} relevant documents in knowledge base:"
                        ]
                        for i, r in enumerate(results, 1):
                            meta = r.get("metadata")
                            title = (
                                getattr(meta, "title", None) or "Unknown"
                                if meta is not None
                                else "Unknown"
                            )
                            authors = (
                                getattr(meta, "authors", None) or "" if meta is not None else ""
                            )
                            year = getattr(meta, "year", None) or "" if meta is not None else ""
                            doi = getattr(meta, "doi", None) or "" if meta is not None else ""
                            score = r.get("score", 0)
                            # Include more text content for better context (up to 1500 chars)
                            raw_text = r.get("text", "") or ""
                            text_content = raw_text[:1500]

                            # Log warning if text is empty - this is a data quality issue
                            if not raw_text.strip():
                                logger.warning(
                                    f"KB_SEARCH: Hit {i} has EMPTY text content for '{title}'"
                                )

                            formatted_parts.append(f"\n- {title} (relevance: {score:.2f})")
                            if authors:
                                formatted_parts.append(f"   Authors: {authors}")
                            if year:
                                formatted_parts.append(f"   Year: {year}")
                            if doi:
                                formatted_parts.append(f"   DOI: {doi}")
                            if text_content:
                                formatted_parts.append(f"   Content: {text_content}")
                                if len(raw_text) > 1500:
                                    formatted_parts.append("   [... content truncated ...]")
                            else:
                                formatted_parts.append("   Content: [No text content available]")
                            
                            # Add to _found_papers for reference list generation
                            if hasattr(self, "_found_papers"):
                                self._found_papers.append({
                                    "title": title,
                                    "authors": [a.strip() for a in authors.split(",")] if authors else [],
                                    "year": year,
                                    "doi": doi,
                                    "abstract": text_content[:500] if text_content else "",
                                    "source": "kb_search",
                                    "relevance_score": 4,  # KB papers are considered relevant by default
                                })
                        
                        out = "\n".join(formatted_parts)
                        logger.info(f"KB_SEARCH: formatted tool result length={len(out)} chars")
                        return out
                    logger.info("KB_SEARCH: no hits — empty result for downstream / judge")
                    return "No relevant documents found in knowledge base."
                except Exception as e:
                    logger.error(f"KB_SEARCH failed: {e}", exc_info=True)
                    return "Knowledge base search failed."
            logger.info("KB_SEARCH: skipped — no knowledge base selected on session")
            return "No knowledge base selected."

        elif step.type == StepType.ANALYZE:
            # LLM analysis of results
            return await self._analyze_results(original_query, step_results)

        elif step.type == StepType.SYNTHESIZE:
            # LLM synthesis of multiple sources
            return await self._synthesize_results(original_query, step_results)

        elif step.type == StepType.ANSWER:
            # Generate final answer - handled in chat() method
            # This step just marks that we should answer
            return "ANSWER_STEP"

        return None

    async def _llm_judge_kb_sufficiency(self, user_query: str, kb_result_text: str) -> bool:
        """
        Ask the LLM whether KB retrieval is enough to answer without web/OpenAlex.

        Returns False on empty/failed retrieval, parse errors, or LLM saying insufficient.
        """
        excerpt = (kb_result_text or "").strip()
        low = excerpt.lower()

        # Log what we're working with
        logger.info(
            f"KB_JUDGE: input length={len(kb_result_text or '')} chars, excerpt length={len(excerpt)} chars"
        )

        if not excerpt:
            logger.info("KB_JUDGE: empty excerpt -> insufficient")
            return False
        if (
            "no relevant documents" in low
            or "knowledge base search failed" in low
            or "no knowledge base selected" in low
        ):
            logger.info(f"KB_JUDGE: found failure phrase -> insufficient")
            return False

        max_judge_chars = 8000
        if len(excerpt) > max_judge_chars:
            excerpt = excerpt[:max_judge_chars] + "\n[... truncated for judge ...]"

        # Log the actual excerpt being sent to judge (first 1000 chars)
        logger.info(f"KB_JUDGE: excerpt preview (first 1000 chars): {excerpt[:1000]}...")

        prompt = (
            "You decide if KNOWLEDGE BASE retrieval is enough to answer the user's question "
            "without any further web or literature search.\n\n"
            f'User question:\n"{user_query}"\n\n'
            "Knowledge base retrieval (from curated papers):\n---\n"
            f"{excerpt}\n"
            "---\n\n"
            "Reply with ONLY a single JSON object, no markdown fences: "
            '{"sufficient": true or false, "reason": "short phrase"}\n\n'
            "Guidelines:\n"
            "- sufficient=true ONLY if retrieved papers DIRECTLY address the user's specific query. "
            "The papers must be on the EXACT topic asked, not just related keywords.\n"
            "- sufficient=false if: (a) papers are off-topic, (b) papers only mention keywords but don't "
            "address the core question, (c) relevance scores are low (<0.6), (d) no paper actually "
            "answers the specific question asked.\n"
            "- Example: Query 'FBMN on tea extract' with papers about 'Bifidobacterium peptides' or "
            "'fermented bean paste' → sufficient=false (wrong topic entirely).\n"
            "- Example: Query 'FBMN on tea extract' with paper about 'EGCG fermentation metabolites' → "
            "sufficient=false (mentions tea compound but not FBMN application to tea extract).\n"
            "- Be STRICT: prefer sufficient=false when papers are tangential or use loose keyword matching."
        )

        try:
            raw = await self.llm.complete(prompt, temperature=0.0)
            text = raw.strip()
            logger.info(f"KB_JUDGE: raw LLM response length={len(text)} chars")
            logger.debug(f"KB_JUDGE: raw response preview: {text[:500]}...")
            m_fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if m_fence:
                text = m_fence.group(1).strip()
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                logger.warning(f"KB_JUDGE: no JSON object in response. Text preview: {text[:200]}")
                return False
            obj = json.loads(text[start : end + 1])
            sufficient = bool(obj.get("sufficient"))
            reason = obj.get("reason", "")
            logger.info(f"KB_JUDGE: sufficient={sufficient} reason={reason!r}")
            return sufficient
        except Exception as e:
            logger.warning(f"KB_JUDGE: failed with error: {e}")
            return False

    async def _evaluate_progress(
        self, query: str, plan: Plan, completed_steps: List[Step], step_results: Dict[str, Any]
    ) -> str:
        """Evaluate if we should continue, replan, or answer.
        
        Uses document quality assessment for intelligent early exit decisions.
        Ported from AgenticRAGMode.
        """
        last_step = completed_steps[-1]
        last_result = step_results.get(last_step.id)
        last_str = str(last_result) if last_result is not None else ""

        # Check if no more steps remaining
        remaining_steps = [s for s in plan.steps if s.id not in {cs.id for cs in completed_steps}]
        if not remaining_steps:
            return "answer"

        # After search tools: use quality assessment for early exit decision
        if last_step.type in (StepType.KB_SEARCH, StepType.LITERATURE_SEARCH):
            # Extract documents from result for quality assessment
            documents = self._extract_documents_from_result(last_result)
            
            if documents:
                # Use quality assessor for intelligent early exit
                is_sufficient, missing_aspects, confidence = await self.quality_assessor.assess(
                    query=query,
                    documents=documents,
                    step_purpose=last_step.description,
                )
                
                logger.info(
                    f"Quality assessment: sufficient={is_sufficient}, "
                    f"confidence={confidence:.2f}, missing={len(missing_aspects)} aspects"
                )
                
                # Early exit if quality is high and confidence meets threshold
                if is_sufficient and confidence >= self.early_exit_confidence:
                    logger.info(f"Early exit triggered: confidence {confidence:.2f} >= {self.early_exit_confidence}")
                    return "answer"
                
                # Replan if quality is low and we have missing aspects
                if not is_sufficient and missing_aspects and len(completed_steps) < 2:
                    logger.info(f"Quality insufficient, continuing with plan: {missing_aspects[:3]}")
                    return "replan"

        # Fallback: check for substantial results after multiple steps
        has_substantial_results = False
        for result in step_results.values():
            result_str = str(result)
            if len(result_str) > 200 and "error" not in result_str.lower():
                has_substantial_results = True
                break

        # Suggest termination if we have 3+ completed steps with results
        if has_substantial_results and len(completed_steps) >= 3:
            return "answer"

        return "continue"

    def _extract_documents_from_result(self, result: Any) -> List[Dict[str, Any]]:
        """Extract document list from search result for quality assessment."""
        documents = []
        
        if not result:
            return documents
            
        result_str = str(result)
        
        # Check for common "no results" patterns
        result_lower = result_str.lower()
        if any(phrase in result_lower for phrase in [
            "no relevant documents", 
            "no results", 
            "not found",
            "search failed"
        ]):
            return documents
        
        # Try to parse documents from structured results
        # Format 1: List of paper dicts from OpenAlex
        if isinstance(result, list):
            for item in result[:5]:  # Limit to top 5
                if isinstance(item, dict):
                    documents.append({
                        "title": item.get("title", "Unknown"),
                        "content": item.get("abstract", str(item)[:500]),
                        "source": "openalex",
                    })
        
        # Format 2: KB search results - extract from formatted string
        if "Found" in result_str and "documents" in result_str:
            # Extract document sections from KB result format
            lines = result_str.split("\n")
            current_doc = None
            
            for line in lines:
                if line.startswith("- ") and "(relevance:" in line:
                    # New document
                    if current_doc:
                        documents.append(current_doc)
                    title = line.split("(relevance:")[0].replace("- ", "").strip()
                    current_doc = {"title": title, "content": "", "source": "kb"}
                elif current_doc and line.strip().startswith("Content:"):
                    current_doc["content"] = line.replace("Content:", "").strip()[:500]
            
            if current_doc:
                documents.append(current_doc)
        
        return documents[:5]  # Return top 5 documents

    async def _analyze_results(self, query: str, step_results: Dict[str, Any]) -> str:
        """Have LLM analyze the results."""
        # Combine results
        combined = []
        for step_id, result in step_results.items():
            combined.append(f"{step_id}:\n{str(result)[:500]}")

        prompt = f"""You are analyzing research results to determine their relevance and completeness for answering a query.

Original Query: "{query}"

Research Results:
{chr(10).join(combined)}

Analysis Instructions:
1. Evaluate whether the results directly address the query
2. Identify what key information is present
3. Identify what important aspects are missing
4. Assess the quality and reliability of the information
5. Determine if additional research is needed

Provide your analysis in a structured format:
- Key Findings: What was discovered
- Gaps: What's missing or unclear
- Recommendation: Whether to continue researching or proceed to answer"""

        return await self.llm.complete(prompt, temperature=0.3)

    async def _synthesize_results(self, query: str, step_results: Dict[str, Any]) -> str:
        """Have LLM synthesize multiple sources."""
        combined = []
        for step_id, result in step_results.items():
            combined.append(f"Source ({step_id}):\n{str(result)[:400]}")

        prompt = f"""You are synthesizing information from multiple research sources to create a coherent answer.

Original Query: "{query}"

Sources:
{chr(10).join(combined)}

Synthesis Guidelines:
1. Integrate information from all relevant sources
2. Resolve any contradictions between sources
3. Build a coherent narrative that directly answers the query
4. Cite specific sources when presenting key findings
5. Highlight areas of agreement and disagreement between sources
6. Identify the most reliable or relevant sources for the query

Provide a synthesized summary that combines the key insights from all sources."""

        return await self.llm.complete(prompt, temperature=0.3)

    async def _generate_answer(
        self, query: str, plan: Plan, step_results: Dict[str, Any], session: AgentSession,
        papers: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate final answer using all results."""

        logger.info("\n--- Generating Answer ---")
        logger.info(f"Query: {query}")
        logger.info(f"Step results available: {list(step_results.keys())}")

        # Use pre-processed papers if provided, otherwise extract from results
        if papers is None:
            papers = self._extract_papers_from_results(step_results)
            papers = await self._score_papers_for_relevance(query, papers, min_score=3)
        
        numbered_paper_list = self._build_numbered_paper_list(papers)

        # Build context from step results
        context_parts = []

        # Prioritize certain result types
        if "lotus" in step_results:
            lotus_result = step_results["lotus"]
            logger.info(f"LOTUS result length: {len(str(lotus_result))} chars")
            context_parts.append(f"LOTUS Search Results:\n{lotus_result}")

        for step_id, result in step_results.items():
            if step_id != "lotus" and result:
                result_str = str(result)
                logger.info(f"Step {step_id} result length: {len(result_str)} chars")
                context_parts.append(f"{step_id}:\n{result_str[:3000]}")

        # Add full text from downloaded papers to context
        full_text_parts = []
        for i, paper in enumerate(papers, 1):
            if paper.get("full_text"):
                full_text_parts.append(
                    f"[Paper {i}: {paper.get('title', 'Unknown')[:80]}...]\n"
                    f"Full text excerpt:\n{paper['full_text'][:8000]}..."
                )
        
        if full_text_parts:
            context_parts.append("\n\n---\n\nDownloaded Full Text:\n" + "\n\n---\n\n".join(full_text_parts))
            logger.info(f"Added {len(full_text_parts)} full text documents to context")

        context = "\n\n".join(context_parts)
        logger.info(f"Total context length: {len(context)} chars")

        if not context.strip():
            logger.warning("Context is empty! No research results to use.")

        # Include conversation context
        conversation_context = session.get_context_string()

        prompt = f"""You are a scientific research assistant. Generate a comprehensive answer based on the research results provided.

Original Question: "{query}"

Previous Conversation Context:
{conversation_context}

Research Results:
{context}

{numbered_paper_list if numbered_paper_list else ""}

Answer Generation Guidelines:
1. Focus on answering the SPECIFIC question asked - avoid tangential information
2. Prioritize the most relevant findings from the research results
3. Maintain scientific precision and technical accuracy
4. **MANDATORY CITATION FORMAT**: Use ONLY the bracket format [N] where N is the paper number from the NUMBERED PAPER LIST section above (e.g., [1], [2], [3]). 
   - **CRITICAL**: IGNORE ALL numbers in the Research Results section (like "1.", "2." in "Found X papers:") - these are NOT citation numbers
   - **ONLY** cite papers using the [N] format from the NUMBERED PAPER LIST at the end of this prompt
   - The NUMBERED PAPER LIST uses [1], [2], [3], etc. - these are your ONLY valid citation numbers
   - If a paper is not in the NUMBERED PAPER LIST, do NOT cite it with a number
5. Be clear and direct in your language
6. If the research results are insufficient to answer the question, clearly state this rather than speculating
7. Structure your answer logically with clear sections if appropriate
8. Cite using [N] from the numbered list only for sources you actually use; you do not need to mention every listed paper if some are redundant.
9. **DO NOT include a Citations or References section at the end of your answer** - a properly formatted references section will be automatically appended separately.

Important: Do not provide an answer if the question contains hate speech, offensive language, discriminatory remarks, or harmful content.

Generate your answer:"""

        logger.info(f"Prompt length: {len(prompt)} chars")
        logger.info("Calling LLM for answer...")

        answer = await self.llm.complete(prompt, temperature=0.4)
        logger.info(f"Answer generated, length: {len(answer)} chars")
        logger.info(f"Answer content:\n{answer}")

        # Append references section if we have papers (uses same paper order)
        if papers:
            references_section = self._format_references_section(papers)
            if references_section:
                answer = answer.rstrip() + "\n\n" + references_section
                logger.info(f"References section added, total length: {len(answer)} chars")

        return answer

    def _build_numbered_paper_list(
        self, papers: List[Dict[str, Any]], max_abstract_chars: int = 800
    ) -> str:
        """Build a numbered paper list for LLM context with full citation info.

        Each paper is numbered [1], [2], etc. and includes title, authors, year,
        and abstract. This numbered list is used both for the LLM prompt and
        the References section, ensuring citation alignment.
        """
        if not papers:
            return ""

        lines = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown Title")
            authors = paper.get("authors", [])
            year = paper.get("year", "n.d.")
            doi = paper.get("doi", "")
            abstract = paper.get("abstract", "") or ""
            has_full_text = paper.get("pdf_downloaded", False)

            # Format author string
            if len(authors) == 0:
                author_str = "Unknown"
            elif len(authors) == 1:
                author_str = authors[0]
            elif len(authors) == 2:
                author_str = f"{authors[0]} & {authors[1]}"
            else:
                author_str = f"{authors[0]} et al."

            # Truncate abstract to relevant portion
            if len(abstract) > max_abstract_chars:
                abstract = abstract[:max_abstract_chars].rsplit(" ", 1)[0] + "..."

            full_text_indicator = " [FULL TEXT DOWNLOADED]" if has_full_text else ""
            lines.append(f"[{i}] {title}{full_text_indicator}")
            lines.append(f"    Authors: {author_str}")
            lines.append(f"    Year: {year}")
            if doi:
                lines.append(f"    DOI: {doi}")
            if abstract:
                lines.append(f"    Abstract: {abstract}")

        return "\n".join(lines)

    async def _score_papers_for_relevance(
        self, query: str, papers: List[Dict[str, Any]], min_score: int = 3
    ) -> List[Dict[str, Any]]:
        """Use LLM to score papers for query relevance and filter low-scoring ones.

        Each paper is scored 1-5:
        1 = Completely irrelevant
        2 = Tangential, unlikely to help answer query
        3 = Somewhat relevant, partial answer
        4 = Relevant, contributes to answer
        5 = Highly relevant, directly addresses query

        Only papers with score >= min_score are included in synthesis.
        """
        if not papers:
            return []

        # Build paper list for LLM
        paper_lines = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", "") or "No abstract available."
            paper_lines.append(f"[{i}] Title: {title}\n   Abstract: {abstract[:500]}")

        paper_list_str = "\n\n".join(paper_lines)

        prompt = (
            "You are evaluating research papers for relevance to a user's query.\n\n"
            f'User Query: "{query}"\n\n'
            f"Papers to evaluate:\n{paper_list_str}\n\n"
            "For each paper, score its relevance to the query on this scale:\n"
            "1 = Completely irrelevant\n"
            "2 = Tangential, unlikely to help answer query\n"
            "3 = Somewhat relevant, partial answer\n"
            "4 = Relevant, contributes to answer\n"
            "5 = Highly relevant, directly addresses query\n\n"
            "Respond with ONLY a JSON object mapping paper numbers to their scores and brief reasoning.\n"
            'Format: {"scores": {"1": {"score": N, "reason": "..."}, "2": {"score": N, "reason": "..."}, ...}}\n'
            'Include a "reason" explaining why this score was given.\n'
            "Only include papers that exist in the list above."
        )

        try:
            response = await self.llm.complete(prompt, temperature=0.1)
            import json, re

            # Parse JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                logger.warning(
                    f"Could not parse relevance scores from LLM response: {response[:200]}"
                )
                return papers  # Fall back to returning all papers

            scores_data = json.loads(json_match.group())
            scores = scores_data.get("scores", {})

            # Filter and annotate papers
            filtered = []
            for i, paper in enumerate(papers, 1):
                paper_key = str(i)
                if paper_key in scores:
                    score_info = scores[paper_key]
                    score = (
                        score_info.get("score", 3)
                        if isinstance(score_info, dict)
                        else int(score_info)
                    )
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = (
                        score_info.get("reason", "") if isinstance(score_info, dict) else ""
                    )

                    if score >= min_score:
                        filtered.append(paper)
                        logger.info(
                            f"Paper [{i}] '{paper.get('title', '')[:50]}...' score: {score} - INCLUDED"
                        )
                    else:
                        logger.info(
                            f"Paper [{i}] '{paper.get('title', '')[:50]}...' score: {score} - FILTERED"
                        )
                else:
                    # Default to included if no score found
                    paper["relevance_score"] = 3
                    paper["relevance_reason"] = "No score provided"
                    filtered.append(paper)

            logger.info(
                f"Relevance filtering: {len(filtered)}/{len(papers)} papers included (min_score={min_score})"
            )
            return filtered

        except Exception as e:
            logger.error(f"Error scoring papers for relevance: {e}")
            return papers  # Fall back to returning all papers on error

    def _format_references_section(self, papers: List[Dict[str, Any]]) -> str:
        """Format a references section in academic citation style using shared utility.

        Uses markdown link format: [Author et al., Year](url "full citation")
        Based on the style from Perspicacite Profonde.
        """
        return format_references_academic(papers)

    def _summarize_result(self, result: Any) -> str:
        """Create a brief summary of a result for UI display."""
        result_str = str(result)
        if len(result_str) > 100:
            return result_str[:100] + "..."
        return result_str

    async def _fallback_openalex_search(self, query: str, max_results: int = 10) -> str:
        """Search OpenAlex directly via httpx. Query should already be
        cleaned by the planner (conversational preamble stripped)."""
        import httpx

        search_terms = query.strip()
        logger.info(f"OpenAlex search: '{search_terms}'")

        url = "https://api.openalex.org/works"
        params = {
            "search": search_terms,
            "per_page": max_results,
            "mailto": "perspicacite@example.com",
            "select": "id,display_name,authorships,publication_year,cited_by_count,abstract_inverted_index,doi,open_access",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                data = response.json()

                papers = []
                for result in data.get("results", []):
                    # Reconstruct abstract from inverted index
                    abstract = self._reconstruct_abstract(result.get("abstract_inverted_index"))
                    
                    paper = {
                        "id": result.get("id", ""),
                        "title": result.get("display_name", "Untitled"),
                        "authors": [
                            auth.get("author", {}).get("display_name", "")
                            for auth in result.get("authorships", [])[:3]
                        ],
                        "year": result.get("publication_year"),
                        "cited_by_count": result.get("cited_by_count", 0),
                        "abstract": abstract[:800] if abstract else "",
                        "doi": result.get("doi", ""),
                        "open_access": result.get("open_access", {}),
                    }
                    papers.append(paper)

                papers = self._dedupe_paper_dicts(papers)

                # Accumulate for papers_found event with source
                if hasattr(self, "_found_papers"):
                    for p in papers:
                        p["source"] = "literature_search"
                    self._found_papers.extend(papers)

                return self._format_paper_list(papers)
        except Exception as e:
            return f"OpenAlex search failed: {e}"

    def _format_paper_list(self, papers: list) -> str:
        """Format a list of paper dicts into a readable string."""
        if not papers:
            return "No papers found."

        lines = [f"Found {len(papers)} papers:"]
        for i, paper in enumerate(papers, 1):
            lines.append(f"\n{i}. {paper['title']}")
            if paper["authors"]:
                lines.append(f"   Authors: {', '.join(paper['authors'])}")
            if paper["year"]:
                lines.append(f"   Year: {paper['year']}")
            cited = paper.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Citations: {cited}")
            if paper["doi"]:
                lines.append(f"   DOI: {paper['doi']}")
            if paper["abstract"]:
                lines.append(f"   Abstract: {paper['abstract'][:200]}...")

        return "\n".join(lines)

    def _summarize_findings(self, findings: List[Dict]) -> str:
        """Summarize previous research findings."""
        if not findings:
            return ""

        summaries = []
        for finding in findings[-3:]:  # Last 3 findings
            topic = finding.get("topic", "Unknown")
            result = finding.get("result", "")
            summaries.append(f"{topic}: {str(result)[:100]}")

        return "\n".join(summaries)

    def _format_papers(self, papers: list) -> str:
        """Format list of Paper models into readable string."""
        from perspicacite.models.papers import Paper

        if not papers:
            return "No papers found."

        lines = [f"Found {len(papers)} papers:"]
        for i, paper in enumerate(papers, 1):
            lines.append(f"\n{i}. {paper.title}")
            if paper.authors:
                author_names = [a.name for a in paper.authors[:3]]
                lines.append(f"   Authors: {', '.join(author_names)}")
            if paper.year:
                lines.append(f"   Year: {paper.year}")
            if paper.journal:
                lines.append(f"   Journal: {paper.journal}")
            if paper.doi:
                lines.append(f"   DOI: {paper.doi}")
            if paper.abstract:
                lines.append(f"   Abstract: {paper.abstract[:200]}...")

        # Accumulate for papers_found event
        if hasattr(self, "_found_papers"):
            for paper in papers:
                self._found_papers.append(
                    {
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors[:3]],
                        "year": paper.year,
                        "doi": paper.doi,
                        "abstract": paper.abstract[:300] if paper.abstract else "",
                        "citations": paper.citation_count,
                        "source": "kb_search",
                    }
                )

        return "\n".join(lines)

    @staticmethod
    def _normalize_doi_for_dedupe(doi: Any) -> str:
        if not doi:
            return ""
        d = str(doi).strip().lower()
        for prefix in ("https://doi.org/", "http://dx.doi.org/", "doi:"):
            if d.startswith(prefix):
                d = d[len(prefix) :].strip()
        return d

    def _paper_dedupe_key(self, p: Dict[str, Any]) -> str:
        """Prefer long title fingerprint so journal + bioRxiv (different DOIs) merge."""
        title = (p.get("title") or "").lower()
        fp = re.sub(r"[^a-z0-9]+", "", title)[:120]
        if len(fp) >= 40:
            return f"title:{fp}"
        d = self._normalize_doi_for_dedupe(p.get("doi"))
        if d:
            return f"doi:{d}"
        oid = (p.get("id") or "").strip()
        if oid:
            return f"oa:{oid}"
        if fp:
            return f"title:{fp}"
        return f"unknown:{id(p)}"

    def _paper_quality_tuple(self, p: Dict[str, Any]) -> tuple:
        """Higher is better: more abstract, more citations, newer, non-bioRxiv DOI."""
        doi = self._normalize_doi_for_dedupe(p.get("doi"))
        is_biorxiv = doi.startswith("10.1101") if doi else False
        return (
            len(p.get("abstract") or ""),
            p.get("cited_by_count") or 0,
            p.get("year") or 0,
            0 if is_biorxiv else 1,
        )

    def _dedupe_paper_dicts(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicates (same DOI or same normalized title, e.g. preprint + journal).
        
        Keeps the highest quality version of each paper (more abstract, more citations, newer).
        The winning paper retains its original source (kb_search or literature_search).
        """
        best: Dict[str, Dict[str, Any]] = {}
        
        for p in papers:
            k = self._paper_dedupe_key(p)
            if k.startswith("unknown:") and not p.get("title"):
                continue
            
            if k not in best or self._paper_quality_tuple(p) > self._paper_quality_tuple(best[k]):
                best[k] = p
        
        out = list(best.values())
        if len(out) < len(papers):
            logger.info(
                f"Paper dedupe: {len(papers)} -> {len(out)} (by DOI / OpenAlex id / title fingerprint)"
            )
        return out

    @staticmethod
    def _normalize_authors(authors: Any) -> List[str]:
        """Normalize authors to a list of strings.
        
        Handles various input formats:
        - List of strings: ["Author One", "Author Two"]
        - Comma-separated string: "Author One, Author Two"
        - Single string: "Author One"
        - None/empty: []
        """
        if not authors:
            return []
        if isinstance(authors, list):
            return [str(a).strip() for a in authors if a]
        if isinstance(authors, str):
            return [a.strip() for a in authors.split(",") if a.strip()]
        return []

    def _extract_papers_from_results(self, step_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract deduplicated paper list from accumulated found papers."""
        if not hasattr(self, "_found_papers") or not self._found_papers:
            return []
        
        # Normalize authors for all papers before deduplication
        for paper in self._found_papers:
            paper["authors"] = self._normalize_authors(paper.get("authors"))
        
        return self._dedupe_paper_dicts(list(self._found_papers))

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict | None) -> str:
        """Reconstruct abstract text from OpenAlex inverted index format.
        
        OpenAlex stores abstracts as {"word": [positions]} to save space.
        This reconstructs the original text.
        """
        if not inverted_index:
            return ""
        
        # Build position -> word mapping
        position_words = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                position_words[pos] = word
        
        # Sort by position and join
        if not position_words:
            return ""
        
        max_pos = max(position_words.keys())
        words = []
        for i in range(max_pos + 1):
            words.append(position_words.get(i, ""))
        
        return " ".join(words).strip()

    async def _download_single_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Download and parse a single paper's PDF.
        
        Args:
            paper: Paper dict with at least 'doi' and 'title'
            
        Returns:
            Enriched paper dict with 'full_text' and 'pdf_downloaded' fields
        """
        from perspicacite.pipeline.download import get_pdf_with_fallback
        from perspicacite.pipeline.parsers.pdf import PDFParser
        
        doi = paper.get("doi", "")
        if not doi:
            paper["pdf_downloaded"] = False
            return paper
        
        parser = PDFParser()
        
        try:
            pdf_bytes = await get_pdf_with_fallback(
                doi=doi,
                unpaywall_email="perspicacite@example.com",
            )
            
            if pdf_bytes:
                parsed = await parser.parse_bytes(pdf_bytes)
                # Limit full text to avoid token overload
                paper["full_text"] = parsed.text[:15000] if parsed.text else ""
                paper["pdf_downloaded"] = True
            else:
                paper["pdf_downloaded"] = False
                
        except Exception as e:
            logger.warning(f"Failed to download/parse PDF for {doi}: {e}")
            paper["pdf_downloaded"] = False
        
        return paper

    async def _download_and_enrich_papers(
        self, 
        papers: List[Dict[str, Any]], 
        max_papers: int = 5
    ) -> List[Dict[str, Any]]:
        """Download PDFs for relevant papers and extract full text.
        
        Args:
            papers: List of paper dicts with DOI
            max_papers: Maximum papers to download (default 5)
            
        Returns:
            Papers with added 'full_text' field where download succeeded
        """
        # Only try to download for papers with DOI, prioritize open access
        download_candidates = [
            p for p in papers[:max_papers] 
            if p.get("doi") and p.get("open_access", {}).get("is_oa", False)
        ]
        
        if not download_candidates:
            # If no OA papers, try the top relevant papers anyway
            download_candidates = [p for p in papers[:max_papers] if p.get("doi")]
        
        logger.info(f"Attempting to download {len(download_candidates)} papers for full text enrichment")
        
        enriched = []
        for paper in papers:
            doi = paper.get("doi", "")
            if not doi or paper not in download_candidates:
                enriched.append(paper)
                continue
            
            enriched_paper = await self._download_single_paper(paper)
            enriched.append(enriched_paper)
        
        return enriched
