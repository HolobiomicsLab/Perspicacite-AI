"""Main agentic orchestrator with session management."""

import uuid
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from .intent import IntentClassifier, Intent
from .planner import ResearchPlanner, Step, StepType, Plan
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

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
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata or {}
        ))
        self.last_active = datetime.now()
    
    def get_conversation_history(self, limit: int = 10) -> List[dict]:
        """Get conversation history as list of dicts."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-limit:]
        ]
    
    def get_context_string(self) -> str:
        """Get recent conversation context as string."""
        context = []
        for msg in self.messages[-4:]:
            context.append(f"{msg.role}: {msg.content[:300]}")
        return "\n".join(context)


class AgenticOrchestrator:
    """
    True agentic orchestrator with LLM-driven planning and execution.
    """
    
    def __init__(
        self,
        llm_client,
        tool_registry,
        embedding_provider,
        vector_store,
        max_iterations: int = 5
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.embeddings = embedding_provider
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        
        self.intent_classifier = IntentClassifier(llm_client)
        self.planner = ResearchPlanner(llm_client)
        
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
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main chat entry point with true agentic behavior.
        
        Yields:
            Dict with type: "thinking", "tool_call", "tool_result", "answer", "papers_found"
        """
        logger.info("=" * 80)
        logger.info("NEW CHAT REQUEST")
        logger.info(f"Query: {query}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"KB: {kb_name or 'none'}")
        
        session = self.get_or_create_session(session_id)
        session.add_message("user", query)
        session.kb_name = kb_name
        logger.info(f"Session messages count: {len(session.messages)}")
        
        # Step 1: Classify intent
        yield {"type": "thinking", "message": "Analyzing your query..."}
        
        intent_result = await self.intent_classifier.classify(
            query=query,
            conversation_history=session.get_conversation_history()
        )
        logger.info(f"Intent classified: {intent_result.intent.name}")
        logger.info(f"Confidence: {intent_result.confidence}")
        logger.info(f"Suggested tools: {intent_result.suggested_tools}")
        
        yield {
            "type": "thinking", 
            "message": f"Intent: {intent_result.intent.name.replace('_', ' ').title()}",
            "details": intent_result.reasoning
        }
        
        # Step 2: Create dynamic plan
        yield {"type": "thinking", "message": "Creating research plan..."}
        
        # Available tools: registered tools (excluding deactivated ones) + built-in
        available_tools = [t for t in self.tools.list_tools() if t != "lotus_search"] + ["openalex_search", "kb_search"]
        logger.info(f"Available tools: {available_tools}")
        previous_findings = self._summarize_findings(session.research_findings)
        
        plan = await self.planner.create_plan(
            query=query,
            intent_result=intent_result,
            available_tools=available_tools,
            conversation_history=session.get_conversation_history(),
            previous_findings=previous_findings
        )
        
        # If a KB is selected, always search it first (don't rely on the LLM planner)
        if kb_name:
            has_kb_step = any(s.type == StepType.KB_SEARCH for s in plan.steps)
            if not has_kb_step:
                from perspicacite.rag.agentic.planner import ResearchPlanner
                clean_query = ResearchPlanner._clean_query_for_search(query)
                kb_step = Step(
                    id="kb_search",
                    type=StepType.KB_SEARCH,
                    description=f"Search knowledge base '{kb_name}'",
                    tool="kb_search",
                    tool_input={"query": clean_query},
                )
                plan.steps.insert(0, kb_step)
                plan.estimated_steps = len(plan.steps)
                logger.info(f"Injected kb_search step for KB '{kb_name}'")
        
        logger.info(f"Plan created with {len(plan.steps)} steps")
        for i, step in enumerate(plan.steps):
            logger.info(f"  Step {i+1}: {step.id} - {step.type.value} - {step.description}")
        
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
            
            logger.info(f"Next step: {next_step.id} ({next_step.type.value}) - {next_step.description}")
            
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
            
            logger.info(f"Executing step {next_step.id}...")
            result = await self._execute_step(
                next_step, 
                query, 
                step_results,
                session
            )
            
            # Log the result
            result_str = str(result)
            logger.info(f"Step {next_step.id} completed")
            logger.info(f"Result length: {len(result_str)} chars")
            logger.info(f"Result preview: {result_str[:500]}...")
            
            step_results[next_step.id] = result
            completed_steps.append(next_step)
            
            yield {
                "type": "tool_result",
                "step": next_step.id,
                "result_summary": self._summarize_result(result)
            }
            
            # Evaluate if we need to replan
            if next_step.type in (StepType.LOTUS_SEARCH, StepType.OPENALEX_SEARCH, StepType.KB_SEARCH):
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
        
        # Step 4: Generate final answer
        yield {"type": "thinking", "message": "Synthesizing answer..."}
        
        answer = await self._generate_answer(
            query=query,
            plan=plan,
            step_results=step_results,
            session=session
        )
        
        session.add_message("assistant", answer, {
            "intent": intent_result.intent.name,
            "steps_completed": len(completed_steps),
            "tools_used": [s.tool for s in completed_steps if s.tool]
        })
        
        yield {"type": "answer", "content": answer, "session_id": session.session_id}
        
        # Yield found papers so the UI can offer "Add to KB"
        found_papers = self._extract_papers_from_results(step_results)
        if found_papers:
            yield {"type": "papers_found", "papers": found_papers}
    
    def _get_next_step(
        self, 
        plan: Plan, 
        completed: List[Step],
        results: Dict[str, Any]
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
        self,
        step: Step,
        original_query: str,
        step_results: Dict[str, Any],
        session: AgentSession
    ) -> Any:
        """Execute a single step."""
        
        if step.type == StepType.LOTUS_SEARCH:
            logger.info("LOTUS_SEARCH: skipped (deactivated)")
            return "LOTUS search is currently deactivated."
        
        elif step.type == StepType.OPENALEX_SEARCH:
            query = step.tool_input.get("query", original_query)
            logger.info(f"OPENALEX_SEARCH: query='{query}'")
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
                    collection_name = f"kb_{session.kb_name}"
                    logger.info(f"KB_SEARCH: Searching collection '{collection_name}'")
                    query = step.tool_input.get("query", original_query)
                    
                    dkb = DynamicKnowledgeBase(
                        vector_store=self.vector_store,
                        embedding_service=self.embeddings,
                    )
                    dkb.collection_name = collection_name
                    dkb._initialized = True
                    
                    results = await dkb.search(query, top_k=5)
                    logger.info(f"KB_SEARCH: Found {len(results)} results")
                    
                    if results:
                        formatted_parts = [f"Found {len(results)} relevant documents in knowledge base:"]
                        for i, r in enumerate(results, 1):
                            title = r.get("metadata", {}).title if hasattr(r.get("metadata", {}), "title") else "Unknown"
                            score = r.get("score", 0)
                            text_preview = r.get("text", "")[:200]
                            formatted_parts.append(f"\n{i}. {title} (relevance: {score:.2f})")
                            formatted_parts.append(f"   {text_preview}...")
                        return "\n".join(formatted_parts)
                    return "No relevant documents found in knowledge base."
                except Exception as e:
                    logger.error(f"KB_SEARCH failed: {e}", exc_info=True)
                    return "Knowledge base search failed."
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
    
    async def _evaluate_progress(
        self,
        query: str,
        plan: Plan,
        completed_steps: List[Step],
        step_results: Dict[str, Any]
    ) -> str:
        """Evaluate if we should continue, replan, or answer."""
        
        # Simple heuristic: if we have good results, answer
        has_substantial_results = False
        for result in step_results.values():
            result_str = str(result)
            if len(result_str) > 200 and "error" not in result_str.lower():
                has_substantial_results = True
                break
        
        if has_substantial_results and len(completed_steps) >= 2:
            return "answer"
        
        return "continue"
    
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
        self,
        query: str,
        plan: Plan,
        step_results: Dict[str, Any],
        session: AgentSession
    ) -> str:
        """Generate final answer using all results."""

        logger.info("\n--- Generating Answer ---")
        logger.info(f"Query: {query}")
        logger.info(f"Step results available: {list(step_results.keys())}")

        # Extract papers and score for relevance
        papers = self._extract_papers_from_results(step_results)
        papers = await self._score_papers_for_relevance(query, papers, min_score=3)
        numbered_paper_list = self._build_numbered_paper_list(papers)

        # Build context from step results
        context_parts = []

        # Prioritize certain result types
        if "lotus" in step_results:
            lotus_result = step_results['lotus']
            logger.info(f"LOTUS result length: {len(str(lotus_result))} chars")
            context_parts.append(f"LOTUS Search Results:\n{lotus_result}")

        for step_id, result in step_results.items():
            if step_id != "lotus" and result:
                result_str = str(result)
                logger.info(f"Step {step_id} result length: {len(result_str)} chars")
                context_parts.append(f"{step_id}:\n{result_str[:3000]}")

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

{numbered_paper_list if numbered_paper_list else ''}

Answer Generation Guidelines:
1. Focus on answering the SPECIFIC question asked - avoid tangential information
2. Prioritize the most relevant findings from the research results
3. Maintain scientific precision and technical accuracy
4. **MANDATORY CITATION FORMAT**: When referencing any paper, you MUST use the bracket format [N] where N is the paper number from the numbered list above (e.g., [1], [2], [3]). ALWAYS use this format for paper citations - do NOT use author-year or other citation styles.
5. Be clear and direct in your language
6. If the research results are insufficient to answer the question, clearly state this rather than speculating
7. Structure your answer logically with clear sections if appropriate
8. List ALL papers [1] through [{len(papers)}] in your answer if they are relevant - do not skip or group papers

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

    def _build_numbered_paper_list(self, papers: List[Dict[str, Any]], max_abstract_chars: int = 800) -> str:
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
                abstract = abstract[:max_abstract_chars].rsplit(' ', 1)[0] + "..."

            lines.append(f"[{i}] {title}")
            lines.append(f"    Authors: {author_str}")
            lines.append(f"    Year: {year}")
            if doi:
                lines.append(f"    DOI: {doi}")
            if abstract:
                lines.append(f"    Abstract: {abstract}")

        return "\n".join(lines)

    async def _score_papers_for_relevance(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        min_score: int = 3
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

        prompt = """You are evaluating research papers for relevance to a user's query.

User Query: "{query}"

Papers to evaluate:
{paper_list_str}

For each paper, score its relevance to the query on this scale:
1 = Completely irrelevant
2 = Tangential, unlikely to help answer query
3 = Somewhat relevant, partial answer
4 = Relevant, contributes to answer
5 = Highly relevant, directly addresses query

Respond with ONLY a JSON object mapping paper numbers to their scores and brief reasoning.
Format: {{"scores": {{"1": {{"score": N, "reason": "..."}}, "2": {{"score": N, "reason": "..."}}, ...}}}
Include a "reason" explaining why this score was given.
Only include papers that exist in the list above.""".format(query=query, paper_list_str=paper_list_str)

        try:
            response = await self.llm.complete(prompt, temperature=0.1)
            import json, re

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                logger.warning(f"Could not parse relevance scores from LLM response: {response[:200]}")
                return papers  # Fall back to returning all papers

            scores_data = json.loads(json_match.group())
            scores = scores_data.get("scores", {})

            # Filter and annotate papers
            filtered = []
            for i, paper in enumerate(papers, 1):
                paper_key = str(i)
                if paper_key in scores:
                    score_info = scores[paper_key]
                    score = score_info.get("score", 3) if isinstance(score_info, dict) else int(score_info)
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = score_info.get("reason", "") if isinstance(score_info, dict) else ""

                    if score >= min_score:
                        filtered.append(paper)
                        logger.info(f"Paper [{i}] '{paper.get('title', '')[:50]}...' score: {score} - INCLUDED")
                    else:
                        logger.info(f"Paper [{i}] '{paper.get('title', '')[:50]}...' score: {score} - FILTERED")
                else:
                    # Default to included if no score found
                    paper["relevance_score"] = 3
                    paper["relevance_reason"] = "No score provided"
                    filtered.append(paper)

            logger.info(f"Relevance filtering: {len(filtered)}/{len(papers)} papers included (min_score={min_score})")
            return filtered

        except Exception as e:
            logger.error(f"Error scoring papers for relevance: {e}")
            return papers  # Fall back to returning all papers on error

    def _format_references_section(self, papers: List[Dict[str, Any]]) -> str:
        """Format a references section in academic citation style.

        Uses markdown link format: [Author et al., Year](url "full citation")
        Based on the style from Perspicacite Profonde.
        """
        if not papers:
            return ""

        ref_lines = ["## References\n"]

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown Title")
            authors = paper.get("authors", [])
            year = paper.get("year", "n.d.")
            doi = paper.get("doi", "")
            url = f"https://doi.org/{doi}" if doi else ""

            # Format authors: "FirstAuthor et al." if >2 authors, else "FirstAuthor & SecondAuthor"
            if len(authors) == 0:
                author_str = "Unknown"
            elif len(authors) == 1:
                author_str = authors[0]
            elif len(authors) == 2:
                author_str = f"{authors[0]} & {authors[1]}"
            else:
                author_str = f"{authors[0]} et al."

            # Format full citation (for title attribute of the link)
            if authors:
                full_citation = f"{', '.join(authors)}. {year}. {title}."
            else:
                full_citation = f"{title}. {year}."

            # Use markdown link format: [Author et al., Year](url "full citation")
            if url:
                ref_lines.append(f"{i}. [{author_str}, {year}]({url} \"{full_citation}\")")
            else:
                ref_lines.append(f"{i}. {author_str}, {year}. {title}.")

        return "\n".join(ref_lines)

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
            "mailto": "perspicacite@example.com"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                data = response.json()
                
                papers = []
                for result in data.get("results", []):
                    paper = {
                        "id": result.get("id", ""),
                        "title": result.get("display_name", "Untitled"),
                        "authors": [
                            auth.get("author", {}).get("display_name", "")
                            for auth in result.get("authorships", [])[:3]
                        ],
                        "year": result.get("publication_year"),
                        "cited_by_count": result.get("cited_by_count", 0),
                        "abstract": result.get("abstract", "")[:500] if result.get("abstract") else "",
                        "doi": result.get("doi", "")
                    }
                    papers.append(paper)
                
                # Accumulate for papers_found event
                if hasattr(self, "_found_papers"):
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
            if paper['authors']:
                lines.append(f"   Authors: {', '.join(paper['authors'])}")
            if paper['year']:
                lines.append(f"   Year: {paper['year']}")
            cited = paper.get('cited_by_count')
            if cited is not None:
                lines.append(f"   Citations: {cited}")
            if paper['doi']:
                lines.append(f"   DOI: {paper['doi']}")
            if paper['abstract']:
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
                self._found_papers.append({
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors[:3]],
                    "year": paper.year,
                    "doi": paper.doi,
                    "abstract": paper.abstract[:300] if paper.abstract else "",
                    "citations": paper.citation_count,
                })
        
        return "\n".join(lines)

    def _extract_papers_from_results(self, step_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract deduplicated paper list from accumulated found papers."""
        if not hasattr(self, "_found_papers") or not self._found_papers:
            return []
        
        seen_titles = set()
        unique = []
        for p in self._found_papers:
            key = p.get("title", "").lower().strip()
            if key and key not in seen_titles:
                seen_titles.add(key)
                unique.append(p)
        return unique
