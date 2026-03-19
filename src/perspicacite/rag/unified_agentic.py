"""Unified Agentic RAG for scientific literature information retrieval.

Combines the best of:
- AgenticOrchestrator: Intent classification, dynamic planning, session management, tool execution
- AgenticRAGMode: Document quality assessment, early exit, self-evaluation

This is a SINGLE agentic system for information retrieval from:
- Knowledge base (vector search)
- Academic papers (OpenAlex, Semantic Scholar via SciLEx)
- Natural products (LOTUS database)
"""

from __future__ import annotations

import uuid
import json
import re
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional

import httpx

from perspicacite.logging import get_logger

if TYPE_CHECKING:
    from perspicacite.llm.client import AsyncLLMClient
    from perspicacite.retrieval.chroma_store import ChromaVectorStore
    from perspicacite.llm.embeddings import EmbeddingProvider

logger = get_logger("perspicacite.rag.unified_agentic")


# =============================================================================
# Intent Classification
# =============================================================================

class Intent(Enum):
    """User query intent types."""
    NATURAL_PRODUCTS_ONLY = auto()
    PAPERS_ONLY = auto()
    COMBINED_RESEARCH = auto()
    FOLLOW_UP = auto()
    CLARIFICATION = auto()
    ANALYSIS = auto()
    UNKNOWN = auto()


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    reasoning: str
    suggested_tools: list[str]
    entities: dict[str, Any]


class IntentClassifier:
    """Classifies user intent using LLM with keyword fallback."""

    def __init__(self, llm: AsyncLLMClient):
        self.llm = llm

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling markdown code blocks."""
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1) if '```' in pattern else match.group(0)
        text = text.strip()
        if text.startswith('{') and text.endswith('}'):
            return text
        return None

    async def classify(
        self,
        query: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> IntentResult:
        """Classify user intent from query."""
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-4:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                history_context += f"{role}: {content}\n"

        prompt = f"""You are an intent classifier for a scientific research assistant.

Query: "{query}"{history_context}

Return JSON:
{{
    "intent": "natural_products_only|papers_only|combined_research|follow_up|clarification|analysis|unknown",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "suggested_tools": ["tool1", "tool2"],
    "entities": {{"compounds": [], "organisms": [], "topics": []}}
}}"""

        try:
            response = await self.llm.complete(prompt, temperature=0.1)
            json_str = self._extract_json(response)
            if not json_str:
                return self._keyword_fallback(query)

            result = json.loads(json_str)
            intent_map = {
                "natural_products_only": Intent.NATURAL_PRODUCTS_ONLY,
                "papers_only": Intent.PAPERS_ONLY,
                "combined_research": Intent.COMBINED_RESEARCH,
                "follow_up": Intent.FOLLOW_UP,
                "clarification": Intent.CLARIFICATION,
                "analysis": Intent.ANALYSIS,
                "unknown": Intent.UNKNOWN,
            }
            return IntentResult(
                intent=intent_map.get(result["intent"], Intent.UNKNOWN),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                suggested_tools=result.get("suggested_tools", []),
                entities=result.get("entities", {}),
            )
        except Exception as e:
            logger.error("intent_classification_error", error=str(e))
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> IntentResult:
        """Simple keyword-based fallback."""
        q = query.lower()
        lotus_kw = ["lotus", "natural product", "compound", "metabolite", "structure", "chemical"]
        paper_kw = ["paper", "article", "research", "literature", "study", "publication"]

        if any(kw in q for kw in lotus_kw):
            return IntentResult(
                intent=Intent.COMBINED_RESEARCH,
                confidence=0.7,
                reasoning="Keyword: natural products mentioned",
                suggested_tools=["lotus_search", "openalex_search"],
                entities={},
            )
        if any(kw in q for kw in paper_kw):
            return IntentResult(
                intent=Intent.PAPERS_ONLY,
                confidence=0.8,
                reasoning="Keyword: explicit paper search",
                suggested_tools=["openalex_search"],
                entities={},
            )
        return IntentResult(
            intent=Intent.COMBINED_RESEARCH,
            confidence=0.6,
            reasoning="Default: combined research",
            suggested_tools=["openalex_search", "kb_search"],
            entities={},
        )


# =============================================================================
# Research Planning
# =============================================================================

class StepType(Enum):
    """Types of research steps."""
    LOTUS_SEARCH = "lotus_search"
    OPENALEX_SEARCH = "openalex_search"
    KB_SEARCH = "kb_search"
    WEB_SEARCH = "web_search"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    ANSWER = "answer"


@dataclass
class Step:
    """A single research step."""
    id: str
    type: StepType
    description: str
    tool: Optional[str] = None
    tool_input: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    condition: Optional[str] = None


@dataclass
class Plan:
    """A research plan."""
    steps: list[Step]
    reasoning: str
    estimated_steps: int
    can_answer_from_history: bool = False


class ResearchPlanner:
    """Generates dynamic research plans using LLM."""

    def __init__(self, llm: AsyncLLMClient):
        self.llm = llm

    async def create_plan(
        self,
        query: str,
        intent_result: IntentResult,
        available_tools: list[str],
        conversation_history: Optional[list[dict]] = None,
        previous_findings: Optional[str] = None,
    ) -> Plan:
        """Create a dynamic research plan."""
        context_parts = []
        if conversation_history:
            context_parts.append("Previous conversation:")
            for msg in conversation_history[-3:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:300]
                context_parts.append(f"  {role}: {content}")
        if previous_findings:
            context_parts.append(f"\nPrevious findings:\n{previous_findings[:500]}")

        context = "\n".join(context_parts)

        prompt = f"""Create a research plan for this query.

Query: "{query}"
Intent: {intent_result.intent.name} (confidence: {intent_result.confidence:.2f})
Reasoning: {intent_result.reasoning}
Entities: {intent_result.entities}

Available Tools: {available_tools}

{context}

Step Types:
- lotus_search: Natural products, chemical structures
- openalex_search: Academic papers via OpenAlex/Semantic Scholar
- kb_search: Search existing knowledge base
- analyze: Process and extract insights
- synthesize: Combine multiple sources
- answer: Generate final response

GUIDELINES FOR BROAD QUERIES:
If the query is broad (like "tell me about X"), create MULTIPLE focused search steps:
- One search for core concepts/methodology
- One search for applications/use cases
- One search for key papers/reviews

GUIDELINES FOR QUERY CLEANING:
- The "query" field in tool_input must be a CLEAN, SEARCH-FRIENDLY string
- Use ONLY terms from the original query — do NOT add new terms like "metabolomics" unless explicitly mentioned
- Remove conversational preamble: "I want to learn about", "tell me about", "what is", etc.
- Use technical/scientific keywords from the original query
- Keep it concise (under 20 words)
- Example: Query "I want to learn about feature based molecular networking" → "feature-based molecular networking"
- Example: Query "what are the applications of GNPS" → "GNPS molecular networking applications"
- WRONG: Adding "metabolomics" when user didn't mention it

Return JSON:
{{
    "reasoning": "research strategy",
    "can_answer_from_history": false,
    "steps": [
        {{
            "id": "step1",
            "type": "lotus_search|openalex_search|kb_search|analyze|synthesize|answer",
            "description": "what this step does",
            "tool": "optional_tool_name",
            "tool_input": {{"query": "CLEAN search query using ONLY terms from original query"}},
            "depends_on": [],
            "condition": null
        }}
    ]
}}"""

        try:
            response = await self.llm.complete(prompt, temperature=0.2)
            result = json.loads(response.strip())

            steps = []
            for step_data in result.get("steps", []):
                steps.append(Step(
                    id=step_data["id"],
                    type=StepType(step_data.get("type", "answer")),
                    description=step_data["description"],
                    tool=step_data.get("tool"),
                    tool_input=step_data.get("tool_input", {}),
                    depends_on=step_data.get("depends_on", []),
                    condition=step_data.get("condition"),
                ))

            return Plan(
                steps=steps,
                reasoning=result.get("reasoning", ""),
                estimated_steps=len(steps),
                can_answer_from_history=result.get("can_answer_from_history", False),
            )
        except Exception as e:
            logger.error("plan_creation_error", error=str(e))
            return self._fallback_plan(query, available_tools)

    def _fallback_plan(self, query: str, available_tools: list[str]) -> Plan:
        """Create a simple fallback plan."""
        steps = []
        step_id = 1

        if "lotus_search" in available_tools:
            steps.append(Step(
                id=f"step{step_id}",
                type=StepType.LOTUS_SEARCH,
                description="Search LOTUS for natural products",
                tool="lotus_search",
                tool_input={"query": query},
            ))
            step_id += 1

        if "openalex_search" in available_tools:
            steps.append(Step(
                id=f"step{step_id}",
                type=StepType.OPENALEX_SEARCH,
                description="Search academic papers",
                tool="openalex_search",
                tool_input={"query": query},
                depends_on=[f"step{step_id-1}"] if step_id > 1 else [],
            ))
            step_id += 1

        if "kb_search" in available_tools:
            steps.append(Step(
                id=f"step{step_id}",
                type=StepType.KB_SEARCH,
                description="Search knowledge base",
                tool="kb_search",
                tool_input={"query": query},
            ))
            step_id += 1

        steps.append(Step(
            id=f"step{step_id}",
            type=StepType.ANSWER,
            description="Generate answer",
            depends_on=[s.id for s in steps[-1:]] if steps else [],
        ))

        return Plan(
            steps=steps,
            reasoning="Fallback plan due to planning error",
            estimated_steps=len(steps),
        )

    async def replan(
        self,
        query: str,
        current_plan: Plan,
        completed_steps: list[Step],
        step_results: dict[str, Any],
        evaluation: str,
    ) -> Plan:
        """Replan based on evaluation of current results."""
        results_summary = []
        for step in completed_steps:
            result = step_results.get(step.id, "No result")
            results_summary.append(f"{step.id}: {str(result)[:200]}")

        prompt = f"""Evaluate and possibly modify the research plan.

Query: "{query}"
Evaluation: {evaluation}

Completed:
{chr(10).join(results_summary)}

Return JSON:
{{
    "action": "continue|add_steps|answer",
    "reasoning": "why",
    "additional_steps": [{{"id": "new1", "type": "...", "description": "...", "tool": "...", "tool_input": {{}}, "depends_on": []}}]
}}"""

        try:
            response = await self.llm.complete(prompt, temperature=0.2)
            result = json.loads(response.strip())

            action = result.get("action", "continue")

            if action == "add_steps":
                for step_data in result.get("additional_steps", []):
                    current_plan.steps.append(Step(
                        id=step_data["id"],
                        type=StepType(step_data["type"]),
                        description=step_data["description"],
                        tool=step_data.get("tool"),
                        tool_input=step_data.get("tool_input", {}),
                        depends_on=step_data.get("depends_on", []),
                    ))
                current_plan.reasoning += f"\nReplanned: {result.get('reasoning', '')}"

            elif action == "answer":
                current_plan.steps = completed_steps + [Step(
                    id="final_answer",
                    type=StepType.ANSWER,
                    description="Generate final answer",
                    depends_on=[s.id for s in completed_steps],
                )]

            return current_plan
        except Exception as e:
            logger.error("replan_error", error=str(e))
            return current_plan


# =============================================================================
# Document Quality Assessment (from AgenticRAGMode)
# =============================================================================

class DocumentQualityAssessor:
    """Assess if retrieved documents are sufficient to answer a query."""

    def __init__(self, llm: AsyncLLMClient):
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

        doc_texts = []
        for i, doc in enumerate(documents[:5]):
            if hasattr(doc, 'chunk'):
                text = doc.chunk.text[:500] if hasattr(doc.chunk, 'text') else str(doc.chunk)[:500]
            else:
                text = str(doc)[:500]
            doc_texts.append(f"Document {i+1}:\n{text}")

        doc_content = "\n\n---\n\n".join(doc_texts)

        prompt = f"""Assess if these documents are sufficient to answer the query.

Purpose: {step_purpose or 'Answer the research question'}
Query: {query}

Documents:
{doc_content}

Return JSON:
{{
    "is_sufficient": true/false,
    "missing_aspects": ["aspect1", "aspect2"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            response = await self.llm.complete(prompt, temperature=0.0, max_tokens=300)

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
            logger.error("quality_assessment_error", error=str(e))
            return False, ["Assessment error"], 0.0

    async def is_question_answered(
        self,
        steps: list[dict[str, Any]],
        original_question: str,
    ) -> tuple[bool, float]:
        """
        Evaluate if completed research steps answer the question.

        Returns:
            Tuple of (is_answered, confidence)
        """
        if not steps:
            return False, 0.0

        steps_info = "\n\n".join([
            f"Step: {s.get('purpose', 'Unknown')}\n"
            f"Success: {s.get('success', False)}\n"
            f"Analysis: {str(s.get('analysis', ''))[:300]}..."
            for s in steps
        ])

        prompt = f"""Evaluate if research steps answer the original question.

Question: {original_question}

Research:
{steps_info}

Return JSON:
{{
    "question_answered": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "remaining_gaps": ["gap1", "gap2"]
}}"""

        try:
            response = await self.llm.complete(prompt, temperature=0.0, max_tokens=300)

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
            logger.error("is_question_answered_error", error=str(e))
            return False, 0.0


# =============================================================================
# Session Management
# =============================================================================

@dataclass
class Message:
    """A conversation message."""
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchStepResult:
    """Result of a single research step."""
    step_id: str
    step_type: StepType
    purpose: str
    tool_used: str
    success: bool
    confidence: float
    documents: list[Any] = field(default_factory=list)
    analysis: str = ""
    key_points: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    raw_result: Any = None


@dataclass
class AgentSession:
    """Persistent session for agent conversations."""
    session_id: str
    messages: list[Message] = field(default_factory=list)
    research_steps: list[ResearchStepResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Add a message to the session."""
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))

    def get_conversation_history(self, limit: int = 10) -> list[dict]:
        """Get conversation history as list of dicts."""
        return [{"role": m.role, "content": m.content} for m in self.messages[-limit:]]

    def get_context_string(self) -> str:
        """Get recent conversation context as string."""
        return "\n".join(
            f"{msg.role}: {msg.content[:300]}" for msg in self.messages[-4:]
        )

    def add_research_step(self, result: ResearchStepResult):
        """Record a research step result."""
        self.research_steps.append(result)


# =============================================================================
# Main Unified Agentic RAG
# =============================================================================

class UnifiedAgenticRAG:
    """
    Unified agentic RAG for scientific literature retrieval.

    Combines the best of both systems:
    - Intent classification for smart routing
    - Dynamic planning with dependencies
    - Document quality assessment with early exit
    - Multi-tool execution with fallback
    - Session-based conversation context
    """

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        tool_registry: dict[str, Any],
        config: Optional[dict[str, Any]] = None,
    ):
        self.llm = llm_client
        self.vector_store = vector_store
        self.embeddings = embedding_provider
        self.tools = tool_registry
        self.config = config or {}

        # Early exit confidence threshold
        self.early_exit_confidence = self.config.get("early_exit_confidence", 0.85)
        self.max_iterations = self.config.get("max_iterations", 5)

        # Initialize components
        self.intent_classifier = IntentClassifier(llm_client)
        self.planner = ResearchPlanner(llm_client)
        self.quality_assessor = DocumentQualityAssessor(llm_client)

        # Session management
        self.sessions: dict[str, AgentSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> AgentSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]

        new_session_id = session_id or str(uuid.uuid4())[:8]
        session = AgentSession(session_id=new_session_id)
        self.sessions[new_session_id] = session
        return session

    async def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        kb_name: str = "default",
        stream: bool = True,
    ) -> dict[str, Any]:
        """
        Main entry point for agentic RAG query.

        Returns a dict with:
        - answer: str
        - sources: list of SourceReference
        - session_id: str
        - intent: str
        - iterations: int
        - steps_completed: list of step info
        """
        session = self.get_or_create_session(session_id)
        session.add_message("user", query)

        logger.info("unified_agentic_query", query=query[:100], session_id=session.session_id)

        # Step 1: Classify intent
        intent_result = await self.intent_classifier.classify(
            query=query,
            conversation_history=session.get_conversation_history(),
        )
        logger.info("intent_classified", intent=intent_result.intent.name, confidence=intent_result.confidence)

        # Step 2: Create dynamic plan
        available_tools = list(self.tools.keys()) + ["openalex_search", "kb_search"]
        plan = await self.planner.create_plan(
            query=query,
            intent_result=intent_result,
            available_tools=available_tools,
            conversation_history=session.get_conversation_history(),
        )
        logger.info("plan_created", steps=len(plan.steps), reasoning=plan.reasoning)

        # Step 3: Execute plan with quality assessment and early exit
        step_results: dict[str, Any] = {}
        completed_steps_meta: list[dict[str, Any]] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info("iteration_start", iteration=iteration, max_iterations=self.max_iterations)

            # Find next executable step
            next_step = self._get_next_step(plan, completed_steps_meta, step_results)
            if not next_step:
                logger.info("no_more_steps")
                break

            # Execute step
            step_result = await self._execute_step(
                step=next_step,
                original_query=query,
                kb_name=kb_name,
                step_results=step_results,
            )
            step_results[next_step.id] = step_result.raw_result
            completed_steps_meta.append({
                "id": next_step.id,
                "purpose": next_step.description,
                "success": step_result.success,
                "analysis": step_result.analysis,
                "confidence": step_result.confidence,
            })
            session.add_research_step(step_result)

            # Quality assessment: should we continue or exit early?
            if next_step.type in (StepType.LOTUS_SEARCH, StepType.OPENALEX_SEARCH, StepType.KB_SEARCH):
                is_answered, confidence = await self.quality_assessor.is_question_answered(
                    completed_steps_meta, query
                )
                logger.info("quality_check", is_answered=is_answered, confidence=confidence)

                if is_answered and confidence >= self.early_exit_confidence:
                    logger.info("early_exit_triggered", confidence=confidence)
                    break

                # If insufficient, try to replan
                if not is_answered and iteration < self.max_iterations:
                    evaluation = f"Confidence {confidence}, need more specific search"
                    plan = await self.planner.replan(
                        query, plan, [Step(id=s["id"], type=StepType.ANSWER, description=s["purpose"])
                                      for s in completed_steps_meta], step_results, evaluation
                    )

        # Step 4: Generate final answer
        answer = await self._generate_answer(
            query=query,
            intent=intent_result,
            step_results=step_results,
            session=session,
        )

        session.add_message("assistant", answer, {
            "intent": intent_result.intent.name,
            "steps_completed": len(completed_steps_meta),
        })

        # Format sources from research steps
        sources = self._extract_sources(completed_steps_meta)

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session.session_id,
            "intent": intent_result.intent.name,
            "iterations": iteration,
            "steps_completed": completed_steps_meta,
        }

    def _get_next_step(
        self,
        plan: Plan,
        completed: list[dict[str, Any]],
        results: dict[str, Any],
    ) -> Optional[Step]:
        """Get next executable step based on dependencies."""
        completed_ids = {s["id"] for s in completed}

        for step in plan.steps:
            if step.id in completed_ids:
                continue
            if all(dep in completed_ids for dep in step.depends_on):
                return step
        return None

    def _clean_search_query(self, query: str, step_description: str = "") -> str:
        """
        Clean conversational query into search-friendly format.

        Removes conversational phrases and extracts key search terms.
        Uses both the query and step description to extract meaningful search terms.
        """
        import re

        # First, try to extract quoted phrases (often the actual topic)
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return quoted[0]

        # Remove conversational preamble
        q = query.lower()
        conversational_phrases = [
            "i want to learn about",
            "i want to know about",
            "i'm interested in",
            "i'm curious about",
            "can you tell me about",
            "tell me about",
            "what is",
            "what are",
            "what's",
            "how does",
            "how do",
            "explain",
            "describe",
            "learn about",
            "find papers about",
            "search for",
            "look up",
            "research on",
            "investigate",
        ]

        for phrase in conversational_phrases:
            if q.startswith(phrase):
                q = q[len(phrase):].strip()

        # Remove trailing prepositions and articles
        q = re.sub(r'\s+(about|on|in|for|to|with|the|a|an)\s*$', '', q, flags=re.IGNORECASE).strip()

        # If the step description contains specific search terms, use them
        if step_description:
            # Extract key terms from description (words after "on", "for", quotes)
            desc_lower = step_description.lower()
            match = re.search(r"['\"]([^'\"]+)['\"]", desc_lower)
            if match:
                return match.group(1)
            match = re.search(r'(?:search|find|look).*?(?:for|about)\s+([^,.\n]+)', desc_lower)
            if match:
                return match.group(1).strip()

        return q if q else query

    async def _execute_step(
        self,
        step: Step,
        original_query: str,
        kb_name: str,
        step_results: dict[str, Any],
    ) -> ResearchStepResult:
        """Execute a single research step."""
        result = ResearchStepResult(
            step_id=step.id,
            step_type=step.type,
            purpose=step.description,
            tool_used=step.tool or step.type.value,
            success=False,
            confidence=0.0,
        )

        try:
            if step.type == StepType.LOTUS_SEARCH:
                tool = self.tools.get("lotus_search")
                if tool:
                    raw_query = step.tool_input.get("query", original_query)
                    query = self._clean_search_query(raw_query, step.description)
                    logger.info("lotus_search", original_query=raw_query, cleaned_query=query)
                    raw = await tool.execute(query=query)
                    result.raw_result = raw
                    result.success = raw and "not found" not in raw.lower() and "error" not in raw.lower()
                    result.confidence = 0.8 if result.success else 0.3

            elif step.type == StepType.OPENALEX_SEARCH:
                raw_query = step.tool_input.get("query", original_query)
                query = self._clean_search_query(raw_query, step.description)
                logger.info("openalex_search", original_query=raw_query, cleaned_query=query)
                raw = await self._search_openalex(query, max_results=5)
                result.raw_result = raw
                result.success = raw and "no papers found" not in raw.lower() and "failed" not in raw.lower()
                result.confidence = 0.7 if result.success else 0.3

            elif step.type == StepType.KB_SEARCH:
                raw_query = step.tool_input.get("query", original_query)
                query = self._clean_search_query(raw_query, step.description)
                logger.info("kb_search", original_query=raw_query, cleaned_query=query)
                query_emb = await self.embeddings.embed([query])
                docs = await self.vector_store.search(
                    collection=f"kb_{kb_name}",
                    query_embedding=query_emb[0],
                    top_k=10,
                )
                # Assess quality
                is_sufficient, missing, confidence = await self.quality_assessor.assess(
                    original_query, docs, step.description
                )
                result.documents = docs
                result.success = is_sufficient
                result.confidence = confidence
                result.missing_aspects = missing
                result.raw_result = docs
                result.analysis = f"Found {len(docs)} documents"

            elif step.type == StepType.ANALYZE:
                result.raw_result = await self._analyze_results(original_query, step_results)
                result.success = True
                result.confidence = 0.6

            elif step.type == StepType.SYNTHESIZE:
                result.raw_result = await self._synthesize_results(original_query, step_results)
                result.success = True
                result.confidence = 0.6

            elif step.type == StepType.ANSWER:
                result.raw_result = "ANSWER_STEP"
                result.success = True

        except Exception as e:
            logger.error("step_execution_error", step=step.id, error=str(e))
            result.raw_result = f"Error: {e}"

        return result

    async def _search_openalex(self, query: str, max_results: int = 5) -> str:
        """Search OpenAlex for academic papers."""
        # Clean query
        q = query.lower()
        for phrase in ["i want to learn about", "tell me about", "what is", "what are"]:
            if q.startswith(phrase):
                q = q[len(phrase):].strip()

        try:
            url = "https://api.openalex.org/works"
            params = {
                "search": q,
                "per_page": max_results,
                "mailto": "perspicacite@example.com",
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                data = response.json()

            papers = data.get("results", [])
            if not papers:
                return "No papers found."

            lines = [f"Found {len(papers)} papers:"]
            for i, paper in enumerate(papers, 1):
                lines.append(f"\n{i}. {paper.get('display_name', 'Untitled')}")
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in paper.get("authorships", [])[:3]
                ]
                if authors:
                    lines.append(f"   Authors: {', '.join(authors)}")
                year = paper.get("publication_year")
                if year:
                    lines.append(f"   Year: {year}")
                abstract = paper.get("abstract", "")
                if abstract:
                    lines.append(f"   Abstract: {abstract[:200]}...")

            return "\n".join(lines)

        except Exception as e:
            logger.error("openalex_search_error", error=str(e))
            return f"OpenAlex search failed: {e}"

    async def _analyze_results(self, query: str, step_results: dict[str, Any]) -> str:
        """Analyze research results."""
        combined = [f"{k}:\n{str(v)[:500]}" for k, v in step_results.items()]

        prompt = f"""Analyze research results for relevance and completeness.

Query: "{query}"

Results:
{chr(10).join(combined)}

Return analysis of key findings, gaps, and recommendation."""

        return await self.llm.complete(prompt, temperature=0.3)

    async def _synthesize_results(self, query: str, step_results: dict[str, Any]) -> str:
        """Synthesize multiple sources."""
        combined = [f"Source ({k}):\n{str(v)[:400]}" for k, v in step_results.items()]

        prompt = f"""Synthesize information from multiple sources.

Query: "{query}"

Sources:
{chr(10).join(combined)}

Provide a coherent synthesis that answers the query."""

        return await self.llm.complete(prompt, temperature=0.3)

    async def _generate_answer(
        self,
        query: str,
        intent: IntentResult,
        step_results: dict[str, Any],
        session: AgentSession,
    ) -> str:
        """Generate final answer using all results."""
        context_parts = []
        for step_id, result in step_results.items():
            if result and str(result) not in ["ANSWER_STEP", "None"]:
                context_parts.append(f"[{step_id}]\n{str(result)[:600]}")

        context = "\n\n".join(context_parts)
        conversation_context = session.get_context_string()

        prompt = f"""Generate a comprehensive answer based on research results.

Original Question: "{query}"
Intent: {intent.intent.name}

Conversation Context:
{conversation_context}

Research Results:
{context}

Guidelines:
1. Focus on answering the SPECIFIC question asked
2. Prioritize the most relevant findings
3. Maintain scientific precision
4. Cite sources when referencing specific information
5. If insufficient information, clearly state this
6. Structure your answer logically

Generate your answer:"""

        return await self.llm.complete(prompt, temperature=0.4)

    def _extract_sources(self, completed_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract source references from completed steps."""
        sources = []
        seen = set()

        for step in completed_steps:
            if step.get("documents"):
                for doc in step["documents"]:
                    if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'metadata'):
                        meta = doc.chunk.metadata
                        title = getattr(meta, 'title', 'Untitled')
                        if title not in seen:
                            seen.add(title)
                            sources.append({
                                "title": title,
                                "authors": getattr(meta, 'authors', []),
                                "year": getattr(meta, 'year', None),
                                "score": getattr(doc, 'score', 0.0),
                            })

        return sources[:10]


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

AgenticOrchestrator = UnifiedAgenticRAG
