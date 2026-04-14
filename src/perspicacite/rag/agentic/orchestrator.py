"""Main agentic orchestrator with session management."""

import json
import re
import time
import uuid
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
from datetime import datetime

from .intent import IntentClassifier, Intent
from .planner import ResearchPlanner, Step, StepType, Plan, _log_steps_detail
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.retrieval.hybrid import hybrid_retrieval
from perspicacite.rag.utils import format_references_academic

# SciLEx integration
from perspicacite.search.scilex_adapter import SciLExAdapter

logger = logging.getLogger(__name__)

# Cap per-paper extraction LLM calls during final answer (map-reduce style).
MAP_REDUCE_MAX_PAPERS = 8

# Maximum number of replan iterations before forcing answer generation.
MAX_REPLANS = 3

# URL detection patterns for pre-fetch.
_URL_RE = re.compile(r"https?://\S+")
_DOI_IN_URL_RE = re.compile(r"10\.\d{4,9}/[^\s\])>'\"]+")


def _query_seeks_workflow_detail(query: str) -> bool:
    """True when the user likely needs ordered pipeline / methods, not abstract themes."""
    q = (query or "").lower()
    needles = (
        "workflow",
        "pipeline",
        "processing pipeline",
        "procedure",
        "step by step",
        "step-by-step",
        "stages",
        "methodology",
        "methods",
        "describe in detail",
        "explain in detail",
        "how does",
        "how do",
        "architecture",
        "sequence of",
        "outline the",
        "outline of",
    )
    return any(n in q for n in needles)


@dataclass
class EvidenceFacet:
    """One facet (sub-question) of a research query."""

    query: str
    step_ids: List[str] = field(default_factory=list)
    entries: List[Dict[str, Any]] = field(default_factory=list)
    _seen_keys: Set[str] = field(default_factory=set, repr=False)

    @property
    def status(self) -> str:
        """gap / partial / covered based on hit count."""
        n = len(self.entries)
        if n == 0:
            return "gap"
        if n <= 2:
            return "partial"
        return "covered"

    @property
    def confidence(self) -> float:
        """Heuristic confidence score in [0, 1] for this facet.

        Blends entry count, average relevance score, and full-text availability.
        """
        if not self.entries:
            return 0.0

        n = len(self.entries)
        count_score = min(n / 5, 1.0)

        relevance_scores = [
            e.get("relevance_score", 3) for e in self.entries
        ]
        avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 3.0
        rel_score = min(max((avg_rel - 1) / 4, 0.0), 1.0)  # map [1,5] → [0,1]

        full_text_count = sum(
            1 for e in self.entries if e.get("pdf_downloaded") or e.get("full_text")
        )
        ft_score = min(full_text_count / max(n, 1), 1.0)

        return round(0.45 * count_score + 0.35 * rel_score + 0.20 * ft_score, 4)

    def _entry_key(self, e: Dict[str, Any]) -> str:
        doi = (e.get("doi") or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        title = (e.get("title") or "").strip().lower()[:120]
        return f"title:{title}" if title else ""

    def _add_entry(self, entry: Dict[str, Any]) -> bool:
        """Add an entry if not a facet-local duplicate. Returns True if added."""
        k = self._entry_key(entry)
        if k and k in self._seen_keys:
            return False
        if k:
            self._seen_keys.add(k)
        self.entries.append(entry)
        return True


@dataclass
class EvidenceStore:
    """Facet-keyed evidence accumulator for gap-driven replanning.

    Each search step is registered under a facet (sub-question).
    Simple queries use one facet ("main"); composite queries get one facet per
    sub-query.  ``to_prompt_block()`` renders per-facet status for the replanner.
    """

    facets: Dict[str, EvidenceFacet] = field(default_factory=dict)

    def register_facet(self, facet_key: str, query: str) -> EvidenceFacet:
        if facet_key not in self.facets:
            self.facets[facet_key] = EvidenceFacet(query=query)
        return self.facets[facet_key]

    def facet_for_step(self, step_id: str) -> Optional["EvidenceFacet"]:
        for f in self.facets.values():
            if step_id in f.step_ids:
                return f
        return None

    def add_hits(
        self,
        hits: List[Dict[str, Any]],
        step_id: str,
        facet_key: str = "main",
    ) -> None:
        facet = self.facets.get(facet_key)
        if facet is None:
            facet = self.register_facet(facet_key, facet_key)
        if step_id not in facet.step_ids:
            facet.step_ids.append(step_id)
        for h in hits:
            facet._add_entry(h)

    # Backward-compat alias
    def add_kb_hits(self, hits: List[Dict[str, Any]], step_id: str = "", facet_key: str = "main") -> None:
        self.add_hits(hits, step_id=step_id, facet_key=facet_key)

    @property
    def all_entries(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for f in self.facets.values():
            out.extend(f.entries)
        return out

    def gap_summary(self) -> Dict[str, str]:
        """Return {facet_key: status} for all facets."""
        return {k: f.status for k, f in self.facets.items()}

    def facet_confidences(self) -> Dict[str, float]:
        """Return {facet_key: confidence} for all facets."""
        return {k: f.confidence for k, f in self.facets.items()}

    def overall_confidence(self) -> float:
        """Weighted-average confidence across facets (equal weight)."""
        if not self.facets:
            return 0.0
        confs = [f.confidence for f in self.facets.values()]
        return sum(confs) / len(confs)

    def to_prompt_block(self, max_entries_per_facet: int = 7, max_chars: int = 3200) -> str:
        """Render per-facet evidence + status for the replanner."""
        if not self.facets:
            return ""
        sections: List[str] = []
        for key, facet in self.facets.items():
            status = facet.status.upper()
            header = f"[{status}] Facet: {facet.query}"
            if not facet.entries:
                sections.append(f"{header}\n  (no evidence yet)")
                continue
            lines = [header]
            for e in facet.entries[-max_entries_per_facet:]:
                title = str(e.get("title", "?"))[:140]
                doi = e.get("doi") or ""
                ex = (e.get("excerpt") or "")[:350]
                line = f"  - {title}"
                if doi:
                    line += f" (DOI: {doi})"
                if ex:
                    line += f"\n    {ex}"
                lines.append(line)
            sections.append("\n".join(lines))
        text = "\n\n".join(sections)
        return text if len(text) <= max_chars else text[:max_chars] + "\n…"


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
    
    # User preferences for research depth (can be set per query)
    max_papers_to_download: Optional[int] = None  # Override orchestrator default
    evidence: Optional[EvidenceStore] = None

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
        # KB / rich hits may need thousands of chars for workflow or methods questions;
        # 500 chars caused false "insufficient" when full text existed but was never shown.
        _assess_doc_cap = 8000
        for i, doc in enumerate(documents[:5]):  # Limit to top 5
            if hasattr(doc, "chunk"):
                raw = doc.chunk.text if hasattr(doc.chunk, "text") else str(doc.chunk)
                text = raw[:_assess_doc_cap]
            elif isinstance(doc, dict):
                raw = doc.get("text", doc.get("content", str(doc)))
                text = raw[:_assess_doc_cap] if isinstance(raw, str) else str(raw)[:_assess_doc_cap]
            else:
                text = str(doc)[:_assess_doc_cap]
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
        use_two_pass: bool = True,
        early_exit_confidence: float = 0.85,
        relevance_threshold: int = 3,
        max_papers_to_download: int = 10,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.embeddings = embedding_provider
        self.vector_store = vector_store
        self.max_iterations = max_iterations
        self.use_hybrid = use_hybrid
        self.use_two_pass = True  # Default, overridden from config if available
        self.early_exit_confidence = early_exit_confidence
        
        # Paper download configuration
        # For literature surveys: lower threshold = more papers, higher max = comprehensive coverage
        # For quick answers: higher threshold = only best papers, lower max = faster
        self.relevance_threshold = relevance_threshold  # Min relevance score to download (1-5)
        self.max_papers_to_download = max_papers_to_download  # Safety cap on downloads

        self.intent_classifier = IntentClassifier(llm_client)
        self.planner = ResearchPlanner(llm_client)
        self.quality_assessor = DocumentQualityAssessor(llm_client)

        # Session management
        self.sessions: Dict[str, AgentSession] = {}
        self._found_papers_lock = asyncio.Lock()

        # SciLEx adapter for literature search (multi-API aggregation)
        self.scilex_adapter = SciLExAdapter()

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
        max_papers_to_download: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main chat entry point with true agentic behavior.

        Args:
            query: User's research question
            session_id: Optional session ID for persistence
            kb_name: Optional knowledge base to search first
            stream: Whether to stream responses
            max_papers_to_download: Override default max papers to download (for user preference)

        Yields:
            Dict with type: "thinking", "tool_call", "tool_result", "answer", "papers_found"
        """
        logger.info("=" * 80)
        logger.info("NEW CHAT REQUEST")
        logger.info(f"Query: {query}")
        logger.info(f"Session ID (client): {session_id!r}")
        logger.info(f"KB: {kb_name or 'none'}")
        logger.info(f"Max papers to download: {max_papers_to_download or self.max_papers_to_download}")

        session = self.get_or_create_session(session_id)
        logger.info(f"Resolved session_id: {session.session_id}")
        session.add_message("user", query)
        session.kb_name = kb_name
        session.evidence = EvidenceStore()
        
        # Store user preference for download cap in session
        if max_papers_to_download is not None:
            session.max_papers_to_download = max_papers_to_download
        
        logger.info(f"Session messages count: {len(session.messages)}")

        # Clear accumulated papers from previous requests
        self._found_papers = []
        if not hasattr(session, 'found_papers_archive'):
            session.found_papers_archive = []
        # Pre-populate with papers from previous turns
        self._found_papers = list(session.found_papers_archive)
        session.original_query = query

        # --- URL pre-processing (deterministic, before planning) ---
        url_paper = await self._try_resolve_url(query)
        if url_paper:
            title = url_paper.get("title", "the paper")
            yield {"type": "thinking", "message": f"Fetching paper: {title[:80]}..."}
            self._found_papers.append(url_paper)
            yield {"type": "thinking", "message": f"Retrieved: {title[:80]}"}
            logger.info(f"URL pre-fetch: {title[:80]} ({len(url_paper.get('full_text', ''))} chars)")
        else:
            if _URL_RE.search(query):
                yield {"type": "thinking", "message": "Could not fetch paper directly, searching databases instead..."}

        # Step 1: Classify intent
        yield {"type": "thinking", "message": "Analyzing your query..."}

        intent_result = await self.intent_classifier.classify(
            query=query,
            conversation_history=session.get_conversation_history(),
            active_kb_name=kb_name,
        )
        logger.info(f"Intent classified: {intent_result.intent.name}")
        logger.info(f"Confidence: {intent_result.confidence}")
        logger.info(f"Query complexity: {getattr(intent_result, 'query_complexity', 'simple')} "
                    f"({getattr(intent_result, 'query_complexity_source', '')})")
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
            "paper_lookup",
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
            available_papers=[url_paper] if url_paper else None,
        )

        # If a KB is selected, always search it first (don't rely on the LLM planner).
        # Composite + active KB: ensure parallel kb_search wave even when the planner emitted only one KB step.
        if kb_name:
            has_kb_step = any(s.type == StepType.KB_SEARCH for s in plan.steps)
            qcomp = getattr(intent_result, "query_complexity", "simple")
            if not has_kb_step:
                clean_query = ResearchPlanner._clean_query_for_search(query)
                if qcomp == "composite":
                    subs = list(dict.fromkeys(
                        await self.planner.composite_subqueries_with_llm(clean_query)
                    ))[:3]
                    for i, sq in enumerate(subs):
                        plan.steps.insert(
                            i,
                            Step(
                                id=f"inject_kb_{i+1}",
                                type=StepType.KB_SEARCH,
                                description=f"Search knowledge base '{kb_name}'",
                                tool="kb_search",
                                tool_input={"query": sq},
                                depends_on=[],
                            ),
                        )
                    logger.info(
                        f"Injected {len(subs)} parallel kb_search step(s) for KB {kb_name!r} "
                        f"(composite): queries={subs!r}"
                    )
                else:
                    kb_step = Step(
                        id="step1",
                        type=StepType.KB_SEARCH,
                        description=f"Search knowledge base '{kb_name}'",
                        tool="kb_search",
                        tool_input={"query": clean_query},
                    )
                    plan.steps.insert(0, kb_step)
                    logger.info(
                        f"Injected kb_search as step1 for KB {kb_name!r} tool_input.query={clean_query!r}"
                    )
            elif qcomp == "composite":
                await self._maybe_upgrade_single_kb_to_composite_parallel(plan, query, kb_name)
            plan.estimated_steps = len(plan.steps)

        logger.info(f"Orchestrator plan reasoning ({len(plan.reasoning)} chars): {plan.reasoning}")
        _log_steps_detail(plan.steps, "Orchestrator plan (final, after KB inject if any)")

        self._register_evidence_facets(session, plan)

        if plan.can_answer_from_history:
            # Planner decided we have enough — check if pre-fetched paper has full text
            if url_paper and url_paper.get("full_text"):
                logger.info("Planner says can_answer_from_history + url_paper has full_text → single-paper answer")
                yield {"type": "thinking", "message": "Generating answer from retrieved paper..."}
                papers = [url_paper]
                is_summary = not any(
                    w in query.lower() for w in ("what", "how", "why", "which", "method", "approach", "result", "compare", "find")
                )
                answer, citation_map = await self._generate_single_paper_answer(
                    query, papers, session, is_summary_request=is_summary,
                )
                session.add_message("assistant", answer, {
                    "intent": intent_result.intent.name,
                    "steps_completed": 0,
                    "tools_used": [],
                    "can_answer_from_history": True,
                })
                yield {
                    "type": "answer",
                    "content": answer,
                    "session_id": session.session_id,
                    "citations": citation_map,
                }
                return
            else:
                yield {"type": "thinking", "message": "I can answer from our conversation history..."}

        # Step 3: Execute plan iteratively
        step_results: Dict[str, Any] = {}
        completed_steps: List[Step] = []
        replan_count = 0

        for iteration in range(self.max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")

            batch = self._get_next_parallel_batch(plan, completed_steps, step_results)
            if not batch:
                logger.info("No more steps to execute")
                break

            to_run: List[Step] = []
            for s in batch:
                if s.condition and not self._evaluate_condition(s.condition, step_results):
                    logger.info(f"Step {s.id} condition not met, skipping")
                    completed_steps.append(s)
                else:
                    to_run.append(s)
            if not to_run:
                continue

            if len(to_run) > 1:
                logger.info(
                    "Parallel batch: "
                    + ", ".join(f"{s.id} ({s.type.value})" for s in to_run)
                )

            async def _run_step(step: Step) -> tuple[Step, Any, float]:
                t0 = time.time()
                res = await self._execute_step(step, query, step_results, session)
                return step, res, time.time() - t0

            for s in to_run:
                yield {
                    "type": "tool_call",
                    "step": s.id,
                    "tool": s.tool or s.type.value,
                    "description": s.description,
                    "query": (s.tool_input or {}).get("query", ""),
                }

            if len(to_run) == 1:
                next_step = to_run[0]
                step, result, step_duration = await _run_step(next_step)
                result_str = str(result)
                logger.info(f"Step {step.id} completed in {step_duration:.2f}s")
                logger.info(f"Result length: {len(result_str)} chars")
                preview_len = min(2000, len(result_str))
                logger.info(
                    f"Result preview: {result_str[:preview_len]}{'...[truncated]' if len(result_str) > preview_len else ''}"
                )
                step_results[step.id] = result
                completed_steps.append(step)
                yield {
                    "type": "tool_result",
                    "step": step.id,
                    "result_summary": self._summarize_result(result),
                }
                batch_for_eval = [step]
            else:
                gathered = await asyncio.gather(*[_run_step(s) for s in to_run])
                for step, result, step_duration in gathered:
                    result_str = str(result)
                    logger.info(f"Step {step.id} completed in {step_duration:.2f}s")
                    logger.info(f"Result length: {len(result_str)} chars")
                    step_results[step.id] = result
                    completed_steps.append(step)
                    yield {
                        "type": "tool_result",
                        "step": step.id,
                        "result_summary": self._summarize_result(result),
                    }
                batch_for_eval = to_run

            trigger_eval = any(
                s.type
                in (
                    StepType.LOTUS_SEARCH,
                    StepType.LITERATURE_SEARCH,
                    StepType.KB_SEARCH,
                )
                for s in batch_for_eval
            )

            if trigger_eval and session.evidence:
                yield {
                    "type": "evidence_update",
                    "facets": {
                        k: {
                            "status": f.status,
                            "entry_count": len(f.entries),
                            "confidence": f.confidence,
                        }
                        for k, f in session.evidence.facets.items()
                    },
                    "overall_confidence": round(session.evidence.overall_confidence(), 3),
                }

            if trigger_eval:
                eval_result = await self._evaluate_progress(
                    query,
                    plan,
                    completed_steps,
                    step_results,
                    session=session,
                    eval_step_ids=[s.id for s in batch_for_eval],
                )
                decision = eval_result["decision"]
                logger.info(
                    f"Progress evaluation: decision={decision}, "
                    f"gaps={eval_result['gap_facets']}, "
                    f"missing={eval_result['missing_aspects'][:3]}"
                )

                if decision == "replan":
                    if replan_count >= MAX_REPLANS:
                        logger.info(
                            f"Replan budget exhausted ({replan_count}/{MAX_REPLANS}), "
                            "forcing answer with current evidence"
                        )
                        break
                    replan_count += 1
                    ev_summary = (
                        session.evidence.to_prompt_block()
                        if session.evidence
                        else ""
                    )
                    plan = await self.planner.replan(
                        query,
                        plan,
                        completed_steps,
                        step_results,
                        eval_result["evaluation_text"] or "Need more specific search",
                        evidence_summary=ev_summary or None,
                    )
                    self._register_evidence_facets(session, plan)
                    yield {"type": "thinking", "message": f"Adjusting research plan (replan {replan_count}/{MAX_REPLANS})..."}
                    yield {
                        "type": "replan",
                        "iteration": replan_count,
                        "max_replans": MAX_REPLANS,
                        "reason": eval_result.get("evaluation_text", ""),
                        "gap_facets": eval_result.get("gap_facets", []),
                        "new_step_count": len(plan.steps),
                    }
                elif decision == "answer":
                    logger.info("Sufficient results, moving to answer")
                    break

        logger.info(f"\n=== Execution complete ===")
        logger.info(f"Completed {len(completed_steps)} steps")
        logger.info(f"Step results keys: {list(step_results.keys())}")

        # Step 4: Extract and process papers with progress updates
        yield {"type": "thinking", "message": "Extracting papers from search results..."}
        papers = self._extract_papers_from_results(step_results)
        logger.info(f"Extracted {len(papers)} papers from search results")

        all_from_kb = all(p.get("source") == "kb_search" for p in papers) if papers else False
        if all_from_kb and papers:
            logger.info("All papers from KB (pre-scored at 4); skipping LLM relevance scoring")
            yield {"type": "thinking", "message": f"Using {len(papers)} KB papers (relevance scoring skipped)"}
        else:
            yield {"type": "thinking", "message": f"Scoring {len(papers)} papers for relevance..."}
            papers = await self._score_papers_for_relevance(query, papers, min_score=3)
            included_count = len([p for p in papers if p.get("relevance_score", 0) >= 3])
            yield {"type": "thinking", "message": f"Relevance filtering: {included_count}/{len(papers)} papers included"}
        
        # Download full text for relevant papers (threshold-based, not hard limit)
        # For literature surveys, comprehensive coverage is important - download ALL relevant papers
        # up to a safety cap. Configurable via relevance_threshold and max_papers_to_download.
        # Use session-specific limit if user provided it, otherwise use orchestrator default
        max_download = session.max_papers_to_download or self.max_papers_to_download
        
        download_candidates = [
            p for p in papers 
            if p.get("relevance_score", 0) >= self.relevance_threshold
        ][:max_download]
        
        if download_candidates:
            yield {"type": "thinking", "message": f"Attempting to download {len(download_candidates)} relevant papers for full text analysis..."}
            
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
        else:
            yield {"type": "thinking", "message": "No papers met the relevance threshold for full-text download"}
        
        # Generate final answer
        yield {"type": "thinking", "message": "Synthesizing answer..."}
        answer, citation_map = await self._generate_answer(
            query=query, plan=plan, step_results=step_results, session=session, papers=papers
        )

        overall_conf = (
            session.evidence.overall_confidence()
            if session.evidence
            else None
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

        answer_event: Dict[str, Any] = {
            "type": "answer",
            "content": answer,
            "session_id": session.session_id,
            "citations": citation_map,
        }
        if overall_conf is not None:
            answer_event["confidence"] = round(overall_conf, 3)
        yield answer_event

        # Yield found papers so the UI can offer "Add to KB"
        # Only include papers that:
        # 1. Passed relevance filtering (score >= 3)
        # 2. Are NOT already in the KB (source != "kb_search")
        # Default 0: missing score means scoring failed or paper was excluded — do not
        # treat as "passed" (previously default 3 falsely showed all hits as relevant).
        relevant_papers = [
            p for p in papers
            if p.get("relevance_score", 0) >= 3 and p.get("source") != "kb_search"
        ]
        if relevant_papers:
            yield {"type": "papers_found", "papers": relevant_papers}

    async def _maybe_upgrade_single_kb_to_composite_parallel(
        self, plan: Plan, query: str, kb_name: str
    ) -> None:
        """Replace one root parallel kb_search with multiple facets when query is composite.

        The LLM planner often emits a single kb_search even for comparisons; this enforces
        the parallel wave when composite_subqueries_with_llm yields 2+ distinct phrases.
        """
        clean = ResearchPlanner._clean_query_for_search(query)
        subs = list(dict.fromkeys(
            await self.planner.composite_subqueries_with_llm(clean)
        ))[:3]
        if len(subs) < 2:
            return

        kb_parallel = [
            (i, s)
            for i, s in enumerate(plan.steps)
            if s.type == StepType.KB_SEARCH and not s.depends_on
        ]
        if len(kb_parallel) != 1:
            return

        idx, lone = kb_parallel[0]
        old_id = lone.id
        top_k = lone.tool_input.get("top_k")
        new_steps: List[Step] = []
        for i, sq in enumerate(subs):
            tool_input: Dict[str, Any] = {"query": sq}
            if top_k is not None:
                tool_input["top_k"] = top_k
            new_steps.append(
                Step(
                    id=f"composite_kb_{i+1}",
                    type=StepType.KB_SEARCH,
                    description=f"Search knowledge base '{kb_name}' (composite facet)",
                    tool="kb_search",
                    tool_input=tool_input,
                    depends_on=[],
                )
            )
        plan.steps[idx : idx + 1] = new_steps
        new_ids = [s.id for s in new_steps]
        for s in plan.steps:
            if old_id in s.depends_on:
                s.depends_on = [d for d in s.depends_on if d != old_id] + new_ids
        logger.info(
            "Composite KB upgrade: replaced single root kb_search %r with %d step(s) subs=%r",
            old_id,
            len(new_steps),
            subs,
        )

    # ------------------------------------------------------------------
    # URL pre-processing
    # ------------------------------------------------------------------

    async def _try_resolve_url(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect paper URL in query, fetch content, return paper dict or None.

        Priority:
        1. arXiv URL → extract ID, fetch HTML version
        2. DOI in URL (doi.org/... or publisher URLs containing a DOI) → unified retrieval
        3. Publisher URL without DOI → give up (future: scrape meta tags)
        """
        url_match = _URL_RE.search(query)
        if not url_match:
            return None

        url = url_match.group(0).rstrip(".,;:)")

        # 1. arXiv URL → extract ID directly
        from perspicacite.pipeline.download.arxiv import (
            get_arxiv_id_from_url,
            fetch_arxiv_html,
        )
        arxiv_id = get_arxiv_id_from_url(url)

        if arxiv_id:
            # Strip version suffix for fetching (arXiv serves latest by default)
            bare_id = arxiv_id.split("v")[0] if re.match(r".*v\d+$", arxiv_id) else arxiv_id
            full_text, sections, html_title = await fetch_arxiv_html(bare_id)
            if full_text:
                title, doi_resolved = await self._resolve_arxiv_metadata(bare_id)
                # Prefer: OpenAlex title > HTML <title> tag > arXiv ID fallback
                final_title = title or html_title or f"arXiv:{arxiv_id}"
                return {
                    "title": final_title,
                    "doi": doi_resolved or f"10.48550/arXiv.{bare_id}",
                    "full_text": full_text,
                    "sections": sections,
                    "source": "url_fetch",
                    "relevance_score": 5,
                    "arxiv_id": bare_id,
                    "_step_id": "url_prefetch",
                }

        # 2. DOI in URL (doi.org/... or publisher URLs containing a DOI)
        doi: Optional[str] = None
        doi_match = _DOI_IN_URL_RE.search(url)
        if doi_match:
            doi = doi_match.group(0)

        if doi:
            from perspicacite.pipeline.download.unified import retrieve_paper_content
            from perspicacite.pipeline.parsers.pdf import PDFParser

            parser = PDFParser()
            try:
                result = await retrieve_paper_content(
                    doi=doi,
                    pdf_parser=parser,
                    unpaywall_email="perspicacite@example.com",
                )
                if result.success and result.full_text:
                    return {
                        "title": (result.metadata or {}).get("title", doi),
                        "doi": doi,
                        "full_text": result.full_text,
                        "sections": getattr(result, "sections", None),
                        "source": "url_fetch",
                        "relevance_score": 5,
                        "content_type": result.content_type,
                        "_step_id": "url_prefetch",
                    }
            except Exception as e:
                logger.warning(f"URL pre-fetch via retrieve_paper_content failed for {doi}: {e}")

        # 3. Fallback: Semantic Scholar metadata lookup (no full text, just title + abstract)
        from perspicacite.search.semantic_scholar import lookup_paper
        lookup_id = None
        if doi:
            lookup_id = doi
        elif arxiv_id:
            lookup_id = arxiv_id
        if lookup_id:
            logger.info(f"URL pre-fetch fallback: S2 lookup for {lookup_id}")
            paper = await lookup_paper(lookup_id)
            if paper:
                logger.info(f"S2 fallback: found '{paper.title[:60]}'")
                return {
                    "title": paper.title,
                    "doi": paper.doi or doi or "",
                    "abstract": paper.abstract or "",
                    "full_text": None,
                    "authors": [a.name for a in paper.authors],
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "source": "semantic_scholar_fallback",
                    "relevance_score": 4,
                    "arxiv_id": arxiv_id,
                    "_step_id": "url_prefetch",
                    "_metadata_only": True,
                }

        return None

    async def _resolve_arxiv_metadata(
        self, arxiv_id: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve title + DOI from arXiv ID via OpenAlex.

        Lightweight — one API call.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(
                    f"https://api.openalex.org/works/arxiv:{arxiv_id}",
                    params={"mailto": "perspicacite@example.com"},
                )
                if r.status_code == 200:
                    work = r.json()
                    title = work.get("title")
                    ids = work.get("ids") or {}
                    doi = (ids.get("doi") or "").replace("https://doi.org/", "")
                    return title, doi or None
        except Exception as e:
            logger.debug(f"OpenAlex arXiv metadata lookup failed for {arxiv_id}: {e}")
        return None, None

    def _register_evidence_facets(self, session: AgentSession, plan: Plan) -> None:
        """Create evidence facets from search steps in the plan.

        Each search step's ``tool_input.query`` becomes a facet.  Steps that
        share identical queries share a facet.  This runs once after the plan is
        finalized (including composite injection/upgrade).
        """
        if session.evidence is None:
            session.evidence = EvidenceStore()
        ev = session.evidence
        for step in plan.steps:
            if step.type not in (StepType.KB_SEARCH, StepType.LITERATURE_SEARCH):
                continue
            q = step.tool_input.get("query", "").strip()
            facet_key = q.lower()[:120] or "main"
            facet = ev.register_facet(facet_key, q or "main")
            if step.id not in facet.step_ids:
                facet.step_ids.append(step.id)
        if not ev.facets:
            ev.register_facet("main", "main")
        logger.info(
            "Evidence facets registered: %s",
            {k: f.step_ids for k, f in ev.facets.items()},
        )

    def _facet_key_for_step(self, session: AgentSession, step: Step) -> str:
        """Resolve which facet key a step belongs to."""
        if session.evidence:
            f = session.evidence.facet_for_step(step.id)
            if f:
                return f.query.lower()[:120] or "main"
        q = step.tool_input.get("query", "").strip().lower()[:120]
        return q or "main"

    def _get_next_parallel_batch(
        self, plan: Plan, completed: List[Step], results: Dict[str, Any]
    ) -> List[Step]:
        """Return all steps whose dependencies are satisfied (general DAG).

        Any ready steps that are *not* of type ANSWER run concurrently.
        ANSWER steps are always executed alone (they need all prior context).
        If only a single step is ready, it runs by itself.
        """
        completed_ids = {s.id for s in completed}
        ready: List[Step] = []
        for step in plan.steps:
            if step.id in completed_ids:
                continue
            if all(dep in completed_ids for dep in step.depends_on):
                ready.append(step)
        if not ready:
            return []
        answers = [s for s in ready if s.type == StepType.ANSWER]
        if answers:
            return [answers[0]]
        return ready

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
            return await self._scilex_search(query, step_id=step.id, session=session)

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
                    # Dynamic top_k: planner can specify via tool_input
                    planner_top_k = step.tool_input.get("top_k")
                    top_k = planner_top_k if planner_top_k is not None else dkb.config.top_k
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
                        # Two-pass enrichment: fetch full paper text (if enabled)
                        paper_ids = []
                        paper_results = []
                        if self.use_two_pass:
                            for r in results:
                                meta = r.get("metadata")
                                pid = getattr(meta, "paper_id", None) if meta else r.get("paper_id")
                                if pid and pid not in paper_ids:
                                    paper_ids.append(pid)

                            if paper_ids:
                                try:
                                    from perspicacite.rag.utils import deduplicate_chunk_overlaps
                                    all_chunks = await self.vector_store.get_chunks_by_paper_ids(
                                        collection_name, paper_ids
                                    )
                                    if all_chunks:
                                        deduped = deduplicate_chunk_overlaps(all_chunks)
                                        from collections import OrderedDict
                                        grouped: OrderedDict = OrderedDict()
                                        for d in deduped:
                                            grouped.setdefault(d["paper_id"], []).append(d)
                                        for pid in paper_ids:
                                            clist = grouped.get(pid, [])
                                            full_text = " ".join(c["text"] for c in clist)
                                            meta_obj = clist[0]["metadata"] if clist else None
                                            # Find score from pass-1 results
                                            score = 0.5
                                            for r in results:
                                                m = r.get("metadata")
                                                rp = getattr(m, "paper_id", None) if m else r.get("paper_id")
                                                if rp == pid:
                                                    score = r.get("score", 0.5)
                                                    break
                                            paper_results.append({
                                                "paper_id": pid,
                                                "paper_score": score,
                                                "title": getattr(meta_obj, "title", None) if meta_obj else None,
                                                "authors": getattr(meta_obj, "authors", None) if meta_obj else None,
                                                "year": getattr(meta_obj, "year", None) if meta_obj else None,
                                                "doi": getattr(meta_obj, "doi", None) if meta_obj else None,
                                                "full_text": full_text,
                                                "source": "kb_search",
                                            })
                                        logger.info(
                                            f"KB_SEARCH: two-pass enrichment: {len(paper_ids)} papers, "
                                            f"{len(all_chunks)} chunks total"
                                        )
                                except Exception as e:
                                    logger.warning(f"KB_SEARCH: two-pass enrichment failed: {e}")

                        # Format results for the agent
                        formatted_parts = [
                            f"Found {len(paper_results or results)} relevant documents in knowledge base:"
                        ]
                        _kb_format_cap = 48000
                        items = paper_results if paper_results else results
                        for i, item in enumerate(items, 1):
                            if isinstance(item, dict) and "full_text" in item:
                                title = item.get("title") or "Unknown"
                                authors = item.get("authors") or ""
                                year = item.get("year") or ""
                                doi = item.get("doi") or ""
                                score = item.get("paper_score", 0)
                                joined_full = item.get("full_text") or ""
                                fmt_cap = _kb_format_cap
                                text_content = joined_full[:fmt_cap]
                                pid = item.get("paper_id")
                            else:
                                meta = item.get("metadata")
                                title = getattr(meta, "title", None) or "Unknown" if meta else "Unknown"
                                authors = getattr(meta, "authors", None) or "" if meta else ""
                                year = getattr(meta, "year", None) or "" if meta else ""
                                doi = getattr(meta, "doi", None) or "" if meta else ""
                                score = item.get("score", 0)
                                joined_full = item.get("text", "") or ""
                                fmt_cap = 1500
                                text_content = joined_full[:fmt_cap]
                                pid = (
                                    getattr(meta, "paper_id", None) if meta else None
                                ) or item.get("paper_id")

                            logger.info(
                                "KB_SEARCH: content_lengths "
                                f"doc={i}/{len(items)} paper_id={pid!r} "
                                f"joined_full_len={len(joined_full)} "
                                f"formatted_content_len={len(text_content)} cap={fmt_cap} "
                                f"truncated={len(joined_full) > len(text_content)}"
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
                                if len(joined_full) > _kb_format_cap:
                                    formatted_parts.append("   [... content truncated ...]")
                            else:
                                formatted_parts.append("   Content: [No text content available]")
                            
                            async with self._found_papers_lock:
                                if hasattr(self, "_found_papers"):
                                    self._found_papers.append({
                                        "title": title,
                                        "authors": [a.strip() for a in authors.split(",")] if authors else [],
                                        "year": year,
                                        "doi": doi,
                                        "abstract": (text_content[:4000] if text_content else ""),
                                        "full_text": joined_full,
                                        "source": "kb_search",
                                        "relevance_score": 4,
                                        "_step_id": step.id,
                                    })
                                if session.evidence is None:
                                    session.evidence = EvidenceStore()
                                fk = self._facet_key_for_step(session, step)
                                session.evidence.add_hits(
                                    [
                                        {
                                            "title": title,
                                            "doi": doi,
                                            "excerpt": (text_content[:4000] if text_content else ""),
                                            "step_id": step.id,
                                            "source": "kb_search",
                                        }
                                    ],
                                    step_id=step.id,
                                    facet_key=fk,
                                )
                        
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

        elif step.type == StepType.PAPER_LOOKUP:
            paper_id = step.tool_input.get("paper_id", "")
            if not paper_id:
                return "No paper ID provided for lookup."

            logger.info(f"PAPER_LOOKUP: paper_id='{paper_id}'")

            from perspicacite.search.semantic_scholar import lookup_paper
            paper = await lookup_paper(paper_id)

            if paper is None:
                logger.info(f"PAPER_LOOKUP: paper not found for {paper_id}")
                return f"Paper not found for identifier: {paper_id}"

            paper_dict = {
                "id": paper.id,
                "title": paper.title,
                "authors": [a.name for a in paper.authors[:10]],
                "year": paper.year,
                "doi": paper.doi or "",
                "abstract": paper.abstract or "",
                "citation_count": paper.citation_count or 0,
                "pdf_url": paper.pdf_url or "",
                "source": "paper_lookup",
                "relevance_score": 5,
                "_step_id": step.id,
            }
            self._found_papers.append(paper_dict)
            logger.info(f"PAPER_LOOKUP: found '{paper.title[:60]}' (citations: {paper.citation_count})")

            parts = [f"Paper: {paper.title}"]
            if paper.authors:
                parts.append(f"Authors: {', '.join(a.name for a in paper.authors[:5])}")
            if paper.year:
                parts.append(f"Year: {paper.year}")
            if paper.doi:
                parts.append(f"DOI: {paper.doi}")
            if paper.abstract:
                parts.append(f"Abstract: {paper.abstract}")
            if paper.citation_count:
                parts.append(f"Citations: {paper.citation_count}")

            return "\n".join(parts)

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
        self,
        query: str,
        plan: Plan,
        completed_steps: List[Step],
        step_results: Dict[str, Any],
        session: Optional[AgentSession] = None,
        eval_step_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate whether to continue, replan, or answer.

        Returns a dict with:
          - ``"decision"``: ``"continue"`` | ``"replan"`` | ``"answer"``
          - ``"gap_facets"``: list of facet queries that are still gap/partial
          - ``"missing_aspects"``: list from the quality assessor (if any)
          - ``"evaluation_text"``: human-readable summary for the replanner
        """
        result: Dict[str, Any] = {
            "decision": "continue",
            "gap_facets": [],
            "missing_aspects": [],
            "evaluation_text": "",
        }

        last_step = completed_steps[-1]

        remaining_steps = [s for s in plan.steps if s.id not in {cs.id for cs in completed_steps}]
        if not remaining_steps:
            result["decision"] = "answer"
            result["evaluation_text"] = "No remaining steps."
            return result

        # --- Facet gap check (Phase 2) ---
        gaps: Dict[str, str] = {}
        if session and session.evidence:
            gaps = session.evidence.gap_summary()
        gap_facets = [k for k, v in gaps.items() if v in ("gap", "partial")]
        result["gap_facets"] = gap_facets

        remaining_search = [
            s for s in remaining_steps
            if s.type in (StepType.KB_SEARCH, StepType.LITERATURE_SEARCH)
        ]

        if gap_facets and remaining_search:
            covered_by_remaining: Set[str] = set()
            if session and session.evidence:
                for s in remaining_search:
                    f = session.evidence.facet_for_step(s.id)
                    if f:
                        covered_by_remaining.add(f.query.lower()[:120] or "main")
            uncovered_gaps = [f for f in gap_facets if f not in covered_by_remaining]
            if uncovered_gaps:
                facet_labels = "; ".join(f'"{f}" ({gaps[f]})' for f in uncovered_gaps[:4])
                logger.info(f"Facet gaps with no remaining steps, forcing replan: {facet_labels}")
                result["decision"] = "replan"
                result["evaluation_text"] = f"Facets still uncovered: {facet_labels}"
                return result
            else:
                logger.info(
                    f"Gap facets exist but remaining plan steps already target them: "
                    f"{list(covered_by_remaining)[:4]} — continuing"
                )

        # --- Quality assessor (per-batch documents) ---
        has_search = any(
            s.type in (StepType.KB_SEARCH, StepType.LITERATURE_SEARCH)
            for s in (
                [s for s in completed_steps if s.id in eval_step_ids]
                if eval_step_ids
                else [last_step]
            )
        )
        if has_search:
            documents = self._get_recent_found_papers(step_ids=eval_step_ids)

            if documents:
                is_sufficient, missing_aspects, confidence = await self.quality_assessor.assess(
                    query=query,
                    documents=documents,
                    step_purpose=last_step.description,
                )
                result["missing_aspects"] = missing_aspects or []

                logger.info(
                    f"Quality assessment: sufficient={is_sufficient}, "
                    f"confidence={confidence:.2f}, missing={len(missing_aspects)} aspects"
                )

                if is_sufficient and confidence >= self.early_exit_confidence:
                    all_covered = all(v == "covered" for v in gaps.values()) if gaps else True
                    if all_covered:
                        logger.info(
                            f"Early exit: confidence {confidence:.2f} >= {self.early_exit_confidence}, "
                            "all facets covered"
                        )
                        result["decision"] = "answer"
                        result["evaluation_text"] = "All facets covered; evidence sufficient."
                        return result
                    else:
                        logger.info(
                            "Quality assessor says sufficient but facets not all covered — "
                            "continuing to fill gaps"
                        )

                if not is_sufficient and missing_aspects:
                    aspect_str = ", ".join(missing_aspects[:3])
                    logger.info(f"Quality insufficient, missing: {aspect_str}")
                    if gap_facets:
                        result["decision"] = "replan"
                        result["evaluation_text"] = (
                            f"Missing aspects: {aspect_str}. "
                            f"Gap facets: {'; '.join(gap_facets[:4])}"
                        )
                        return result

        has_substantial_results = False
        for r in step_results.values():
            result_str = str(r)
            if len(result_str) > 200 and "error" not in result_str.lower():
                has_substantial_results = True
                break

        if has_substantial_results and len(completed_steps) >= 3:
            result["decision"] = "answer"
            result["evaluation_text"] = "Sufficient results accumulated."

        return result

    def _get_recent_found_papers(self, step_ids: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Return structured documents from _found_papers for quality assessment.

        Uses the accumulated structured paper data rather than parsing formatted
        output strings.  Optionally filters to papers added by specific step_ids
        (stored on evidence entries).
        """
        if not hasattr(self, "_found_papers") or not self._found_papers:
            return []

        papers = list(self._found_papers)
        if step_ids:
            papers = [p for p in papers if p.get("_step_id") in step_ids] or papers

        docs: List[Dict[str, Any]] = []
        seen: set[str] = set()
        _content_cap = 12000
        for p in papers[-limit * 2 :]:
            title = str(p.get("title") or "Unknown")
            key = title.strip().lower()[:100]
            if key in seen:
                continue
            seen.add(key)
            ft = (p.get("full_text") or "").strip()
            ab = (p.get("abstract") or "").strip()
            content = (ft[:_content_cap] if ft else ab[:_content_cap]) or ab[:500] or ""
            docs.append({
                "title": title,
                "content": content,
                "source": p.get("source", "unknown"),
            })
            if len(docs) >= limit:
                break
        return docs

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

        return await self.llm.complete(prompt, temperature=0.15)

    async def _per_paper_extraction_bullet(
        self, query: str, paper: Dict[str, Any], list_index: int
    ) -> str:
        """Single-paper LLM pass: bullets aligned to the user question."""
        title = str(paper.get("title", "Unknown"))[:220]
        body = (paper.get("full_text") or paper.get("abstract") or "").strip()
        if len(body) > 14000:
            body = body[:14000] + "\n[truncated]"
        if len(body) < 400:
            return ""
        if _query_seeks_workflow_detail(query):
            extract_instructions = (
                "The research question asks for workflow, pipeline, methodology, or procedural detail.\n"
                "Output **numbered steps (1., 2., 3., …)** in the order given in the text (chronological "
                "or data-flow order). Each step: one or two sentences, grounded only in the excerpt.\n"
                "Include named components when present (software, databases, file formats, algorithms, "
                "repositories, scoring rules). Prefer 4–12 steps when the text supports them.\n"
                "If the excerpt only states high-level goals without concrete operational stages, give "
                "the **maximum detail the text allows** in numbered form and end with a line: "
                "(excerpt lacks finer pipeline detail)\n"
                "If nothing in the text helps at all, reply exactly: (no relevant content for this question)\n"
                "Do not invent steps; use only the text above."
            )
        else:
            extract_instructions = (
                "Output 3–8 bullet points of facts that directly help answer the research question. "
                "Each bullet must be self-contained. If nothing is relevant, reply exactly: "
                "(no relevant content for this question)\n"
                "Do not invent information; use only the text above."
            )
        prompt = (
            f'Research question: "{query}"\n\n'
            f"You extract from ONE paper. Citation index for this paper in the final answer: [{list_index}]\n"
            f"Title: {title}\n\n"
            "Text:\n---\n"
            f"{body}\n"
            "---\n\n"
            f"{extract_instructions}"
        )
        return await self.llm.complete(prompt, temperature=0.15)

    async def _map_reduce_paper_bullets(
        self, query: str, papers: List[Dict[str, Any]]
    ) -> str:
        """Top-N papers: parallel extraction bullets for final synthesis."""
        indexed: List[tuple[int, Dict[str, Any]]] = []
        for i, p in enumerate(papers, 1):
            ft = (p.get("full_text") or "").strip()
            ab = (p.get("abstract") or "").strip()
            if len(ft) >= 500 or len(ab) >= 200:
                indexed.append((i, p))
        indexed = indexed[:MAP_REDUCE_MAX_PAPERS]
        if not indexed:
            return ""

        sem = asyncio.Semaphore(4)

        async def _one(li: int, p: Dict[str, Any]) -> str:
            async with sem:
                return await self._per_paper_extraction_bullet(query, p, li)

        parts = await asyncio.gather(*[_one(li, p) for li, p in indexed])
        blocks: List[str] = []
        for (li, p), text in zip(indexed, parts):
            t = (text or "").strip()
            if not t:
                continue
            low = t.lower()
            if "no relevant content for this question" in low and len(t) < 160:
                continue
            blocks.append(f"### Paper [{li}] {str(p.get('title', ''))[:90]}\n{t}")
        return "\n\n".join(blocks)

    def _build_facet_overview(self, session: AgentSession) -> str:
        """Build a structured overview of evidence per facet for the answer prompt.

        For composite queries this gives the LLM a roadmap: which sub-questions
        were investigated, what evidence quality each has, guiding it to cover
        all facets and flag gaps.
        """
        if not session.evidence or len(session.evidence.facets) <= 1:
            return ""
        sections: List[str] = []
        sections.append("Research facets investigated:")
        for key, facet in session.evidence.facets.items():
            status = facet.status.upper()
            n = len(facet.entries)
            titles = [str(e.get("title", ""))[:80] for e in facet.entries[:4]]
            title_list = "; ".join(t for t in titles if t) or "(none)"
            sections.append(f"  [{status}] \"{facet.query}\" — {n} source(s): {title_list}")
        return "\n".join(sections)

    async def _generate_answer(
        self, query: str, plan: Plan, step_results: Dict[str, Any], session: AgentSession,
        papers: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """Generate final answer and return ``(answer_text, citation_map)``."""

        logger.info("\n--- Generating Answer ---")
        logger.info(f"Query: {query}")
        logger.info(f"Step results available: {list(step_results.keys())}")

        if papers is None:
            papers = self._extract_papers_from_results(step_results)
            papers = await self._score_papers_for_relevance(query, papers, min_score=3)

        # --- Single-paper fast path: skip map-reduce, pass full text directly ---
        single_paper_with_full_text = (
            len(papers) == 1
            and (papers[0].get("full_text") or "").strip()
        )
        if single_paper_with_full_text:
            return await self._generate_single_paper_answer(query, papers, session)

        numbered_paper_list = self._build_numbered_paper_list(papers)

        context_parts = []

        facet_overview = self._build_facet_overview(session)
        if facet_overview:
            context_parts.append(facet_overview)

        if "lotus" in step_results:
            lotus_result = step_results["lotus"]
            logger.info(f"LOTUS result length: {len(str(lotus_result))} chars")
            context_parts.append(f"LOTUS Search Results:\n{lotus_result}")

        map_reduce_block = await self._map_reduce_paper_bullets(query, papers)
        if map_reduce_block:
            context_parts.append(
                "\n\n---\n\nQuery-focused extractions (per paper; cite using [N] from the numbered list):\n"
                + map_reduce_block
            )
            logger.info("Answer context: using map-reduce per-paper extractions (raw step results suppressed)")
        else:
            for step_id, result in step_results.items():
                if step_id != "lotus" and result:
                    result_str = str(result)
                    logger.info(f"Step {step_id} result length: {len(result_str)} chars")
                    context_parts.append(f"{step_id}:\n{result_str[:3000]}")

            full_text_parts = []
            ft_cap = 14000 if _query_seeks_workflow_detail(query) else 8000
            for i, paper in enumerate(papers, 1):
                if paper.get("full_text"):
                    full_text_parts.append(
                        f"[Paper {i}: {paper.get('title', 'Unknown')[:80]}...]\n"
                        f"Full text excerpt:\n{paper['full_text'][:ft_cap]}..."
                    )
            if full_text_parts:
                context_parts.append(
                    "\n\n---\n\nDownloaded Full Text:\n"
                    + "\n\n---\n\n".join(full_text_parts)
                )
                logger.info(f"Added {len(full_text_parts)} full text documents to context")

        context = "\n\n".join(context_parts)
        logger.info(f"Total context length: {len(context)} chars")

        if not context.strip():
            logger.warning("Context is empty! No research results to use.")

        conversation_context = session.get_context_string()

        facet_guideline = ""
        if facet_overview:
            facet_guideline = (
                "10. The query has multiple facets (see \"Research facets investigated\" above). "
                "Address EACH facet in your answer. If a facet is marked [GAP] or [PARTIAL], "
                "acknowledge the limited evidence rather than omitting that aspect.\n"
            )

        workflow_guideline = ""
        if _query_seeks_workflow_detail(query):
            workflow_guideline = (
                "10. The question asks for workflow, methodology, pipeline, or procedural detail. "
                "Answer with a **clear numbered list of stages** (1., 2., …) in the order described "
                "in the research results, using concrete names and actions from the text (systems, "
                "data objects, algorithms, repositories). Do **not** replace this with a short "
                "paragraph of generic themes (e.g. only \"tracking\" and \"prioritization\") when "
                "the results contain more specific steps—surface those steps.\n"
                "11. If the research results lack operational detail (only motivation or high-level "
                "goals), say so explicitly and summarize only what is literally supported; do not "
                "fill gaps with plausible-sounding but unsourced narrative.\n"
            )
            if facet_guideline:
                # Renumber so facet rule stays 10 and workflow rules follow.
                workflow_guideline = workflow_guideline.replace("10. ", "11. ", 1).replace(
                    "11. If the research",
                    "12. If the research",
                    1,
                )

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
{facet_guideline}{workflow_guideline}
Important: Do not provide an answer if the question contains hate speech, offensive language, discriminatory remarks, or harmful content.

Generate your answer:"""

        logger.info(f"Prompt length: {len(prompt)} chars")
        logger.info("Calling LLM for answer...")

        answer = await self.llm.complete(prompt, temperature=0.25)
        logger.info(f"Answer generated, length: {len(answer)} chars")
        logger.info(f"Answer content:\n{answer}")

        answer, citation_map = self._verify_citations(answer, papers)

        cited_indices = {c["index"] for c in citation_map["cited"]}
        if cited_indices and cited_indices != set(range(1, len(papers) + 1)):
            answer, papers = self._compact_citations(answer, papers, cited_indices)
            citation_map["compacted"] = True
            logger.info(
                f"Compacted citations: {len(cited_indices)} cited out of "
                f"{citation_map['total_papers']} → renumbered to [1]-[{len(papers)}]"
            )

        if papers:
            references_section = self._format_references_section(papers)
            if references_section:
                answer = answer.rstrip() + "\n\n" + references_section
                logger.info(f"References section added, total length: {len(answer)} chars")

        return answer, citation_map

    _CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    @classmethod
    def _verify_citations(
        cls, answer: str, papers: List[Dict[str, Any]]
    ) -> tuple[str, Dict[str, Any]]:
        """Validate citation markers in the generated answer.

        Handles both single ``[N]`` and multi-citation ``[N, M]`` brackets.
        Returns ``(cleaned_answer, citation_map)`` where:
        - Invalid references (N > number of papers or N < 1) are stripped.
        - Empty brackets left after stripping are removed.
        - ``citation_map`` records which papers were cited, uncited, and any
          invalid references that were removed.
        """
        valid_range = set(range(1, len(papers) + 1))

        found_refs: Set[int] = set()
        for m in cls._CITE_RE.finditer(answer):
            for num_str in m.group(1).split(","):
                found_refs.add(int(num_str.strip()))

        valid_refs = found_refs & valid_range
        invalid_refs = found_refs - valid_range

        if invalid_refs:
            def _strip_invalid(m: re.Match) -> str:
                nums = [int(s.strip()) for s in m.group(1).split(",")]
                kept = [n for n in nums if n in valid_range]
                if not kept:
                    return ""
                return "[" + ", ".join(str(n) for n in kept) + "]"

            answer = cls._CITE_RE.sub(_strip_invalid, answer)
            answer = re.sub(r"  +", " ", answer)
            logger.warning(
                f"Citation verification: stripped {len(invalid_refs)} invalid ref(s): "
                f"{sorted(invalid_refs)}"
            )

        cited_indices = sorted(valid_refs)
        uncited_indices = sorted(valid_range - valid_refs)

        citation_map: Dict[str, Any] = {
            "cited": [
                {"index": i, "title": papers[i - 1].get("title", ""), "doi": papers[i - 1].get("doi", "")}
                for i in cited_indices
            ],
            "uncited": [
                {"index": i, "title": papers[i - 1].get("title", "")}
                for i in uncited_indices
            ],
            "invalid_stripped": sorted(invalid_refs),
            "total_papers": len(papers),
            "cited_count": len(cited_indices),
        }

        if papers and len(uncited_indices) > len(papers) * 0.5:
            logger.warning(
                f"Citation coverage low: {len(cited_indices)}/{len(papers)} papers cited "
                f"({len(uncited_indices)} uncited) — possible over-retrieval"
            )
        else:
            logger.info(
                f"Citation verification: {len(cited_indices)}/{len(papers)} papers cited, "
                f"{len(invalid_refs)} invalid stripped"
            )

        return answer, citation_map

    @classmethod
    def _compact_citations(
        cls,
        answer: str,
        papers: List[Dict[str, Any]],
        cited_indices: set[int],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Remove uncited papers and renumber citation markers to be consecutive.

        Handles both single ``[N]`` and multi-citation ``[N, M]`` brackets.
        Given ``cited_indices={1, 3}`` out of 3 papers, the answer's ``[1]``
        stays ``[1]``, ``[3]`` becomes ``[2]``, and ``[1, 3]`` becomes
        ``[1, 2]``.  Only cited papers are returned in the new list.
        """
        old_to_new: Dict[int, int] = {}
        compacted_papers: List[Dict[str, Any]] = []
        for new_idx, old_idx in enumerate(sorted(cited_indices), 1):
            old_to_new[old_idx] = new_idx
            compacted_papers.append(papers[old_idx - 1])

        def _renumber(m: re.Match) -> str:
            nums = [int(s.strip()) for s in m.group(1).split(",")]
            renumbered = [old_to_new.get(n, n) for n in nums]
            return "[" + ", ".join(str(n) for n in renumbered) + "]"

        answer = cls._CITE_RE.sub(_renumber, answer)
        return answer, compacted_papers

    async def _generate_single_paper_answer(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        session: AgentSession,
        is_summary_request: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        """Generate answer for a single paper with full text — no map-reduce.

        Uses the full paper text directly. If is_summary_request, produces a
        comprehensive summary. Otherwise answers the specific question.
        """
        paper = papers[0]
        full_text = (paper.get("full_text") or "").strip()
        title = paper.get("title", "Unknown")
        doi = paper.get("doi", "")

        logger.info(
            f"Single-paper answer path: title={title[:60]!r}, "
            f"full_text_len={len(full_text)}, is_summary={is_summary_request}"
        )

        conversation_context = session.get_context_string()

        if is_summary_request:
            effective_question = (
                "Provide a comprehensive overview of this paper, covering: "
                "motivation and problem statement, methodology and approach, "
                "key contributions and innovations, main results and findings, "
                "limitations, and significance."
            )
            logger.info("Single-paper: URL-only query → summary mode")
        else:
            effective_question = query
            logger.info(f"Single-paper: specific question → focused answer")

        prompt = f"""You are a scientific research assistant. You have the full text of a single research paper.

Paper: {title}
DOI: {doi or 'N/A'}

{"Previous Conversation Context:" + chr(10) + conversation_context + chr(10) if conversation_context.strip() else ""}

Full Paper Text:
---
{full_text}
---

{effective_question}

Guidelines:
1. Be thorough and comprehensive — you have the full paper text
2. Maintain scientific precision and technical accuracy
3. Structure your answer with clear sections using markdown headers
4. Include specific details: named methods, numerical results, key equations or frameworks
5. If the paper describes a multi-component system, describe each component
6. Do NOT invent information — use only what is in the paper text above
7. Do NOT include a References section at the end — it will be appended separately

Generate your answer:"""

        logger.info(f"Single-paper prompt length: {len(prompt)} chars")
        answer = await self.llm.complete(prompt, temperature=0.25)
        logger.info(f"Single-paper answer generated: {len(answer)} chars")

        # Build citation map for single paper (always [1])
        citation_map: Dict[str, Any] = {
            "cited": [{"index": 1, "title": title, "doi": doi}],
            "uncited": [],
            "invalid_stripped": [],
            "total_papers": 1,
            "cited_count": 1,
            "single_paper": True,
        }

        # Append references section
        references_section = self._format_references_section(papers)
        if references_section:
            answer = answer.rstrip() + "\n\n" + references_section

        return answer, citation_map

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
        2 = Tangential -- shares field keywords but does not address the question
        3 = Substantively addresses a significant subtopic
        4 = Relevant, contributes to answer
        5 = Highly relevant, directly addresses query

        Only papers with score >= min_score are included in synthesis.
        KB-sourced papers (source=="kb_search") are always retained.
        """
        if not papers:
            return []

        n_papers = len(papers)

        # Build paper list for LLM -- use full abstract (no truncation)
        paper_lines = []
        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", "") or "No abstract available."
            paper_lines.append(f"[{i}] Title: {title}\n   Abstract: {abstract}")

        paper_list_str = "\n\n".join(paper_lines)

        prompt = (
            "You are evaluating research papers for relevance to a user's query.\n\n"
            f'User Query: "{query}"\n\n'
            f"Papers to evaluate:\n{paper_list_str}\n\n"
            "Score each paper's relevance on this scale:\n"
            "1 = Completely irrelevant to the query\n"
            "2 = Tangential -- shares general field keywords but does not address "
            "the specific question\n"
            "3 = Substantively addresses a significant subtopic of the query, but "
            "may not fully answer it. Mere domain overlap is a 2, not a 3.\n"
            "4 = Relevant -- directly contributes to answering the query\n"
            "5 = Highly relevant -- directly and comprehensively addresses the query\n\n"
            "Rules:\n"
            "- If the query names a specific system, method, or entity, only score 4+ "
            "if the paper discusses THAT entity. Named-entity mismatch = max 2.\n"
            "- Sharing general field keywords (e.g. 'metabolomics', 'pipeline') without "
            "addressing the specific question is NOT sufficient for 3+.\n"
            "- Base your score ONLY on the title and abstract text provided. "
            "Do not infer relevance from the paper's general research area.\n"
            f"- You MUST provide a score for EVERY paper numbered 1 through {n_papers}. "
            "Do not skip any paper.\n\n"
            "Example calibration: Query \"CRISPR gene editing in wheat\" ->\n"
            "  Paper \"RNA editing in rice\" = 2 (related technique, wrong organism "
            "and mechanism)\n"
            "  Paper \"CRISPR-Cas9 optimization in bread wheat\" = 5\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"scores": {"1": {"score": N, "reason": "..."}, '
            '"2": {"score": N, "reason": "..."}, ...}}'
        )

        def _parse_and_filter(response_text: str) -> Tuple[list, bool]:
            """Parse LLM response and return (filtered_papers, is_complete)."""
            import json as _json, re as _re

            json_match = _re.search(r"\{.*\}", response_text, _re.DOTALL)
            if not json_match:
                return papers, False

            scores_data = _json.loads(json_match.group())
            scores = scores_data.get("scores", {})

            # Check completeness: every paper 1..n must have a score
            all_keys = {str(i) for i in range(1, n_papers + 1)}
            returned_keys = set(scores.keys())
            is_complete = returned_keys >= all_keys

            filtered = []
            for i, paper in enumerate(papers, 1):
                paper_key = str(i)
                is_kb = paper.get("source") == "kb_search"

                if paper_key in scores:
                    score_info = scores[paper_key]
                    score = (
                        score_info.get("score", 0)
                        if isinstance(score_info, dict)
                        else int(score_info)
                    )
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = (
                        score_info.get("reason", "") if isinstance(score_info, dict) else ""
                    )
                else:
                    # No score returned for this paper
                    if is_kb:
                        paper["relevance_score"] = min_score
                        paper["relevance_reason"] = "No LLM score (KB paper, retained)"
                    else:
                        paper["relevance_score"] = 0
                        paper["relevance_reason"] = "No LLM score provided"
                        logger.info(
                            f"Paper [{i}] '{paper.get('title', '')[:50]}...' "
                            f"score: MISSING - DISCARDED (non-KB, no score)"
                        )
                        continue

                score = paper["relevance_score"]

                # KB papers are always retained
                if is_kb:
                    paper["relevance_score"] = max(score, min_score)
                    if "KB paper" not in paper.get("relevance_reason", ""):
                        paper["relevance_reason"] = (
                            f"{paper['relevance_reason']} (KB paper, retained)"
                        ).strip()
                    filtered.append(paper)
                    logger.info(
                        f"Paper [{i}] '{paper.get('title', '')[:50]}...' "
                        f"score: {paper['relevance_score']} - INCLUDED (KB paper)"
                    )
                elif score >= min_score:
                    filtered.append(paper)
                    logger.info(
                        f"Paper [{i}] '{paper.get('title', '')[:50]}...' "
                        f"score: {score} - INCLUDED"
                    )
                else:
                    logger.info(
                        f"Paper [{i}] '{paper.get('title', '')[:50]}...' "
                        f"score: {score} - FILTERED"
                    )

            logger.info(
                f"Relevance filtering: {len(filtered)}/{n_papers} papers included "
                f"(min_score={min_score}, complete={is_complete})"
            )
            return filtered, is_complete

        try:
            response = await self.llm.complete(prompt, temperature=0.1)
            filtered, is_complete = _parse_and_filter(response)

            # If scores are incomplete, retry once with stricter instruction
            if not is_complete:
                logger.warning(
                    f"Relevance scorer returned incomplete scores "
                    f"({len(filtered)}/{n_papers} kept). Retrying once."
                )
                retry_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous response was missing scores for "
                    f"some papers. You MUST return exactly {n_papers} entries with "
                    f'keys "1" through "{n_papers}". Try again.'
                )
                response = await self.llm.complete(retry_prompt, temperature=0.1)
                retry_filtered, retry_complete = _parse_and_filter(response)
                if retry_complete or len(retry_filtered) > len(filtered):
                    filtered = retry_filtered
                    is_complete = retry_complete

            return filtered

        except Exception as e:
            logger.error(f"Error scoring papers for relevance: {e}")
            # Never treat unscored literature as relevance-approved (UI + downstream use
            # relevance_score). KB hits are pre-trusted; literature without a successful
            # score is excluded from the paper list so synthesis falls back to raw step
            # text rather than unfiltered OpenAlex/SciLEx blobs.
            retained: List[Dict[str, Any]] = []
            for p in papers:
                if p.get("source") == "kb_search":
                    p.setdefault("relevance_score", min_score)
                    p.setdefault(
                        "relevance_reason",
                        "KB paper retained after relevance scoring error",
                    )
                    retained.append(p)
                else:
                    p["relevance_score"] = 0
                    p["relevance_reason"] = "Excluded: relevance scoring error"
            if retained:
                logger.warning(
                    "Relevance scoring failed: returning %d KB paper(s) only; "
                    "excluding %d literature paper(s) without scores.",
                    len(retained),
                    len(papers) - len(retained),
                )
                return retained
            logger.warning(
                "Relevance scoring failed with no KB papers; returning no scored papers "
                "(answer may use raw search result text only)."
            )
            return []

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

    async def _scilex_search(
        self,
        query: str,
        max_results: int = 10,
        step_id: str = "",
        session: Optional[AgentSession] = None,
    ) -> str:
        """Search academic literature using SciLEx (multi-API aggregation).

        Falls back to direct OpenAlex if SciLEx is not available.
        """
        logger.info(f"SciLEx search: '{query}'")

        try:
            papers = await self.scilex_adapter.search(
                query=query,
                max_results=max_results,
                apis=["semantic_scholar", "openalex", "pubmed"],
            )

            if papers:
                paper_dicts = []
                for p in papers:
                    paper_dict = {
                        "id": p.id,
                        "title": p.title,
                        "authors": [a.name for a in p.authors[:3]],
                        "year": p.year,
                        "cited_by_count": p.citation_count or 0,
                        "abstract": p.abstract[:800] if p.abstract else "",
                        "doi": p.doi or "",
                        "pdf_url": p.pdf_url or "",
                        "source": "literature_search",
                        "_step_id": step_id,
                    }
                    paper_dicts.append(paper_dict)

                async with self._found_papers_lock:
                    if hasattr(self, "_found_papers"):
                        self._found_papers.extend(paper_dicts)
                    self._accumulate_lit_evidence(paper_dicts, step_id, session)

                logger.info(f"SciLEx search found {len(paper_dicts)} papers")
                return self._format_paper_list(paper_dicts)
            else:
                logger.warning("SciLEx returned no results, falling back to OpenAlex")
                return await self._fallback_openalex_search(query, max_results, step_id=step_id, session=session)

        except Exception as e:
            logger.error(f"SciLEx search failed: {e}, falling back to OpenAlex")
            return await self._fallback_openalex_search(query, max_results, step_id=step_id, session=session)

    async def _fallback_openalex_search(
        self,
        query: str,
        max_results: int = 10,
        step_id: str = "",
        session: Optional[AgentSession] = None,
    ) -> str:
        """Fallback: Search OpenAlex directly via httpx."""
        import httpx

        search_terms = query.strip()
        logger.info(f"OpenAlex fallback search: '{search_terms}'")

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
                        "source": "literature_search",
                        "_step_id": step_id,
                    }
                    papers.append(paper)

                papers = self._dedupe_paper_dicts(papers)

                async with self._found_papers_lock:
                    if hasattr(self, "_found_papers"):
                        self._found_papers.extend(papers)
                    self._accumulate_lit_evidence(papers, step_id, session)

                logger.info(f"OpenAlex fallback found {len(papers)} papers")
                return self._format_paper_list(papers)
        except Exception as e:
            logger.error(f"OpenAlex fallback failed: {e}")
            return f"Literature search failed: {e}"

    def _accumulate_lit_evidence(
        self,
        paper_dicts: List[Dict[str, Any]],
        step_id: str,
        session: Optional[AgentSession],
    ) -> None:
        """Push literature search results into the faceted evidence store."""
        if not session or session.evidence is None or not paper_dicts:
            return
        ev = session.evidence
        facet = ev.facet_for_step(step_id) if step_id else None
        fk = facet.query.lower()[:120] if facet else "main"
        hits = [
            {
                "title": p.get("title", ""),
                "doi": p.get("doi", ""),
                "excerpt": (p.get("abstract") or "")[:600],
                "step_id": step_id,
                "source": "literature_search",
            }
            for p in paper_dicts
        ]
        ev.add_hits(hits, step_id=step_id, facet_key=fk)

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
        """Higher is better: more in-corpus text, more abstract, citations, newer."""
        doi = self._normalize_doi_for_dedupe(p.get("doi"))
        is_biorxiv = doi.startswith("10.1101") if doi else False
        return (
            len(p.get("full_text") or ""),
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
        """Download and parse a single paper's content.

        Args:
            paper: Paper dict with at least 'doi' and 'title'

        Returns:
            Enriched paper dict with 'full_text' and 'pdf_downloaded' fields
        """
        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.pipeline.parsers.pdf import PDFParser

        doi = paper.get("doi", "")
        if not doi:
            paper["pdf_downloaded"] = False
            return paper

        parser = PDFParser()

        try:
            result = await retrieve_paper_content(
                doi=doi,
                pdf_parser=parser,
                unpaywall_email="perspicacite@example.com",
            )

            if result.success and result.full_text:
                paper["full_text"] = result.full_text
                paper["pdf_downloaded"] = True
            else:
                paper["pdf_downloaded"] = False

        except Exception as e:
            logger.warning(f"Failed to download/parse content for {doi}: {e}")
            paper["pdf_downloaded"] = False

        return paper

    async def _download_and_enrich_papers(
        self, 
        papers: List[Dict[str, Any]], 
        relevance_threshold: int = 3,
        max_papers: int = 10
    ) -> List[Dict[str, Any]]:
        """Download PDFs for relevant papers and extract full text.
        
        For literature surveys, downloads ALL relevant papers (score >= threshold)
        up to a safety cap. This ensures comprehensive coverage rather than
        arbitrary limits.
        
        Args:
            papers: List of paper dicts with DOI and relevance_score
            relevance_threshold: Minimum relevance score to download (default 3)
            max_papers: Safety cap on downloads (default 10)
            
        Returns:
            Papers with added 'full_text' field where download succeeded
        """
        # Filter to relevant papers only (threshold-based, not hard limit)
        relevant_papers = [
            p for p in papers 
            if p.get("relevance_score", 0) >= relevance_threshold and p.get("doi")
        ][:max_papers]
        
        # Prioritize open access papers first (they're more likely to download successfully)
        download_candidates = sorted(
            relevant_papers,
            key=lambda p: (not p.get("open_access", {}).get("is_oa", False), p.get("relevance_score", 0)),
            reverse=True
        )
        
        if not download_candidates:
            logger.info("No papers met the relevance threshold for download")
            return papers
        
        logger.info(f"Attempting to download {len(download_candidates)} relevant papers (threshold >= {relevance_threshold})")
        
        enriched = []
        for paper in papers:
            doi = paper.get("doi", "")
            if not doi or paper not in download_candidates:
                enriched.append(paper)
                continue
            
            enriched_paper = await self._download_single_paper(paper)
            enriched.append(enriched_paper)
        
        return enriched

    async def _get_citation_network(self, doi: str, direction: str = "both") -> Dict[str, Any]:
        """
        Get citation network for a paper using SciLEx/OpenCitations.
        
        Args:
            doi: DOI of the paper
            direction: "citations" (papers citing this), "references" (papers cited by this), 
                      or "both"
            
        Returns:
            Dict with citation network information
        """
        logger.info(f"Getting citation network for DOI: {doi} (direction: {direction})")
        
        # Run in thread pool since SciLEx citation tools are synchronous
        import asyncio
        
        try:
            # Try to import SciLEx citation tools
            from scilex.citations.citations_tools import getRefandCitFormatted
            
            result = await asyncio.to_thread(getRefandCitFormatted, doi)
            citations_data, stats = result
            
            network = {
                "doi": doi,
                "citing": [],  # Papers that cite this paper
                "cited": [],   # Papers that this paper cites (references)
                "stats": stats,
            }
            
            if direction in ("citations", "both"):
                network["citing"] = citations_data.get("citing", [])
                
            if direction in ("references", "both"):
                network["cited"] = citations_data.get("cited", [])
            
            logger.info(
                f"Citation network for {doi}: "
                f"{len(network['citing'])} citing, {len(network['cited'])} references"
            )
            
            return network
            
        except ImportError:
            logger.warning("SciLEx citation tools not available")
            return {"doi": doi, "citing": [], "cited": [], "error": "Citation tools not available"}
        except Exception as e:
            logger.error(f"Failed to get citation network: {e}")
            return {"doi": doi, "citing": [], "cited": [], "error": str(e)}
