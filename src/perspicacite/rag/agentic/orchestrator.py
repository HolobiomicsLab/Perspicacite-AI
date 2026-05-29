"""Main agentic orchestrator with session management."""

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from perspicacite.logging import get_logger
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.provenance.context import get_collector
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
from perspicacite.rag.utils import format_references_academic
from perspicacite.retrieval.hybrid import hybrid_retrieval

# SciLEx integration
from perspicacite.search.scilex_adapter import SciLExAdapter

from .intent import IntentClassifier
from .planner import Plan, ResearchPlanner, Step, StepType, _log_steps_detail

logger = get_logger("perspicacite.rag.agentic.orchestrator")

# Cap per-paper extraction LLM calls during final answer (map-reduce style).
MAP_REDUCE_MAX_PAPERS = 8


def _strip_html_if_needed(text: str) -> str:
    """If text is raw HTML (starts with <!DOCTYPE/<html/<head), strip tags.

    Some upstream content paths can leak the raw publisher landing page into
    ``paper.full_text`` (e.g. a Nature.com HTML page passed through when no
    PDF could be downloaded). Surfacing that into events bloats the SSE
    payload with markup, breaks downstream chunking, and leaks site
    navigation cruft into the UI. Defensive cleanup at the emit boundary.
    """
    if not text or not isinstance(text, str):
        return text
    stripped = text.lstrip()[:200].lower()
    if not (
        stripped.startswith("<!doctype")
        or stripped.startswith("<html")
        or stripped.startswith("<head")
        or stripped.startswith("<?xml")
    ):
        return text
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "lxml" if _has_lxml() else "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
            tag.decompose()
        cleaned = soup.get_text(separator=" ", strip=True)
        # Collapse runs of whitespace
        import re as _re
        return _re.sub(r"\s+", " ", cleaned).strip()
    except Exception:
        return text


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def _clean_papers_for_event(papers: list[dict]) -> list[dict]:
    """Sanitize papers list before emitting in a `papers_found` event."""
    if not papers:
        return papers
    out = []
    for p in papers:
        if not isinstance(p, dict):
            out.append(p)
            continue
        ft = p.get("full_text")
        if isinstance(ft, str):
            cleaned = _strip_html_if_needed(ft)
            if cleaned != ft:
                p = {**p, "full_text": cleaned}
        out.append(p)
    return out

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
    step_ids: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    _seen_keys: set[str] = field(default_factory=set, repr=False)

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

        relevance_scores = [e.get("relevance_score", 3) for e in self.entries]
        avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 3.0
        rel_score = min(max((avg_rel - 1) / 4, 0.0), 1.0)  # map [1,5] → [0,1]

        full_text_count = sum(
            1 for e in self.entries if e.get("pdf_downloaded") or e.get("full_text")
        )
        ft_score = min(full_text_count / max(n, 1), 1.0)

        return round(0.45 * count_score + 0.35 * rel_score + 0.20 * ft_score, 4)

    def _entry_key(self, e: dict[str, Any]) -> str:
        doi = (e.get("doi") or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        title = (e.get("title") or "").strip().lower()[:120]
        return f"title:{title}" if title else ""

    def _add_entry(self, entry: dict[str, Any]) -> bool:
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

    facets: dict[str, EvidenceFacet] = field(default_factory=dict)

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
        hits: list[dict[str, Any]],
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
    def add_kb_hits(
        self, hits: list[dict[str, Any]], step_id: str = "", facet_key: str = "main"
    ) -> None:
        self.add_hits(hits, step_id=step_id, facet_key=facet_key)

    @property
    def all_entries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for f in self.facets.values():
            out.extend(f.entries)
        return out

    def gap_summary(self) -> dict[str, str]:
        """Return {facet_key: status} for all facets."""
        return {k: f.status for k, f in self.facets.items()}

    def facet_confidences(self) -> dict[str, float]:
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
        sections: list[str] = []
        for _key, facet in self.facets.items():
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
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSession:
    """Persistent session for agent conversations."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    knowledge_base: DynamicKnowledgeBase | None = None
    research_findings: list[dict[str, Any]] = field(default_factory=list)
    kb_name: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    # User preferences for research depth (can be set per query)
    max_papers_to_download: int | None = None  # Override orchestrator default
    evidence: EvidenceStore | None = None

    # MCP cancellation tracking: when set, the orchestrator's iteration loop
    # checks ``cancellation.is_cancelled(task_id)`` at the top of each
    # iteration and returns early. Threaded from ``RAGRequest.task_id`` via
    # AgenticRAGMode → orchestrator.chat(task_id=...).
    task_id: str | None = None

    def add_message(self, role: str, content: str, metadata: dict | None = None):
        """Add a message to the session."""
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
        self.last_active = datetime.now()

    def get_conversation_history(self, limit: int = 10) -> list[dict]:
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
            if not response:
                logger.warning(
                    "agentic_quality_assessment_error",
                    error="empty_response_from_llm",
                )
                return False, ["Assessment error: empty LLM response"], 0.0

            # Parse JSON response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            result = json.loads(json_match.group()) if json_match else json.loads(response)

            return (
                result.get("is_sufficient", False),
                result.get("missing_aspects", []),
                result.get("confidence", 0.5),
            )

        except Exception as e:
            logger.warning("agentic_quality_assessment_error", error=str(e))
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
        map_reduce_max_papers: int = 8,
        recency_weight: float | None = None,
        recency_half_life_years: float | None = None,
        kb_metas: list | None = None,
        config: Any = None,
        app_state: Any = None,
    ):
        self.llm = llm_client
        self.config = config
        self.app_state = app_state
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

        # Cap on per-paper LLM extraction calls during final map-reduce answer synthesis
        self.map_reduce_max_papers = map_reduce_max_papers

        # Recency weighting params (passed by callers; stored for use in retrieval)
        self.recency_weight = recency_weight
        self.recency_half_life_years = recency_half_life_years
        self.kb_metas: list = list(kb_metas or [])
        # Optional multi-KB names for KB_SEARCH step (set per-request by callers
        # via the agentic_request_overrides context manager or by direct attr set).
        self.kb_names: list[str] = []

        self.intent_classifier = IntentClassifier(llm_client)
        self.planner = ResearchPlanner(llm_client)
        self.quality_assessor = DocumentQualityAssessor(llm_client)

        # Session management
        self.sessions: dict[str, AgentSession] = {}
        self._found_papers_lock = asyncio.Lock()

        # SciLEx adapter for literature search (multi-API aggregation)
        self.scilex_adapter = SciLExAdapter()

    def _build_kb_retriever(self, default_kb_name: str | None = None):
        """Return a retriever for the KB_SEARCH step.

        Multi-KB request → MultiKBRetriever fanning across collections.
        Single KB → DynamicKnowledgeBase (backward compatible).

        Resolves the KB name list in this order:
          1. agentic_request_overrides context var kb_metas (per-request, race-free)
          2. self.kb_names (set on the orchestrator)
          3. [default_kb_name] (typically session.kb_name)
        """
        from types import SimpleNamespace

        from perspicacite.rag.agentic.context import get_current_kb_metas
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
        from perspicacite.retrieval.multi_kb import MultiKBRetriever

        ctx_metas = get_current_kb_metas()
        if ctx_metas and len(ctx_metas) > 1:
            return MultiKBRetriever(
                vector_store=self.vector_store,
                embedding_service=self.embeddings,
                kb_metas=ctx_metas,
            )

        names = list(getattr(self, "kb_names", None) or [])
        if not names and default_kb_name:
            names = [default_kb_name]

        if len(names) > 1:
            metas = [
                SimpleNamespace(
                    name=n,
                    collection_name=chroma_collection_name_for_kb(n),
                    embedding_model=None,
                )
                for n in names
            ]
            return MultiKBRetriever(
                vector_store=self.vector_store,
                embedding_service=self.embeddings,
                kb_metas=metas,
            )

        single = names[0] if names else default_kb_name
        dkb = DynamicKnowledgeBase(
            vector_store=self.vector_store,
            embedding_service=self.embeddings,
        )
        if single:
            dkb.collection_name = chroma_collection_name_for_kb(single)
            dkb._initialized = True
        return dkb

    def get_or_create_session(self, session_id: str | None = None) -> AgentSession:
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
        session_id: str | None = None,
        kb_name: str | None = None,
        stream: bool = True,
        max_papers_to_download: int | None = None,
        databases: list[str] | None = None,
        task_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Main chat entry point with true agentic behavior.

        Args:
            query: User's research question
            session_id: Optional session ID for persistence
            kb_name: Optional knowledge base to search first
            stream: Whether to stream responses
            max_papers_to_download: Override default max papers to download (for user preference)
            databases: Optional list of user-selected databases (e.g. ``["semantic_scholar", "europepmc"]``)
                used to scope SciLEx and standalone-provider fan-out. When None, defaults to
                ``["semantic_scholar", "openalex", "pubmed"]`` (legacy behaviour).

        Yields:
            Dict with type: "thinking", "tool_call", "tool_result", "answer", "papers_found"
        """
        logger.info(
            "agentic_new_chat_request",
            query=query,
            session_id=session_id,
            kb=kb_name or "none",
            max_papers_to_download=max_papers_to_download or self.max_papers_to_download,
            databases=databases or "default",
        )

        session = self.get_or_create_session(session_id)
        logger.info("agentic_session_resolved", session_id=session.session_id)
        session.add_message("user", query)
        session.kb_name = kb_name
        # Stash the user's database selection on the session so tool-execution
        # helpers (``_scilex_search``) can honour it without changing every
        # internal call signature. ``None`` means "use defaults".
        session.databases = databases  # type: ignore[attr-defined]
        # MCP cancellation: the iteration loop reads session.task_id and
        # bails out when cancellation.is_cancelled(task_id) flips True.
        # Threaded here from RAGRequest.task_id via AgenticRAGMode.
        session.task_id = task_id
        session.evidence = EvidenceStore()

        # Store user preference for download cap in session
        if max_papers_to_download is not None:
            session.max_papers_to_download = max_papers_to_download

        logger.info("agentic_session_message_count", count=len(session.messages))

        # Clear accumulated papers from previous requests
        self._found_papers = []
        if not hasattr(session, "found_papers_archive"):
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
            logger.info(
                "agentic_url_prefetch",
                title=title[:80],
                full_text_chars=len(url_paper.get("full_text", "")),
            )
        else:
            if _URL_RE.search(query):
                yield {
                    "type": "thinking",
                    "message": "Could not fetch paper directly, searching databases instead...",
                }

        # Step 1: Classify intent
        yield {"type": "phase_transition", "phase": "classify"}
        yield {"type": "thinking", "message": "Analyzing your query..."}

        intent_result = await self.intent_classifier.classify(
            query=query,
            conversation_history=session.get_conversation_history(),
            active_kb_name=kb_name,
        )
        logger.info(
            "agentic_intent_classified",
            intent=intent_result.intent.name,
            confidence=intent_result.confidence,
            query_complexity=getattr(intent_result, "query_complexity", "simple"),
            query_complexity_source=getattr(intent_result, "query_complexity_source", ""),
            suggested_tools=intent_result.suggested_tools,
        )

        # Provenance: trace intent classification
        _c = get_collector()
        if _c is not None:
            _c.add_trace("intent", detail={"value": intent_result.intent.name})

        yield {
            "type": "intent_result",
            "intent": intent_result.intent.name,
            "confidence": round(float(intent_result.confidence), 3),
            "reasoning": intent_result.reasoning,
            "suggested_tools": list(intent_result.suggested_tools or []),
            "query_complexity": getattr(intent_result, "query_complexity", "simple"),
        }
        yield {
            "type": "thinking",
            "message": f"Intent: {intent_result.intent.name.replace('_', ' ').title()}",
            "details": intent_result.reasoning,
        }

        # Step 2: Create dynamic plan
        yield {"type": "phase_transition", "phase": "plan"}
        yield {"type": "thinking", "message": "Creating research plan..."}

        # Available tools: registered tools (excluding deactivated ones) + built-in
        available_tools = [t for t in self.tools.list_tools() if t != "lotus_search"] + [
            "literature_search",
            "kb_search",
            "paper_lookup",
        ]
        logger.info("agentic_available_tools", tools=available_tools)
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
                    subs = list(
                        dict.fromkeys(await self.planner.composite_subqueries_with_llm(clean_query))
                    )[:3]
                    for i, sq in enumerate(subs):
                        plan.steps.insert(
                            i,
                            Step(
                                id=f"inject_kb_{i + 1}",
                                type=StepType.KB_SEARCH,
                                description=f"Search knowledge base '{kb_name}'",
                                tool="kb_search",
                                tool_input={"query": sq},
                                depends_on=[],
                            ),
                        )
                    logger.info(
                        "agentic_kb_search_injected_composite",
                        count=len(subs),
                        kb_name=kb_name,
                        queries=subs,
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
                        "agentic_kb_search_injected",
                        kb_name=kb_name,
                        query=clean_query,
                    )
            elif qcomp == "composite":
                await self._maybe_upgrade_single_kb_to_composite_parallel(plan, query, kb_name)
            plan.estimated_steps = len(plan.steps)

        logger.info(
            "agentic_plan_reasoning",
            reasoning_chars=len(plan.reasoning),
            reasoning=plan.reasoning,
        )
        _log_steps_detail(plan.steps, "Orchestrator plan (final, after KB inject if any)")

        # Provenance: trace plan
        if _c is not None:
            _c.add_trace("plan", detail={"steps": len(plan.steps)})

        yield {
            "type": "plan",
            "reasoning": plan.reasoning,
            "can_answer_from_history": bool(getattr(plan, "can_answer_from_history", False)),
            "estimated_steps": len(plan.steps),
            "steps": [
                {
                    "id": s.id,
                    "type": s.type.value,
                    "tool": s.tool or s.type.value,
                    "description": s.description,
                    "query": (s.tool_input or {}).get("query", ""),
                    "depends_on": list(s.depends_on or []),
                }
                for s in plan.steps
            ],
        }

        self._register_evidence_facets(session, plan)

        if plan.can_answer_from_history:
            # Planner decided we have enough — check if pre-fetched paper has full text
            if url_paper and url_paper.get("full_text"):
                logger.info("agentic_plan_answer_from_history_with_url_paper")
                yield {"type": "phase_transition", "phase": "synth"}
                yield {"type": "thinking", "message": "Generating answer from retrieved paper..."}
                papers = [url_paper]
                is_summary = not any(
                    w in query.lower()
                    for w in (
                        "what",
                        "how",
                        "why",
                        "which",
                        "method",
                        "approach",
                        "result",
                        "compare",
                        "find",
                    )
                )
                answer, citation_map = await self._generate_single_paper_answer(
                    query,
                    papers,
                    session,
                    is_summary_request=is_summary,
                )
                session.add_message(
                    "assistant",
                    answer,
                    {
                        "intent": intent_result.intent.name,
                        "steps_completed": 0,
                        "tools_used": [],
                        "can_answer_from_history": True,
                    },
                )
                yield {
                    "type": "answer",
                    "content": answer,
                    "session_id": session.session_id,
                    "citations": citation_map,
                }
                # Emit papers_found AFTER answer so the UI can append the
                # "Add to KB" curation block. The answer handler replaces
                # assistantDiv.innerHTML, so a papers_found emitted before
                # the answer would be wiped.
                yield {"type": "papers_found", "papers": _clean_papers_for_event(papers)}
                yield {"type": "phase_transition", "phase": "done"}
                return
            else:
                yield {
                    "type": "thinking",
                    "message": "I can answer from our conversation history...",
                }

        # Step 3: Execute plan iteratively
        yield {"type": "phase_transition", "phase": "retrieve"}
        step_results: dict[str, Any] = {}
        completed_steps: list[Step] = []
        replan_count = 0

        for iteration in range(self.max_iterations):
            from perspicacite.rag.cancellation import is_cancelled
            _tid = getattr(session, "task_id", None)
            if _tid and is_cancelled(_tid):
                logger.info("agentic_cancelled", task_id=_tid, iteration=iteration)
                return

            logger.info(
                "agentic_iteration_start",
                iteration=iteration + 1,
                max_iterations=self.max_iterations,
            )

            # Provenance: trace iteration start
            if _c is not None:
                _c.add_trace("iteration", detail={"n": iteration})

            batch = self._get_next_parallel_batch(plan, completed_steps, step_results)
            if not batch:
                logger.info("agentic_no_more_steps")
                break

            to_run: list[Step] = []
            for s in batch:
                if s.condition and not self._evaluate_condition(s.condition, step_results):
                    logger.info("agentic_step_condition_not_met", step_id=s.id)
                    completed_steps.append(s)
                else:
                    to_run.append(s)
            if not to_run:
                continue

            if len(to_run) > 1:
                logger.info(
                    "agentic_parallel_batch",
                    steps=[f"{s.id} ({s.type.value})" for s in to_run],
                )

            async def _run_step(step: Step) -> tuple[Step, Any, float]:
                t0 = time.time()
                res = await self._execute_step(step, query, step_results, session)
                return step, res, time.time() - t0

            for s in to_run:
                # Provenance: trace each tool invocation
                if _c is not None:
                    _c.add_trace("tool", detail={"name": s.tool or s.type.value})
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
                preview_len = min(2000, len(result_str))
                logger.info(
                    "agentic_step_complete",
                    step_id=step.id,
                    duration_s=round(step_duration, 2),
                    result_chars=len(result_str),
                    result_preview=(
                        result_str[:preview_len]
                        + ("...[truncated]" if len(result_str) > preview_len else "")
                    ),
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
                    logger.info(
                        "agentic_step_complete",
                        step_id=step.id,
                        duration_s=round(step_duration, 2),
                        result_chars=len(result_str),
                        result_preview=result_str[:100] if result_str else "",
                    )
                    step_results[step.id] = result
                    completed_steps.append(step)
                    yield {
                        "type": "tool_result",
                        "step": step.id,
                        "result_summary": self._summarize_result(result),
                    }
                batch_for_eval = to_run

            # Surface any search notices (e.g. "arXiv rate-limited, drew from
            # OpenAlex instead") so the trail explains unexpected sources.
            _notices = getattr(session, "search_notices", None)
            if _notices:
                for _n in _notices:
                    yield {"type": "thinking", "message": _n}
                session.search_notices = []  # type: ignore[attr-defined]

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
                    "agentic_progress_evaluation",
                    decision=decision,
                    gap_facets=eval_result["gap_facets"],
                    missing_aspects=eval_result["missing_aspects"][:3],
                )

                if decision == "replan":
                    if replan_count >= MAX_REPLANS:
                        logger.info(
                            "agentic_replan_budget_exhausted",
                            replan_count=replan_count,
                            max_replans=MAX_REPLANS,
                        )
                        break
                    replan_count += 1
                    ev_summary = session.evidence.to_prompt_block() if session.evidence else ""
                    plan = await self.planner.replan(
                        query,
                        plan,
                        completed_steps,
                        step_results,
                        eval_result["evaluation_text"] or "Need more specific search",
                        evidence_summary=ev_summary or None,
                    )
                    self._register_evidence_facets(session, plan)
                    # Provenance: trace replan event
                    if _c is not None:
                        _c.add_trace(
                            "replan",
                            detail={
                                "iteration": replan_count,
                                "gap_facets": eval_result.get("gap_facets", []),
                            },
                        )
                    yield {
                        "type": "thinking",
                        "message": f"Adjusting research plan (replan {replan_count}/{MAX_REPLANS})...",
                    }
                    yield {
                        "type": "replan",
                        "iteration": replan_count,
                        "max_replans": MAX_REPLANS,
                        "reason": eval_result.get("evaluation_text", ""),
                        "gap_facets": eval_result.get("gap_facets", []),
                        "new_step_count": len(plan.steps),
                    }
                elif decision == "answer":
                    logger.info("agentic_sufficient_results")
                    break

        logger.info(
            "agentic_execution_complete",
            completed_steps=len(completed_steps),
            step_result_keys=list(step_results.keys()),
        )

        # Step 4: Extract and process papers with progress updates
        yield {"type": "thinking", "message": "Extracting papers from search results..."}
        papers = self._extract_papers_from_results(step_results)
        logger.info("agentic_papers_extracted", count=len(papers))

        all_from_kb = all(p.get("source") == "kb_search" for p in papers) if papers else False
        if all_from_kb and papers:
            logger.info("agentic_kb_papers_skip_relevance_scoring", count=len(papers))
            # Ensure KB papers carry a non-zero relevance_score so they pass the
            # download-candidates filter below (which requires >= relevance_threshold).
            for p in papers:
                if not p.get("relevance_score"):
                    p["relevance_score"] = 4  # KB papers are pre-trusted
            yield {
                "type": "thinking",
                "message": f"Using {len(papers)} KB papers (relevance scoring skipped)",
            }
        else:
            yield {"type": "thinking", "message": f"Scoring {len(papers)} papers for relevance..."}
            papers = await self._score_papers_for_relevance(query, papers, min_score=3)
            included_count = len([p for p in papers if p.get("relevance_score", 0) >= 3])
            yield {
                "type": "thinking",
                "message": f"Relevance filtering: {included_count}/{len(papers)} papers included",
            }

        # Download full text for relevant papers (threshold-based, not hard limit)
        # For literature surveys, comprehensive coverage is important - download ALL relevant papers
        # up to a safety cap. Configurable via relevance_threshold and max_papers_to_download.
        # Use session-specific limit if user provided it, otherwise use orchestrator default
        max_download = session.max_papers_to_download or self.max_papers_to_download

        download_candidates = [
            p for p in papers if p.get("relevance_score", 0) >= self.relevance_threshold
        ][:max_download]

        if download_candidates:
            yield {
                "type": "thinking",
                "message": f"Attempting to download {len(download_candidates)} relevant papers for full text analysis...",
            }

            downloaded_count = 0
            for i, paper in enumerate(download_candidates, 1):
                title = re.sub(r"[{}]", "", paper.get("title", "Unknown") or "Unknown")[:50]
                yield {
                    "type": "thinking",
                    "message": f"Downloading paper {i}/{len(download_candidates)}: {title}...",
                }

                enriched = await self._download_single_paper(paper)
                if enriched.get("pdf_downloaded"):
                    downloaded_count += 1
                    yield {"type": "thinking", "message": f"✓ Downloaded paper {i}: {title}"}
                else:
                    yield {"type": "thinking", "message": f"✗ Paper {i} not available: {title}"}

            yield {
                "type": "thinking",
                "message": f"Downloaded {downloaded_count}/{len(download_candidates)} papers successfully",
            }
        else:
            yield {
                "type": "thinking",
                "message": "No papers met the relevance threshold for full-text download",
            }

        # Generate final answer
        yield {"type": "phase_transition", "phase": "synth"}
        yield {"type": "thinking", "message": "Synthesizing answer..."}
        answer, citation_map = await self._generate_answer(
            query=query, plan=plan, step_results=step_results, session=session, papers=papers
        )

        overall_conf = session.evidence.overall_confidence() if session.evidence else None

        # Bug B-9: when synthesis returns an empty / whitespace-only answer
        # (most often because every download_papers call failed and the
        # answer-generation prompt ended up content-less), the chat router
        # ships answer.content="" + sources=[]. Replace with a graceful
        # fallback so the user sees what *was* retrieved.
        if not (answer or "").strip():
            if papers:
                bullet_list = "\n".join(
                    f"- {(p.get('title') or 'Untitled')[:160]}"
                    for p in papers[:8]
                )
                answer = (
                    "I could not synthesize a coherent narrative from the "
                    "retrieved sources, but the following papers were found "
                    "to be relevant to your query:\n\n" + bullet_list
                )
                logger.warning(
                    "agentic_empty_answer_replaced_with_paper_list",
                    n_papers=len(papers),
                )
            else:
                answer = (
                    "I could not find any relevant papers for this query. "
                    "Try rephrasing or broadening the search terms."
                )
                logger.warning("agentic_empty_answer_no_papers")

        session.add_message(
            "assistant",
            answer,
            {
                "intent": intent_result.intent.name,
                "steps_completed": len(completed_steps),
                "tools_used": [s.tool for s in completed_steps if s.tool],
            },
        )

        answer_event: dict[str, Any] = {
            "type": "answer",
            "content": answer,
            "session_id": session.session_id,
            "citations": citation_map,
        }
        if overall_conf is not None:
            answer_event["confidence"] = round(overall_conf, 3)
        yield answer_event

        # Yield found papers so the UI can offer "Add to KB" and so SSE
        # clients can render source cards.
        #
        # Two emissions:
        #   1. papers_found  → ALL retrieved papers, regardless of relevance
        #      filtering. Previously this was filtered to (source=="kb_search"
        #      OR relevance_score>=3), which silently dropped papers when the
        #      LLM relevance scorer was strict — leading to empty source cards
        #      even when KB retrieval found the right paper (Bug B-9 in the
        #      2026-05-25 mode-consistency audit). UI clients can distinguish
        #      KB vs web by inspecting source=="kb_search" / relevance_score.
        #   2. papers_to_curate → the original "relevance ≥ 3 and not in KB"
        #      subset. Kept as a separate event so the UI's "Add to KB"
        #      block (which calls /api/kb/{name}/papers) doesn't re-ingest
        #      papers that are already there.
        if papers:
            yield {"type": "papers_found", "papers": _clean_papers_for_event(papers)}
        elif getattr(self, "_found_papers", None):
            # papers list was filtered to empty but raw retrievals exist —
            # surface them so the UI shows *something* (with the caveat that
            # the LLM judged them low-relevance).
            yield {
                "type": "papers_found",
                "papers": _clean_papers_for_event(list(self._found_papers)[:10]),
            }

        relevant_papers = [
            p for p in papers if p.get("relevance_score", 0) >= 3 and p.get("source") != "kb_search"
        ]
        if relevant_papers:
            yield {"type": "papers_to_curate", "papers": _clean_papers_for_event(relevant_papers)}

        yield {"type": "phase_transition", "phase": "done"}
        yield {
            "type": "completion",
            "iteration_count": locals().get("iteration", 0) + 1,
            "completion_reason": "complete",
        }

    async def _maybe_upgrade_single_kb_to_composite_parallel(
        self, plan: Plan, query: str, kb_name: str
    ) -> None:
        """Replace one root parallel kb_search with multiple facets when query is composite.

        The LLM planner often emits a single kb_search even for comparisons; this enforces
        the parallel wave when composite_subqueries_with_llm yields 2+ distinct phrases.
        """
        clean = ResearchPlanner._clean_query_for_search(query)
        subs = list(dict.fromkeys(await self.planner.composite_subqueries_with_llm(clean)))[:3]
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
        new_steps: list[Step] = []
        for i, sq in enumerate(subs):
            tool_input: dict[str, Any] = {"query": sq}
            if top_k is not None:
                tool_input["top_k"] = top_k
            new_steps.append(
                Step(
                    id=f"composite_kb_{i + 1}",
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
            "agentic_kb_composite_upgrade",
            old_step_id=old_id,
            new_step_count=len(new_steps),
            subqueries=subs,
        )

    # ------------------------------------------------------------------
    # URL pre-processing
    # ------------------------------------------------------------------

    async def _try_resolve_url(self, query: str) -> dict[str, Any] | None:
        """Detect paper URL in query, fetch content, return normalized paper dict or None.

        Resolves the URL to a DOI (mapping arXiv IDs to their 10.48550/arXiv.<id> form)
        and runs the unified retrieval pipeline once. The pipeline already queries
        OpenAlex for metadata as part of discovery, so authors/year come back populated.
        If the unified pipeline returns no full text, falls back to a Semantic Scholar
        metadata-only lookup.

        Returns:
            Normalized paper dict with keys: id, title, authors[], year, doi, abstract,
            full_text, source, relevance_score, _step_id, etc.
        """
        url_match = _URL_RE.search(query)
        if not url_match:
            return None

        url = url_match.group(0).rstrip(".,;:)")
        from perspicacite.models import PaperSource, normalize_paper_dict
        from perspicacite.pipeline.download.arxiv import get_arxiv_id_from_url

        # Resolve URL → (arxiv_id, doi). arXiv IDs map to their canonical DOI form
        # so retrieve_paper_content's discovery step can look them up in OpenAlex.
        arxiv_id = get_arxiv_id_from_url(url)
        bare_id = None
        doi: str | None = None
        if arxiv_id:
            bare_id = arxiv_id.split("v")[0] if re.match(r".*v\d+$", arxiv_id) else arxiv_id
            doi = f"10.48550/arXiv.{bare_id}"
        else:
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
                    paper_dict = normalize_paper_dict(
                        {
                            **(result.metadata or {}),
                            "doi": doi,
                            "full_text": result.full_text,
                            "abstract": result.abstract,
                        },
                        source=PaperSource.OPENALEX,
                    )
                    paper_dict["relevance_score"] = 5
                    paper_dict["_step_id"] = "url_prefetch"
                    paper_dict["sections"] = result.sections
                    paper_dict["content_type"] = result.content_type
                    return paper_dict
            except Exception as e:
                logger.warning("agentic_url_prefetch_failed", doi=doi, error=str(e))

        # Fallback: Semantic Scholar metadata lookup (no full text, just title + abstract).
        # Triggered when the unified pipeline returned no full text (very new papers, paywalled
        # publishers we can't access, etc.).
        from perspicacite.search.semantic_scholar import lookup_paper

        lookup_id = doi or bare_id or arxiv_id
        if lookup_id:
            logger.info("agentic_url_prefetch_s2_fallback", lookup_id=lookup_id)
            paper = await lookup_paper(lookup_id)
            if paper:
                logger.info("agentic_url_prefetch_s2_found", title=paper.title[:60])
                paper_dict = normalize_paper_dict(
                    {
                        "id": paper.id,
                        "title": paper.title,
                        "doi": paper.doi or doi or "",
                        "abstract": paper.abstract or "",
                        "authors": [a.name for a in paper.authors],
                        "year": paper.year,
                        "citation_count": paper.citation_count,
                        "arxiv_id": bare_id,
                    },
                    source=PaperSource.SEMANTIC_SCHOLAR,
                )
                paper_dict["relevance_score"] = 4
                paper_dict["_step_id"] = "url_prefetch"
                paper_dict["_metadata_only"] = True
                return paper_dict

        return None

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
            "agentic_evidence_facets_registered",
            facets={k: f.step_ids for k, f in ev.facets.items()},
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
        self, plan: Plan, completed: list[Step], results: dict[str, Any]
    ) -> list[Step]:
        """Return all steps whose dependencies are satisfied (general DAG).

        Any ready steps that are *not* of type ANSWER run concurrently.
        ANSWER steps are always executed alone (they need all prior context).
        If only a single step is ready, it runs by itself.
        """
        completed_ids = {s.id for s in completed}
        ready: list[Step] = []
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

    def _evaluate_condition(self, condition: str, results: dict[str, Any]) -> bool:
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
        self, step: Step, original_query: str, step_results: dict[str, Any], session: AgentSession
    ) -> Any:
        """Execute a single step."""

        if step.type == StepType.LOTUS_SEARCH:
            logger.info("agentic_lotus_search_skipped")
            return "LOTUS search is currently deactivated."

        elif step.type == StepType.LITERATURE_SEARCH:
            query = step.tool_input.get("query", original_query)
            logger.info("agentic_literature_search", query=query)
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

                    logger.info(
                        "agentic_kb_search_start",
                        kb_name=session.kb_name,
                        collection=collection_name,
                        step_id=step.id,
                        query=kb_query,
                        query_chars=len(kb_query),
                    )

                    dkb = self._build_kb_retriever(default_kb_name=session.kb_name)
                    # Dynamic top_k: planner can specify via tool_input
                    planner_top_k = step.tool_input.get("top_k")
                    # DynamicKnowledgeBase exposes .config; MultiKBRetriever does not.
                    _kb_cfg = getattr(dkb, "config", None)
                    _default_top_k = getattr(_kb_cfg, "top_k", 5)
                    _default_min_rel = getattr(_kb_cfg, "min_relevance_score", 0.0)
                    top_k = planner_top_k if planner_top_k is not None else _default_top_k
                    logger.info(
                        "agentic_kb_search_config",
                        top_k=top_k,
                        min_relevance_score=_default_min_rel,
                        embedding_model=getattr(self.embeddings, "model_name", "?"),
                        retriever=type(dkb).__name__,
                    )

                    results = await dkb.search(kb_query, top_k=top_k)
                    logger.info("agentic_kb_search_vector_hits", hit_count=len(results))

                    # Apply hybrid retrieval if enabled
                    if self.use_hybrid and results:
                        try:
                            logger.info("agentic_kb_search_hybrid_retrieval")
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

                            logger.info("agentic_kb_search_hybrid_complete", result_count=len(results))
                        except Exception as e:
                            logger.warning("agentic_kb_search_hybrid_failed", error=str(e), exc_info=True)

                    # Filter results by minimum relevance score (0.5 = medium relevance)
                    min_relevance_threshold = 0.5
                    filtered_results = [
                        r for r in results if r.get("score", 0) >= min_relevance_threshold
                    ]
                    if len(filtered_results) < len(results):
                        logger.info(
                            "agentic_kb_search_relevance_filtered",
                            filtered_out=len(results) - len(filtered_results),
                            kept=len(filtered_results),
                            threshold=min_relevance_threshold,
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
                                "agentic_kb_search_hit_empty_text",
                                hit_index=j,
                                paper_id=pid,
                                title=title,
                            )

                        logger.info(
                            "agentic_kb_search_hit",
                            hit_index=j,
                            total_hits=len(results),
                            paper_id=pid,
                            score=round(r.get("score", 0), 4),
                            title=title,
                            text_len=len(txt),
                        )
                        preview = txt[:280].replace("\n", " ")
                        if preview.strip():
                            logger.info(
                                "agentic_kb_search_hit_preview",
                                hit_index=j,
                                preview=preview + ("…" if len(txt) > 280 else ""),
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

                                    # Multi-KB: fan get_chunks across all member
                                    # collections; single-KB: original code path.
                                    _is_multi = type(dkb).__name__ == "MultiKBRetriever"
                                    if _is_multi:
                                        from perspicacite.retrieval.multi_kb import (
                                            get_chunks_by_paper_ids_across,
                                        )

                                        _coll_names = [
                                            getattr(m, "collection_name", "")
                                            for m in getattr(dkb, "kb_metas", [])
                                            if getattr(m, "collection_name", None)
                                        ]
                                        all_chunks = await get_chunks_by_paper_ids_across(
                                            self.vector_store,
                                            collection_names=_coll_names,
                                            paper_ids=paper_ids,
                                        )
                                    else:
                                        all_chunks = (
                                            await self.vector_store.get_chunks_by_paper_ids(
                                                collection_name, paper_ids
                                            )
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
                                                rp = (
                                                    getattr(m, "paper_id", None)
                                                    if m
                                                    else r.get("paper_id")
                                                )
                                                if rp == pid:
                                                    score = r.get("score", 0.5)
                                                    break
                                            paper_results.append(
                                                {
                                                    "paper_id": pid,
                                                    "paper_score": score,
                                                    "title": getattr(meta_obj, "title", None)
                                                    if meta_obj
                                                    else None,
                                                    "authors": getattr(meta_obj, "authors", None)
                                                    if meta_obj
                                                    else None,
                                                    "year": getattr(meta_obj, "year", None)
                                                    if meta_obj
                                                    else None,
                                                    "doi": getattr(meta_obj, "doi", None)
                                                    if meta_obj
                                                    else None,
                                                    "full_text": full_text,
                                                    "source": "kb_search",
                                                }
                                            )
                                        logger.info(
                                            "agentic_kb_search_two_pass_enrichment",
                                            paper_count=len(paper_ids),
                                            chunk_count=len(all_chunks),
                                        )
                                except Exception as e:
                                    logger.warning("agentic_kb_search_two_pass_failed", error=str(e))

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
                                title = (
                                    getattr(meta, "title", None) or "Unknown" if meta else "Unknown"
                                )
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
                                "agentic_kb_search_content_lengths",
                                doc_index=i,
                                total_docs=len(items),
                                paper_id=pid,
                                joined_full_len=len(joined_full),
                                formatted_content_len=len(text_content),
                                cap=fmt_cap,
                                truncated=len(joined_full) > len(text_content),
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
                                    self._found_papers.append(
                                        {
                                            "title": title,
                                            "authors": [a.strip() for a in authors.split(",")]
                                            if authors
                                            else [],
                                            "year": year,
                                            "doi": doi,
                                            "abstract": (
                                                text_content[:4000] if text_content else ""
                                            ),
                                            "full_text": joined_full,
                                            "source": "kb_search",
                                            "relevance_score": 4,
                                            "_step_id": step.id,
                                        }
                                    )
                                if session.evidence is None:
                                    session.evidence = EvidenceStore()
                                fk = self._facet_key_for_step(session, step)
                                session.evidence.add_hits(
                                    [
                                        {
                                            "title": title,
                                            "doi": doi,
                                            "excerpt": (
                                                text_content[:4000] if text_content else ""
                                            ),
                                            "step_id": step.id,
                                            "source": "kb_search",
                                        }
                                    ],
                                    step_id=step.id,
                                    facet_key=fk,
                                )
                                # Provenance: per-chunk retrieval event for KB search
                                _prov_c = get_collector()
                                if _prov_c is not None:
                                    _prov_c.add_retrieval(
                                        paper_id=pid if isinstance(pid, str) else None,
                                        doi=doi or None,
                                        title=title or None,
                                        score=float(score),
                                        kb_name=session.kb_name,
                                        content_type="kb_chunk",
                                        pipeline_step=step.id,
                                        rank=i - 1,
                                        stage_label="agentic.tool.search_kb",
                                    )

                        out = "\n".join(formatted_parts)
                        logger.info("agentic_kb_search_result_formatted", result_chars=len(out))
                        return out
                    logger.info("agentic_kb_search_no_hits")
                    return "No relevant documents found in knowledge base."
                except Exception as e:
                    logger.error("agentic_kb_search_failed", error=str(e), exc_info=True)
                    return "Knowledge base search failed."
            logger.info("agentic_kb_search_skipped_no_kb")
            return "No knowledge base selected."

        elif step.type == StepType.PAPER_LOOKUP:
            paper_id = step.tool_input.get("paper_id", "")
            if not paper_id:
                return "No paper ID provided for lookup."

            logger.info("agentic_paper_lookup", paper_id=paper_id)

            from perspicacite.search.semantic_scholar import lookup_paper

            paper = await lookup_paper(paper_id)

            if paper is None:
                logger.info("agentic_paper_lookup_not_found", paper_id=paper_id)
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
            logger.info(
                "agentic_paper_lookup_found",
                title=paper.title[:60],
                citation_count=paper.citation_count,
            )

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
            "agentic_kb_judge_input",
            input_chars=len(kb_result_text or ""),
            excerpt_chars=len(excerpt),
        )

        if not excerpt:
            logger.info("agentic_kb_judge_empty_excerpt")
            return False
        if (
            "no relevant documents" in low
            or "knowledge base search failed" in low
            or "no knowledge base selected" in low
        ):
            logger.info("agentic_kb_judge_failure_phrase")
            return False

        max_judge_chars = 8000
        if len(excerpt) > max_judge_chars:
            excerpt = excerpt[:max_judge_chars] + "\n[... truncated for judge ...]"

        # Log the actual excerpt being sent to judge (first 1000 chars)
        logger.info("agentic_kb_judge_excerpt_preview", preview=excerpt[:1000])

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
            logger.info("agentic_kb_judge_llm_response", response_chars=len(text))
            logger.debug("agentic_kb_judge_llm_response_preview", preview=text[:500])
            m_fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if m_fence:
                text = m_fence.group(1).strip()
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                logger.warning("agentic_kb_judge_no_json", text_preview=text[:200])
                return False
            obj = json.loads(text[start : end + 1])
            sufficient = bool(obj.get("sufficient"))
            reason = obj.get("reason", "")
            logger.info("agentic_kb_judge_result", sufficient=sufficient, reason=reason)
            return sufficient
        except Exception as e:
            logger.warning("agentic_kb_judge_failed", error=str(e))
            return False

    async def _evaluate_progress(
        self,
        query: str,
        plan: Plan,
        completed_steps: list[Step],
        step_results: dict[str, Any],
        session: AgentSession | None = None,
        eval_step_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate whether to continue, replan, or answer.

        Returns a dict with:
          - ``"decision"``: ``"continue"`` | ``"replan"`` | ``"answer"``
          - ``"gap_facets"``: list of facet queries that are still gap/partial
          - ``"missing_aspects"``: list from the quality assessor (if any)
          - ``"evaluation_text"``: human-readable summary for the replanner
        """
        result: dict[str, Any] = {
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
        gaps: dict[str, str] = {}
        if session and session.evidence:
            gaps = session.evidence.gap_summary()
        gap_facets = [k for k, v in gaps.items() if v in ("gap", "partial")]
        result["gap_facets"] = gap_facets

        remaining_search = [
            s for s in remaining_steps if s.type in (StepType.KB_SEARCH, StepType.LITERATURE_SEARCH)
        ]

        if gap_facets and remaining_search:
            covered_by_remaining: set[str] = set()
            if session and session.evidence:
                for s in remaining_search:
                    f = session.evidence.facet_for_step(s.id)
                    if f:
                        covered_by_remaining.add(f.query.lower()[:120] or "main")
            uncovered_gaps = [f for f in gap_facets if f not in covered_by_remaining]
            if uncovered_gaps:
                facet_labels = "; ".join(f'"{f}" ({gaps[f]})' for f in uncovered_gaps[:4])
                logger.info("agentic_facet_gaps_force_replan", facet_labels=facet_labels)
                result["decision"] = "replan"
                result["evaluation_text"] = f"Facets still uncovered: {facet_labels}"
                return result
            else:
                logger.info(
                    "agentic_gap_facets_covered_by_remaining",
                    covered=list(covered_by_remaining)[:4],
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
                    "agentic_quality_assessment",
                    sufficient=is_sufficient,
                    confidence=round(confidence, 2),
                    missing_aspects_count=len(missing_aspects),
                )

                if is_sufficient and confidence >= self.early_exit_confidence:
                    all_covered = all(v == "covered" for v in gaps.values()) if gaps else True
                    if all_covered:
                        logger.info(
                            "agentic_early_exit",
                            confidence=round(confidence, 2),
                            threshold=self.early_exit_confidence,
                        )
                        result["decision"] = "answer"
                        result["evaluation_text"] = "All facets covered; evidence sufficient."
                        return result
                    else:
                        logger.info("agentic_quality_sufficient_facets_incomplete")

                if not is_sufficient and missing_aspects:
                    aspect_str = ", ".join(missing_aspects[:3])
                    logger.info("agentic_quality_insufficient", missing=aspect_str)
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

    def _get_recent_found_papers(
        self, step_ids: list[str] | None = None, limit: int = 5
    ) -> list[dict[str, Any]]:
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

        docs: list[dict[str, Any]] = []
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
            docs.append(
                {
                    "title": title,
                    "content": content,
                    "source": p.get("source", "unknown"),
                }
            )
            if len(docs) >= limit:
                break
        return docs

    async def _analyze_results(self, query: str, step_results: dict[str, Any]) -> str:
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

    async def _synthesize_results(self, query: str, step_results: dict[str, Any]) -> str:
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

COPYRIGHT & ATTRIBUTION:
- Paraphrase rather than reproducing source text verbatim. Papers may be from copyrighted publications.
- Quotations must be short (≤ 15 words), in quotation marks, and cited.
- Do not reproduce abstracts, full paragraphs, tables, or extended passages from any source.

Provide a synthesized summary that combines the key insights from all sources."""

        return await self.llm.complete(prompt, temperature=0.15)

    async def _per_paper_extraction_bullet(
        self, query: str, paper: dict[str, Any], list_index: int
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

    async def _map_reduce_paper_bullets(self, query: str, papers: list[dict[str, Any]]) -> str:
        """Top-N papers: parallel extraction bullets for final synthesis."""
        indexed: list[tuple[int, dict[str, Any]]] = []
        for i, p in enumerate(papers, 1):
            ft = (p.get("full_text") or "").strip()
            ab = (p.get("abstract") or "").strip()
            if len(ft) >= 500 or len(ab) >= 200:
                indexed.append((i, p))
        indexed = indexed[: self.map_reduce_max_papers]
        if not indexed:
            return ""

        sem = asyncio.Semaphore(4)

        async def _one(li: int, p: dict[str, Any]) -> str:
            async with sem:
                return await self._per_paper_extraction_bullet(query, p, li)

        parts = await asyncio.gather(*[_one(li, p) for li, p in indexed])
        blocks: list[str] = []
        for (li, p), text in zip(indexed, parts, strict=True):
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
        sections: list[str] = []
        sections.append("Research facets investigated:")
        for _key, facet in session.evidence.facets.items():
            status = facet.status.upper()
            n = len(facet.entries)
            titles = [str(e.get("title", ""))[:80] for e in facet.entries[:4]]
            title_list = "; ".join(t for t in titles if t) or "(none)"
            sections.append(f'  [{status}] "{facet.query}" — {n} source(s): {title_list}')
        return "\n".join(sections)

    async def _generate_answer(
        self,
        query: str,
        plan: Plan,
        step_results: dict[str, Any],
        session: AgentSession,
        papers: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate final answer and return ``(answer_text, citation_map)``."""

        logger.info(
            "agentic_generating_answer",
            query=query,
            step_result_keys=list(step_results.keys()),
        )

        if papers is None:
            papers = self._extract_papers_from_results(step_results)
            papers = await self._score_papers_for_relevance(query, papers, min_score=3)

        # --- Single-paper fast path: skip map-reduce, pass full text directly ---
        single_paper_with_full_text = (
            len(papers) == 1 and (papers[0].get("full_text") or "").strip()
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
            logger.info("agentic_lotus_result", result_chars=len(str(lotus_result)))
            context_parts.append(f"LOTUS Search Results:\n{lotus_result}")

        map_reduce_block = await self._map_reduce_paper_bullets(query, papers)
        if map_reduce_block:
            context_parts.append(
                "\n\n---\n\nQuery-focused extractions (per paper; cite using [N] from the numbered list):\n"
                + map_reduce_block
            )
            logger.info("agentic_answer_context_map_reduce")
        else:
            for step_id, result in step_results.items():
                if step_id != "lotus" and result:
                    result_str = str(result)
                    logger.info(
                        "agentic_answer_step_result_length",
                        step_id=step_id,
                        result_chars=len(result_str),
                    )
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
                    "\n\n---\n\nDownloaded Full Text:\n" + "\n\n---\n\n".join(full_text_parts)
                )
                logger.info("agentic_answer_full_text_docs", count=len(full_text_parts))

        context = "\n\n".join(context_parts)
        logger.info("agentic_answer_context_length", context_chars=len(context))

        if not context.strip() and papers:
            # Abstract fallback: use paper abstracts when all context was
            # dropped (e.g. every paper scored below min_score, leaving no
            # map-reduce block and no step results). This prevents an empty
            # synthesis call that returns an uninformative "no information" answer.
            abstract_parts = []
            for i, p in enumerate(papers[:10], 1):
                abstract = p.get("abstract") or ""
                title = p.get("title", "")
                if abstract:
                    abstract_parts.append(f"[{i}] {title}\n{abstract[:800]}")
            if abstract_parts:
                context = "Paper abstracts (KB source):\n\n" + "\n\n".join(abstract_parts)
                logger.info(
                    "agentic_answer_fallback_to_abstracts",
                    count=len(abstract_parts),
                )

        if not context.strip():
            logger.warning("agentic_answer_context_empty")

        conversation_context = session.get_context_string()

        facet_guideline = ""
        if facet_overview:
            facet_guideline = (
                '10. The query has multiple facets (see "Research facets investigated" above). '
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
                'paragraph of generic themes (e.g. only "tracking" and "prioritization") when '
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
6. If the research results are insufficient to answer the question, clearly state INSUFFICIENT EVIDENCE rather than speculating or inferring a negative conclusion
7. Structure your answer logically with clear sections if appropriate
8. Cite using [N] from the numbered list only for sources you actually use; you do not need to mention every listed paper if some are redundant.
9. **DO NOT include a Citations or References section at the end of your answer** - a properly formatted references section will be automatically appended separately.
10. EVIDENCE EPISTEMICS: Only conclude a claim is REFUTED when a retrieved paper EXPLICITLY and DIRECTLY contradicts it. Papers that do not mention a topic, only discuss related concepts, or report null/absence results are NOT evidence of refutation — treat them as INSUFFICIENT EVIDENCE for that specific claim.
{facet_guideline}{workflow_guideline}
Important: Do not provide an answer if the question contains hate speech, offensive language, discriminatory remarks, or harmful content.

Generate your answer:"""

        logger.info("agentic_answer_prompt_length", prompt_chars=len(prompt))
        logger.info("agentic_answer_llm_call")

        mm_chunks = (
            self._chunks_with_figure_refs(step_results)
            if getattr(self, "config", None) is not None
            else None
        )
        answer = await self.llm.complete(
            prompt,
            temperature=0.25,
            chunks=mm_chunks,
            config=getattr(self, "config", None),
        )
        logger.info("agentic_answer_generated", answer_chars=len(answer), answer=answer)

        answer, citation_map = self._verify_citations(answer, papers)

        cited_indices = {c["index"] for c in citation_map["cited"]}
        if cited_indices and cited_indices != set(range(1, len(papers) + 1)):
            answer, papers = self._compact_citations(answer, papers, cited_indices)
            citation_map["compacted"] = True
            logger.info(
                "agentic_citations_compacted",
                cited=len(cited_indices),
                total_papers=citation_map["total_papers"],
                renumbered_to=len(papers),
            )

        if papers:
            references_section = self._format_references_section(papers)
            if references_section:
                answer = answer.rstrip() + "\n\n" + references_section
                logger.info("agentic_references_section_added", answer_chars=len(answer))

        return answer, citation_map

    _CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

    @classmethod
    def _verify_citations(
        cls, answer: str, papers: list[dict[str, Any]]
    ) -> tuple[str, dict[str, Any]]:
        """Validate citation markers in the generated answer.

        Handles both single ``[N]`` and multi-citation ``[N, M]`` brackets.
        Returns ``(cleaned_answer, citation_map)`` where:
        - Invalid references (N > number of papers or N < 1) are stripped.
        - Empty brackets left after stripping are removed.
        - ``citation_map`` records which papers were cited, uncited, and any
          invalid references that were removed.
        """
        valid_range = set(range(1, len(papers) + 1))

        found_refs: set[int] = set()
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
                "agentic_citations_invalid_stripped",
                invalid_count=len(invalid_refs),
                invalid_refs=sorted(invalid_refs),
            )

        cited_indices = sorted(valid_refs)
        uncited_indices = sorted(valid_range - valid_refs)

        citation_map: dict[str, Any] = {
            "cited": [
                {
                    "index": i,
                    "title": papers[i - 1].get("title", ""),
                    "doi": papers[i - 1].get("doi", ""),
                }
                for i in cited_indices
            ],
            "uncited": [
                {"index": i, "title": papers[i - 1].get("title", "")} for i in uncited_indices
            ],
            "invalid_stripped": sorted(invalid_refs),
            "total_papers": len(papers),
            "cited_count": len(cited_indices),
        }

        if papers and len(uncited_indices) > len(papers) * 0.5:
            logger.warning(
                "agentic_citation_coverage_low",
                cited=len(cited_indices),
                total=len(papers),
                uncited=len(uncited_indices),
            )
        else:
            logger.info(
                "agentic_citation_verification",
                cited=len(cited_indices),
                total=len(papers),
                invalid_stripped=len(invalid_refs),
            )

        return answer, citation_map

    @classmethod
    def _compact_citations(
        cls,
        answer: str,
        papers: list[dict[str, Any]],
        cited_indices: set[int],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Remove uncited papers and renumber citation markers to be consecutive.

        Handles both single ``[N]`` and multi-citation ``[N, M]`` brackets.
        Given ``cited_indices={1, 3}`` out of 3 papers, the answer's ``[1]``
        stays ``[1]``, ``[3]`` becomes ``[2]``, and ``[1, 3]`` becomes
        ``[1, 2]``.  Only cited papers are returned in the new list.
        """
        old_to_new: dict[int, int] = {}
        compacted_papers: list[dict[str, Any]] = []
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
        papers: list[dict[str, Any]],
        session: AgentSession,
        is_summary_request: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Generate answer for a single paper with full text — no map-reduce.

        Uses the full paper text directly. If is_summary_request, produces a
        comprehensive summary. Otherwise answers the specific question.
        """
        paper = papers[0]
        full_text = (paper.get("full_text") or "").strip()
        title = paper.get("title", "Unknown")
        doi = paper.get("doi", "")

        logger.info(
            "agentic_single_paper_answer",
            title=title[:60],
            full_text_len=len(full_text),
            is_summary=is_summary_request,
        )

        conversation_context = session.get_context_string()

        if is_summary_request:
            effective_question = (
                "Provide a comprehensive overview of this paper, covering: "
                "motivation and problem statement, methodology and approach, "
                "key contributions and innovations, main results and findings, "
                "limitations, and significance."
            )
            logger.info("agentic_single_paper_summary_mode")
        else:
            effective_question = query
            logger.info("agentic_single_paper_question_mode")

        prompt = f"""You are a scientific research assistant. You have the full text of a single research paper.

Paper: {title}
DOI: {doi or "N/A"}

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

        logger.info("agentic_single_paper_prompt_length", prompt_chars=len(prompt))
        # Single-paper path doesn't surface step_results / DocumentChunks here,
        # so multimodal degrades to text-only. Pass None for chunks; the wrap
        # function won't be invoked.
        answer = await self.llm.complete(
            prompt,
            temperature=0.25,
            chunks=None,
            config=getattr(self, "config", None),
        )
        logger.info("agentic_single_paper_answer_generated", answer_chars=len(answer))

        # Build citation map for single paper (always [1])
        citation_map: dict[str, Any] = {
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
        self, papers: list[dict[str, Any]], max_abstract_chars: int = 800
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
        self, query: str, papers: list[dict[str, Any]], min_score: int = 3
    ) -> list[dict[str, Any]]:
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
            'Example calibration: Query "CRISPR gene editing in wheat" ->\n'
            '  Paper "RNA editing in rice" = 2 (related technique, wrong organism '
            "and mechanism)\n"
            '  Paper "CRISPR-Cas9 optimization in bread wheat" = 5\n\n'
            "Respond with ONLY a JSON object:\n"
            '{"scores": {"1": {"score": N, "reason": "..."}, '
            '"2": {"score": N, "reason": "..."}, ...}}'
        )

        def _parse_and_filter(response_text: str) -> tuple[list, bool]:
            """Parse LLM response and return (filtered_papers, is_complete)."""
            import json as _json
            import re as _re

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
                            "agentic_relevance_score_missing_discarded",
                            paper_index=i,
                            title=paper.get("title", "")[:50],
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
                        "agentic_relevance_paper_included_kb",
                        paper_index=i,
                        title=paper.get("title", "")[:50],
                        score=paper["relevance_score"],
                    )
                elif score >= min_score:
                    filtered.append(paper)
                    logger.info(
                        "agentic_relevance_paper_included",
                        paper_index=i,
                        title=paper.get("title", "")[:50],
                        score=score,
                    )
                else:
                    logger.info(
                        "agentic_relevance_paper_filtered",
                        paper_index=i,
                        title=paper.get("title", "")[:50],
                        score=score,
                    )

            logger.info(
                "agentic_relevance_filtering_complete",
                included=len(filtered),
                total=n_papers,
                min_score=min_score,
                complete=is_complete,
            )
            return filtered, is_complete

        try:
            response = await self.llm.complete(prompt, temperature=0.1)
            filtered, is_complete = _parse_and_filter(response)

            # If scores are incomplete, retry once with stricter instruction
            if not is_complete:
                logger.warning(
                    "agentic_relevance_scorer_incomplete",
                    kept=len(filtered),
                    total=n_papers,
                )
                retry_prompt = (
                    prompt + "\n\nIMPORTANT: Your previous response was missing scores for "
                    f"some papers. You MUST return exactly {n_papers} entries with "
                    f'keys "1" through "{n_papers}". Try again.'
                )
                response = await self.llm.complete(retry_prompt, temperature=0.1)
                retry_filtered, retry_complete = _parse_and_filter(response)
                if retry_complete or len(retry_filtered) > len(filtered):
                    filtered = retry_filtered
                    is_complete = retry_complete

            # === Graceful threshold relaxation ===
            # If the strict scorer rejected almost everything (common for
            # "What is X" definitional queries — the prompt is calibrated
            # for narrow questions and over-penalises "applies X" papers),
            # rescue the top-scoring non-KB papers by including any that
            # scored ``min_score - 1`` (typically 2). This avoids the
            # degenerate "agentic returned 1 paper" outcome the user hit
            # while keeping the strict floor when there are plenty of
            # truly-relevant papers.
            #
            # Triggers when:
            #   - fewer than 2 non-KB papers passed AND
            #   - the candidate pool was at least 3 (otherwise no rescue
            #     can help) AND
            #   - we have papers at relaxed_floor (e.g. score=2) to promote
            non_kb_kept = [
                p for p in filtered if p.get("source") != "kb_search"
            ]
            if len(non_kb_kept) < 2 and n_papers >= 3 and min_score > 1:
                relaxed_floor = min_score - 1
                rescued: list[dict[str, Any]] = []
                for p in papers:
                    if p in filtered:
                        continue  # already kept
                    if p.get("source") == "kb_search":
                        continue
                    sc = int(p.get("relevance_score") or 0)
                    if sc == relaxed_floor:
                        rescued.append(p)
                # Cap rescues so we don't drown the strict picks in noise.
                rescued.sort(
                    key=lambda x: int(x.get("relevance_score") or 0),
                    reverse=True,
                )
                rescued = rescued[:5]
                if rescued:
                    logger.info(
                        "agentic_relevance_threshold_relaxed",
                        kept_strict=len(filtered),
                        rescued=len(rescued),
                        relaxed_floor=relaxed_floor,
                    )
                    filtered = filtered + rescued

            return filtered

        except Exception as e:
            logger.error("agentic_relevance_scoring_error", error=str(e))
            # Never treat unscored literature as relevance-approved (UI + downstream use
            # relevance_score). KB hits are pre-trusted; literature without a successful
            # score is excluded from the paper list so synthesis falls back to raw step
            # text rather than unfiltered OpenAlex/SciLEx blobs.
            retained: list[dict[str, Any]] = []
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
                    "agentic_relevance_scoring_failed_kb_only",
                    kb_papers=len(retained),
                    excluded_literature=len(papers) - len(retained),
                )
                return retained
            logger.warning("agentic_relevance_scoring_failed_no_papers")
            return []

    def _format_references_section(self, papers: list[dict[str, Any]]) -> str:
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

    async def _rerank_papers_by_relevance(
        self, query: str, papers: list, top_k: int
    ) -> list:
        """Semantically rerank a candidate paper pool by query relevance.

        SciLEx returns papers in arbitrary collection order. When several
        databases are searched, a noisy source (e.g. OpenAlex/PubMed for an AI
        query) floods the pool and pushes genuinely-relevant papers past the
        top-N cut before they ever reach the LLM relevance scorer. A cross-
        encoder rerank here keeps the most on-topic papers regardless of
        source. Lexical ranking is insufficient — "Evaluation of
        Chemotherapeutic Agents" matches the words "agent"/"evaluation" — so we
        use a semantic query-document model. Falls back to the original order
        on any failure (missing model, etc.) so search never hard-breaks.
        """
        if len(papers) <= top_k:
            return papers
        # Honour the master reranker switch — when disabled (e.g. strong
        # instruction-tuned embedder configs), keep the original order.
        _rag_modes_cfg = getattr(getattr(self, "config", None), "rag_modes", None)
        _reranker_enabled = getattr(_rag_modes_cfg, "reranker_enabled", True)
        _reranker_model = getattr(_rag_modes_cfg, "reranker_model", None) \
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if not (_reranker_enabled and _reranker_model):
            return papers[:top_k]
        try:
            from perspicacite.retrieval.reranker import CrossEncoderReranker
            if getattr(self, "_relevance_reranker", None) is None:
                self._relevance_reranker = CrossEncoderReranker(model_name=_reranker_model)
            texts = [
                f"{p.title or ''}. {(p.abstract or '')[:1000]}" for p in papers
            ]
            scores = await self._relevance_reranker.score_texts(query, texts)
            # Sort by index to avoid comparing Paper objects on score ties.
            order = sorted(range(len(papers)), key=lambda i: scores[i], reverse=True)
            logger.info(
                "agentic_scilex_reranked", pool=len(papers), kept=top_k
            )
            return [papers[i] for i in order[:top_k]]
        except Exception as e:
            logger.warning("agentic_scilex_rerank_failed", error=str(e))
            return papers[:top_k]

    def _note_search_fallback(
        self, session: AgentSession | None, apis: list[str]
    ) -> None:
        """Stash a user-facing notice when SciLEx yields nothing and we fall
        back to OpenAlex — usually because a selected source rate-limited.

        Drained as a ``thinking`` event by the ``chat()`` loop so a user who
        picked e.g. arXiv-only isn't surprised to see OpenAlex results.
        """
        if session is None:
            return
        errs = getattr(self.scilex_adapter, "last_errors_by_database", None) or {}
        failed = [db for db in apis if db in errs]
        if failed:
            parts = [
                f"{db} ({'rate-limited' if '429' in errs[db] else 'unavailable'})"
                for db in failed
            ]
            msg = (
                f"Note: {', '.join(parts)} returned nothing right now, "
                "so results were drawn from OpenAlex instead."
            )
        else:
            msg = (
                "Note: your selected source(s) returned nothing, "
                "so results were drawn from OpenAlex instead."
            )
        notices = getattr(session, "search_notices", None)
        if notices is None:
            notices = []
            session.search_notices = notices  # type: ignore[attr-defined]
        notices.append(msg)

    async def _scilex_search(
        self,
        query: str,
        max_results: int = 10,
        step_id: str = "",
        session: AgentSession | None = None,
    ) -> str:
        """Search academic literature using SciLEx (multi-API aggregation).

        Falls back to direct OpenAlex if SciLEx is not available.
        """
        logger.info("agentic_scilex_search", query=query)

        # -- query optimization --
        _app_state = getattr(self, "app_state", None)
        effective_query = query
        if _app_state is not None and getattr(_app_state, "config", None) is not None:
            import perspicacite.search.query_optimizer as _qo_mod
            try:
                opt = await _qo_mod.optimize_query(
                    query=query,
                    context=None,
                    app_state=_app_state,
                    optimize_enabled=True,  # always rewrite for web search (item 2)
                )
                effective_query = opt.searched_query
                if opt.applied:
                    logger.info(
                        "agentic_scilex_query_rewritten",
                        original=query,
                        rewritten=effective_query,
                    )
            except Exception as _opt_exc:
                logger.warning("agentic_scilex_optimizer_failed", error=str(_opt_exc))

        # Honour the user's database selection threaded through ``chat()``;
        # falls back to the legacy default when no session-level override is
        # set. Unknown providers (e.g. ``google_scholar``) are filtered out by
        # ``SciLExAdapter.search`` so we can pass the raw list through.
        _selected_dbs = (
            getattr(session, "databases", None)
            if session is not None
            else None
        )
        _apis = _selected_dbs or ["semantic_scholar", "openalex", "pubmed"]
        try:
            # Over-fetch a wider candidate pool than we ultimately keep. The
            # adapter truncates by arbitrary collection order, so when several
            # databases are searched the relevant papers can sit deep in the
            # pool (measured as far as position ~36 for a multi-DB AI query).
            # We fetch wide, then semantically rerank down to ``max_results``.
            _overfetch = max(max_results * 4, 40)

            # Prefer the domain aggregator so user-selected non-SciLEx
            # providers (Google Scholar via SerpApi, EuropePMC, CORE, ...) are
            # actually queried — not just SciLEx's built-in APIs. It honours
            # the same ``databases`` selection and forwards SciLEx sub-provider
            # names down to SciLEx. Falls back to the raw SciLEx adapter when
            # no config/aggregator is available (e.g. unit tests).
            papers = None
            _cfg = getattr(self, "config", None) or getattr(
                getattr(self, "app_state", None), "config", None
            )
            if _cfg is not None:
                try:
                    from perspicacite.search.domain_aggregator import build_aggregator
                    if getattr(self, "_aggregator", None) is None:
                        self._aggregator = build_aggregator(_cfg)
                    if self._aggregator.available:
                        papers = await self._aggregator.search(
                            query=effective_query,
                            max_results=_overfetch,
                            databases=_selected_dbs,
                        )
                except Exception as _agg_exc:
                    logger.warning("agentic_aggregator_failed", error=str(_agg_exc))
                    papers = None
            if papers is None:
                papers = await self.scilex_adapter.search(
                    query=effective_query,
                    max_results=_overfetch,
                    apis=_apis,
                )

            # Rerank the (possibly noisy, multi-source) pool by semantic
            # relevance to the user's query, then keep only the top
            # ``max_results``, so relevant papers aren't buried by a flooding
            # database before the LLM scorer sees them. Uses the original
            # query (intent), not the search-optimized rewrite.
            papers = await self._rerank_papers_by_relevance(
                query, papers, top_k=max_results
            )

            # Crossref-enrich: fills missing abstracts (Google Scholar /
            # SciLEx don't always include them), canonicalises author lists
            # and journal names. Same enrichment basic/advanced web-fallback
            # uses; ensures agentic synthesis works on clean records.
            try:
                from perspicacite.pipeline.enrichment.crossref_enrich import enrich_papers
                papers = await enrich_papers(papers)
            except Exception as _ee:
                logger.warning("agentic_enrich_failed", error=str(_ee))

            # arXiv DataCite-DOI fallback: any preprint that Crossref couldn't
            # match by title still has a real citable identifier — arXiv's
            # DataCite DOI 10.48550/arxiv.<id>. Populating it here gives the
            # references section a clickable DOI and prevents downstream dedup
            # collisions in consumers that key by DOI.
            try:
                from perspicacite.pipeline.download.arxiv import get_arxiv_id_from_url
                for _p in papers:
                    if (_p.doi or "").strip():
                        continue
                    _aid = (
                        get_arxiv_id_from_url(_p.pdf_url or "")
                        or get_arxiv_id_from_url(_p.url or "")
                    )
                    if _aid:
                        _p.doi = f"10.48550/arxiv.{_aid.split('v')[0]}"
            except Exception as _ae:
                logger.warning("agentic_arxiv_doi_fallback_failed", error=str(_ae))

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

                logger.info("agentic_scilex_search_found", count=len(paper_dicts))
                return self._format_paper_list(paper_dicts)
            else:
                logger.warning("agentic_scilex_search_no_results")
                self._note_search_fallback(session, _apis)
                return await self._fallback_openalex_search(
                    effective_query, max_results, step_id=step_id, session=session
                )

        except Exception as e:
            logger.error("agentic_scilex_search_failed", error=str(e))
            self._note_search_fallback(session, _apis)
            return await self._fallback_openalex_search(
                effective_query, max_results, step_id=step_id, session=session
            )

    async def _fallback_openalex_search(
        self,
        query: str,
        max_results: int = 10,
        step_id: str = "",
        session: AgentSession | None = None,
    ) -> str:
        """Fallback: Search OpenAlex directly via httpx."""
        import httpx

        search_terms = query.strip()
        logger.info("agentic_openalex_fallback_search", search_terms=search_terms)

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

                logger.info("agentic_openalex_fallback_found", count=len(papers))
                return self._format_paper_list(papers)
        except Exception as e:
            logger.error("agentic_openalex_fallback_failed", error=str(e))
            return f"Literature search failed: {e}"

    def _accumulate_lit_evidence(
        self,
        paper_dicts: list[dict[str, Any]],
        step_id: str,
        session: AgentSession | None,
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

    def _summarize_findings(self, findings: list[dict]) -> str:
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

    def _paper_dedupe_key(self, p: dict[str, Any]) -> str:
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

    def _paper_quality_tuple(self, p: dict[str, Any]) -> tuple:
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

    def _dedupe_paper_dicts(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge duplicates (same DOI or same normalized title, e.g. preprint + journal).

        Keeps the highest quality version of each paper (more abstract, more citations, newer).
        The winning paper retains its original source (kb_search or literature_search).
        """
        best: dict[str, dict[str, Any]] = {}

        for p in papers:
            k = self._paper_dedupe_key(p)
            if k.startswith("unknown:") and not p.get("title"):
                continue

            if k not in best or self._paper_quality_tuple(p) > self._paper_quality_tuple(best[k]):
                best[k] = p

        out = list(best.values())
        if len(out) < len(papers):
            logger.info("agentic_paper_dedupe", before=len(papers), after=len(out))
        return out

    @staticmethod
    def _normalize_authors(authors: Any) -> list[str]:
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

    def _chunks_with_figure_refs(self, step_results: dict[str, Any] | None) -> list[Any]:
        """Best-effort scan of ``step_results`` for DocumentChunk-like items with figure_refs.

        Used to surface multimodal candidates into the final-answer LLM call. The
        agentic flow today stores mostly paper dicts in ``step_results``, so this
        commonly returns ``[]`` — that's the graceful degradation path (text-only).
        """
        out: list[Any] = []
        if not step_results:
            return out
        for v in step_results.values():
            if not isinstance(v, list):
                continue
            for item in v:
                md = getattr(item, "metadata", None)
                if md is None:
                    continue
                refs = getattr(md, "figure_refs", None) or []
                if refs:
                    out.append(item)
        return out

    def _extract_papers_from_results(self, step_results: dict[str, Any]) -> list[dict[str, Any]]:
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

    async def _download_single_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Download and parse a single paper's content.

        Args:
            paper: Paper dict with at least 'doi' and 'title'

        Returns:
            Enriched paper dict with 'full_text' and 'pdf_downloaded' fields
        """
        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.pipeline.download.arxiv import get_arxiv_id_from_url
        from perspicacite.pipeline.parsers.pdf import PDFParser

        doi = (paper.get("doi") or "").strip()
        # Defensive: some adapters historically emitted the literal "NA" string
        # for missing DOIs; treat that as no-DOI to avoid 404-spam downstream.
        if doi.lower() == "na":
            doi = ""

        # Build a download chain: try the Crossref-mapped DOI first (preserves
        # metadata fidelity — the IEEE/Springer/etc. published version is often
        # what we want to cite), then fall back to the arXiv preprint when the
        # publisher version is paywalled. The original paper["doi"] is never
        # overwritten so downstream citations keep the canonical reference.
        arxiv_id = get_arxiv_id_from_url(paper.get("pdf_url") or paper.get("url") or "")
        arxiv_doi = ""
        if arxiv_id:
            bare = arxiv_id.split("v")[0]
            arxiv_doi = f"10.48550/arxiv.{bare}"

        attempts: list[str] = []
        for candidate in (doi, arxiv_doi):
            if candidate and candidate not in attempts:
                attempts.append(candidate)

        if not attempts:
            paper["pdf_downloaded"] = False
            return paper

        parser = PDFParser()
        result = None
        for attempt_doi in attempts:
            try:
                result = await retrieve_paper_content(
                    doi=attempt_doi,
                    pdf_parser=parser,
                    unpaywall_email="perspicacite@example.com",
                )
                if result.success and result.full_text:
                    break
            except Exception as e:
                logger.warning("agentic_download_failed", doi=attempt_doi, error=str(e))
                result = None

        if result and result.success and result.full_text:
            paper["full_text"] = result.full_text
            paper["pdf_downloaded"] = True
        else:
            paper["pdf_downloaded"] = False

        return paper

    async def _download_and_enrich_papers(
        self, papers: list[dict[str, Any]], relevance_threshold: int = 3, max_papers: int = 10
    ) -> list[dict[str, Any]]:
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
            p for p in papers if p.get("relevance_score", 0) >= relevance_threshold and p.get("doi")
        ][:max_papers]

        # Prioritize open access papers first (they're more likely to download successfully)
        download_candidates = sorted(
            relevant_papers,
            key=lambda p: (
                not p.get("open_access", {}).get("is_oa", False),
                p.get("relevance_score", 0),
            ),
            reverse=True,
        )

        if not download_candidates:
            logger.info("agentic_download_no_candidates")
            return papers

        logger.info(
            "agentic_download_candidates",
            count=len(download_candidates),
            relevance_threshold=relevance_threshold,
        )

        enriched = []
        for paper in papers:
            doi = paper.get("doi", "")
            if not doi or paper not in download_candidates:
                enriched.append(paper)
                continue

            enriched_paper = await self._download_single_paper(paper)
            enriched.append(enriched_paper)

        return enriched

