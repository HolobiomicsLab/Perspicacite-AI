"""Dynamic research planning with LLM."""

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import re

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output."""
    stripped = text.strip()
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


class StepType(Enum):
    """Types of research steps."""
    LOTUS_SEARCH = "lotus_search"
    LITERATURE_SEARCH = "literature_search"  # Generic academic literature search (may use OpenAlex, SciLEx, or fallback)
    DOWNLOAD_PAPERS = "download_papers"
    KB_SEARCH = "kb_search"
    WEB_SEARCH = "web_search"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    ANSWER = "answer"


_STEP_TYPE_VALUE_SET = frozenset(m.value for m in StepType)


def coerce_step_type(raw: Any, *, tool: Optional[str] = None) -> StepType:
    """Map LLM output to ``StepType`` (handles synonyms the model invents)."""
    if isinstance(raw, StepType):
        return raw
    key = (raw if isinstance(raw, str) else str(raw or "")).strip().lower()
    key = re.sub(r"[\s-]+", "_", key)

    tool_key = (tool or "").strip().lower()
    tool_key = re.sub(r"[\s-]+", "_", tool_key)

    # Explicit canonical type from the model wins over the tool field.
    if key in _STEP_TYPE_VALUE_SET:
        return StepType(key)

    type_aliases: dict[str, StepType] = {
        # Replanner often emits these; they are not enum values.
        "document_retrieval": StepType.KB_SEARCH,
        "document_retrieval_step": StepType.KB_SEARCH,
        "doc_retrieval": StepType.KB_SEARCH,
        "retrieve_documents": StepType.KB_SEARCH,
        "vector_search": StepType.KB_SEARCH,
        "rag_search": StepType.KB_SEARCH,
        "kb_retrieval": StepType.KB_SEARCH,
        "corpus_search": StepType.KB_SEARCH,
        "knowledge_base_search": StepType.KB_SEARCH,
        "paper_retrieval": StepType.LITERATURE_SEARCH,
        "literature_retrieval": StepType.LITERATURE_SEARCH,
        "academic_search": StepType.LITERATURE_SEARCH,
        "scholar_search": StepType.LITERATURE_SEARCH,
        "pubmed_search": StepType.LITERATURE_SEARCH,
        "openalex_search": StepType.LITERATURE_SEARCH,
        "final_answer": StepType.ANSWER,
        "respond": StepType.ANSWER,
    }
    tool_aliases: dict[str, StepType] = {
        "document_retrieval": StepType.KB_SEARCH,
        "doc_retrieval": StepType.KB_SEARCH,
        "retrieve_documents": StepType.KB_SEARCH,
        "vector_search": StepType.KB_SEARCH,
        "rag": StepType.KB_SEARCH,
    }
    if key in type_aliases:
        coerced = type_aliases[key]
        logger.info(f"Planner: coerced step type {raw!r} → {coerced.value}")
        return coerced
    if tool_key in tool_aliases:
        coerced = tool_aliases[tool_key]
        logger.info(f"Planner: coerced step type from tool {tool!r} → {coerced.value}")
        return coerced
    if tool_key in _STEP_TYPE_VALUE_SET:
        return StepType(tool_key)
    try:
        return StepType(key)
    except ValueError:
        logger.warning(
            f"Planner: unknown step type {raw!r} tool={tool!r}; defaulting to literature_search"
        )
        return StepType.LITERATURE_SEARCH


@dataclass
class Step:
    """A single research step."""
    id: str
    type: StepType
    description: str
    tool: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Execute only if condition met


@dataclass
class Plan:
    """A research plan."""
    steps: List[Step]
    reasoning: str
    estimated_steps: int
    can_answer_from_history: bool = False


def _log_steps_detail(steps: List[Step], label: str) -> None:
    """Structured log of every planned step (tool_input, deps, full description)."""
    logger.info(f"{label}: {len(steps)} step(s)")
    for i, s in enumerate(steps, 1):
        logger.info(
            f"{label} [{i}/{len(steps)}] id={s.id!r} type={s.type.value} tool={s.tool!r} "
            f"depends_on={s.depends_on!r} condition={s.condition!r}"
        )
        logger.info(f"{label} [{i}] description: {s.description}")
        if s.tool_input:
            try:
                logger.info(
                    f"{label} [{i}] tool_input: {json.dumps(s.tool_input, ensure_ascii=False)}"
                )
            except (TypeError, ValueError):
                logger.info(f"{label} [{i}] tool_input: {s.tool_input!r}")


class ResearchPlanner:
    """Generates dynamic research plans using LLM."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def create_plan(
        self,
        query: str,
        intent_result,
        available_tools: List[str],
        conversation_history: Optional[List[dict]] = None,
        previous_findings: Optional[str] = None,
        active_kb_name: Optional[str] = None,
    ) -> Plan:
        """
        Create a dynamic research plan.
        
        Args:
            query: User query
            intent_result: Classified intent
            available_tools: List of available tool names
            conversation_history: Previous messages
            previous_findings: Summary of previous research
            active_kb_name: If set, planner must lead with kb_search for that KB

        Returns:
            Plan with steps to execute
        """
        
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

        query_complexity = getattr(intent_result, "query_complexity", "simple")
        query_complexity_source = getattr(intent_result, "query_complexity_source", "default")

        kb_rules = ""
        if active_kb_name:
            kb_rules = f"""

ACTIVE KNOWLEDGE BASE: The user selected curated KB "{active_kb_name}".
- Step 1 MUST be type "kb_search", tool "kb_search", with tool_input.query = the cleaned core topic (same phrase you would use for OpenAlex — no invented qualifiers).
- You SHOULD add step 2 as "literature_search" on that same core topic as a fallback when the KB is sparse (the system may skip step 2 automatically if KB retrieval is already sufficient).
- Do not emit duplicate kb_search steps.
"""

        composite_rules = ""
        if query_complexity == "composite":
            composite_rules = """

QUERY COMPLEXITY: composite (multi-aspect, comparison, or pros/cons).
For the FIRST wave of kb_search steps ONLY (when ACTIVE KNOWLEDGE BASE applies):
- Emit 2–3 kb_search steps, each with "depends_on": [] (parallel), with DISTINCT tool_input.query strings.
- Each sub-query MUST use ONLY words and phrases copied from the Original Query — no invented synonyms or new topics.
- Example: "FBMN vs GNPS workflows" → separate kb_search queries for each side (minimal phrases from the query).
- After that wave, use literature_search / answer as usual; set depends_on so later steps wait for ALL parallel kb steps when needed.
"""
        
        simple_search_rules = """
SIMPLE query path (query complexity is simple — one focused topic, no explicit comparison):
1. CLEAN THE QUERY: Strip conversational preamble ("what is", "tell me about", etc.).
   Use ONLY terms from the Original Query. NEVER invent new terms.
2. ONE SEARCH FIRST: The first search step must be the core topic with NOTHING appended.
   Do NOT add "methodology", "review", "applications", or extra qualifiers.
   - "what is feature based molecular networking and its application"
     → search: "feature-based molecular networking" (NOT "...methodology", NOT "...review")
   - "tell me about CRISPR gene editing" → search: "CRISPR gene editing"
3. DO NOT OVER-DECOMPOSE: Do NOT pre-plan multiple parallel searches for broad simple queries.
   One strong search beats several narrowed ones; replanning adds more if results are thin.
4. Exception — genuinely distinct aspect only: if the user clearly asks for a second facet with different
   search terms (rare for simple complexity), add ONE additional search for that facet — not parallel batches.
"""

        composite_search_rules = """
COMPOSITE query path (query complexity is composite — see COMPOSITE block above):
- Ignore the "one search first" and "do not over-decompose" rules for the initial kb_search wave.
- Follow the COMPOSITE block: 2–3 parallel kb_search steps with distinct sub-queries from the Original Query only.
- After that wave, prefer a single literature_search on the cleaned core topic unless the query clearly needs more.
"""

        mode_rules = composite_search_rules if query_complexity == "composite" else simple_search_rules

        prompt = f"""You are a research planner for a scientific research assistant. Create an effective research plan based on the user's query and intent.

Original Query: "{query}"
Classified Intent: {intent_result.intent.name} (confidence: {intent_result.confidence:.2f})
Intent Reasoning: {intent_result.reasoning}
Query complexity: {query_complexity} (detector: {query_complexity_source})
Extracted Entities: {intent_result.entities}

Available Tools: {available_tools}{kb_rules}{composite_rules}

{context}

Your task is to create a research plan following the strategy below.

SEARCH STRATEGY — apply the path that matches query complexity (stated above):

{mode_rules.strip()}

Step Types:
- lotus_search: Natural products, chemical structures
- literature_search: Academic literature search via SciLEx (multi-API: Semantic Scholar, OpenAlex, PubMed, etc.)
- kb_search: Search existing knowledge base. Use top_k (1-20) in tool_input to control how many papers to retrieve (3-5 for targeted queries, 10-20 for broad surveys).
- analyze: Process and extract insights
- answer: Final response

Intent-Specific:
- NATURAL_PRODUCTS_ONLY: lotus_search → answer
- PAPERS_ONLY: If ACTIVE KNOWLEDGE BASE is set: kb_search → literature_search → answer; else literature_search → answer
- COMBINED_RESEARCH: If ACTIVE KNOWLEDGE BASE is set: kb_search → literature_search (same core topic) → answer; else literature_search (core topic) → answer (replan adds more if needed)
- FOLLOW_UP: Focus on gaps from previous research

Return JSON only (no markdown):
{{
    "reasoning": "research strategy",
    "can_answer_from_history": false,
    "steps": [
        {{
            "id": "step1",
            "type": "lotus_search|literature_search|kb_search|analyze|synthesize|answer",
            "description": "what this step does",
            "tool": "tool_name",
            "tool_input": {{"query": "CLEAN search query using ONLY original query terms", "top_k": 5}},
            "depends_on": [],
            "condition": null
        }}
    ]
}}"""

        try:
            logger.info("========== PLANNER create_plan ==========")
            q_preview = query if len(query) <= 2000 else query[:2000] + "…"
            logger.info(f"Query ({len(query)} chars): {q_preview}")
            logger.info(
                f"Intent: {intent_result.intent.name} "
                f"confidence={intent_result.confidence:.3f}"
            )
            logger.info(f"Intent reasoning: {intent_result.reasoning}")
            logger.info(f"Intent entities: {intent_result.entities!r}")
            logger.info(f"Available tools: {available_tools}")
            if context.strip():
                ctx_prev = context if len(context) <= 1200 else context[:1200] + "…"
                logger.info(f"Planner conversation/previous_findings context:\n{ctx_prev}")
            logger.info(f"Planner prompt length: {len(prompt)} chars")

            response = await self.llm.complete(prompt, temperature=0.2)
            logger.info(f"Planner raw LLM response length: {len(response)} chars")
            head, tail = 1600, 600
            if len(response) <= head + tail:
                logger.info(f"Planner raw LLM response (full):\n{response}")
            else:
                logger.info(f"Planner raw LLM response (first {head} chars):\n{response[:head]}…")
                logger.info(f"Planner raw LLM response (last {tail} chars):\n…{response[-tail:]}")

            cleaned_response = _strip_markdown_fences(response)
            logger.info(f"Planner fenced-stripped length: {len(cleaned_response)} chars")
            result = json.loads(cleaned_response)

            reasoning = result.get("reasoning", "N/A")
            logger.info(f"Plan reasoning ({len(reasoning)} chars): {reasoning}")
            cfh = result.get("can_answer_from_history", False)
            logger.info(f"Plan can_answer_from_history: {cfh}")

            steps_data = result.get("steps", [])
            logger.info(f"LLM returned {len(steps_data)} step object(s)")

            steps = []
            for step_data in steps_data:
                step_type = coerce_step_type(
                    step_data.get("type", "answer"),
                    tool=step_data.get("tool"),
                )
                steps.append(Step(
                    id=step_data["id"],
                    type=step_type,
                    description=step_data["description"],
                    tool=step_data.get("tool"),
                    tool_input=step_data.get("tool_input", {}),
                    depends_on=step_data.get("depends_on", []),
                    condition=step_data.get("condition")
                ))

            _log_steps_detail(steps, "Planned")

            return Plan(
                steps=steps,
                reasoning=result.get("reasoning", ""),
                estimated_steps=len(steps),
                can_answer_from_history=result.get("can_answer_from_history", False)
            )
            
        except Exception as e:
            logger.error(f"Error in planning: {e}", exc_info=True)
            if "response" in locals() and response:
                rs = response
                logger.info(
                    f"Planner LLM response on failure: length={len(rs)} "
                    f"head={rs[:800]!r}{'…' if len(rs) > 800 else ''}"
                )
            else:
                logger.info("Planner LLM response on failure: (none)")
            
            return await self._build_fallback_plan(
                query,
                intent_result,
                available_tools,
                e,
                active_kb_name=active_kb_name,
                query_complexity=getattr(intent_result, "query_complexity", "simple"),
            )

    async def _build_fallback_plan(
        self,
        query: str,
        intent_result,
        available_tools,
        error=None,
        active_kb_name: Optional[str] = None,
        query_complexity: str = "simple",
    ):
        """Build an intent-aware fallback plan when LLM planning fails."""
        from .intent import Intent
        
        clean_query = self._clean_query_for_search(query)
        intent = intent_result.intent
        fallback_steps: List[Step] = []
        
        if intent == Intent.NATURAL_PRODUCTS_ONLY:
            if "lotus_search" in available_tools:
                fallback_steps.append(Step(
                    id="step1",
                    type=StepType.LOTUS_SEARCH,
                    description="Search LOTUS for natural products",
                    tool="lotus_search",
                    tool_input={"query": clean_query}
                ))
        
        elif intent == Intent.PAPERS_ONLY:
            if "literature_search" in available_tools:
                sub_queries = (
                    await self.composite_subqueries_with_llm(clean_query)
                    if query_complexity == "composite"
                    else [clean_query]
                )
                sub_queries = list(dict.fromkeys(sub_queries))[:3]
                for i, sub_q in enumerate(sub_queries):
                    fallback_steps.append(Step(
                        id=f"step_lit_{i+1}",
                        type=StepType.LITERATURE_SEARCH,
                        description=f"Search for papers: {sub_q}",
                        tool="literature_search",
                        tool_input={"query": sub_q},
                        depends_on=[],
                    ))
        
        elif intent == Intent.COMBINED_RESEARCH:
            step_counter = 1
            composite_subs = (
                await self.composite_subqueries_with_llm(clean_query)
                if query_complexity == "composite"
                else None
            )
            if active_kb_name and "kb_search" in available_tools:
                kb_queries = list(dict.fromkeys(
                    composite_subs if composite_subs else [clean_query]
                ))[:3]
                for sub_q in kb_queries:
                    fallback_steps.append(Step(
                        id=f"step_kb_{step_counter}",
                        type=StepType.KB_SEARCH,
                        description=f"Search knowledge base '{active_kb_name}'",
                        tool="kb_search",
                        tool_input={"query": sub_q, "top_k": 10},
                        depends_on=[],
                    ))
                    step_counter += 1
            if "lotus_search" in available_tools:
                fallback_steps.append(Step(
                    id=f"step{step_counter}",
                    type=StepType.LOTUS_SEARCH,
                    description="Search LOTUS for natural products",
                    tool="lotus_search",
                    tool_input={"query": clean_query},
                    depends_on=[],
                ))
                step_counter += 1

            if "literature_search" in available_tools:
                sub_queries = list(dict.fromkeys(
                    composite_subs if composite_subs else self._decompose_query(clean_query)
                ))[:3]
                for sub_q in sub_queries:
                    fallback_steps.append(Step(
                        id=f"step{step_counter}",
                        type=StepType.LITERATURE_SEARCH,
                        description=f"Search papers: {sub_q}",
                        tool="literature_search",
                        tool_input={"query": sub_q},
                        depends_on=[],
                    ))
                    step_counter += 1
        
        else:
            if "literature_search" in available_tools:
                fallback_steps.append(Step(
                    id="step1",
                    type=StepType.LITERATURE_SEARCH,
                    description="Search for papers",
                    tool="literature_search",
                    tool_input={"query": clean_query},
                    depends_on=[],
                ))

        pre_answer_ids = [s.id for s in fallback_steps]
        fallback_steps.append(Step(
            id="final",
            type=StepType.ANSWER,
            description="Generate answer",
            depends_on=pre_answer_ids,
        ))
        
        logger.warning(
            f"Using fallback plan with {len(fallback_steps)} steps (intent: {intent.name}); "
            f"error={error!r}"
        )
        _log_steps_detail(fallback_steps, "Fallback plan")

        return Plan(
            steps=fallback_steps,
            reasoning=f"LLM planning failed ({error}). Fallback for intent {intent.name}.",
            estimated_steps=len(fallback_steps)
        )
    
    @staticmethod
    def _clean_query_for_search(query: str) -> str:
        """Remove conversational preamble from a query for use as a search term."""
        cleaned = query.strip()
        prefixes = [
            "i want to learn about", "i want to know about",
            "tell me about", "what is", "what are",
            "how does", "how do", "explain", "describe",
            "can you tell me about", "i'd like to know about",
        ]
        lower = cleaned.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        return cleaned
    
    @staticmethod
    def _decompose_query(clean_query: str) -> list:
        """Extract the core topic from a query. Only adds a second sub-query
        if the user explicitly mentioned a distinct aspect (e.g., "X and its Y").
        
        Default is a single search on the core topic — one good search beats
        multiple narrowed ones. The replan loop handles follow-ups.
        """
        parts = re.split(r'\band\b(?:\s+(?:its|their|the))?\s+', clean_query, maxsplit=1)
        if len(parts) == 2 and len(parts[0].strip()) > 5 and len(parts[1].strip()) > 3:
            base_topic = parts[0].strip()
            aspect = parts[1].strip()
            return [base_topic, f"{base_topic} {aspect}"]
        
        return [clean_query]

    @staticmethod
    def _composite_subqueries(clean_query: str) -> List[str]:
        """Split a comparison / multi-entity query into 2-3 search phrases (regex only)."""
        q = (clean_query or "").strip()
        if not q:
            return [clean_query]
        split_vs = re.split(r"(?i)\s+vs\.?\s+|\s+versus\s+", q, maxsplit=1)
        if len(split_vs) == 2:
            a, b = split_vs[0].strip(), split_vs[1].strip()
            if len(a) > 2 and len(b) > 2:
                return [a, b][:3]
        dec = ResearchPlanner._decompose_query(q)
        if len(dec) >= 2:
            return dec[:3]
        split_and = re.split(r"(?i)\s+and\s+", q, maxsplit=1)
        if len(split_and) == 2:
            a, b = split_and[0].strip(), split_and[1].strip()
            if len(a) > 3 and len(b) > 3:
                return [a, b][:3]
        return [q]

    async def composite_subqueries_with_llm(self, clean_query: str) -> List[str]:
        """Split a composite query into 2-3 search phrases.

        Tries regex first via ``_composite_subqueries``.  If that yields only a
        single phrase (regex couldn't find a split point), falls back to a
        lightweight LLM call to decompose the query.
        """
        regex_result = self._composite_subqueries(clean_query)
        if len(regex_result) >= 2:
            return regex_result

        prompt = (
            "Split the following research query into 2-3 distinct, "
            "non-overlapping search phrases. Each phrase should target a "
            "different aspect or entity in the query. Use ONLY words from "
            "the original query — do not invent synonyms or add terms.\n\n"
            f'Query: "{clean_query}"\n\n'
            'Return JSON: {"sub_queries": ["phrase1", "phrase2"]}\n'
            "Valid JSON only:"
        )
        try:
            raw = await self.llm.complete(prompt, temperature=0.0, max_tokens=256)
            cleaned = _strip_markdown_fences(raw)
            data = json.loads(cleaned)
            subs = data.get("sub_queries", [])
            subs = [s.strip() for s in subs if isinstance(s, str) and len(s.strip()) > 2]
            if len(subs) >= 2:
                logger.info(
                    f"LLM composite decomposition: {clean_query!r} → {subs[:3]}"
                )
                return subs[:3]
        except Exception as e:
            logger.warning(f"LLM composite decomposition failed: {e}")

        return regex_result
    
    async def replan(
        self,
        query: str,
        current_plan: Plan,
        completed_steps: List[Step],
        step_results: Dict[str, Any],
        evaluation: str,
        evidence_summary: Optional[str] = None,
    ) -> Plan:
        """
        Replan based on evaluation of current results.
        
        Args:
            query: Original query
            current_plan: Current plan
            completed_steps: Steps already executed
            step_results: Results from completed steps
            evaluation: LLM evaluation of whether more research needed
            
        Returns:
            Updated plan
        """
        
        results_summary = []
        for step in completed_steps:
            result = step_results.get(step.id, "No result")
            results_summary.append(f"{step.id} ({step.type.value}): {str(result)[:200]}")

        evidence_block = ""
        if evidence_summary and evidence_summary.strip():
            evidence_block = (
                "\n--- Per-facet evidence & gap status ---\n"
                f"{evidence_summary.strip()[:3200]}\n"
                "--- end evidence ---\n"
            )

        prompt = f"""Evaluate and replan if needed.

Query: "{query}"
Evaluation: {evaluation}
{evidence_block}
Completed steps:
{chr(10).join(results_summary)}

Current plan steps remaining: {len(current_plan.steps) - len(completed_steps)}

Instructions:
- Review the per-facet evidence status above.  Facets marked [GAP] have no
  evidence yet; facets marked [PARTIAL] have limited evidence.
- If any facet is [GAP] or [PARTIAL], consider adding targeted search steps
  for that specific facet query.
- If all facets are [COVERED], prefer "answer" unless the evaluation
  specifically says more depth is needed.
- When adding steps, target the gap facets with specific queries.

For each added step, "type" MUST be exactly one of these strings (no synonyms):
lotus_search, literature_search, download_papers, kb_search, web_search, analyze, synthesize, answer
- Use kb_search for curated knowledge-base / in-corpus retrieval (including "full paper in KB").
- Use literature_search for academic API search (OpenAlex, Semantic Scholar, PubMed, etc.).
- Do NOT use invented names such as document_retrieval — use kb_search or literature_search instead.

Choose ONE action:
1. "continue" – current plan still has useful remaining steps
2. "add_steps" – add targeted searches for gap/partial facets
3. "answer" – evidence is sufficient, generate final answer

Return JSON:
{{
    "action": "continue|add_steps|answer",
    "reasoning": "why",
    "additional_steps": [
        {{
            "id": "new_step1",
            "type": "kb_search|literature_search|lotus_search|download_papers|web_search|analyze|synthesize|answer",
            "description": "...",
            "tool": "kb_search|literature_search|... (must match type)",
            "tool_input": {{}},
            "depends_on": []
        }}
    ]
}}

Valid JSON only:"""

        logger.info("========== PLANNER replan (pre-LLM) ==========")
        for step in completed_steps:
            raw = step_results.get(step.id, "No result")
            sr = str(raw)
            prev = sr[:700] + ("…" if len(sr) > 700 else "")
            logger.info(
                f"Replan prior result step={step.id!r} type={step.type.value} "
                f"len={len(sr)} preview={prev!r}"
            )

        try:
            logger.info("========== PLANNER replan (LLM) ==========")
            logger.info(f"Replan query ({len(query)} chars): {query[:1500]}{'…' if len(query) > 1500 else ''}")
            logger.info(f"Replan evaluation: {evaluation}")
            logger.info(
                f"Replan: remaining steps in current plan ≈ "
                f"{len(current_plan.steps) - len(completed_steps)} (by count)"
            )
            logger.info(f"Replan prompt length: {len(prompt)} chars")

            response = await self.llm.complete(prompt, temperature=0.2)
            logger.info(f"Replan raw LLM response length: {len(response)} chars")
            r_head = 1400
            if len(response) <= r_head:
                logger.info(f"Replan raw LLM response (full):\n{response}")
            else:
                logger.info(f"Replan raw LLM response (first {r_head} chars):\n{response[:r_head]}…")

            cleaned_response = _strip_markdown_fences(response)
            result = json.loads(cleaned_response)

            action = result.get("action", "continue")
            logger.info(
                f"Replan action={action!r} reasoning={result.get('reasoning', '')!r}"
            )

            if action == "add_steps":
                new_steps = []
                for step_data in result.get("additional_steps", []):
                    st = coerce_step_type(
                        step_data.get("type"),
                        tool=step_data.get("tool"),
                    )
                    tool_name = step_data.get("tool")
                    if st == StepType.KB_SEARCH and (not tool_name or tool_name == step_data.get("type")):
                        tool_name = "kb_search"
                    elif st == StepType.LITERATURE_SEARCH and (
                        not tool_name or tool_name == step_data.get("type")
                    ):
                        tool_name = "literature_search"
                    new_steps.append(Step(
                        id=step_data["id"],
                        type=st,
                        description=step_data["description"],
                        tool=tool_name,
                        tool_input=step_data.get("tool_input", {}),
                        depends_on=step_data.get("depends_on", [])
                    ))
                
                # Append new steps to current plan
                current_plan.steps.extend(new_steps)
                current_plan.estimated_steps = len(current_plan.steps)
                current_plan.reasoning += f"\nReplanned: {result.get('reasoning', '')}"
                _log_steps_detail(new_steps, "Replan added steps")

            elif action == "answer":
                # Remove remaining steps, just add answer step
                current_plan.steps = completed_steps + [Step(
                    id="answer",
                    type=StepType.ANSWER,
                    description="Generate final answer",
                    depends_on=[s.id for s in completed_steps]
                )]
                logger.info("Replan: action=answer — truncating plan to completed + ANSWER step")

            else:
                logger.info("Replan: action=continue — plan unchanged except reasoning append if any")

            _log_steps_detail(current_plan.steps, "Replan final plan")

            return current_plan

        except Exception as e:
            logger.error(f"Replan failed, returning unchanged plan: {e}", exc_info=True)
            return current_plan
