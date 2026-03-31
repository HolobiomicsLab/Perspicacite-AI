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

        kb_rules = ""
        if active_kb_name:
            kb_rules = f"""

ACTIVE KNOWLEDGE BASE: The user selected curated KB "{active_kb_name}".
- Step 1 MUST be type "kb_search", tool "kb_search", with tool_input.query = the cleaned core topic (same phrase you would use for OpenAlex — no invented qualifiers).
- You SHOULD add step 2 as "literature_search" on that same core topic as a fallback when the KB is sparse (the system may skip step 2 automatically if KB retrieval is already sufficient).
- Do not emit duplicate kb_search steps.
"""
        
        prompt = f"""You are a research planner for a scientific research assistant. Create an effective research plan based on the user's query and intent.

Original Query: "{query}"
Classified Intent: {intent_result.intent.name} (confidence: {intent_result.confidence:.2f})
Intent Reasoning: {intent_result.reasoning}
Extracted Entities: {intent_result.entities}

Available Tools: {available_tools}{kb_rules}

{context}

Your task is to create a research plan following the strategy below.

SEARCH STRATEGY (follow this like a senior researcher would):

1. CLEAN THE QUERY: Strip conversational preamble ("what is", "tell me about", etc.).
   Use ONLY terms from the Original Query. NEVER invent new terms.

2. ONE SEARCH FIRST: Your first search step must be the core topic with NOTHING appended.
   Do NOT add "methodology", "review", "applications", or any qualifier.
   One well-targeted search on the exact topic finds the original paper, reviews,
   and related work — all of which contain the topic name.
   - "what is feature based molecular networking and its application"
     → search: "feature-based molecular networking" (NOT "...methodology", NOT "...review")
   - "tell me about CRISPR gene editing" → search: "CRISPR gene editing"

3. DO NOT OVER-DECOMPOSE: Do NOT pre-plan multiple parallel searches for broad queries.
   A single search on the core topic covers more ground than several narrowed searches.
   If the initial results are insufficient, the system will replan and add targeted searches.

4. ONLY exception for a second search: if the user explicitly asks about a DISTINCT aspect
   that would require genuinely different search terms (e.g., "compare X with Y",
   "X and its effect on Z"). In that case, add ONE additional search for that specific aspect.

Step Types:
- lotus_search: Natural products, chemical structures
- literature_search: Academic literature search via SciLEx (multi-API: Semantic Scholar, OpenAlex, PubMed, etc.)
- kb_search: Search existing knowledge base
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
            "tool_input": {{"query": "CLEAN search query using ONLY original query terms"}},
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
                step_type = StepType(step_data.get("type", "answer"))
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
            
            return self._build_fallback_plan(query, intent_result, available_tools, e)

    def _build_fallback_plan(self, query: str, intent_result, available_tools, error=None):
        """Build an intent-aware fallback plan when LLM planning fails."""
        from .intent import Intent
        
        clean_query = self._clean_query_for_search(query)
        intent = intent_result.intent
        fallback_steps = []
        
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
                fallback_steps.append(Step(
                    id="step1",
                    type=StepType.LITERATURE_SEARCH,
                    description="Search for papers on core topic",
                    tool="literature_search",
                    tool_input={"query": clean_query}
                ))
        
        elif intent == Intent.COMBINED_RESEARCH:
            step_counter = 1
            if "lotus_search" in available_tools:
                fallback_steps.append(Step(
                    id=f"step{step_counter}",
                    type=StepType.LOTUS_SEARCH,
                    description="Search LOTUS for natural products",
                    tool="lotus_search",
                    tool_input={"query": clean_query}
                ))
                step_counter += 1
            
            if "literature_search" in available_tools:
                sub_queries = self._decompose_query(clean_query)
                for sub_q in sub_queries:
                    fallback_steps.append(Step(
                        id=f"step{step_counter}",
                        type=StepType.LITERATURE_SEARCH,
                        description=f"Search papers: {sub_q}",
                        tool="literature_search",
                        tool_input={"query": sub_q}
                    ))
                    step_counter += 1
        
        else:
            if "literature_search" in available_tools:
                fallback_steps.append(Step(
                    id="step1",
                    type=StepType.LITERATURE_SEARCH,
                    description="Search for papers",
                    tool="literature_search",
                    tool_input={"query": clean_query}
                ))
        
        fallback_steps.append(Step(
            id="final",
            type=StepType.ANSWER,
            description="Generate answer",
            depends_on=[s.id for s in fallback_steps[-1:]] if fallback_steps else []
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
    
    async def replan(
        self,
        query: str,
        current_plan: Plan,
        completed_steps: List[Step],
        step_results: Dict[str, Any],
        evaluation: str
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
        
        prompt = f"""Evaluate and replan if needed.

Query: "{query}"
Evaluation: {evaluation}

Completed steps:
{chr(10).join(results_summary)}

Current plan steps remaining: {len(current_plan.steps) - len(completed_steps)}

Should we:
1. Continue with current plan
2. Add more research steps
3. Answer with what we have

Return JSON:
{{
    "action": "continue|add_steps|answer",
    "reasoning": "why",
    "additional_steps": [  # Only if action is "add_steps"
        {{
            "id": "new_step1",
            "type": "tool_type",
            "description": "...",
            "tool": "tool_name",
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
                    new_steps.append(Step(
                        id=step_data["id"],
                        type=StepType(step_data["type"]),
                        description=step_data["description"],
                        tool=step_data.get("tool"),
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
