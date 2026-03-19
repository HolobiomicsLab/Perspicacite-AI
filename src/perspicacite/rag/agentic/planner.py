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
    OPENALEX_SEARCH = "openalex_search"
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
        previous_findings: Optional[str] = None
    ) -> Plan:
        """
        Create a dynamic research plan.
        
        Args:
            query: User query
            intent_result: Classified intent
            available_tools: List of available tool names
            conversation_history: Previous messages
            previous_findings: Summary of previous research
            
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
        
        prompt = f"""You are a research planner for a scientific research assistant. Create an effective research plan based on the user's query and intent.

Original Query: "{query}"
Classified Intent: {intent_result.intent.name} (confidence: {intent_result.confidence:.2f})
Intent Reasoning: {intent_result.reasoning}
Extracted Entities: {intent_result.entities}

Available Tools: {available_tools}

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
- openalex_search: Academic papers via OpenAlex/Semantic Scholar
- kb_search: Search existing knowledge base
- analyze: Process and extract insights
- answer: Final response

Intent-Specific:
- NATURAL_PRODUCTS_ONLY: lotus_search → answer
- PAPERS_ONLY: openalex_search → answer
- COMBINED_RESEARCH: openalex_search (core topic) → answer (replan adds more if needed)
- FOLLOW_UP: Focus on gaps from previous research

Return JSON only (no markdown):
{{
    "reasoning": "research strategy",
    "can_answer_from_history": false,
    "steps": [
        {{
            "id": "step1",
            "type": "lotus_search|openalex_search|kb_search|analyze|synthesize|answer",
            "description": "what this step does",
            "tool": "tool_name",
            "tool_input": {{"query": "CLEAN search query using ONLY original query terms"}},
            "depends_on": [],
            "condition": null
        }}
    ]
}}"""

        try:
            logger.info(f"Creating plan for query: {query}")
            logger.info(f"Intent: {intent_result.intent.name}, Tools: {available_tools}")
            
            response = await self.llm.complete(prompt, temperature=0.2)
            logger.debug(f"Planner LLM response: {response[:500]}...")
            
            cleaned_response = _strip_markdown_fences(response)
            result = json.loads(cleaned_response)
            logger.info(f"Plan reasoning: {result.get('reasoning', 'N/A')}")
            
            steps_data = result.get("steps", [])
            logger.info(f"LLM generated {len(steps_data)} steps")
            
            steps = []
            for step_data in steps_data:
                step_type = StepType(step_data.get("type", "answer"))
                logger.info(f"  Step: {step_data.get('id')} - {step_type.value} - {step_data.get('description', '')[:50]}")
                steps.append(Step(
                    id=step_data["id"],
                    type=step_type,
                    description=step_data["description"],
                    tool=step_data.get("tool"),
                    tool_input=step_data.get("tool_input", {}),
                    depends_on=step_data.get("depends_on", []),
                    condition=step_data.get("condition")
                ))
            
            return Plan(
                steps=steps,
                reasoning=result.get("reasoning", ""),
                estimated_steps=len(steps),
                can_answer_from_history=result.get("can_answer_from_history", False)
            )
            
        except Exception as e:
            logger.error(f"Error in planning: {e}", exc_info=True)
            logger.info(f"LLM response was: {response[:500] if 'response' in locals() else 'N/A'}...")
            
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
            if "openalex_search" in available_tools:
                fallback_steps.append(Step(
                    id="step1",
                    type=StepType.OPENALEX_SEARCH,
                    description="Search for papers on core topic",
                    tool="openalex_search",
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
            
            if "openalex_search" in available_tools:
                sub_queries = self._decompose_query(clean_query)
                for sub_q in sub_queries:
                    fallback_steps.append(Step(
                        id=f"step{step_counter}",
                        type=StepType.OPENALEX_SEARCH,
                        description=f"Search papers: {sub_q}",
                        tool="openalex_search",
                        tool_input={"query": sub_q}
                    ))
                    step_counter += 1
        
        else:
            if "openalex_search" in available_tools:
                fallback_steps.append(Step(
                    id="step1",
                    type=StepType.OPENALEX_SEARCH,
                    description="Search for papers",
                    tool="openalex_search",
                    tool_input={"query": clean_query}
                ))
        
        fallback_steps.append(Step(
            id="final",
            type=StepType.ANSWER,
            description="Generate answer",
            depends_on=[s.id for s in fallback_steps[-1:]] if fallback_steps else []
        ))
        
        logger.info(f"Using fallback plan with {len(fallback_steps)} steps (intent: {intent.name})")
        for s in fallback_steps:
            logger.info(f"  Fallback step: {s.id} - {s.type.value} - {s.description}")
        
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

        try:
            response = await self.llm.complete(prompt, temperature=0.2)
            cleaned_response = _strip_markdown_fences(response)
            result = json.loads(cleaned_response)
            
            action = result.get("action", "continue")
            
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
                
            elif action == "answer":
                # Remove remaining steps, just add answer step
                current_plan.steps = completed_steps + [Step(
                    id="answer",
                    type=StepType.ANSWER,
                    description="Generate final answer",
                    depends_on=[s.id for s in completed_steps]
                )]
            
            return current_plan
            
        except Exception:
            return current_plan
