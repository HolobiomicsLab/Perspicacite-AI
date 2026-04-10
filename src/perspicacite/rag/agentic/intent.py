"""Intent classification for query routing."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Tuple
import json
import re
import logging

logger = logging.getLogger(__name__)

# Heuristic triggers for multi-aspect / comparison queries (parallel search + map-reduce path).
_COMPOSITE_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvs\.?\b", re.I), "vs"),
    (re.compile(r"\bversus\b", re.I), "versus"),
    (re.compile(r"\bcompare\b", re.I), "compare"),
    (re.compile(r"\bcomparison\b", re.I), "comparison"),
    (re.compile(r"\bpros and cons\b", re.I), "pros_cons"),
    (re.compile(r"\badvantages and disadvantages\b", re.I), "adv_disadv"),
    (re.compile(r"\bdifference between\b", re.I), "difference_between"),
    (re.compile(r"\bdifferences between\b", re.I), "differences_between"),
    (re.compile(r"\btrade-?offs?\b", re.I), "tradeoffs"),
    (re.compile(r"\beffect of .{3,80} on\b", re.I), "effect_on"),
    (re.compile(r"\band its effect on\b", re.I), "and_effect_on"),
]

# Heuristic tag is a weak signal: do not mark composite on this alone (needs LLM agreement).
HEURISTIC_WEAK_COMPLEXITY_TAGS = frozenset({"effect_on"})


def heuristic_query_complexity(query: str) -> Tuple[str, str]:
    """Return (complexity, reason_tag). complexity is 'composite' or 'simple'."""
    if not (query or "").strip():
        return "simple", ""
    for pat, tag in _COMPOSITE_PATTERNS:
        if pat.search(query):
            return "composite", tag
    return "simple", ""


class Intent(Enum):
    """User query intent types."""
    NATURAL_PRODUCTS_ONLY = auto()  # Only search LOTUS (e.g., "search LOTUS for X")
    PAPERS_ONLY = auto()             # Only search papers (e.g., "find papers about X")
    COMBINED_RESEARCH = auto()       # Full research (e.g., "tell me about X")
    FOLLOW_UP = auto()               # References previous context
    CLARIFICATION = auto()           # User asking for clarification
    ANALYSIS = auto()                # Analyze/synthesize existing data
    UNKNOWN = auto()                 # Unclear intent


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    reasoning: str
    suggested_tools: List[str]
    entities: dict  # Extracted entities (compounds, organisms, etc.)
    query_complexity: str = "simple"  # "simple" | "composite"
    query_complexity_source: str = "default"  # heuristic, llm, heuristic+llm, simple, default


class IntentClassifier:
    """Classifies user intent using LLM."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling markdown code blocks."""
        # Try to find JSON in markdown code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',  # JSON code block
            r'```\s*(.*?)\s*```',      # Generic code block
            r'\{.*\}',                  # Raw JSON object
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1) if '```' in pattern else match.group(0)
        
        # If no patterns match, return the text if it starts with {
        text = text.strip()
        if text.startswith('{') and text.endswith('}'):
            return text
        
        return None
    
    async def classify(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None,
        active_kb_name: Optional[str] = None,
    ) -> IntentResult:
        """
        Classify user intent from query.
        
        Args:
            query: User's current query
            conversation_history: Previous messages for context
            active_kb_name: If set, user selected a curated knowledge base to search first
            
        Returns:
            IntentResult with classification and metadata
        """
        
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-4:]:  # Last 4 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                history_context += f"{role}: {content}\n"

        kb_block = ""
        if active_kb_name:
            kb_block = f"""

ACTIVE KNOWLEDGE BASE: The user has selected curated KB "{active_kb_name}".
- Always include "kb_search" in suggested_tools (preferably first) for any question that can be answered from scientific literature, methods, or prior curated papers.
- You may also suggest literature_search for external discovery when KB might be incomplete.
"""
        
        prompt = f"""You are an intent classifier for a scientific research assistant. Analyze the user query and determine the most appropriate research approach.

Query: "{query}"{history_context}{kb_block}

Available tools:
- lotus_search: Search for natural products, chemical structures, and taxonomy in the LOTUS database
- literature_search: Search for academic papers via SciLEx (multi-API: Semantic Scholar, OpenAlex, PubMed)
- kb_search: Search within previously downloaded papers in the knowledge base
- web_search: General web search for supplementary information

Classify the intent by considering:
1. Does the query mention specific chemicals, compounds, or natural products? → natural_products_only
2. Does the query ask for papers, research, or literature explicitly? → papers_only  
3. Is this a general research question requiring multiple sources? → combined_research
4. Does this reference a previous conversation? → follow_up
5. Is the user asking for clarification? → clarification
6. Does the user want to analyze or synthesize existing data? → analysis

Return ONLY a JSON object (no markdown, no text before or after):
{{
    "intent": "natural_products_only|papers_only|combined_research|follow_up|clarification|analysis|unknown",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why this intent was chosen",
    "query_complexity": "simple|composite",
    "suggested_tools": ["tool1", "tool2"],
    "entities": {{
        "compounds": ["extracted chemical names"],
        "organisms": ["extracted organism names"],
        "topics": ["research topics"]
    }}
}}

query_complexity rules:
- "composite" if the question compares two or more entities, asks for pros/cons, differences, or multiple distinct sub-questions in one utterance.
- "simple" for a single focused topic without explicit comparison or multi-part structure."""

        try:
            response = await self.llm.complete(prompt, temperature=0.1)
            
            # Log the raw response for debugging
            logger.debug(f"Intent classification raw response: {response[:200]}...")
            
            # Extract JSON from response
            json_str = self._extract_json(response)
            
            if not json_str:
                logger.warning(f"Could not extract JSON from response: {response[:200]}")
                # Try simple keyword-based fallback
                return self._keyword_fallback(query, active_kb_name=active_kb_name)
            
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
            
            tools = list(result.get("suggested_tools", []))
            if active_kb_name and "kb_search" not in tools:
                tools.insert(0, "kb_search")
                logger.info(
                    "Intent: prepended kb_search because active_kb_name=%r",
                    active_kb_name,
                )

            llm_qc = result.get("query_complexity", "simple")
            if llm_qc not in ("simple", "composite"):
                llm_qc = "simple"
            h_qc, h_tag = heuristic_query_complexity(query)
            if h_tag in HEURISTIC_WEAK_COMPLEXITY_TAGS and llm_qc != "composite":
                h_qc = "simple"
                if h_tag:
                    logger.info(
                        "Intent: weak heuristic tag %r ignored (LLM query_complexity=%s)",
                        h_tag,
                        llm_qc,
                    )
            final_qc = "composite" if (h_qc == "composite" or llm_qc == "composite") else "simple"
            if final_qc == "composite":
                if h_qc == "composite" and llm_qc == "composite":
                    qc_src = "heuristic+llm"
                elif h_qc == "composite":
                    qc_src = f"heuristic:{h_tag}"
                else:
                    qc_src = "llm"
            else:
                qc_src = "simple"

            return IntentResult(
                intent=intent_map.get(result["intent"], Intent.UNKNOWN),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                suggested_tools=tools,
                entities=result.get("entities", {}),
                query_complexity=final_qc,
                query_complexity_source=qc_src,
            )
            
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            # Fallback to keyword-based classification
            return self._keyword_fallback(query, active_kb_name=active_kb_name)
    
    def _keyword_fallback(self, query: str, active_kb_name: Optional[str] = None) -> IntentResult:
        """Simple keyword-based intent classification as fallback."""
        query_lower = query.lower()
        h_qc, h_tag = heuristic_query_complexity(query)
        if h_tag in HEURISTIC_WEAK_COMPLEXITY_TAGS:
            h_qc = "simple"
            h_tag = ""
        qc = h_qc
        qc_src = f"heuristic:{h_tag}" if h_tag else ("simple" if h_qc == "simple" else "heuristic")
        
        # Check for LOTUS-specific keywords
        lotus_keywords = ["lotus", "natural product", "compound", "metabolite", "structure", "chemical"]
        if any(kw in query_lower for kw in lotus_keywords):
            # Check if it's LOTUS-only
            if any(kw in query_lower for kw in ["lotus only", "only lotus", "search lotus"]):
                return IntentResult(
                    intent=Intent.NATURAL_PRODUCTS_ONLY,
                    confidence=0.8,
                    reasoning="Keyword-based: Explicit LOTUS request",
                    suggested_tools=self._tools_with_kb(["lotus_search"], active_kb_name),
                    entities={},
                    query_complexity=qc,
                    query_complexity_source=qc_src,
                )
            # Otherwise combined
            return IntentResult(
                intent=Intent.COMBINED_RESEARCH,
                confidence=0.7,
                reasoning="Keyword-based: Natural products mentioned, checking both sources",
                suggested_tools=self._tools_with_kb(
                    ["lotus_search", "openalex_search"], active_kb_name
                ),
                entities={},
                query_complexity=qc,
                query_complexity_source=qc_src,
            )
        
        # Check for paper-specific keywords
        paper_keywords = ["paper", "article", "research", "literature", "study", "publication"]
        if any(kw in query_lower for kw in paper_keywords):
            return IntentResult(
                intent=Intent.PAPERS_ONLY,
                confidence=0.8,
                reasoning="Keyword-based: Explicit paper search request",
                suggested_tools=self._tools_with_kb(["openalex_search"], active_kb_name),
                entities={},
                query_complexity=qc,
                query_complexity_source=qc_src,
            )
        
        # Default to combined research
        return IntentResult(
            intent=Intent.COMBINED_RESEARCH,
            confidence=0.6,
            reasoning="Keyword-based: Defaulting to combined research",
            suggested_tools=self._tools_with_kb(
                ["lotus_search", "openalex_search"], active_kb_name
            ),
            entities={},
            query_complexity=qc,
            query_complexity_source=qc_src,
        )

    @staticmethod
    def _tools_with_kb(base: List[str], active_kb_name: Optional[str]) -> List[str]:
        if not active_kb_name or "kb_search" in base:
            return base
        return ["kb_search"] + base
