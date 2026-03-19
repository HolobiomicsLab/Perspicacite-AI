"""
True Agentic RAG with LLM-driven orchestration.

This module implements a ReAct-style agent that:
1. Analyzes user intent
2. Plans steps dynamically
3. Executes tools iteratively
4. Reflects on results
5. Maintains conversation context
"""

from .orchestrator import AgenticOrchestrator, AgentSession
from .planner import ResearchPlanner, Step, Plan, StepType
from .intent import IntentClassifier, Intent, IntentResult
from .llm_adapter import LLMAdapter

__all__ = [
    "AgenticOrchestrator",
    "AgentSession", 
    "ResearchPlanner",
    "Step",
    "StepType",
    "Plan",
    "IntentClassifier",
    "Intent",
    "IntentResult",
    "LLMAdapter",
]
