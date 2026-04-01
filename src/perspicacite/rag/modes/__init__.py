"""RAG mode implementations for benchmark comparison.

Modes:
- BasicRAGMode: Simple retrieval + generation (single query)
- AdvancedRAGMode: Query rephrasing + WRRF scoring + optional refinement
- ProfoundRAGMode: Multi-cycle research with planning (from v1)
- AgenticRAGMode: Intent-based agentic RAG with tool use
- LiteratureSurveyRAGMode: Systematic field mapping with theme identification
"""

from perspicacite.rag.modes.agentic import AgenticRAGMode
from perspicacite.rag.modes.basic import BasicRAGMode
from perspicacite.rag.modes.advanced import AdvancedRAGMode
from perspicacite.rag.modes.profound import ProfoundRAGMode
from perspicacite.rag.modes.literature_survey import LiteratureSurveyRAGMode

__all__ = [
    "BasicRAGMode",
    "AdvancedRAGMode", 
    "ProfoundRAGMode",
    "AgenticRAGMode",
    "LiteratureSurveyRAGMode",
]
