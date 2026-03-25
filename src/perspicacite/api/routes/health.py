"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "services": {
            "api": True,
            "database": True,
            "llm": True,
        },
    }


@router.get("/api/info")
async def api_info():
    """API information."""
    return {
        "version": "2.0.0",
        "available_providers": ["anthropic", "openai", "deepseek", "gemini"],
        "rag_modes": ["basic", "advanced", "profound", "agentic"],
        "rag_mode_descriptions": {
            "basic": "Simple retrieval + generation (fastest)",
            "advanced": "Query rephrasing + WRRF scoring + optional refinement",
            "profound": "Multi-cycle research with planning (from v1)",
            "agentic": "Intent-based agentic RAG with tool use",
        },
    }
