"""Unit tests for ReasoningRAGMode dispatch + extra-check + default strategy."""

import json

from perspicacite.config.schema import Config
from perspicacite.models.rag import RAGMode, RAGRequest


async def _drain(stream):
    return [ev async for ev in stream]


async def test_default_strategy_after_subplan_b_is_evidence_graded():
    from perspicacite.rag.modes.reasoning import (
        DEFAULT_REASONING_STRATEGY,
        SHIPPED_STRATEGIES,
    )

    assert DEFAULT_REASONING_STRATEGY == "evidence_graded"
    assert (
        frozenset({"provenance", "contradiction", "graph", "evidence_graded"}) == SHIPPED_STRATEGIES
    )


async def test_missing_indicium_yields_error(monkeypatch):
    from perspicacite.rag.modes import reasoning as reasoning_mod

    monkeypatch.setattr(reasoning_mod, "_HAS_INDICIA", False)
    mode = reasoning_mod.ReasoningRAGMode(Config())
    req = RAGRequest(query="q", mode=RAGMode.REASONING)
    events = await _drain(
        mode.execute_stream(req, llm=None, vector_store=None, embedding_provider=None, tools=None)
    )
    err = next(e for e in events if e.event == "error")
    payload = json.loads(err.data)
    assert "indicium" in payload["message"]


async def test_reasoning_mode_registered_in_engine():
    """RAGEngine._modes must contain RAGMode.REASONING after construction."""
    from unittest.mock import MagicMock

    from perspicacite.config.schema import Config
    from perspicacite.models.rag import RAGMode
    from perspicacite.rag.engine import RAGEngine

    engine = RAGEngine(
        llm_client=MagicMock(),
        vector_store=MagicMock(),
        embedding_provider=MagicMock(),
        tool_registry=MagicMock(),
        config=Config(),
    )
    assert RAGMode.REASONING in engine._modes


async def test_reasoning_in_chat_route_map():
    """RAG_MODE_MAP in the chat router must map 'reasoning' to RAGMode.REASONING."""
    from perspicacite.models.rag import RAGMode
    from perspicacite.web.routers.chat import RAG_MODE_MAP

    assert "reasoning" in RAG_MODE_MAP
    assert RAG_MODE_MAP["reasoning"] == RAGMode.REASONING
