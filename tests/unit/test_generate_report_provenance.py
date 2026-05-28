"""generate_report must attach a non-breaking ``_provenance`` envelope (issue #1).

Researchers see ``sources``/``papers_used`` next to a ``report`` field and
assume the whole report is citation-grounded. In reality the ``report`` text is
LLM synthesis. ``_provenance`` makes that explicit to the host agent: which
fields are model-authored, which provider/model produced them, how many RAG
cycles ran, and which retrieved sources are attacker-influenceable.
"""

from __future__ import annotations

import inspect
import json as _json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _real_config() -> MagicMock:
    cfg = MagicMock()
    cfg.llm.default_provider = "anthropic"
    cfg.llm.default_model = "claude-3-5-sonnet"
    return cfg


def _kb_meta(name: str, model: str = "text-embedding-3-small") -> SimpleNamespace:
    return SimpleNamespace(name=name, embedding_model=model, collection_name=f"coll_{name}")


def test_generate_report_body_builds_provenance() -> None:
    """White-box: the generate_report body must wire a _provenance envelope."""
    from perspicacite.mcp import server as mcp_mod

    src = inspect.getsource(mcp_mod)
    gr_idx = src.index("async def generate_report(")
    next_tool = src.index("@mcp.tool()", gr_idx + 1)
    gr_body = src[gr_idx:next_tool]
    assert "_provenance" in gr_body, "generate_report return missing _provenance key"


@pytest.mark.asyncio
async def test_generate_report_returns_provenance_envelope() -> None:
    import perspicacite.mcp.server as mcp_server
    import perspicacite.rag.engine as _engine_mod
    from perspicacite.mcp.server import MCPState, generate_report
    from perspicacite.models.rag import StreamEvent
    from perspicacite.rag.engine import RAGEngine

    state = MCPState()
    state.initialized = True
    state.config = _real_config()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None

    meta_solo = _kb_meta("solo")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(return_value=meta_solo)
    state.session_store = ss_mock

    src_a = {
        "title": "Paper A",
        "authors": ["A. Author"],
        "year": 2024,
        "doi": "10.1/aaa",
        "relevance_score": 0.9,
        "section": None,
        "kb_name": "solo",
        "metadata": None,
    }
    src_b = {
        "title": "Paper B (no doi)",
        "authors": ["B. Author"],
        "year": 2023,
        "doi": None,
        "relevance_score": 0.8,
        "section": None,
        "kb_name": "solo",
        "metadata": None,
    }

    class _CapturingRAGEngine(RAGEngine):
        async def query_stream(
            self, req, *, message_id=None, conversation_id=None
        ) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(event="content", data=_json.dumps({"delta": "synthesized report"}))
            yield StreamEvent(event="source", data=_json.dumps(src_a))
            yield StreamEvent(event="source", data=_json.dumps(src_b))
            yield StreamEvent(event="done", data="{}")

    original_cls = _engine_mod.RAGEngine
    _engine_mod.RAGEngine = _CapturingRAGEngine  # type: ignore[assignment]
    try:
        with patch.object(mcp_server, "mcp_state", state):
            result_str = await generate_report(query="q", kb_name="solo", mode="advanced")
    finally:
        _engine_mod.RAGEngine = original_cls

    result = _json.loads(result_str)
    assert result.get("success") is True, f"tool returned error: {result}"

    prov = result.get("_provenance")
    assert isinstance(prov, dict), f"_provenance missing: {result.keys()}"
    assert prov["provider"] == "anthropic"
    assert prov["model"] == "claude-3-5-sonnet"
    assert prov["ai_generated_fields"] == ["report"]
    assert prov["rag_cycles_executed"] == result["iteration_count"]
    # All retrieved sources are attacker-influenceable; DOI preferred, title fallback.
    assert prov["untrusted_sources"] == ["10.1/aaa", "Paper B (no doi)"]
