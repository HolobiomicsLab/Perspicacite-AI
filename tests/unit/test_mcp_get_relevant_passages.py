"""Tests for the ``get_relevant_passages`` MCP tool, including adaptive mode."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from perspicacite.mcp import server as mcp_server


def _state():
    state = MagicMock()
    state.session_store.get_kb_metadata = AsyncMock(
        return_value=MagicMock(embedding_model="text-embedding-3-small")
    )
    state.vector_store = MagicMock()
    state.embedding_provider = MagicMock(dimension=1536)
    return state


async def test_non_adaptive_returns_passages():
    fake_hits = [
        {"text": "a", "score": 0.5, "paper_id": "x", "metadata": {}, "kb_name": "kb"}
    ]
    with patch.object(mcp_server, "_require_state", return_value=_state()), patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase.search",
        new=AsyncMock(return_value=fake_hits),
    ):
        raw = await mcp_server.get_relevant_passages(
            query="enzyme kinetics", kb_name="kb", k=5
        )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert len(payload["passages"]) == 1
    assert payload["attempts"][0]["query"] == "enzyme kinetics"
    assert payload["attempts"][0]["hit_count"] == 1
    assert payload.get("refined_query") is None


async def test_adaptive_retries_on_empty():
    sequence = [[], [{"text": "found", "score": 0.7, "paper_id": "x", "metadata": {}, "kb_name": "kb"}]]
    search_mock = AsyncMock(side_effect=lambda *a, **kw: sequence.pop(0))

    with patch.object(mcp_server, "_require_state", return_value=_state()), patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase.search", new=search_mock
    ), patch.object(
        mcp_server, "_rephrase_query", AsyncMock(return_value="rephrased q")
    ):
        raw = await mcp_server.get_relevant_passages(
            query="obscure terms", kb_name="kb", k=5, adaptive=True
        )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert len(payload["passages"]) == 1
    assert payload["refined_query"] == "rephrased q"
    assert [a["query"] for a in payload["attempts"]] == [
        "obscure terms",
        "rephrased q",
    ]
    assert [a["hit_count"] for a in payload["attempts"]] == [0, 1]
    assert search_mock.await_count == 2


async def test_adaptive_disabled_does_not_retry():
    with patch.object(mcp_server, "_require_state", return_value=_state()), patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase.search",
        new=AsyncMock(return_value=[]),
    ) as search_mock, patch.object(
        mcp_server, "_rephrase_query", AsyncMock(return_value="never used")
    ) as rephrase_mock:
        raw = await mcp_server.get_relevant_passages(
            query="zero hits", kb_name="kb", k=5, adaptive=False
        )
    payload = json.loads(raw)
    assert payload["passages"] == []
    assert search_mock.await_count == 1
    rephrase_mock.assert_not_awaited()
