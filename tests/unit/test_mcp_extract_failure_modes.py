"""Tests for the ``extract_failure_modes_from_passages`` MCP tool."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from perspicacite.mcp import server as mcp_server


async def test_extract_failure_modes_returns_records():
    state = MagicMock()
    state.llm_client = AsyncMock()
    state.llm_client.complete = AsyncMock(
        return_value=(
            '[{"symptom":"fails on dilute samples",'
            '"root_cause":"detection limit","mitigation":"concentrate first",'
            '"source_doi":"10/a","confidence":0.9}]'
        )
    )
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.extract_failure_modes_from_passages(
            passages=[
                {
                    "text": "Method fails on dilute samples below the LOD.",
                    "source_doi": "10/a",
                    "license_id": "CC-BY",
                }
            ],
            context="LC-MS quantification",
        )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert len(payload["failure_modes"]) == 1
    assert "dilute" in payload["failure_modes"][0]["symptom"]


async def test_extract_failure_modes_empty_returns_empty():
    state = MagicMock()
    state.llm_client = AsyncMock()
    state.llm_client.complete = AsyncMock()
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.extract_failure_modes_from_passages(passages=[])
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["failure_modes"] == []
    state.llm_client.complete.assert_not_awaited()
