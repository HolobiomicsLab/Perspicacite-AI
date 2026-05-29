"""Tests for the ``extract_parameters_from_passages`` MCP tool."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from perspicacite.mcp import server as mcp_server


async def test_extract_parameters_returns_records():
    state = MagicMock()
    state.llm_client = AsyncMock()
    state.llm_client.complete = AsyncMock(
        return_value='[{"name":"temperature","typical":"37","units":"C","source_doi":"10/a"}]'
    )
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.extract_parameters_from_passages(
            passages=[
                {
                    "text": "Cells grown at 37 C",
                    "source_doi": "10/a",
                    "license_id": "CC-BY",
                }
            ],
            context="cell-culture",
        )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert len(payload["parameters"]) == 1
    assert payload["parameters"][0]["name"] == "temperature"
    assert payload["parameters"][0]["units"] == "C"


async def test_extract_parameters_empty_passages_returns_empty():
    state = MagicMock()
    state.llm_client = AsyncMock()
    state.llm_client.complete = AsyncMock()
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.extract_parameters_from_passages(passages=[])
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["parameters"] == []
    state.llm_client.complete.assert_not_awaited()


async def test_extract_parameters_license_tier_c_drops_quote():
    state = MagicMock()
    state.llm_client = AsyncMock()
    state.llm_client.complete = AsyncMock(
        return_value=(
            '[{"name":"pH","typical":"7.4","units":"","source_doi":"10/x",'
            '"source_quote":"verbatim text from closed paper"}]'
        )
    )
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.extract_parameters_from_passages(
            passages=[
                {
                    "text": "pH 7.4 buffer used",
                    "source_doi": "10/x",
                    "license_id": "all rights reserved",
                }
            ],
        )
    payload = json.loads(raw)
    assert payload["success"] is True
    p = payload["parameters"][0]
    # Quote either omitted or paraphrased; never verbatim closed-source text.
    assert p.get("source_quote") != "verbatim text from closed paper"
