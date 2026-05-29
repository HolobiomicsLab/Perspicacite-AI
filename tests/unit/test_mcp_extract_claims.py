"""Tests for the ``extract_claims_from_passages`` MCP tool."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.mcp import server as mcp_server


@pytest.mark.unit
async def test_extract_claims_tool_returns_typed_claims():
    state = MagicMock()
    state.llm_client = MagicMock()
    state.llm_client.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "A", "qualifier": "inhibits",
        "relation": "inhibits", "object": "B", "evidence_type": "data",
        "quote": "A inhibits B", "source_doi": "10.1/x"}]}))
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.extract_claims_from_passages(
            passages=[{"chunk_text": "A inhibits B", "source": {"doi": "10.1/x"}}],
            context="onc")
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["claims"][0]["qualifier"] == "inhibits"
    assert payload["claims_valid"] is True
