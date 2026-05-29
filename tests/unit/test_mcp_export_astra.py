import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("indicium", reason="indicia extra not installed")

from perspicacite.mcp import server as mcp_server


@pytest.mark.unit
async def test_export_astra_projects_claims_to_insights():
    state = MagicMock()
    with patch.object(mcp_server, "_require_state", return_value=state):
        raw = await mcp_server.export_astra(claims=[{
            "id": "c1", "context": "in vitro", "subject": "A",
            "qualifier": "inhibits", "relation": "inhibits", "object": "B",
            "evidence": [{"doi": "10.1/x", "quote": "A inhibits B"}]}])
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["insights"][0]["evidence"][0]["doi"] == "10.1/x"
    assert payload["insights"][0]["evidence"][0]["quote"]["exact"] == "A inhibits B"
