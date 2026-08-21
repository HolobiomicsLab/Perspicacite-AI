"""Tests for Perspicacite indicium_layer v1.3 ClaimLink queries and MCP tool.

Run with:
    cd ~/git/Perspicacite-AI/.worktrees/reasoning
    uv run pytest tests/unit/test_claim_link_query.py -v
"""
from __future__ import annotations
import json
import pytest

# perspicacite.indicium_layer.queries imports the private, maintainer-only
# indicium package at module scope, so these tests cannot run on a public
# checkout.
pytest.importorskip("indicium", reason="indicium is a private, maintainer-only package")


def test_claim_link_iri_constants_exist():
    from perspicacite.indicium_layer.queries import (
        IRI_CLAIM_LINK, IRI_FROM_CLAIM, IRI_TO_CLAIM,
        IRI_LINK_TYPE, IRI_CLAIM_STATUS, IRI_ASSERTED_BY, IRI_DECISION_CONTEXT,
        INDICIUM_NS,
    )
    assert IRI_CLAIM_LINK == "https://w3id.org/indicium/ClaimLink", (
        f"IRI_CLAIM_LINK must be indicium:ClaimLink (https://w3id.org/indicium/ClaimLink), "
        f"got {IRI_CLAIM_LINK!r}. The ClaimLink class_uri was fixed in indicium v1.4."
    )
    assert IRI_FROM_CLAIM       == f"{INDICIUM_NS}from_claim"
    assert IRI_TO_CLAIM         == f"{INDICIUM_NS}to_claim"
    assert IRI_LINK_TYPE        == f"{INDICIUM_NS}link_type"
    assert IRI_CLAIM_STATUS     == f"{INDICIUM_NS}claim_status"
    assert IRI_ASSERTED_BY      == "http://www.w3.org/ns/prov#wasAttributedTo"
    assert IRI_DECISION_CONTEXT == f"{INDICIUM_NS}decision_context"


def test_claim_links_for_claim_sparql_returns_list():
    """claim_links_for_claim returns a list (possibly empty) for any store."""
    from perspicacite.indicium_layer.queries import claim_links_for_claim

    # Use a minimal mock store
    class MockStore:
        def select(self, sparql):
            return []

    rows = claim_links_for_claim(MockStore(), kb_name="test_kb", claim_iri="urn:test:claim:001")
    assert isinstance(rows, list)


@pytest.mark.asyncio
async def test_get_claim_links_mcp_tool_response_structure():
    """get_claim_links MCP tool returns valid JSON with success envelope."""
    from unittest.mock import patch, MagicMock

    # Import server module first so patch targets resolve
    from perspicacite.mcp import server as _server  # noqa: F401

    mock_store = MagicMock()
    mock_store.close = MagicMock()

    mock_rows = [
        {"link_iri": "urn:link:1", "from_claim": "urn:a", "to_claim": "urn:b",
         "link_type": "supports", "direction": "outgoing"}
    ]

    with patch("perspicacite.mcp.server._open_claim_graph_store_for_kb",
               return_value=mock_store), \
         patch("perspicacite.indicium_layer.queries.claim_links_for_claim",
               return_value=mock_rows):

        # get_claim_links is wrapped by fastmcp; access the underlying callable
        tool_fn = getattr(_server.get_claim_links, "fn",
                          getattr(_server.get_claim_links, "func",
                                  _server.get_claim_links))
        result_str = await tool_fn(kb_name="test_kb", claim_iri="urn:a")
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["kb_name"] == "test_kb"
        assert result["claim_iri"] == "urn:a"
        assert "links" in result
