"""Unit tests for query_claim_graph + claim_graph_export MCP tools."""

import json

import pytest

pytest.importorskip("rdflib", reason="rdflib (indicia extra) not installed")


async def test_query_claim_graph_dispatches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    from perspicacite.indicium_layer.queries import (
        ASB_NS,
        IRI_CLAIM,
        IRI_OBJECT,
        IRI_RDF_TYPE,
        IRI_SUBJECT,
        IRI_WAS_DERIVED_FROM,
    )
    from perspicacite.indicium_layer.store import ClaimGraphStore
    from perspicacite.mcp import server as mcp_server

    store = ClaimGraphStore("kb", backend="memory")
    iri = "kb://kb/claim/x"
    store.add(iri, IRI_RDF_TYPE, IRI_CLAIM)
    store.add(iri, IRI_SUBJECT, ("literal", "compound X", None))
    store.add(iri, IRI_OBJECT, ("literal", "enzyme Y", None))
    store.add(iri, IRI_WAS_DERIVED_FROM, "doi:10.1/p1")
    store.add(
        iri,
        f"{ASB_NS}evidenceTypeIri",
        "http://purl.obolibrary.org/obo/ECO_0000006",
    )

    monkeypatch.setattr(
        "perspicacite.mcp.server._open_claim_graph_store_for_kb",
        lambda kb: store,
    )

    raw = await mcp_server.query_claim_graph(
        kb_name="kb",
        query_name="claims_supporting",
        kwargs={"subject_or_iri": "compound X"},
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["rows"]
    assert any(r.get("subject") == "compound X" for r in payload["rows"])


async def test_query_claim_graph_rejects_unknown_query():
    from perspicacite.mcp.server import query_claim_graph

    raw = await query_claim_graph(kb_name="kb", query_name="garbage")
    payload = json.loads(raw)
    assert payload["success"] is False
    assert "unknown" in payload["error"].lower()


def _make_store():
    """Return an in-memory ClaimGraphStore seeded with one claim triple."""
    from perspicacite.indicium_layer.queries import (
        ASB_NS,
        IRI_CLAIM,
        IRI_RDF_TYPE,
        IRI_SUBJECT,
    )
    from perspicacite.indicium_layer.store import ClaimGraphStore

    store = ClaimGraphStore("kb", backend="memory")
    iri = "kb://kb/claim/x"
    store.add(iri, IRI_RDF_TYPE, IRI_CLAIM)
    store.add(iri, IRI_SUBJECT, ("literal", "compound X", None))
    store.add(iri, f"{ASB_NS}evidenceTypeIri", "http://purl.obolibrary.org/obo/ECO_0000006")
    return store


async def test_claim_graph_export_turtle(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    from perspicacite.mcp import server as mcp_server

    monkeypatch.setattr(
        "perspicacite.mcp.server._open_claim_graph_store_for_kb",
        lambda kb: _make_store(),
    )
    raw = await mcp_server.claim_graph_export(kb_name="kb", format="turtle")
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["format"] == "turtle"
    # The claim IRI must appear in the serialised Turtle
    assert "kb://kb/claim/x" in payload["data"]


async def test_claim_graph_export_jsonld(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    from perspicacite.mcp import server as mcp_server

    monkeypatch.setattr(
        "perspicacite.mcp.server._open_claim_graph_store_for_kb",
        lambda kb: _make_store(),
    )
    raw = await mcp_server.claim_graph_export(kb_name="kb", format="jsonld")
    payload = json.loads(raw)
    assert payload["success"] is True
    data = payload["data"]
    assert isinstance(data, list)
    # At least one node must carry a type ending in "Claim"
    assert any(any(t.endswith("Claim") for t in node.get("@type", [])) for node in data)
