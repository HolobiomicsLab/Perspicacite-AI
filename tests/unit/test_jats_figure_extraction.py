"""Tests for JATS <fig> extraction and get_claim_figures MCP tool."""
import json

import pytest

JATS_SAMPLE = b"""<?xml version="1.0"?>
<article>
  <body>
    <sec>
      <p>Results described in Figure 1 and Table 1.</p>
      <fig id="fig1" fig-type="figure">
        <label>Figure 1</label>
        <caption><p>Dose-response curve for compound X in HEK293 cells.</p></caption>
      </fig>
      <table-wrap id="tbl1">
        <label>Table 1</label>
        <caption><p>Summary statistics for all experiments.</p></caption>
      </table-wrap>
    </sec>
  </body>
</article>"""


def test_extract_figures_finds_fig():
    from perspicacite.pipeline.download.pmc import _extract_figures_from_xml

    figs = _extract_figures_from_xml(JATS_SAMPLE)
    fig_items = [f for f in figs if f["fig_type"] != "table"]
    assert len(fig_items) == 1
    f = fig_items[0]
    assert f["fig_id"] == "fig1"
    assert f["label"] == "Figure 1"
    assert f["caption"] is not None and "Dose-response" in f["caption"]
    assert f["fig_type"] == "figure"
    assert f["page"] is None


def test_extract_figures_finds_table_wrap():
    from perspicacite.pipeline.download.pmc import _extract_figures_from_xml

    figs = _extract_figures_from_xml(JATS_SAMPLE)
    tables = [f for f in figs if f["fig_type"] == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t["fig_id"] == "tbl1"
    assert t["fig_type"] == "table"
    assert t["caption"] is not None and "statistics" in t["caption"]


def test_extract_figures_empty_on_no_body():
    from perspicacite.pipeline.download.pmc import _extract_figures_from_xml

    result = _extract_figures_from_xml(b"<?xml version='1.0'?><article/>")
    assert result == []


@pytest.mark.asyncio
async def test_get_claim_figures_returns_rows(monkeypatch, tmp_path):
    pytest.importorskip("rdflib", reason="rdflib (indicia extra) not installed")
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    from perspicacite.indicium_layer.queries import (
        IRI_CLAIM,
        IRI_FIGURE,
        IRI_FIGURE_ID,
        IRI_RDF_TYPE,
        IRI_WAS_DERIVED_FROM,
    )
    from perspicacite.indicium_layer.store import ClaimGraphStore
    from perspicacite.mcp import server as mcp_server

    store = ClaimGraphStore("kb", backend="memory")
    claim_iri_str = "kb://kb/claim/test001"
    fig_iri = "kb://kb/figure/abc123"

    store.add(claim_iri_str, IRI_RDF_TYPE, IRI_CLAIM)
    store.add(fig_iri, IRI_RDF_TYPE, IRI_FIGURE)
    store.add(fig_iri, IRI_FIGURE_ID, ("literal", "fig1", None))
    store.add(claim_iri_str, IRI_WAS_DERIVED_FROM, fig_iri)

    monkeypatch.setattr(
        "perspicacite.mcp.server._open_claim_graph_store_for_kb",
        lambda kb: store,
    )

    raw = await mcp_server.get_claim_figures(
        kb_name="kb",
        claim_iri=claim_iri_str,
    )
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["kb_name"] == "kb"
    assert payload["claim_iri"] == claim_iri_str
    assert isinstance(payload["figures"], list)
    assert len(payload["figures"]) >= 1
    row = payload["figures"][0]
    assert row.get("figure_id") == "fig1"
