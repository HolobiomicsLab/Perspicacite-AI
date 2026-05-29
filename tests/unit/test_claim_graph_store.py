"""Unit tests for indicium_layer.store.ClaimGraphStore (rdflib backend)."""

import pytest

pytest.importorskip("rdflib", reason="rdflib (indicia extra) not installed")

from perspicacite.indicium_layer.store import ClaimGraphStore

PROV_WAS_GEN_BY = "http://www.w3.org/ns/prov#wasGeneratedBy"
ASB_CONFIDENCE = "https://asb.holobiomics.org/ns/asb#confidence"
RDF_SUBJECT = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
CITO_SUPPORTS = "http://purl.org/spar/cito/supports"


def test_add_and_select_default_graph():
    store = ClaimGraphStore("kbA", backend="memory")
    store.add("kb://kbA/claim/x", "http://example/p", "kb://kbA/claim/y")
    rows = store.select("SELECT ?o WHERE { <kb://kbA/claim/x> <http://example/p> ?o }")
    assert any(r["o"] == "kb://kbA/claim/y" for r in rows)
    store.close()


def test_add_idempotent_dedup():
    store = ClaimGraphStore("kbA", backend="memory")
    for _ in range(3):
        store.add("kb://kbA/claim/x", "http://example/p", "kb://kbA/claim/y")
    rows = store.select(
        "SELECT (COUNT(*) AS ?c) WHERE { <kb://kbA/claim/x> <http://example/p> ?o }"
    )
    assert int(rows[0]["c"]) == 1
    store.close()


def test_literal_with_datatype():
    store = ClaimGraphStore("kbA", backend="memory")
    xsd_decimal = "http://www.w3.org/2001/XMLSchema#decimal"
    store.add(
        "kb://kbA/x",
        "https://asb.holobiomics.org/ns/asb#confidence",
        ("literal", "0.82", xsd_decimal),
    )
    rows = store.select(
        "SELECT ?v WHERE { <kb://kbA/x> <https://asb.holobiomics.org/ns/asb#confidence> ?v }"
    )
    assert rows[0]["v"] == "0.82"
    store.close()


def test_named_graph_isolation():
    store = ClaimGraphStore("kbA", backend="memory")
    store.add("kb://kbA/c/x", "http://p", "kb://kbA/c/y", graph="kb://kbA/graphs/cito")
    store.add("kb://kbA/c/x", "http://p", "kb://kbA/c/z")  # default graph
    # Restrict to cito graph
    rows = store.select(
        "SELECT ?o FROM <kb://kbA/graphs/cito> WHERE { <kb://kbA/c/x> <http://p> ?o }"
    )
    objs = sorted(r["o"] for r in rows)
    assert objs == ["kb://kbA/c/y"]
    store.close()


def test_contains_iri():
    store = ClaimGraphStore("kbA", backend="memory")
    store.add("kb://kbA/claim/abc", "http://p", "http://o")
    assert store.contains_iri("kb://kbA/claim/abc") is True
    assert store.contains_iri("kb://kbA/claim/missing") is False
    store.close()


def test_add_edge_with_confidence_reification():
    store = ClaimGraphStore("kbA", backend="memory")
    store.add_edge_with_confidence(
        "kb://kbA/claim/a",
        CITO_SUPPORTS,
        "kb://kbA/claim/b",
        confidence=0.82,
        run_iri="kb://kbA/run/2026-05-23/abc",
        graph="kb://kbA/graphs/cito",
    )
    rows = store.select(
        f"""
        SELECT ?conf ?run FROM <kb://kbA/graphs/cito> WHERE {{
            ?meta <{RDF_SUBJECT}> <kb://kbA/claim/a> ;
                  <{ASB_CONFIDENCE}> ?conf ;
                  <{PROV_WAS_GEN_BY}> ?run .
        }}
        """
    )
    assert rows and rows[0]["conf"] == "0.8200"
    assert rows[0]["run"] == "kb://kbA/run/2026-05-23/abc"
    # And the asserted edge itself
    edge_rows = store.select(
        f"SELECT ?o FROM <kb://kbA/graphs/cito> WHERE {{ <kb://kbA/claim/a> <{CITO_SUPPORTS}> ?o }}"
    )
    assert any(r["o"] == "kb://kbA/claim/b" for r in edge_rows)
    store.close()
