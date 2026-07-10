"""Unit tests for indicium_layer.queries."""

import pytest

from perspicacite.indicium_layer.queries import (
    ASB_NS,
    CITO_NS,
    SPARQL_PREFIXES,
    cito_edges_for_claim,
    cito_graph_iri,
)
from perspicacite.indicium_layer.store import ClaimGraphStore

CITO_SUPPORTS = f"{CITO_NS}supports"
CITO_DISPUTES = f"{CITO_NS}disputes"


def test_namespace_constants():
    # Guards the bug class: a stale copy means SHACL matches nothing and
    # every claim graph "conforms" without a constraint being checked.
    import indicium

    assert ASB_NS == indicium.ASB_BASE
    assert CITO_NS == "http://purl.org/spar/cito/"
    assert "asb:" in SPARQL_PREFIXES
    assert "cito:" in SPARQL_PREFIXES


def test_cito_graph_iri():
    assert cito_graph_iri("my-kb") == "kb://my-kb/graphs/cito"


def test_cito_edges_for_claim_empty():
    store = ClaimGraphStore("kb", backend="memory")
    edges = cito_edges_for_claim(store, "kb", "kb://kb/claim/x")
    assert edges == []
    store.close()


def test_cito_edges_for_claim_returns_supports_and_disputes():
    store = ClaimGraphStore("kb", backend="memory")
    g = cito_graph_iri("kb")
    store.add_edge_with_confidence(
        "kb://kb/claim/a",
        CITO_SUPPORTS,
        "kb://kb/claim/b",
        confidence=0.9,
        run_iri="kb://kb/run/r1",
        graph=g,
    )
    store.add_edge_with_confidence(
        "kb://kb/claim/a",
        CITO_DISPUTES,
        "kb://kb/claim/c",
        confidence=0.55,
        run_iri="kb://kb/run/r1",
        graph=g,
    )
    edges = cito_edges_for_claim(store, "kb", "kb://kb/claim/a")
    edges_sorted = sorted(edges, key=lambda e: e["predicate"])
    assert len(edges_sorted) == 2
    assert edges_sorted[0]["object"] == "kb://kb/claim/c"
    assert edges_sorted[0]["predicate"].endswith("disputes")
    assert float(edges_sorted[0]["confidence"]) == pytest.approx(0.55, abs=0.01)
    store.close()
