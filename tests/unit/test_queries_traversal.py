"""Unit tests for the five typed traversal queries."""

import pytest

pytest.importorskip("rdflib", reason="rdflib (indicia extra) not installed")

from perspicacite.indicium_layer.queries import (
    ASB_NS,
    CITO_NS,
    IRI_CLAIM,
    IRI_OBJECT,
    IRI_RDF_TYPE,
    IRI_SUBJECT,
    IRI_WAS_DERIVED_FROM,
    cito_graph_iri,
    claims_disputing,
    claims_supporting,
    evidence_trace,
    neighbors,
    papers_with_claim_pattern,
)
from perspicacite.indicium_layer.store import ClaimGraphStore

CITO_SUPPORTS = f"{CITO_NS}supports"
CITO_DISPUTES = f"{CITO_NS}disputes"
CITO_QUALIFIES = f"{CITO_NS}qualifies"
ECO_DATA = "http://purl.obolibrary.org/obo/ECO_0000006"
ECO_CITATION = "http://purl.obolibrary.org/obo/ECO_0000033"


def _seed(store: ClaimGraphStore, kb: str):
    g = cito_graph_iri(kb)
    for cid, subj, obj, eco, paper in [
        ("a", "compound X", "enzyme Y", ECO_DATA, "doi:10.1/p1"),
        ("b", "compound X", "enzyme Z", ECO_CITATION, "doi:10.1/p2"),
        ("c", "compound W", "enzyme V", ECO_DATA, "doi:10.1/p3"),
        ("d", "compound X", "enzyme Y", ECO_DATA, "doi:10.1/p4"),
    ]:
        iri = f"kb://{kb}/claim/{cid}"
        store.add(iri, IRI_RDF_TYPE, IRI_CLAIM)
        store.add(iri, IRI_SUBJECT, ("literal", subj, None))
        store.add(iri, IRI_OBJECT, ("literal", obj, None))
        store.add(iri, f"{ASB_NS}evidenceTypeIri", eco)
        store.add(iri, IRI_WAS_DERIVED_FROM, paper)
    store.add_edge_with_confidence(
        f"kb://{kb}/claim/a",
        CITO_SUPPORTS,
        f"kb://{kb}/claim/d",
        confidence=0.9,
        run_iri=f"kb://{kb}/run/r1",
        graph=g,
    )
    store.add_edge_with_confidence(
        f"kb://{kb}/claim/d",
        CITO_SUPPORTS,
        f"kb://{kb}/claim/b",
        confidence=0.8,
        run_iri=f"kb://{kb}/run/r1",
        graph=g,
    )
    store.add_edge_with_confidence(
        f"kb://{kb}/claim/c",
        CITO_DISPUTES,
        f"kb://{kb}/claim/a",
        confidence=0.85,
        run_iri=f"kb://{kb}/run/r1",
        graph=g,
    )


def test_claims_supporting_by_subject():
    store = ClaimGraphStore("kb", backend="memory")
    _seed(store, "kb")
    rows = claims_supporting(store, "kb", "compound X")
    subjects = {r["subject"] for r in rows}
    assert "compound X" in subjects
    assert all("compound W" not in r["subject"] for r in rows)
    store.close()


def test_claims_supporting_with_min_eco_grade():
    store = ClaimGraphStore("kb", backend="memory")
    _seed(store, "kb")
    rows = claims_supporting(store, "kb", "compound X", min_eco_grade="data")
    # Should only include data-grade claims
    assert all(r.get("eco") == ECO_DATA for r in rows)
    store.close()


def test_claims_disputing():
    store = ClaimGraphStore("kb", backend="memory")
    _seed(store, "kb")
    rows = claims_disputing(store, "kb", "kb://kb/claim/a")
    assert any(r["from"] == "kb://kb/claim/c" for r in rows)
    store.close()


def test_evidence_trace_bfs():
    store = ClaimGraphStore("kb", backend="memory")
    _seed(store, "kb")
    path = evidence_trace(store, "kb", "kb://kb/claim/a", max_depth=3)
    iris = {step["claim"] for step in path}
    assert "kb://kb/claim/a" in iris
    assert "kb://kb/claim/d" in iris
    assert "kb://kb/claim/b" in iris
    # disputes is NOT followed by trace (only supports/qualifies)
    assert "kb://kb/claim/c" not in iris
    store.close()


def test_papers_with_claim_pattern():
    store = ClaimGraphStore("kb", backend="memory")
    _seed(store, "kb")
    rows = papers_with_claim_pattern(
        store,
        "kb",
        subject="compound X",
        object="enzyme Y",
    )
    papers = {r["paper"] for r in rows}
    assert "doi:10.1/p1" in papers
    assert "doi:10.1/p4" in papers
    assert "doi:10.1/p2" not in papers
    store.close()


def test_neighbors_with_edge_type_filter():
    store = ClaimGraphStore("kb", backend="memory")
    _seed(store, "kb")
    sup = neighbors(store, "kb", "kb://kb/claim/a", edge_types=["supports"])
    sup_iris = {n["neighbor"] for n in sup}
    assert "kb://kb/claim/d" in sup_iris
    dis = neighbors(store, "kb", "kb://kb/claim/a", edge_types=["disputes"])
    # Note: claim/a is the disputed target, not the disputing source
    dis_iris = {n["neighbor"] for n in dis}
    assert "kb://kb/claim/c" in dis_iris  # incoming dispute counts as a neighbor
    store.close()
