import pytest
import rdflib
from perspicacite.pipeline.claims import claims_to_graph, validate_claims

ASB = rdflib.Namespace("https://asb.holobiomics.org/ns/asb#")


@pytest.mark.unit
def test_claims_to_graph_routes_core_qualifier_to_asb_qualifier():
    """A core Bucur qualifier is emitted on the closed asb:qualifier slot."""
    claims = [{"id": "x", "context": "c", "subject": "s",
               "qualifier": "causes", "relation": "r", "object": "o"}]
    g = claims_to_graph(claims)
    assert [str(o) for o in g.objects(None, ASB.qualifier)] == ["causes"]
    assert list(g.objects(None, ASB.domainQualifier)) == []


@pytest.mark.unit
def test_claims_to_graph_routes_domain_qualifier_to_domain_slot():
    """A non-Bucur (domain-adapter) qualifier is routed to the open asb:domainQualifier
    slot, NOT the closed asb:qualifier enum (which would fail indicium SHACL)."""
    claims = [{"id": "x", "context": "c", "subject": "s",
               "qualifier": "aligned_with", "relation": "r", "object": "o"}]
    g = claims_to_graph(claims)
    assert [str(o) for o in g.objects(None, ASB.domainQualifier)] == ["aligned_with"]
    assert list(g.objects(None, ASB.qualifier)) == []


@pytest.mark.unit
def test_domain_qualifier_claim_conforms_end_to_end():
    """A domain-qualifier claim round-trips through indicium SHACL (Reading 1).

    Requires the `indicia` extra; skipped otherwise. Proves the routed
    asb:domainQualifier value is accepted by indicium's open snake_case slot."""
    pytest.importorskip("indicium")
    claims = [{"context": "c", "subject": "s", "qualifier": "aligned_with",
               "relation": "r", "object": "o"}]
    conforms, report = validate_claims(claims)
    assert conforms is True, report


@pytest.mark.unit
def test_valid_claim_conforms():
    pytest.importorskip("indicium")
    claims = [{"context": "c", "subject": "s", "qualifier": "causes",
               "relation": "r", "object": "o"}]
    conforms, _ = validate_claims(claims)
    assert conforms is True


@pytest.mark.unit
def test_claim_missing_slots_is_rejected():
    pytest.importorskip("indicium")
    claims = [{"context": "c"}]  # missing 4 of 5 required slots
    conforms, _ = validate_claims(claims)
    assert conforms is False
