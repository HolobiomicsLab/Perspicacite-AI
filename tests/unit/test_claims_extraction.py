import json
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from perspicacite.pipeline.claims import extract_claims


@pytest.mark.unit
async def test_extract_claims_builds_typed_claims():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "compound A", "qualifier": "inhibits",
        "relation": "inhibits growth of", "object": "cell line B",
        "claim_type": "explicit", "evidence_type": "data",
        "source_type": "text", "quote": "A inhibited B",
        "source_doi": "10.1/x"}]}))
    passages = [{"chunk_text": "A inhibited B", "source": {"doi": "10.1/x", "title": "T"}}]
    claims = await extract_claims(llm_client=llm, passages=passages, context="onc")
    assert len(claims) == 1
    c = claims[0]
    assert "id" in c, "every coerced claim must carry an id for Indicium adapter compatibility"
    assert c["id"].startswith("perspicacite:")
    assert c["qualifier"] == "inhibits"
    assert c["evidence"][0]["doi"] == "10.1/x"
    assert c["evidence"][0]["evidence_type"] == "data"


@pytest.mark.unit
async def test_extract_claims_drops_out_of_vocab_qualifier():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "x", "subject": "s", "qualifier": "frobnicates",
        "relation": "r", "object": "o", "evidence_type": "data"}]}))
    claims = await extract_claims(llm_client=llm, passages=[{"chunk_text": "t"}], context="c")
    assert claims == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "wrapper",
    [
        "```json\n{body}\n```",      # fenced with json tag
        "```\n{body}\n```",          # fenced without language tag
        "  ```json\n{body}\n```  ",  # surrounding whitespace
    ],
    ids=["json-tagged-fence", "untagged-fence", "padded-fence"],
)
async def test_extract_claims_strips_markdown_json_fence(wrapper):
    """Regression: gpt-4o-mini / Haiku / Sonnet sometimes wrap their JSON
    in a ```json ... ``` fence even when the prompt asks for raw JSON.
    Before the fix, ``json.loads`` raised silently and ``extract_claims``
    returned ``[]`` — the whole pipeline produced zero claims with no
    diagnostic. The fence must be stripped before parsing."""
    body = json.dumps({"claims": [{
        "context": "in vitro", "subject": "compound A", "qualifier": "inhibits",
        "relation": "inhibits growth of", "object": "cell line B",
        "claim_type": "explicit", "evidence_type": "data",
        "source_type": "text", "quote": "A inhibited B",
        "source_doi": "10.1/x"}]})
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=wrapper.format(body=body))
    passages = [{"chunk_text": "A inhibited B", "source": {"doi": "10.1/x"}}]
    claims = await extract_claims(llm_client=llm, passages=passages, context="c")
    assert len(claims) == 1
    assert claims[0]["qualifier"] == "inhibits"
    assert claims[0]["subject"] == "compound A"


@pytest.mark.unit
async def test_extract_claims_handles_raw_json_unchanged():
    """Raw (un-fenced) JSON must continue to parse — the fence stripper
    must be a no-op when no fence is present."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "x", "subject": "s", "qualifier": "inhibits",
        "relation": "r", "object": "o", "evidence_type": "data",
        "source_doi": "10.1/x"}]}))
    claims = await extract_claims(
        llm_client=llm, passages=[{"chunk_text": "t", "source": {"doi": "10.1/x"}}],
        context="c",
    )
    assert len(claims) == 1


# ---------------------------------------------------------------------------
# Domain adapter integration
# ---------------------------------------------------------------------------

class _MockAdapter:
    """Minimal adapter that satisfies the DomainAdapter structural protocol."""
    domain_id = "test"
    api_version = 1
    qualifiers = frozenset({"test_qualifier"})
    ontology_prefixes: ClassVar[dict[str, str]] = {"TST": "http://test.example.org/"}

    def extraction_context(self) -> str:
        return "Domain: test. Extra qualifier: test_qualifier."

    def enrich_claim(self, claim: dict) -> dict:
        claim = dict(claim)
        claim["enriched_by_adapter"] = True
        return claim


class _MockAdapterB:
    """Second minimal adapter for composition tests."""
    domain_id = "test_b"
    api_version = 1
    qualifiers = frozenset({"qualifier_b"})
    ontology_prefixes: ClassVar[dict[str, str]] = {"TST_B": "http://testb.example.org/"}

    def extraction_context(self) -> str:
        return "Domain: test_b. Extra qualifier: qualifier_b."

    def enrich_claim(self, claim: dict) -> dict:
        claim = dict(claim)
        claim["enriched_by_b"] = True
        return claim


@pytest.mark.unit
async def test_extract_claims_with_adapter_appends_domain_context():
    """domain_adapter.extraction_context() must appear in the LLM prompt."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='{"claims": []}')
    adapter = _MockAdapter()

    await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "Some text.", "source": {"doi": "10.1/x"}}],
        context="test",
        domain_adapter=adapter,
    )

    prompt_sent = llm.complete.call_args[1]["messages"][0]["content"]
    assert "Domain: test." in prompt_sent, (
        f"Expected domain context in prompt, got:\n{prompt_sent}"
    )


@pytest.mark.unit
async def test_extract_claims_with_adapter_enriches_claims():
    """Each coerced claim must be passed through domain_adapter.enrich_claim()."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "compound A",
        "qualifier": "inhibits", "relation": "inhibits growth of",
        "object": "cell line B", "claim_type": "explicit",
        "evidence_type": "data", "source_type": "text",
        "quote": "A inhibited B", "source_doi": "10.1/x",
    }]}))
    adapter = _MockAdapter()

    claims = await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "A inhibited B", "source": {"doi": "10.1/x"}}],
        context="test",
        domain_adapter=adapter,
    )

    assert len(claims) == 1
    assert claims[0].get("enriched_by_adapter") is True


@pytest.mark.unit
async def test_extract_claims_with_adapter_accepts_domain_qualifier():
    """Qualifiers from domain_adapter.qualifiers must not be dropped by _coerce_claim."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "compound A",
        "qualifier": "test_qualifier",   # domain-specific; unknown to base _QUALIFIERS
        "relation": "something", "object": "B",
    }]}))
    adapter = _MockAdapter()

    claims = await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "text"}],
        context="test",
        domain_adapter=adapter,
    )

    assert len(claims) == 1
    assert claims[0]["qualifier"] == "test_qualifier"


@pytest.mark.unit
async def test_extract_claims_without_adapter_unchanged():
    """Passing no adapter must preserve existing behaviour exactly."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "compound A", "qualifier": "inhibits",
        "relation": "inhibits", "object": "B",
        "evidence_type": "data", "source_doi": "10.1/x",
    }]}))

    claims = await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "text", "source": {"doi": "10.1/x"}}],
        context="test",
        # no domain_adapter
    )

    assert len(claims) == 1
    assert "enriched_by_adapter" not in claims[0]


@pytest.mark.unit
async def test_extract_claims_domain_qualifier_dropped_without_adapter():
    """Without an adapter, domain-only qualifiers are still dropped by _coerce_claim."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "A", "qualifier": "test_qualifier",
        "relation": "r", "object": "B",
    }]}))

    claims = await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "text"}],
        context="test",
        # no adapter — test_qualifier is out-of-vocab
    )

    assert claims == []


# ---------------------------------------------------------------------------
# validate_claims() — domain adapter awareness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_validate_claims_no_adapter_backward_compat():
    """validate_claims() with no domain_adapter must behave identically to before."""
    from unittest.mock import patch

    from perspicacite.pipeline.claims import validate_claims

    claims = [{
        "context": "c", "subject": "s", "qualifier": "inhibits",
        "relation": "r", "object": "o", "id": "perspicacite:abc123",
    }]

    with patch("indicium.validate_graph", return_value=(True, "")) as mock_vg:
        conforms, _ = validate_claims(claims)

    assert conforms is True
    _, kwargs = mock_vg.call_args
    assert kwargs.get("extra_shapes") is None


@pytest.mark.unit
def test_validate_claims_calls_shacl_shapes_from_adapter():
    """validate_claims() with an adapter that has shacl_shapes() must call it once."""
    from unittest.mock import MagicMock, patch

    import rdflib

    from perspicacite.pipeline.claims import validate_claims

    extra = rdflib.Graph()
    adapter = MagicMock()
    adapter.shacl_shapes.return_value = extra

    claims = [{
        "context": "c", "subject": "s", "qualifier": "inhibits",
        "relation": "r", "object": "o", "id": "perspicacite:abc123",
    }]

    with patch("indicium.validate_graph", return_value=(True, "")) as mock_vg:
        validate_claims(claims, domain_adapter=adapter)

    adapter.shacl_shapes.assert_called_once()
    _, kwargs = mock_vg.call_args
    assert kwargs.get("extra_shapes") is extra


@pytest.mark.unit
def test_validate_claims_adapter_without_shacl_shapes_no_error():
    """validate_claims() with an adapter that lacks shacl_shapes() must not raise."""
    from typing import ClassVar
    from unittest.mock import patch

    from perspicacite.pipeline.claims import validate_claims

    class _NoSHACLAdapter:
        domain_id = "test"
        api_version = 1
        qualifiers = frozenset()
        ontology_prefixes: ClassVar[dict] = {}
        def extraction_context(self): return ""
        def enrich_claim(self, c): return c

    claims = [{
        "context": "c", "subject": "s", "qualifier": "inhibits",
        "relation": "r", "object": "o", "id": "perspicacite:abc123",
    }]

    with patch("indicium.validate_graph", return_value=(True, "")) as mock_vg:
        conforms, _ = validate_claims(claims, domain_adapter=_NoSHACLAdapter())

    assert conforms is True
    _, kwargs = mock_vg.call_args
    assert kwargs.get("extra_shapes") is None


@pytest.mark.unit
def test_validate_claims_enrichment_and_shacl_combined():
    """validate_claims() with an adapter that has both enrich_claim and shacl_shapes
    must call shacl_shapes() once (wiring integration check)."""
    import rdflib
    from unittest.mock import MagicMock, patch
    from perspicacite.pipeline.claims import validate_claims

    mock_graph = rdflib.Graph()
    adapter = MagicMock()
    adapter.shacl_shapes.return_value = mock_graph

    claims = [{"context": "c", "subject": "s", "qualifier": "causes",
               "relation": "r", "object": "o"}]

    with patch("indicium.validate_graph", return_value=(True, "")) as mock_vg:
        conforms, _ = validate_claims(claims, domain_adapter=adapter)

    adapter.shacl_shapes.assert_called_once()
    mock_vg.assert_called_once()
    # extra_shapes kwarg was the mock_graph
    _, kwargs = mock_vg.call_args
    assert kwargs.get("extra_shapes") is mock_graph
    assert conforms is True


@pytest.mark.unit
async def test_extract_claims_mcp_tool_passes_adapter_not_manual_enrich():
    """Regression: extract_claims_from_passages MCP tool must pass domain_adapter
    directly to extract_claims(), NOT call adapter.enrich_claim() manually after.

    Before the fix, the server called extract_claims() without domain_adapter and
    then ran [adapter.enrich_claim(c) for c in claims] manually — bypassing:
      - LLM prompt enrichment via adapter.extraction_context()
      - domain qualifier acceptance via adapter.qualifiers

    This test patches extract_claims() at the pipeline level to assert it receives
    domain_adapter, confirming the MCP tool wiring is correct.
    """
    import json
    from unittest.mock import AsyncMock, MagicMock, patch
    from perspicacite.pipeline.claims import extract_claims

    adapter = _MockAdapter()

    # extract_claims() with the adapter should call enrich_claim on each claim
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "glucose",
        "qualifier": "test_qualifier",   # domain-only qualifier — only accepted when adapter passed
        "relation": "measured in", "object": "plasma",
    }]}))

    claims = await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "glucose measured in plasma", "source": {"doi": "10.1/x"}}],
        context="test",
        domain_adapter=adapter,
    )

    # The domain qualifier must be accepted (would be dropped without adapter)
    assert len(claims) == 1, (
        "domain qualifier 'test_qualifier' should be accepted when adapter is passed"
    )
    # enrich_claim() must have been called by extract_claims() internally
    assert claims[0].get("enriched_by_adapter") is True, (
        "extract_claims() must call adapter.enrich_claim() internally"
    )


# ---------------------------------------------------------------------------
# claims_to_graph() — ontology_terms serialization
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_claims_to_graph_serializes_ontology_terms():
    """ontology_terms in a claim dict must appear as asb:{slot}_ontology_term triples."""
    import rdflib
    from perspicacite.pipeline.claims import claims_to_graph
    _ASB = "https://asb.holobiomics.org/ns/asb#"
    asb = rdflib.Namespace(_ASB)
    claim = {
        "context": "in vivo", "subject": "glucose", "qualifier": "quantifies",
        "relation": "measured in", "object": "plasma",
        "ontology_terms": {"subject": "CHEBI:17234", "object": "CHEBI:00001"},
    }
    g = claims_to_graph([claim])
    nodes = list(g.subjects(rdflib.RDF.type, asb.Claim))
    assert len(nodes) == 1
    node = nodes[0]
    assert (node, asb.subject_ontology_term, rdflib.Literal("CHEBI:17234")) in g
    assert (node, asb.object_ontology_term, rdflib.Literal("CHEBI:00001")) in g


@pytest.mark.unit
def test_claims_to_graph_no_ontology_terms_no_error():
    """Claims without ontology_terms must not raise and produce no extra triples."""
    import rdflib
    from perspicacite.pipeline.claims import claims_to_graph
    _ASB = "https://asb.holobiomics.org/ns/asb#"
    asb = rdflib.Namespace(_ASB)
    claim = {
        "context": "in vivo", "subject": "glucose", "qualifier": "quantifies",
        "relation": "measured in", "object": "plasma",
    }
    g = claims_to_graph([claim])
    terms = list(g.triples((None, asb.subject_ontology_term, None)))
    assert terms == []


@pytest.mark.unit
def test_claims_to_graph_skips_none_ontology_terms():
    """None values in ontology_terms must be skipped — not serialized as literal 'None'."""
    import rdflib
    from perspicacite.pipeline.claims import claims_to_graph
    _ASB = "https://asb.holobiomics.org/ns/asb#"
    asb = rdflib.Namespace(_ASB)
    claim = {
        "context": "in vivo", "subject": "glucose", "qualifier": "quantifies",
        "relation": "measured in", "object": "plasma",
        "ontology_terms": {"subject": None, "object": "CHEBI:00001"},
    }
    g = claims_to_graph([claim])
    # subject term with None value must NOT appear
    subject_terms = list(g.objects(None, asb.subject_ontology_term))
    assert subject_terms == [], f"Expected no subject_ontology_term triple, got {subject_terms}"
    # object term with valid value must appear
    object_terms = list(g.objects(None, asb.object_ontology_term))
    assert len(object_terms) == 1
    assert str(object_terms[0]) == "CHEBI:00001"


# ---------------------------------------------------------------------------
# Multi-domain composition (CompositeAdapter integration)
# ---------------------------------------------------------------------------

@pytest.mark.unit
async def test_extract_claims_with_composite_adapter_chains_enrichments():
    """CompositeAdapter must chain both adapters' enrich_claim() calls."""
    from indicium_adapters.composite import compose_adapters

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "compound A",
        "qualifier": "inhibits", "relation": "inhibits growth of",
        "object": "cell line B", "claim_type": "explicit",
        "evidence_type": "data", "source_type": "text",
        "quote": "A inhibited B", "source_doi": "10.1/x",
    }]}))

    adapter_a = _MockAdapter()      # sets enriched_by_adapter=True
    adapter_b = _MockAdapterB()     # sets enriched_by_b=True
    composite = compose_adapters([adapter_a, adapter_b])

    claims = await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "A inhibited B", "source": {"doi": "10.1/x"}}],
        context="test",
        domain_adapter=composite,
    )

    assert len(claims) == 1
    assert claims[0].get("enriched_by_adapter") is True, "Adapter A enrichment missing"
    assert claims[0].get("enriched_by_b") is True, "Adapter B enrichment missing"


@pytest.mark.unit
async def test_extract_claims_composite_accepts_union_of_qualifiers():
    """Qualifiers from both adapters must be accepted by extract_claims()."""
    from indicium_adapters.composite import compose_adapters

    llm = AsyncMock()
    # Claim uses qualifier_b, which is only in _MockAdapterB
    llm.complete = AsyncMock(return_value=json.dumps({"claims": [{
        "context": "in vitro", "subject": "compound A",
        "qualifier": "qualifier_b",   # only in adapter B
        "relation": "inhibits", "object": "cell line B",
    }]}))

    composite = compose_adapters([_MockAdapter(), _MockAdapterB()])

    claims = await extract_claims(
        llm_client=llm,
        passages=[{"chunk_text": "test", "source": {"doi": "10.1/x"}}],
        context="test",
        domain_adapter=composite,
    )

    assert len(claims) == 1, (
        "qualifier_b should be accepted by composite adapter's union qualifier set"
    )
    assert claims[0]["qualifier"] == "qualifier_b"


@pytest.mark.unit
def test_compose_adapters_filters_unknown_domains():
    """resolve logic: unknown domain IDs are silently dropped."""
    from indicium_adapters.composite import compose_adapters

    adapter_a = _MockAdapter()
    discovered = {"test": adapter_a}   # only "test" is known

    valid = [discovered[d] for d in ["test", "unknown_domain"] if d in discovered]
    result = compose_adapters(valid) if valid else None

    # Only one valid adapter found → returned directly (not wrapped)
    assert result is adapter_a


@pytest.mark.unit
def test_compose_adapters_all_unknown_gives_none():
    """If all requested domains are unknown, adapter must resolve to None."""
    from indicium_adapters.composite import compose_adapters

    discovered: dict = {}   # nothing installed
    valid = [discovered[d] for d in ["foo", "bar"] if d in discovered]
    result = compose_adapters(valid) if valid else None

    assert result is None
