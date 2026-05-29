"""Unit tests for indicium_layer.builder.build_claim_graph."""

import json

import pytest

pytest.importorskip("indicium", reason="indicium (indicia extra) not installed")
pytest.importorskip("rdflib", reason="rdflib (indicia extra) not installed")

from perspicacite.indicium_layer.builder import (
    BuildResult,
    build_claim_graph,
    claim_iri,
    paper_iri,
    passage_iri,
)
from perspicacite.indicium_layer.queries import (
    IRI_CLAIM,
    IRI_RDF_TYPE,
    cito_graph_iri,
)
from perspicacite.indicium_layer.store import ClaimGraphStore


class _FakeLLM:
    """Returns claim-extraction JSON then CiTO-classification JSON."""

    def __init__(self):
        self._calls = 0

    async def complete(self, *, messages, stage=None, **kw):
        self._calls += 1
        # Extraction calls have no stage; CiTO classifier calls use
        # stage="cito_classifier". Detect extraction by the absence of a
        # downstream-classifier stage.
        if stage is None or (stage or "").endswith("extract"):
            # Use call count to produce unique claims per paper so that the
            # pruner can form at least one cross-paper candidate pair.
            return json.dumps({
                "claims": [{
                    "context": "in vitro",
                    "subject": f"compound_{self._calls}",
                    "qualifier": "inhibits",
                    "relation": "binds_to",
                    "object": "enzyme Y",
                    "claim_type": "explicit",
                    "evidence_type": "data",
                    "source_type": "text",
                    "source_doi": "10.1/p1",
                    "quote": "X inhibits Y",
                }],
            })
        return json.dumps([
            {"pair_id": 0, "label": "supports", "confidence": 0.9}
        ])


def _papers_provider():
    return {
        "10.1/p1": {
            "paper_id": "10.1/p1",
            "doi": "10.1/p1",
            "title": "Paper 1",
            "year": 2024,
        },
        "10.1/p2": {
            "paper_id": "10.1/p2",
            "doi": "10.1/p2",
            "title": "Paper 2",
            "year": 2024,
        },
    }


def _passages_provider(paper_id):
    return [
        {"chunk_idx": 0, "text": f"text for {paper_id} chunk 0",
         "char_start": 0, "char_end": 30},
    ]


def test_iri_helpers():
    assert paper_iri("kb", {"doi": "10.1/x"}) == "doi:10.1/x"
    assert paper_iri("kb", {"title": "T", "year": 2024}).startswith("kb://kb/paper/")
    p_iri = passage_iri("kb", "10.1/x", 3)
    assert p_iri.startswith("kb://kb/passage/")
    assert p_iri.endswith("/3")
    c_iri = claim_iri("kb", {
        "context": "c", "subject": "s",
        "qualifier": "inhibits", "relation": "r", "object": "o",
    })
    assert c_iri.startswith("kb://kb/claim/")


async def test_build_creates_claim_node_with_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    store = ClaimGraphStore("kb", backend="memory")
    llm = _FakeLLM()
    result = await build_claim_graph(
        kb_name="kb",
        store=store,
        llm_client=llm,
        papers_provider=_papers_provider,
        passages_provider=_passages_provider,
        max_pairs_per_claim=20,
        model=None,
        builder_version="t",
    )
    assert isinstance(result, BuildResult)
    assert result.claims_added >= 2  # one per paper
    rows = store.select(
        f"SELECT ?c WHERE {{ ?c <{IRI_RDF_TYPE}> <{IRI_CLAIM}> }}"
    )
    assert len(rows) == result.claims_added


async def test_build_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    from perspicacite.indicium_layer.manifest import read_manifest

    store = ClaimGraphStore("kb", backend="memory")
    await build_claim_graph(
        kb_name="kb",
        store=store,
        llm_client=_FakeLLM(),
        papers_provider=_papers_provider,
        passages_provider=_passages_provider,
    )
    m = read_manifest("kb")
    assert set(m.paper_hashes.keys()) == {"10.1/p1", "10.1/p2"}
    assert m.last_build_iso is not None


async def test_build_incremental_skips_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    store = ClaimGraphStore("kb", backend="memory")
    r1 = await build_claim_graph(
        kb_name="kb",
        store=store,
        llm_client=_FakeLLM(),
        papers_provider=_papers_provider,
        passages_provider=_passages_provider,
    )
    r2 = await build_claim_graph(
        kb_name="kb",
        store=store,
        llm_client=_FakeLLM(),
        papers_provider=_papers_provider,
        passages_provider=_passages_provider,
    )
    assert r1.claims_added > 0
    assert r2.claims_added == 0  # nothing new


async def test_build_writes_cito_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "perspicacite.indicium_layer.manifest._DATA_DIR",
        tmp_path / "claim_graphs",
    )
    store = ClaimGraphStore("kb", backend="memory")
    await build_claim_graph(
        kb_name="kb",
        store=store,
        llm_client=_FakeLLM(),
        papers_provider=_papers_provider,
        passages_provider=_passages_provider,
    )
    g = cito_graph_iri("kb")
    edges = store.select(
        f"SELECT ?o FROM <{g}> WHERE {{ ?s <http://purl.org/spar/cito/supports> ?o }}"
    )
    # FakeLLM classifies every candidate pair as supports@0.9 → expect >= 1
    assert len(edges) >= 1
