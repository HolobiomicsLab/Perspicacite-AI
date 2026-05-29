"""Unit tests for Phase 2 (graph_traversal) strategy."""

import json

import pytest

pytest.importorskip("rdflib", reason="rdflib (indicia extra) not installed")

from perspicacite.config.schema import Config
from perspicacite.models.rag import RAGMode, RAGRequest


class _FakeLLM:
    """First call: intent classifier returns 'graph'.
    Second call: planner picks claims_supporting.
    Third call: composer renders the answer."""

    def __init__(self):
        self._calls = 0

    async def complete(self, *, messages, stage=None, **kw):
        self._calls += 1
        s = stage or ""
        if s.endswith("graph.intent"):
            return json.dumps({"intent": "graph", "rationale": "asks which claims"})
        if s.endswith("graph.plan"):
            return json.dumps(
                {
                    "query": "claims_supporting",
                    "kwargs": {"subject_or_iri": "compound X"},
                }
            )
        if s.endswith("graph.compose"):
            return "Compound X supports enzyme inhibition in two studies."
        return "{}"


async def _drain(stream):
    return [ev async for ev in stream]


def _seed_store(kb):
    from perspicacite.indicium_layer.queries import (
        ASB_NS,
        IRI_CLAIM,
        IRI_OBJECT,
        IRI_RDF_TYPE,
        IRI_SUBJECT,
        IRI_WAS_DERIVED_FROM,
    )
    from perspicacite.indicium_layer.store import ClaimGraphStore

    store = ClaimGraphStore(kb, backend="memory")
    iri = f"kb://{kb}/claim/seed"
    store.add(iri, IRI_RDF_TYPE, IRI_CLAIM)
    store.add(iri, IRI_SUBJECT, ("literal", "compound X", None))
    store.add(iri, IRI_OBJECT, ("literal", "enzyme Y", None))
    store.add(iri, IRI_WAS_DERIVED_FROM, "doi:10.1/p1")
    store.add(
        iri,
        f"{ASB_NS}evidenceTypeIri",
        "http://purl.obolibrary.org/obo/ECO_0000006",
    )
    return store


async def test_graph_strategy_runs_planner_query_and_composes(monkeypatch):
    from perspicacite.rag.modes.reasoning import graph_traversal as gt

    monkeypatch.setattr(gt, "_open_store", lambda kb: _seed_store(kb))

    req = RAGRequest(
        query="Which claims support enzyme inhibition by X?",
        mode=RAGMode.REASONING,
        reasoning_strategy="graph",
        kb_name="kb",
    )
    events = await _drain(
        gt.run_graph_traversal_stream(
            request=req,
            llm=_FakeLLM(),
            vector_store=None,
            embedding_provider=None,
            config=Config(),
            session_store=None,
        )
    )
    query_status = [e for e in events if e.event == "status" and "graph_query" in (e.data or "")]
    assert query_status, "expected a graph_query telemetry status event"
    payload = json.loads(query_status[0].data)
    assert payload["kind"] == "graph_query"
    assert payload["query"] == "claims_supporting"

    content = "".join(json.loads(e.data).get("delta", "") for e in events if e.event == "content")
    assert "Compound X" in content


async def test_graph_strategy_falls_back_to_provenance_when_non_graph(monkeypatch):
    """When the intent classifier returns non-graph, we degrade to provenance."""
    from perspicacite.rag.modes.reasoning import graph_traversal as gt

    class _NonGraphLLM:
        async def complete(self, *, messages, stage=None, **kw):
            if (stage or "").endswith("graph.intent"):
                return json.dumps({"intent": "narrative", "rationale": "asks for prose summary"})
            return "{}"

    called = {"prov": False}

    async def _fake_prov_stream(**kw):
        called["prov"] = True
        from perspicacite.models.rag import StreamEvent

        yield StreamEvent.content("provenance-fallback content")
        yield StreamEvent.done(conversation_id="", tokens_used=0, mode="reasoning", iterations=1)

    monkeypatch.setattr(
        "perspicacite.rag.modes.reasoning.provenance.run_provenance_stream",
        _fake_prov_stream,
    )

    req = RAGRequest(
        query="Summarise studies on X.",
        mode=RAGMode.REASONING,
        reasoning_strategy="graph",
        kb_name="kb",
    )
    events = await _drain(
        gt.run_graph_traversal_stream(
            request=req,
            llm=_NonGraphLLM(),
            vector_store=None,
            embedding_provider=None,
            config=Config(),
            session_store=None,
        )
    )
    assert called["prov"] is True
    content = "".join(json.loads(e.data).get("delta", "") for e in events if e.event == "content")
    assert "provenance-fallback content" in content
