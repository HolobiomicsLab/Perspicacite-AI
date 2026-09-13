"""Exercise real passage retrievers through both MCP tools without external I/O."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _passage_fixture import Embedding, Store, hit, metadata

from perspicacite.llm.embeddings import EmbeddingFailedError
from perspicacite.mcp import server
from perspicacite.models.kb import chroma_collection_name_for_kb


def setup_state(monkeypatch, names, rows, *, model="fixture-embedding", embedding=None):
    metas = {name: metadata(name, model) for name in names}
    for name, meta in metas.items():
        meta.collection_name = chroma_collection_name_for_kb(name)
    store = Store(
        {metas[name].collection_name: row for name, row in zip(names, rows, strict=False)}
    )
    state = SimpleNamespace(
        session_store=SimpleNamespace(get_kb_metadata=AsyncMock(side_effect=metas.get)),
        vector_store=store,
        embedding_provider=embedding or Embedding(),
    )
    monkeypatch.setattr(server, "_require_state", lambda: state)
    rephrase = AsyncMock(side_effect=AssertionError("failed retrieval must not call an LLM"))
    monkeypatch.setattr(server, "_rephrase_query", rephrase)
    return state, rephrase


async def invoke(tool, names, **kwargs):
    if tool == "search":
        raw = await server.search_by_passage(text="measure uncertainty", kb_names=names, k=10)
    else:
        raw = await server.get_relevant_passages(
            query="measure uncertainty", kb_names=names, k=10, **kwargs
        )
    return json.loads(raw)


@pytest.mark.parametrize("tool", ["search", "relevant"])
@pytest.mark.parametrize("names", [["alpha"], ["alpha", "beta"]])
async def test_real_mcp_keeps_all_passage_identities(monkeypatch, tool, names):
    rows = [[hit("one", "first passage", 0.9), hit("two", "second passage", 0.8)] for _ in names]
    state, rephrase = setup_state(monkeypatch, names, rows)
    payload = await invoke(tool, names)
    assert payload["success"] is True
    items = payload["results" if tool == "search" else "passages"]
    assert len(items) == 2 * len(names)
    for item in items:
        content = item["chunk_text" if tool == "search" else "text"]
        assert item["chunk_id"] in {"one", "two"}
        assert item["kb_name"] in names
        assert item["collection_name"] == chroma_collection_name_for_kb(item["kb_name"])
        assert item["content_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert len(state.embedding_provider.calls) == 1
    assert len(state.vector_store.calls) == len(names)
    rephrase.assert_not_awaited()


@pytest.mark.parametrize("tool", ["search", "relevant"])
@pytest.mark.parametrize("failures", [1, 2])
async def test_collection_failures_cannot_be_complete_success(monkeypatch, tool, failures):
    names = ["alpha", "beta"]
    rows = [RuntimeError("collection unavailable")] * failures
    rows += [[hit("one", "available passage", 0.9)]] * (2 - failures)
    state, rephrase = setup_state(monkeypatch, names, rows)
    payload = await invoke(tool, names, adaptive=True)
    assert payload["success"] is False
    assert payload["ok"] is False
    assert "results" not in payload and "passages" not in payload
    diagnostics = payload["collection_errors"]
    assert {d["kb_name"] for d in diagnostics} == set(names[:failures])
    assert all(
        d["collection_name"] == chroma_collection_name_for_kb(d["kb_name"]) for d in diagnostics
    )
    assert len(state.vector_store.calls) == 2
    rephrase.assert_not_awaited()


@pytest.mark.parametrize("tool", ["search", "relevant"])
@pytest.mark.parametrize("model", [None, "different-model", "fixture-embedding+code:other"])
async def test_mcp_refuses_unknown_or_foreign_embedding_before_external_io(
    monkeypatch, tool, model
):
    state, rephrase = setup_state(monkeypatch, ["alpha"], [[]], model=model)
    payload = await invoke(tool, ["alpha"], adaptive=True)
    assert payload["success"] is False
    assert "embedding" in payload["error"].lower()
    assert state.embedding_provider.calls == []
    assert state.vector_store.calls == []
    rephrase.assert_not_awaited()


@pytest.mark.parametrize("tool", ["search", "relevant"])
async def test_embedding_failure_stops_without_adaptive_retry(monkeypatch, tool):
    embedding = Embedding()
    embedding.embed_query = AsyncMock(side_effect=EmbeddingFailedError("query embedding failed"))
    state, rephrase = setup_state(monkeypatch, ["alpha", "beta"], [[], []], embedding=embedding)
    payload = await invoke(tool, ["alpha", "beta"], adaptive=True)
    assert payload["success"] is False
    assert state.vector_store.calls == []
    rephrase.assert_not_awaited()


@pytest.mark.parametrize("tool", ["search", "relevant"])
async def test_empty_success_is_distinct_from_collection_failure(monkeypatch, tool):
    state, rephrase = setup_state(monkeypatch, ["alpha", "beta"], [[], []])
    payload = await invoke(tool, ["alpha", "beta"])
    assert payload["success"] is True
    assert payload["results" if tool == "search" else "passages"] == []
    assert "collection_errors" not in payload
    assert len(state.vector_store.calls) == 2
    rephrase.assert_not_awaited()
