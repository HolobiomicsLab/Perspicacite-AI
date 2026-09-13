"""Real local Chroma distinguishes unavailable collections from empty results."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _passage_fixture import Embedding, metadata

from perspicacite.mcp import server
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.retrieval.chroma_store import ChromaVectorStore


def state_for(monkeypatch, tmp_path: Path, names):
    store = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), embedding_provider=Embedding())
    metas = {name: metadata(name) for name in names}
    for name, meta in metas.items():
        meta.collection_name = chroma_collection_name_for_kb(name)
    state = SimpleNamespace(
        session_store=SimpleNamespace(get_kb_metadata=AsyncMock(side_effect=metas.get)),
        vector_store=store,
        embedding_provider=Embedding(),
    )
    monkeypatch.setattr(server, "_require_state", lambda: state)
    return state, metas


async def invoke(tool, names):
    if tool == "search":
        raw = await server.search_by_passage(text="uncertainty", kb_names=names)
    else:
        raw = await server.get_relevant_passages(query="uncertainty", kb_names=names)
    return json.loads(raw)


@pytest.mark.parametrize("tool", ["search", "relevant"])
@pytest.mark.parametrize("names", [["alpha"], ["alpha", "beta"]])
async def test_missing_real_collection_is_error_and_legacy_search_stays_empty(
    monkeypatch, tmp_path, tool, names
):
    state, metas = state_for(monkeypatch, tmp_path, names)
    if len(names) > 1:
        await state.vector_store.create_collection(metas["beta"].collection_name, embedding_dim=2)
    payload = await invoke(tool, names)
    assert payload["success"] is False
    assert payload["collection_errors"][0]["collection_name"] == metas["alpha"].collection_name
    assert await state.vector_store.search(metas["alpha"].collection_name, [0.2, 0.4]) == []


@pytest.mark.parametrize("tool", ["search", "relevant"])
async def test_existing_empty_real_collections_are_success(monkeypatch, tmp_path, tool):
    names = ["alpha", "beta"]
    state, metas = state_for(monkeypatch, tmp_path, names)
    for meta in metas.values():
        await state.vector_store.create_collection(meta.collection_name, embedding_dim=2)
    payload = await invoke(tool, names)
    assert payload["success"] is True
    assert payload["results" if tool == "search" else "passages"] == []


@pytest.mark.parametrize("tool", ["search", "relevant"])
async def test_real_chroma_mcp_keeps_same_paper_passages_from_each_kb(monkeypatch, tmp_path, tool):
    names = ["alpha", "beta"]
    state, metas = state_for(monkeypatch, tmp_path, names)
    for meta in metas.values():
        await state.vector_store.create_collection(meta.collection_name, embedding_dim=2)
        state.vector_store.client.get_collection(meta.collection_name).add(
            ids=["chunk-one", "chunk-two"],
            embeddings=[[0.2, 0.4], [0.3, 0.4]],
            documents=["first computation passage", "second computation passage"],
            metadatas=[{"paper_id": "same-paper", "doi": "10.0000/source"}] * 2,
        )
    payload = await invoke(tool, names)
    assert payload["success"] is True
    items = payload["results" if tool == "search" else "passages"]
    assert len(items) == 4
    assert {(p["kb_name"], p["chunk_id"]) for p in items} == {
        (name, chunk) for name in names for chunk in ("chunk-one", "chunk-two")
    }
    for item in items:
        content = item["chunk_text" if tool == "search" else "text"]
        assert item["content_sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert state.embedding_provider.calls == [["uncertainty"]]
