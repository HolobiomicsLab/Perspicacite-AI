"""Ambiguous wrappers and malformed vectors cannot cross a passage boundary."""

from unittest.mock import AsyncMock

import pytest
from _passage_fixture import Embedding, retriever

from perspicacite.llm.embeddings import CachedEmbeddingProvider, TypedEmbeddingProvider


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("wrapper", ["cached", "typed"])
async def test_wrappers_without_query_producer_identity_are_explicitly_unsupported(kind, wrapper):
    r, store, embedding = retriever(kind, {"kb_a": []})
    if wrapper == "cached":
        cache = AsyncMock()
        r.embedding_service = CachedEmbeddingProvider(inner=embedding, cache=cache)
    else:
        r.embedding_service = TypedEmbeddingProvider(default=embedding, by_content_type={})
    with pytest.raises(ValueError, match=r"(?i)(embedding|identity|provider)"):
        await r.search_chunks("measure uncertainty")
    assert embedding.calls == []
    assert store.calls == []


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[]],
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
        [[float("inf"), 1.0]],
        [[1.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ],
)
async def test_malformed_query_vectors_are_refused_before_vector_search(kind, vectors):
    r, store, embedding = retriever(kind, {"kb_a": []})
    embedding.embed_query = AsyncMock(return_value=vectors)
    with pytest.raises((ValueError, RuntimeError), match=r"(?i)(embedding|vector|dimension)"):
        await r.search_chunks("measure uncertainty")
    embedding.embed_query.assert_awaited_once()
    assert store.calls == []


@pytest.mark.parametrize("kind", ["single", "multi"])
async def test_stateless_configured_provider_retains_existing_protocol(kind):
    r, store, _ = retriever(kind, {"kb_a": []})
    provider = Embedding()
    del provider.last_used_model
    provider.embed_query = AsyncMock(return_value=[[0.2, 0.4]])
    r.embedding_service = provider
    assert await r.search_chunks("measure uncertainty") == []
    assert len(store.calls) == 1
