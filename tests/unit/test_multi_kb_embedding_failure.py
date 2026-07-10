"""A degenerate query must not be swallowed by the multi-KB fan-out.

`MultiKBRetriever.search` and `query_chunks_across_collections` skip a collection
whose search fails, so the query can still be answered from the others. That is
right for a missing collection and wrong for a dead embedder: the query vector is
degenerate for *every* collection, so skipping them one by one turns an error into
an empty result reported as a success.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from perspicacite.llm.embeddings import EmbeddingFailedError
from perspicacite.retrieval.multi_kb import MultiKBRetriever, query_chunks_across_collections


def _embedding_service(vector):
    service = MagicMock()
    service.embed_query = AsyncMock(return_value=[vector])
    return service


def _kb_meta(collection_name: str):
    return SimpleNamespace(collection_name=collection_name, name=collection_name)


@pytest.mark.asyncio
async def test_retriever_propagates_embedding_failure():
    """A zero-norm query must surface as an error, not an empty result set."""
    store = MagicMock()
    store.search = AsyncMock(side_effect=EmbeddingFailedError("zero-norm query embedding"))
    retriever = MultiKBRetriever(
        vector_store=store,
        embedding_service=_embedding_service([0.0, 0.0]),
        kb_metas=[_kb_meta("kb-one"), _kb_meta("kb-two")],
    )

    with pytest.raises(EmbeddingFailedError):
        await retriever.search("anything")


@pytest.mark.asyncio
async def test_retriever_still_skips_a_single_broken_collection():
    """An ordinary per-collection failure is still tolerated."""
    hit = SimpleNamespace(
        chunk=SimpleNamespace(
            id="c1",
            text="text",
            metadata=SimpleNamespace(paper_id="p1", title="T", year=2024),
        ),
        score=0.9,
    )

    async def _search(collection, **_kwargs):
        if collection == "kb-broken":
            raise RuntimeError("collection missing")
        return [hit]

    store = MagicMock()
    store.search = AsyncMock(side_effect=_search)
    retriever = MultiKBRetriever(
        vector_store=store,
        embedding_service=_embedding_service([1.0, 0.0]),
        kb_metas=[_kb_meta("kb-broken"), _kb_meta("kb-good")],
    )

    results = await retriever.search("anything")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_fanout_helper_propagates_embedding_failure():
    """The standalone fan-out helper has the same contract."""
    store = MagicMock()
    store.search = AsyncMock(side_effect=EmbeddingFailedError("zero-norm query embedding"))

    with pytest.raises(EmbeddingFailedError):
        await query_chunks_across_collections(
            vector_store=store,
            embedding_service=_embedding_service([0.0, 0.0]),
            collection_names=["kb-one", "kb-two"],
            query="anything",
            top_k=5,
        )
