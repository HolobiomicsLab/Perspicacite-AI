"""Passage retrieval preserves evidence units independently of paper search."""

import hashlib

import pytest
from _passage_fixture import hit, retriever

from perspicacite.models.search import SearchFilters
from perspicacite.retrieval.passage_search import search_passages


def adapter(value):
    from perspicacite.retrieval.passage_search import PassageRetriever

    return PassageRetriever(value)


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("science", ["proteomics", "astronomy", "climate", "electrophysiology"])
async def test_three_passages_from_one_article_survive_for_each_science(kind, science):
    rows = [
        hit(f"chunk-{n}", f"{science} condition {n}", score)
        for n, score in [(1, 0.7), (2, 0.9), (3, 0.8)]
    ]
    r, _, embedding = retriever(kind, {"kb_a": rows})
    results = await search_passages(adapter(r), text="conditions", k=3)
    assert [v.chunk_id for v in results] == ["chunk-2", "chunk-3", "chunk-1"]
    assert all(v.kb_name == "a" and v.collection_name == "kb_a" for v in results)
    assert all(
        v.content_sha256 == hashlib.sha256(v.chunk_text.encode()).hexdigest() for v in results
    )
    assert len(embedding.calls) == 1
    assert len(await r.search("conditions", top_k=3)) == 1


async def test_duplicate_identity_is_local_to_collection_and_uses_highest_score():
    r, _, _ = retriever(
        "multi",
        {
            "kb_b": [hit("same", "second source", 0.9), hit("third", "third source", 0.8)],
            "kb_a": [hit("same", "first source", 0.8), hit("same", "first source", 0.9)],
        },
    )
    results = await search_passages(adapter(r), text="condition", k=3)
    assert [(r.collection_name, r.chunk_id, r.score) for r in results] == [
        ("kb_a", "same", 0.9),
        ("kb_b", "same", 0.9),
        ("kb_b", "third", 0.8),
    ]


@pytest.mark.parametrize("kind", ["single", "multi"])
async def test_explicit_zero_floor_and_filters_reach_the_same_query_boundary(kind):
    r, store, _ = retriever(
        kind, {"kb_a": [hit("low", "low score", 0.1), hit("high", "high score", 0.9)]}, floor=0.5
    )
    filters = SearchFilters(year_min=2020)
    results = await r.search_chunks("condition", top_k=3, min_score=0, filters=filters)
    assert [row["chunk_id"] for row in results] == ["high", "low"]
    assert store.calls[-1][3] is filters
    assert len(await r.search_chunks("condition", min_score=None)) == 1


async def test_common_multi_kb_search_accepts_filters_without_changing_paper_unit():
    r, store, _ = retriever("multi", {"kb_a": [hit("a", "first", 0.9), hit("b", "second", 0.8)]})
    filters = SearchFilters(year_min=2020)
    results = await r.search("condition", filters=filters)
    assert len(results) == 1
    assert store.calls[0][3] is filters


@pytest.mark.parametrize("kind", ["single", "multi"])
async def test_missing_source_metadata_is_unknown_without_losing_actual_chunk_identity(kind):
    r, _, _ = retriever(
        kind, {"kb_a": [hit("known-chunk", "unattributed passage", 0.9, metadata=False)]}
    )
    result = (await search_passages(adapter(r), text="condition"))[0]
    assert result.chunk_id == "known-chunk" and result.kb_name == "a"
    assert result.source.doi is None and result.source.license_id is None


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_scores_are_errors_not_ranked_passages(kind, score):
    r, _, _ = retriever(kind, {"kb_a": [hit("chunk", "passage", score)]})
    with pytest.raises(ValueError):
        await r.search_chunks("condition")


@pytest.mark.parametrize("kind", ["single", "multi"])
async def test_missing_chunk_id_is_explicitly_unusable(kind):
    r, _, _ = retriever(kind, {"kb_a": [hit(None, "passage", 0.9)]})
    with pytest.raises(ValueError):
        await r.search_chunks("condition")


@pytest.mark.parametrize("kind", ["single", "multi"])
async def test_conflicting_content_for_one_chunk_identity_is_an_error(kind):
    r, _, _ = retriever(kind, {"kb_a": [hit("same", "first", 0.9), hit("same", "changed", 0.8)]})
    with pytest.raises(ValueError):
        await r.search_chunks("condition")


@pytest.mark.parametrize("kind", ["single", "multi"])
@pytest.mark.parametrize("model", [None, "foreign-model", "fixture-embedding|fallback"])
async def test_unknown_or_incompatible_vector_model_refuses_before_embedding(kind, model):
    r, store, embedding = retriever(kind, {"kb_a": [hit("chunk", "passage", 0.9)]}, model=model)
    with pytest.raises(ValueError):
        await r.search_chunks("condition")
    assert embedding.calls == [] and store.calls == []


@pytest.mark.parametrize("kind", ["single", "multi"])
async def test_fallback_vector_model_cannot_query_original_collection(kind):
    r, store, embedding = retriever(
        kind, {"kb_a": [hit("chunk", "passage", 0.9)]}, served_model="different-model"
    )
    with pytest.raises(ValueError):
        await r.search_chunks("condition")
    assert len(embedding.calls) == 1 and store.calls == []


@pytest.mark.parametrize("kind", ["single", "multi"])
async def test_successful_empty_query_is_empty_and_keeps_paper_mode(kind):
    r, store, _ = retriever(kind, {"kb_a": []})
    assert await search_passages(adapter(r), text="condition") == []
    assert len(store.calls) == 1
    assert await r.search("condition") == []
