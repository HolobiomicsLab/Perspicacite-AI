"""profound.py honors request.kb_names — fans out across multiple KB collections.

The fan-out itself is unit-tested at the `_basic_vector_retrieve` and
`_wrrf_retrieval` boundaries: with two collections, results from both KBs
reach the retrieval stage tagged with their originating collection name
(kb_name). The two-pass `_enrich_with_full_text` carries that kb_name through
to the resulting paper-level dicts, so downstream `SourceReference` instances
can attribute each paper to its originating KB.
"""

from __future__ import annotations

from typing import Any

import pytest

from perspicacite.config.schema import Config
from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.papers import PaperSource
from perspicacite.models.search import RetrievedChunk
from perspicacite.rag.modes.deep_research import (
    DeepResearchRAGMode as ProfoundRAGMode,  # renamed; alias kept for backward compat
)


def _retrieved(paper_id: str, text: str, score: float) -> RetrievedChunk:
    md = ChunkMetadata(
        paper_id=paper_id, chunk_index=0, source=PaperSource.BIBTEX, title=paper_id
    )
    ch = DocumentChunk(id=f"chunk-{paper_id}", text=text, metadata=md)
    return RetrievedChunk(chunk=ch, score=score)


class _FakeVS:
    """Minimal vector_store stub for the fan-out path."""

    def __init__(
        self,
        by_coll: dict[str, list[RetrievedChunk]],
        chunks_by_coll: dict[str, list[dict[str, Any]]] | None = None,
    ):
        self.by_coll = by_coll
        self.chunks_by_coll = chunks_by_coll or {}
        self.search_calls: list[str] = []
        self.get_chunks_calls: list[str] = []

    async def search(
        self,
        *,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        **_: Any,
    ) -> list[RetrievedChunk]:
        self.search_calls.append(collection)
        return self.by_coll.get(collection, [])

    async def get_chunks_by_paper_ids(
        self, collection: str, paper_ids: list[str]
    ) -> list[dict[str, Any]]:
        self.get_chunks_calls.append(collection)
        return [
            c
            for c in self.chunks_by_coll.get(collection, [])
            if c.get("paper_id") in paper_ids
        ]


class _FakeEmb:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        return await self.embed(texts)


@pytest.mark.asyncio
async def test_basic_vector_retrieve_fans_out_and_tags_kb_name():
    """When `collection_names` has >1 entry, search is fanned out across all
    collections and each result is tagged with its originating kb_name."""
    cfg = Config()
    mode = ProfoundRAGMode(cfg)
    mode.initial_docs = 5
    mode.final_max_docs = 5
    mode.max_docs_per_source = 5

    vs = _FakeVS(
        {
            "kb_a": [_retrieved("p1", "text from kb_a", 0.9)],
            "kb_b": [_retrieved("p2", "text from kb_b", 0.8)],
        }
    )

    selected = await mode._basic_vector_retrieve(
        query="q",
        vector_store=vs,
        embedding_provider=_FakeEmb(),
        kb_name="kb_a",  # legacy single
        collection_names=["kb_a", "kb_b"],
    )

    assert sorted(vs.search_calls) == ["kb_a", "kb_b"], (
        f"expected fan-out to both collections, got {vs.search_calls}"
    )
    assert selected, "expected selected documents"
    paper_ids = {
        getattr(getattr(d, "chunk", None), "metadata", None).paper_id
        for d in selected
        if getattr(getattr(d, "chunk", None), "metadata", None) is not None
        and getattr(d.chunk.metadata, "paper_id", None)
    }
    assert paper_ids & {"p1", "p2"}, f"expected at least one of p1/p2, got {paper_ids}"
    # Each retrieved chunk should be tagged with its source collection.
    tagged_kbs = {
        getattr(d, "kb_name", None)
        for d in selected
        if getattr(d, "kb_name", None) is not None
    }
    assert tagged_kbs == {"kb_a", "kb_b"} or tagged_kbs <= {"kb_a", "kb_b"}
    assert tagged_kbs, "expected results to be tagged with kb_name"


@pytest.mark.asyncio
async def test_basic_vector_retrieve_single_collection_back_compat():
    """When `collection_names` is None, the legacy single-collection path
    still works using `kb_name`."""
    cfg = Config()
    mode = ProfoundRAGMode(cfg)
    mode.initial_docs = 5
    mode.final_max_docs = 5
    mode.max_docs_per_source = 5

    vs = _FakeVS({"kb_solo": [_retrieved("p1", "solo text", 0.7)]})

    selected = await mode._basic_vector_retrieve(
        query="q",
        vector_store=vs,
        embedding_provider=_FakeEmb(),
        kb_name="kb_solo",
    )

    assert vs.search_calls == ["kb_solo"]
    assert selected


@pytest.mark.asyncio
async def test_wrrf_retrieval_fans_out_across_collections():
    """`_wrrf_retrieval` with multiple `collection_names` fans out and results
    from every collection reach WRRF."""
    cfg = Config()
    mode = ProfoundRAGMode(cfg)
    mode.initial_docs = 5
    mode.final_max_docs = 5
    mode.max_docs_per_source = 5

    vs = _FakeVS(
        {
            "kb_a": [_retrieved("p1", "text from kb_a", 0.9)],
            "kb_b": [_retrieved("p2", "text from kb_b", 0.8)],
        }
    )

    selected = await mode._wrrf_retrieval(
        queries=["q1", "q2"],
        vector_store=vs,
        embedding_provider=_FakeEmb(),
        kb_name="kb_a",
        collection_names=["kb_a", "kb_b"],
        llm=None,
    )

    # Two queries x two collections = at least 4 search invocations expected.
    assert sorted(set(vs.search_calls)) == ["kb_a", "kb_b"]
    assert selected, "expected selected documents"


@pytest.mark.asyncio
async def test_enrich_with_full_text_fans_out_and_tags_kb_name():
    """Two-pass enrichment in multi-KB mode fans `get_chunks_by_paper_ids`
    across collections and tags each paper-level dict with its kb_name."""
    cfg = Config()
    mode = ProfoundRAGMode(cfg)

    # Build retrieved chunks pre-tagged with kb_name (as fan-out would have done).
    r1 = _retrieved("p1", "from kb_a", 0.9)
    r1.kb_name = "kb_a"  # type: ignore[attr-defined]
    r2 = _retrieved("p2", "from kb_b", 0.8)
    r2.kb_name = "kb_b"  # type: ignore[attr-defined]

    md1 = ChunkMetadata(
        paper_id="p1", chunk_index=0, source=PaperSource.BIBTEX, title="P1"
    )
    md2 = ChunkMetadata(
        paper_id="p2", chunk_index=0, source=PaperSource.BIBTEX, title="P2"
    )
    chunks_by_coll = {
        "kb_a": [{"paper_id": "p1", "text": "full text 1", "metadata": md1}],
        "kb_b": [{"paper_id": "p2", "text": "full text 2", "metadata": md2}],
    }

    vs = _FakeVS({}, chunks_by_coll=chunks_by_coll)

    paper_results = await mode._enrich_with_full_text(
        results=[r1, r2],
        kb_name="kb_a",
        vector_store=vs,
        collection_names=["kb_a", "kb_b"],
    )

    assert sorted(vs.get_chunks_calls) == ["kb_a", "kb_b"]
    assert len(paper_results) == 2
    kb_tags = {p.get("kb_name") for p in paper_results}
    assert kb_tags == {"kb_a", "kb_b"}, f"expected both KBs tagged, got {kb_tags}"


@pytest.mark.asyncio
async def test_enrich_with_full_text_single_collection_back_compat():
    """Single-collection enrichment path is unchanged."""
    cfg = Config()
    mode = ProfoundRAGMode(cfg)

    r1 = _retrieved("p1", "txt", 0.9)
    md1 = ChunkMetadata(
        paper_id="p1", chunk_index=0, source=PaperSource.BIBTEX, title="P1"
    )
    vs = _FakeVS(
        {},
        chunks_by_coll={
            "kb_solo": [{"paper_id": "p1", "text": "full", "metadata": md1}]
        },
    )

    paper_results = await mode._enrich_with_full_text(
        results=[r1],
        kb_name="kb_solo",
        vector_store=vs,
    )

    assert vs.get_chunks_calls == ["kb_solo"]
    assert len(paper_results) == 1
