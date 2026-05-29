"""Round-trip ASB-style ``paper.metadata`` through ingestion → chroma → retrieval.

Pins:
- ChunkMetadata exposes ``paper_metadata_json``.
- _chunk_to_metadata + _metadata_to_chunk preserve it.
- search_two_pass paper-result dicts include a decoded ``paper_metadata`` field.
"""
from __future__ import annotations

import json

from perspicacite.models.documents import ChunkMetadata
from perspicacite.models.papers import PaperSource


def test_chunk_metadata_has_paper_metadata_json_field():
    cm = ChunkMetadata(
        paper_id="asb_skill:foo",
        chunk_index=0,
        source=PaperSource.SKILL_BUNDLE,
        paper_metadata_json=json.dumps({"content_kind": "skill_body", "skill_id": "foo"}),
    )
    assert cm.paper_metadata_json
    assert json.loads(cm.paper_metadata_json)["skill_id"] == "foo"


def test_chunk_to_chroma_metadata_round_trip_preserves_paper_metadata_json():
    """_chunk_to_metadata(...) → _metadata_to_chunk(dict) preserves the field."""
    from perspicacite.retrieval.chroma_store import _chunk_to_metadata, _metadata_to_chunk

    payload = {"content_kind": "workflow_card", "task_id": "task_001"}
    cm_in = ChunkMetadata(
        paper_id="p1", chunk_index=0, source=PaperSource.SKILL_BUNDLE,
        paper_metadata_json=json.dumps(payload),
    )
    flat = _chunk_to_metadata(cm_in)
    assert flat.get("paper_metadata_json") == json.dumps(payload)

    cm_out = _metadata_to_chunk(flat)
    assert cm_out.paper_metadata_json == json.dumps(payload)
    assert json.loads(cm_out.paper_metadata_json)["task_id"] == "task_001"


def test_search_two_pass_exposes_decoded_paper_metadata():
    """Synthetic: stub vector store so search_two_pass sees a hit with
    ``paper_metadata_json``; result dict must expose decoded ``paper_metadata``."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from perspicacite.models.documents import DocumentChunk
    from perspicacite.models.search import RetrievedChunk
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

    payload = {"content_kind": "skill_body", "skill_id": "abc", "tools": []}
    fake_meta = ChunkMetadata(
        paper_id="asb_skill:abc", chunk_index=0,
        source=PaperSource.SKILL_BUNDLE, title="Abc",
        paper_metadata_json=json.dumps(payload),
    )
    fake_chunk = DocumentChunk(
        id="asb_skill:abc_metadata",
        text="...",
        metadata=fake_meta,
    )
    retrieved = RetrievedChunk(chunk=fake_chunk, score=0.9, retrieval_method="vector")

    fake_vs = MagicMock()
    fake_vs.search = AsyncMock(return_value=[retrieved])
    fake_vs.peek_paper_metadata_row = AsyncMock(return_value=None)
    fake_vs.get_chunks_by_paper_ids = AsyncMock(return_value=[
        {"paper_id": "asb_skill:abc", "chunk_index": 0, "text": "..."}
    ])

    fake_emb = MagicMock()
    fake_emb.embed = AsyncMock(return_value=[[0.0] * 8])
    fake_emb.embed_query = AsyncMock(return_value=[[0.0] * 8])

    dkb = DynamicKnowledgeBase(fake_vs, fake_emb, config=KnowledgeBaseConfig(vector_size=8))
    dkb.collection_name = "test"
    dkb._initialized = True

    results = asyncio.run(dkb.search_two_pass("anything", top_k=5))
    assert results, "expected at least one result"
    r0 = results[0]
    assert "paper_metadata" in r0, f"missing paper_metadata key in {list(r0)}"
    assert r0["paper_metadata"]["skill_id"] == "abc"


def test_search_two_pass_decodes_paper_metadata_via_peek_row():
    """When no vector hit exists, ``search_two_pass`` resolves paper metadata via
    ``peek_paper_metadata_row`` (forced_paper_ids / merged-scope candidates branch).
    The decoded ``paper_metadata`` must still appear on the result dict."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig
    from perspicacite.rag.query_scope import PaperScopeResult

    payload = {"content_kind": "workflow_card", "task_id": "task_007", "tools": ["t1"]}
    # Raw chroma metadata row (flat dict, as stored) — including the JSON blob.
    peek_row = {
        "paper_id": "asb_skill:peeked",
        "chunk_index": 0,
        "source": "skill_bundle",
        "title": "Peeked Skill",
        "authors": None,
        "year": None,
        "doi": None,
        "paper_metadata_json": json.dumps(payload),
    }

    fake_vs = MagicMock()
    # Vector search returns no hits — forces the forced_paper_ids branch.
    fake_vs.search = AsyncMock(return_value=[])
    fake_vs.peek_paper_metadata_row = AsyncMock(return_value=peek_row)
    fake_vs.get_chunks_by_paper_ids = AsyncMock(return_value=[
        {"paper_id": "asb_skill:peeked", "chunk_index": 0, "text": "body text"}
    ])

    fake_emb = MagicMock()
    fake_emb.embed = AsyncMock(return_value=[[0.0] * 8])
    fake_emb.embed_query = AsyncMock(return_value=[[0.0] * 8])

    dkb = DynamicKnowledgeBase(fake_vs, fake_emb, config=KnowledgeBaseConfig(vector_size=8))
    dkb.collection_name = "test"
    dkb._initialized = True

    scope = PaperScopeResult(forced_paper_ids=["asb_skill:peeked"], max_papers=1)
    results = asyncio.run(dkb.search_two_pass("anything", top_k=5, paper_scope=scope))

    assert results, "expected at least one result via peek_paper_metadata_row path"
    fake_vs.peek_paper_metadata_row.assert_awaited()
    r0 = results[0]
    assert r0["paper_id"] == "asb_skill:peeked"
    assert "paper_metadata" in r0, f"missing paper_metadata key in {list(r0)}"
    assert r0["paper_metadata"]["task_id"] == "task_007"
    assert r0["paper_metadata"]["content_kind"] == "workflow_card"
