import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from perspicacite.integrations.capsule_reader import ingest_capsule


def _write_capsule(root, *, blocks, resources=None):
    (root / "figures").mkdir(parents=True)
    (root / "figures" / "index.json").write_text("[]")
    (root / "metadata.json").write_text(json.dumps({
        "capsule_version": "0.1", "producer": "asb", "paper_id": "doi:10.1/x",
        "title": "Test paper", "authors": [{"family": "Doe", "given": "Jane"}],
        "year": 2025, "doi": "10.1/x", "source": "local",
    }))
    (root / "text").mkdir()
    (root / "text" / "blocks.jsonl").write_text(
        "\n".join(json.dumps(b) for b in blocks)
    )
    if resources is not None:
        (root / "resources.json").write_text(json.dumps(resources))


def _app_state(chunk_size=500, chunk_overlap=50, embed_dim=3):
    kb = MagicMock()
    kb.collection_name = "col1"
    kb.chunk_count = 0
    kb.name = "kb1"

    app_state = MagicMock()
    app_state.session_store.get_kb_metadata = AsyncMock(return_value=kb)
    app_state.session_store.save_kb_metadata = AsyncMock()
    # Embed returns one vector per chunk
    app_state.embedding_provider.embed = AsyncMock(
        side_effect=lambda texts: [[0.0] * embed_dim for _ in texts]
    )
    app_state.vector_store.add_documents = AsyncMock()
    app_state.config.knowledge_base.chunk_size = chunk_size
    app_state.config.knowledge_base.chunk_overlap = chunk_overlap
    return app_state, kb


def _registry():
    r = MagicMock()
    r.publish = AsyncMock()
    r.finish = AsyncMock()
    r.fail = AsyncMock()
    return r


@pytest.mark.asyncio
async def test_ingest_chunks_blocks_jsonl(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    _write_capsule(cap, blocks=[
        {"block_id": "p001_b0", "section": "abstract", "page": 1,
         "content": "intro paragraph " * 80, "figure_refs": [], "table_refs": []},
        {"block_id": "p002_b0", "section": "results", "page": 2,
         "content": "results paragraph " * 80,
         "figure_refs": ["pdf_p2_i0"], "table_refs": []},
    ])

    app_state, _kb = _app_state()
    registry = _registry()

    result = await ingest_capsule(
        capsule_dir=cap, kb_name="kb1",
        app_state=app_state, registry=registry, job_id="j1",
    )

    assert result["files"] == 1
    assert result["added_chunks"] > 0
    registry.finish.assert_awaited_once()

    # The chunks passed to vector_store.add_documents should include figure_refs
    add_calls = app_state.vector_store.add_documents.call_args_list
    assert add_calls
    all_chunks = [c for call in add_calls for c in call.args[1]]
    assert any("pdf_p2_i0" in (c.metadata.figure_refs or []) for c in all_chunks)
    sections = {c.metadata.source_section for c in all_chunks if c.metadata.source_section}
    assert {"abstract", "results"}.issubset(sections)


@pytest.mark.asyncio
async def test_ingest_propagates_resource_ids(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    _write_capsule(
        cap,
        blocks=[
            {"block_id": "p001_b0", "section": "methods", "page": 1,
             "content": "methods text " * 80, "figure_refs": [], "table_refs": []},
        ],
        resources=[
            {"resource_id": "github:owner/repo", "kind": "github"},
            {"resource_id": "doi:10.5281/zenodo.123", "kind": "zenodo"},
        ],
    )
    app_state, _ = _app_state()
    registry = _registry()

    await ingest_capsule(
        capsule_dir=cap, kb_name="kb1",
        app_state=app_state, registry=registry, job_id="j1",
    )

    all_chunks = [
        c for call in app_state.vector_store.add_documents.call_args_list
        for c in call.args[1]
    ]
    assert all_chunks
    assert any(
        "github:owner/repo" in (c.metadata.resource_refs or []) for c in all_chunks
    )


@pytest.mark.asyncio
async def test_ingest_unknown_kb_calls_fail(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    _write_capsule(cap, blocks=[
        {"block_id": "p001_b0", "section": "results", "page": 1,
         "content": "x", "figure_refs": [], "table_refs": []},
    ])
    app_state, _ = _app_state()
    app_state.session_store.get_kb_metadata = AsyncMock(return_value=None)
    registry = _registry()

    await ingest_capsule(
        capsule_dir=cap, kb_name="missing",
        app_state=app_state, registry=registry, job_id="j1",
    )
    registry.fail.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_finalize_false_skips_finish(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    _write_capsule(cap, blocks=[
        {"block_id": "p001_b0", "section": "results", "page": 1,
         "content": "results " * 80, "figure_refs": [], "table_refs": []},
    ])
    app_state, _ = _app_state()
    registry = _registry()

    await ingest_capsule(
        capsule_dir=cap, kb_name="kb1",
        app_state=app_state, registry=registry, job_id="j1",
        finalize=False,
    )
    registry.finish.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_no_text_source_returns_zero(tmp_path):
    cap = tmp_path / "cap"
    (cap / "figures").mkdir(parents=True)
    (cap / "figures" / "index.json").write_text("[]")
    (cap / "metadata.json").write_text(json.dumps({
        "capsule_version": "0.1", "paper_id": "doi:10.1/x",
        "title": "Test", "authors": [], "year": 2025, "doi": "10.1/x",
        "source": "local",
    }))
    app_state, _ = _app_state()
    registry = _registry()

    result = await ingest_capsule(
        capsule_dir=cap, kb_name="kb1",
        app_state=app_state, registry=registry, job_id="j1",
    )
    assert result == {"added_chunks": 0, "files": 1}
    registry.finish.assert_awaited_once()
