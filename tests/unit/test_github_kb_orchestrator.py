"""Tests for github_kb orchestrator (mocked dependencies)."""
from __future__ import annotations

from pathlib import Path  # noqa: TC003
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.pipeline.github_kb import (
    IngestSummary,
    ingest_skill_bundle,
    ingest_skill_bundles_batch,
)


_EMBED_DIM = 384


def _mock_embedder():
    """Embedding service stub whose ``embed`` is awaitable.

    The symbol-wise code path embeds in the orchestrator itself, so a bare
    MagicMock is not enough. ``side_effect`` returns one vector per text —
    a plain AsyncMock would return a MagicMock that zips to nothing, letting
    the tests pass without exercising the chunks.
    """
    embedder = MagicMock()
    embedder.model_name = "all-MiniLM-L6-v2"
    embedder.dimension = _EMBED_DIM
    embedder.embed = AsyncMock(
        side_effect=lambda texts: [[0.1] * _EMBED_DIM for _ in texts]
    )
    return embedder


def _short_embedder():
    """Embedder that returns one vector too few, as providers do for blank text."""
    embedder = _mock_embedder()
    embedder.embed = AsyncMock(
        side_effect=lambda texts: [[0.1] * _EMBED_DIM for _ in texts[:-1]]
    )
    return embedder


def _make_bundle(tmp_path: Path, name: str = "test-bundle") -> Path:
    bundle_dir = tmp_path / name
    bundle_dir.mkdir()
    (bundle_dir / "bundle.yml").write_text(
        f"name: {name}\n"
        "papers:\n"
        "  - doi: 10.1/a\n"
        "  - doi: 10.2/b\n"
    )
    (bundle_dir / "README.md").write_text("# Bundle\nSee 10.3/c for details.")
    (bundle_dir / "main.py").write_text('"""Main module."""\n\ndef run():\n    """Run it."""\n    pass\n')  # noqa: E501
    return bundle_dir


def _mock_config(tmp_path: Path):
    return SimpleNamespace(
        knowledge_base=SimpleNamespace(
            log_dir=tmp_path / "logs",
            chunk_size=500,
            chunk_overlap=50,
            embedding_model="all-MiniLM-L6-v2",
        ),
        github=SimpleNamespace(token_env_var="GITHUB_TOKEN", cache_dir=tmp_path / "cache"),
        bundles=SimpleNamespace(default_kb_name_template="{name}"),
    )


@pytest.mark.asyncio
async def test_ingest_skill_bundle_calls_add_papers(tmp_path):
    bundle_dir = _make_bundle(tmp_path)
    config = _mock_config(tmp_path)

    captured_dois: list[str] = []

    async def fake_ingest(app_state, kb_name, dois, **kw):
        captured_dois.extend(dois)
        return {"added_papers": len(dois), "added_chunks": 0, "skipped_duplicates": 0, "failed": [], "pdf_download": {}}  # noqa: E501

    mock_dkb = MagicMock()
    mock_dkb.add_papers = AsyncMock(return_value=5)
    mock_session = AsyncMock()
    mock_session.get_kb_metadata = AsyncMock(return_value=None)
    mock_embed = _mock_embedder()
    vector_store = AsyncMock()

    with patch("perspicacite.pipeline.github_kb.ingest_dois_into_kb", new=fake_ingest), \
         patch("perspicacite.rag.dynamic_kb.DynamicKnowledgeBase", return_value=mock_dkb):
        summary = await ingest_skill_bundle(
            source=bundle_dir,
            kb_name="test-kb",
            config=config,
            vector_store=vector_store,
            embedding_service=mock_embed,
            session_store=mock_session,
            ingest_linked_papers=True,
            app_state_for_doi_ingest=MagicMock(),
        )

    assert summary.files_added >= 2  # README.md + main.py
    # DOIs from manifest should have been ingested
    assert "10.1/a" in captured_dois
    assert "10.2/b" in captured_dois

    # The symbol-wise path embeds in the orchestrator, so prove it ran and that
    # every chunk carries its own vector — a bare AsyncMock would pass without
    # this, embedding nothing.
    mock_embed.embed.assert_awaited()
    code_calls = [
        call.kwargs["chunks"]
        for call in vector_store.add_documents.await_args_list
        if all(c.metadata.content_type == "code" for c in call.kwargs["chunks"])
    ]
    assert code_calls, "the symbol-wise code path never reached the vector store"
    assert all(
        len(chunk.embedding) == _EMBED_DIM
        for chunks in code_calls
        for chunk in chunks
    )


@pytest.mark.asyncio
async def test_ingest_skill_bundle_no_linked_papers(tmp_path):
    bundle_dir = _make_bundle(tmp_path)
    config = _mock_config(tmp_path)
    mock_dkb = MagicMock()
    mock_dkb.add_papers = AsyncMock(return_value=3)
    mock_session = AsyncMock()
    mock_session.get_kb_metadata = AsyncMock(return_value=None)
    mock_embed = _mock_embedder()

    with patch("perspicacite.rag.dynamic_kb.DynamicKnowledgeBase", return_value=mock_dkb):
        summary = await ingest_skill_bundle(
            source=bundle_dir,
            kb_name="test-kb",
            config=config,
            vector_store=AsyncMock(),
            embedding_service=mock_embed,
            session_store=mock_session,
            ingest_linked_papers=False,
        )

    assert summary.linked_papers_added == 0


@pytest.mark.asyncio
async def test_ingest_skill_bundles_batch_processes_all(tmp_path):
    dirs = [_make_bundle(tmp_path, f"bundle-{i}") for i in range(3)]
    config = _mock_config(tmp_path)
    mock_dkb = MagicMock()
    mock_dkb.add_papers = AsyncMock(return_value=1)
    mock_session = AsyncMock()
    mock_session.get_kb_metadata = AsyncMock(return_value=None)
    mock_embed = _mock_embedder()

    async def fake_ingest(app_state, kb_name, dois, **kw):
        return {"added_papers": len(dois), "added_chunks": 0, "skipped_duplicates": 0, "failed": [], "pdf_download": {}}  # noqa: E501

    with patch("perspicacite.pipeline.github_kb.ingest_dois_into_kb", new=fake_ingest), \
         patch("perspicacite.rag.dynamic_kb.DynamicKnowledgeBase", return_value=mock_dkb):
        summaries = await ingest_skill_bundles_batch(
            root=tmp_path,
            config=config,
            vector_store=AsyncMock(),
            embedding_service=mock_embed,
            session_store=mock_session,
            ingest_linked_papers=True,
            app_state_for_doi_ingest=MagicMock(),
        )

    assert len(summaries) == 3
    assert all(isinstance(s, IngestSummary) for s in summaries)


@pytest.mark.asyncio
async def test_short_embedding_list_is_refused_not_misaligned(tmp_path):
    """A provider that drops an input must not shift vectors onto wrong chunks.

    The vector store's own length guard cannot catch this: these chunks arrive
    already embedded, so nothing downstream re-checks the pairing.
    """
    bundle_dir = _make_bundle(tmp_path)
    config = _mock_config(tmp_path)
    mock_dkb = MagicMock()
    mock_dkb.add_papers = AsyncMock(return_value=0)
    mock_session = AsyncMock()
    mock_session.get_kb_metadata = AsyncMock(return_value=None)

    with patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase", return_value=mock_dkb
    ), pytest.raises(ValueError):
        await ingest_skill_bundle(
            source=bundle_dir,
            kb_name="test-kb",
            config=config,
            vector_store=AsyncMock(),
            embedding_service=_short_embedder(),
            session_store=mock_session,
            ingest_linked_papers=False,
        )
