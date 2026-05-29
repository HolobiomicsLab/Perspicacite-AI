"""Tests for github_kb orchestrator (mocked dependencies)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.pipeline.github_kb import (
    IngestSummary,
    ingest_skill_bundle,
    ingest_skill_bundles_batch,
)


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
    (bundle_dir / "main.py").write_text('"""Main module."""\n\ndef run():\n    """Run it."""\n    pass\n')
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
        return {"added_papers": len(dois), "added_chunks": 0, "skipped_duplicates": 0, "failed": [], "pdf_download": {}}

    mock_dkb = MagicMock()
    mock_dkb.add_papers = AsyncMock(return_value=5)
    mock_session = AsyncMock()
    mock_session.get_kb_metadata = AsyncMock(return_value=None)
    mock_embed = MagicMock()
    mock_embed.model_name = "all-MiniLM-L6-v2"
    mock_embed.dimension = 384

    with patch("perspicacite.pipeline.github_kb.ingest_dois_into_kb", new=fake_ingest), \
         patch("perspicacite.rag.dynamic_kb.DynamicKnowledgeBase", return_value=mock_dkb):
        summary = await ingest_skill_bundle(
            source=bundle_dir,
            kb_name="test-kb",
            config=config,
            vector_store=AsyncMock(),
            embedding_service=mock_embed,
            session_store=mock_session,
            ingest_linked_papers=True,
            app_state_for_doi_ingest=MagicMock(),
        )

    assert summary.files_added >= 2  # README.md + main.py
    # DOIs from manifest should have been ingested
    assert "10.1/a" in captured_dois
    assert "10.2/b" in captured_dois


@pytest.mark.asyncio
async def test_ingest_skill_bundle_no_linked_papers(tmp_path):
    bundle_dir = _make_bundle(tmp_path)
    config = _mock_config(tmp_path)
    mock_dkb = MagicMock()
    mock_dkb.add_papers = AsyncMock(return_value=3)
    mock_session = AsyncMock()
    mock_session.get_kb_metadata = AsyncMock(return_value=None)
    mock_embed = MagicMock()
    mock_embed.model_name = "all-MiniLM-L6-v2"
    mock_embed.dimension = 384

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
    for i in range(3):
        _make_bundle(tmp_path, f"bundle-{i}")
    config = _mock_config(tmp_path)
    mock_dkb = MagicMock()
    mock_dkb.add_papers = AsyncMock(return_value=1)
    mock_session = AsyncMock()
    mock_session.get_kb_metadata = AsyncMock(return_value=None)
    mock_embed = MagicMock()
    mock_embed.model_name = "all-MiniLM-L6-v2"
    mock_embed.dimension = 384

    async def fake_ingest(app_state, kb_name, dois, **kw):
        return {"added_papers": len(dois), "added_chunks": 0, "skipped_duplicates": 0, "failed": [], "pdf_download": {}}

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
