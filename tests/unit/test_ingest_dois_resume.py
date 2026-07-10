"""Verify ingest_dois_into_kb wires CheckpointStore + resumes on re-run (Wave 3.3)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.pipeline.checkpoint import CheckpointStore
from perspicacite.pipeline.search_to_kb import ingest_dois_into_kb


def _app_state(tmp_path: Path) -> SimpleNamespace:
    """Minimal app_state stand-in. Avoids pulling the full DI graph."""
    config = SimpleNamespace(
        pdf_download=None,
        knowledge_base=SimpleNamespace(checkpoint_dir=tmp_path / "ck"),
    )
    session_store = MagicMock()
    session_store.get_kb_metadata = AsyncMock(return_value=SimpleNamespace(
        paper_count=0, chunk_count=0,
    ))
    session_store.save_kb_metadata = AsyncMock()
    vector_store = MagicMock()
    vector_store.paper_exists = AsyncMock(return_value=False)
    return SimpleNamespace(
        config=config,
        session_store=session_store,
        vector_store=vector_store,
        embedding_provider=MagicMock(),
        pdf_parser=MagicMock(),
    )


@pytest.mark.asyncio
async def test_resume_skips_already_processed(tmp_path):
    """If a checkpoint already shows 2 of 3 DOIs added, the next call
    only processes the remaining 1."""
    # Seed a checkpoint with 2 of 3 already added.
    ck_dir = tmp_path / "ck"
    ck_dir.mkdir()
    store = CheckpointStore(
        path=ck_dir / "kb1__ingest_dois.json",
        kb_name="kb1",
        operation="ingest_dois",
    )
    state = store.load_or_create(planned_ids=["10.1/a", "10.2/b", "10.3/c"])
    state.record("10.1/a", "added")
    state.record("10.2/b", "added")
    store.save(state)

    # Mock everything that goes over the wire.
    fetched_dois: list[str] = []

    async def fake_retrieve(doi, **kw):
        fetched_dois.append(doi)
        return SimpleNamespace(
            success=True, full_text="x", abstract=None, metadata={},
        )

    state = _app_state(tmp_path)
    with patch(
        "perspicacite.pipeline.download.retrieve_paper_content",
        new=fake_retrieve,
    ), patch(
        "perspicacite.pipeline.download.cookies.build_authenticated_client",
    ) as mock_client_ctx, patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase",
    ) as mock_dkb:
        # async context manager that yields a mocked http client.
        mock_client_ctx.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_client_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_dkb.return_value.add_papers = AsyncMock(return_value=1)

        result = await ingest_dois_into_kb(
            state, "kb1",
            ["10.1/a", "10.2/b", "10.3/c"],
        )

    # Only "10.3/c" should have been fetched — the other two were
    # already in the checkpoint.
    assert fetched_dois == ["10.3/c"]
    # The checkpoint should have been deleted on clean completion.
    assert not (ck_dir / "kb1__ingest_dois.json").exists()


@pytest.mark.asyncio
async def test_embedding_failure_unmarks_dois_so_a_resume_retries_them(tmp_path):
    """DOIs are marked "added" on retrieval, before the batch is embedded.

    If the embedder dies, the batch never lands. remaining_ids() never re-offers an
    "added" id, so leaving the mark would make a resume skip those DOIs for ever.
    """
    from perspicacite.llm.embeddings import EmbeddingFailedError

    ck_dir = tmp_path / "ck"
    ck_dir.mkdir()
    checkpoint_path = ck_dir / "kb1__ingest_dois.json"

    async def fake_retrieve(doi, **kw):
        return SimpleNamespace(success=True, full_text="x", abstract=None, metadata={})

    app_state = _app_state(tmp_path)
    with patch(
        "perspicacite.pipeline.download.retrieve_paper_content", new=fake_retrieve
    ), patch(
        "perspicacite.pipeline.download.cookies.build_authenticated_client"
    ) as mock_client_ctx, patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase"
    ) as mock_dkb:
        mock_client_ctx.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_client_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_dkb.return_value.add_papers = AsyncMock(
            side_effect=EmbeddingFailedError("quota exhausted")
        )

        with pytest.raises(EmbeddingFailedError):
            await ingest_dois_into_kb(app_state, "kb1", ["10.1/a", "10.2/b"])

    # The checkpoint survives the abort, but must not claim the DOIs were added.
    assert checkpoint_path.exists()
    store = CheckpointStore(path=checkpoint_path, kb_name="kb1", operation="ingest_dois")
    state = store.load_or_create(planned_ids=["10.1/a", "10.2/b"])
    assert sorted(state.remaining_ids()) == ["10.1/a", "10.2/b"]
