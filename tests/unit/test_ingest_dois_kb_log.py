"""Verify ingest_dois_into_kb emits KBLog events (Wave 4.3)."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.pipeline.search_to_kb import ingest_dois_into_kb


def _app_state(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            pdf_download=None,
            knowledge_base=SimpleNamespace(
                checkpoint_dir=tmp_path / "ckpt",
                log_dir=tmp_path / "logs",
            ),
        ),
        session_store=MagicMock(
            get_kb_metadata=AsyncMock(return_value=SimpleNamespace(
                paper_count=0, chunk_count=0,
            )),
            save_kb_metadata=AsyncMock(),
        ),
        vector_store=MagicMock(paper_exists=AsyncMock(return_value=False)),
        embedding_provider=MagicMock(),
        pdf_parser=MagicMock(),
    )


@pytest.mark.asyncio
async def test_paper_added_event_recorded_on_success(tmp_path):
    state = _app_state(tmp_path)

    async def fake_retrieve(doi, **kw):
        return SimpleNamespace(
            success=True, full_text="x", abstract=None, metadata={"title": "T"},
        )

    with patch(
        "perspicacite.pipeline.download.retrieve_paper_content",
        new=fake_retrieve,
    ), patch(
        "perspicacite.pipeline.download.cookies.build_authenticated_client",
    ) as ctx, patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase",
    ) as mock_dkb:
        ctx.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_dkb.return_value.add_papers = AsyncMock(return_value=5)

        await ingest_dois_into_kb(state, "kb1", ["10.1/a"])

    log_path = tmp_path / "logs" / "kb1.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().split("\n")
    events = [json.loads(l) for l in lines]
    kinds = [e["event"] for e in events]
    assert "paper_added" in kinds
    added = next(e for e in events if e["event"] == "paper_added")
    assert added["paper_id"] == "10.1/a"
    assert added["source_command"] == "ingest_dois_into_kb"


@pytest.mark.asyncio
async def test_paper_skipped_event_for_duplicate(tmp_path):
    state = _app_state(tmp_path)
    # Pretend the paper already exists.
    state.vector_store.paper_exists = AsyncMock(return_value=True)

    with patch(
        "perspicacite.pipeline.download.cookies.build_authenticated_client",
    ) as ctx:
        ctx.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        await ingest_dois_into_kb(state, "kb1", ["10.1/dup"])

    log_path = tmp_path / "logs" / "kb1.jsonl"
    events = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    assert any(e["event"] == "paper_skipped" and e["paper_id"] == "10.1/dup" for e in events)


@pytest.mark.asyncio
async def test_paper_failed_event_with_reason(tmp_path):
    state = _app_state(tmp_path)

    async def fake_retrieve(doi, **kw):
        raise RuntimeError("network down")

    with patch(
        "perspicacite.pipeline.download.retrieve_paper_content",
        new=fake_retrieve,
    ), patch(
        "perspicacite.pipeline.download.cookies.build_authenticated_client",
    ) as ctx:
        ctx.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        await ingest_dois_into_kb(state, "kb1", ["10.1/x"])

    log_path = tmp_path / "logs" / "kb1.jsonl"
    events = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
    failed = [e for e in events if e["event"] == "paper_failed"]
    assert len(failed) == 1
    assert failed[0]["paper_id"] == "10.1/x"
    assert "network down" in (failed[0].get("reason") or "")


# --- retryability of throttled papers (rate-limit fix) ---------------------
#
# A 429 and a genuine abstract-only paper look identical downstream: same
# content_type, same abstract, no full text. The only thing that separates
# them is ``rate_limited_hosts``, so both cases are asserted below.

THROTTLED_HOST = "api.publisher.example"


def _pdf_config(**overrides):
    """A PDFDownloadConfig-shaped stub with every field the builder reads.

    Args:
        **overrides: Field values to replace the neutral defaults.

    Returns:
        A SimpleNamespace usable wherever ``config.pdf_download`` is read.
    """
    fields = {
        "unpaywall_email": None, "alternative_endpoint": None,
        "wiley_tdm_token": None, "elsevier_api_key": None,
        "aaas_api_key": None, "rsc_api_key": None, "springer_api_key": None,
        "cookies_path": None, "cache_pdfs": False, "cache_dir": "/nonexistent",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _abstract_result(hosts: list[str]) -> SimpleNamespace:
    """A successful abstract-only retrieval that saw ``hosts`` throttle it."""
    return SimpleNamespace(
        success=True, full_text=None, abstract="An abstract.",
        metadata={"title": "T"}, content_type="abstract",
        rate_limited_hosts=list(hosts),
    )


async def _run_ingest(state, doi: str, result: SimpleNamespace) -> dict:
    """Ingest one DOI with a stubbed retrieval, keeping the checkpoint file.

    ``CheckpointStore.delete`` is neutralised so the completed run's
    checkpoint survives for the assertions.
    """
    async def fake_retrieve(_doi, **kw):
        return result

    with patch(
        "perspicacite.pipeline.download.retrieve_paper_content", new=fake_retrieve,
    ), patch(
        "perspicacite.pipeline.download.cookies.build_authenticated_client",
    ) as ctx, patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase",
    ) as mock_dkb, patch(
        "perspicacite.pipeline.checkpoint.CheckpointStore.delete",
    ):
        ctx.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_dkb.return_value.add_papers = AsyncMock(return_value=1)
        return await ingest_dois_into_kb(state, "kb1", [doi])


def _load_checkpoint(tmp_path: Path):
    """The persisted CheckpointState for the ``kb1`` ingest_dois run."""
    from perspicacite.pipeline.checkpoint import CheckpointStore

    return CheckpointStore(
        path=tmp_path / "ckpt" / "kb1__ingest_dois.json",
        kb_name="kb1",
        operation="ingest_dois",
    ).load()


@pytest.mark.asyncio
async def test_rate_limited_abstract_is_failed_and_reoffered(tmp_path):
    state = _app_state(tmp_path)

    report = await _run_ingest(
        state, "10.1/throttled", _abstract_result([THROTTLED_HOST]),
    )

    assert report["added_papers"] == 0
    assert report["pdf_download"]["failed"] == 1
    assert THROTTLED_HOST in report["failed"][0]["reason"]

    ck = _load_checkpoint(tmp_path)
    assert ck.processed["10.1/throttled"].startswith("failed")
    assert THROTTLED_HOST in ck.processed["10.1/throttled"]
    assert list(ck.remaining_ids(retry_failed=True)) == ["10.1/throttled"]


@pytest.mark.asyncio
async def test_genuine_abstract_only_paper_is_still_added(tmp_path):
    """Negative case: same shape as above but nothing throttled us."""
    state = _app_state(tmp_path)

    report = await _run_ingest(state, "10.1/abs", _abstract_result([]))

    assert report["added_papers"] == 1
    assert report["pdf_download"]["success"] == 1
    assert report["failed"] == []

    ck = _load_checkpoint(tmp_path)
    assert ck.processed["10.1/abs"] == "added"
    assert list(ck.remaining_ids(retry_failed=True)) == []


def test_pdf_kwargs_carry_elsevier_key_and_landing_capture():
    from perspicacite.pipeline.search_to_kb import _build_pdf_download_kwargs

    kwargs = _build_pdf_download_kwargs(
        _pdf_config(elsevier_api_key="EK", cookies_path="/tmp/cookies.txt"),
    )

    assert kwargs["elsevier_api_key"] == "EK"
    assert kwargs["enable_landing_capture"] is True


def test_pdf_kwargs_omit_new_keys_without_pdf_config():
    from perspicacite.pipeline.search_to_kb import _build_pdf_download_kwargs

    assert _build_pdf_download_kwargs(None) == {}


def test_pdf_kwargs_stay_clean_when_config_sets_neither():
    """Config present but empty must not enable landing capture."""
    from perspicacite.pipeline.search_to_kb import _build_pdf_download_kwargs

    kwargs = _build_pdf_download_kwargs(_pdf_config())

    assert kwargs["elsevier_api_key"] is None
    assert kwargs["enable_landing_capture"] is False


@pytest.mark.asyncio
async def test_pdf_kwargs_reach_retrieve_paper_content(tmp_path):
    state = _app_state(tmp_path)
    state.config.pdf_download = _pdf_config(
        elsevier_api_key="EK", cookies_path="/tmp/cookies.txt",
    )
    seen: dict = {}

    async def fake_retrieve(_doi, **kw):
        seen.update(kw)
        return _abstract_result([])

    with patch(
        "perspicacite.pipeline.download.retrieve_paper_content", new=fake_retrieve,
    ), patch(
        "perspicacite.pipeline.download.cookies.build_authenticated_client",
    ) as ctx, patch(
        "perspicacite.rag.dynamic_kb.DynamicKnowledgeBase",
    ) as mock_dkb:
        ctx.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        ctx.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_dkb.return_value.add_papers = AsyncMock(return_value=1)
        await ingest_dois_into_kb(state, "kb1", ["10.1/kw"])

    assert seen["elsevier_api_key"] == "EK"
    assert seen["enable_landing_capture"] is True
