"""Tests for abstract_only ingestion flag (ingest_mode='abstract_only')."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.pipeline.download.unified import retrieve_paper_content

# ── retrieve_paper_content abstract_only flag ─────────────────────────────────


@pytest.mark.asyncio
async def test_abstract_only_returns_after_discovery():
    """abstract_only=True must return PaperContent after discovery,
    never calling PMC/PDF steps."""
    from perspicacite.pipeline.download.base import PaperDiscovery

    fake_disc = PaperDiscovery(
        doi="10.1/test",
        title="Test paper",
        authors=["A Author"],
        year=2023,
        abstract="A concise summary of the work.",
        is_oa=False,
        pmcid=None,
        arxiv_id=None,
        oa_url=None,
    )

    with patch(
        "perspicacite.pipeline.download.unified.discover_paper_sources",
        new=AsyncMock(return_value=fake_disc),
    ), patch(
        "perspicacite.pipeline.download.unified.get_fulltext_from_pmc",
        new=AsyncMock(return_value=("", [])),
    ) as mock_pmc:
        result = await retrieve_paper_content("10.1/test", abstract_only=True)

    assert result.success is True
    assert result.content_type == "abstract"
    assert result.abstract == "A concise summary of the work."
    assert result.full_text is None
    # PMC (and any PDF step) must never have been called
    mock_pmc.assert_not_called()


@pytest.mark.asyncio
async def test_abstract_only_fails_when_no_abstract():
    """abstract_only=True with no abstract → success=False."""
    from perspicacite.pipeline.download.base import PaperDiscovery

    fake_disc = PaperDiscovery(
        doi="10.1/test",
        title="Test paper",
        authors=[],
        year=2023,
        abstract=None,  # no abstract available
        is_oa=False,
        pmcid=None,
        arxiv_id=None,
        oa_url=None,
    )

    with patch(
        "perspicacite.pipeline.download.unified.discover_paper_sources",
        new=AsyncMock(return_value=fake_disc),
    ):
        result = await retrieve_paper_content("10.1/test", abstract_only=True)

    assert result.success is False


@pytest.mark.asyncio
async def test_abstract_only_false_proceeds_to_pmc():
    """When abstract_only=False (default), the pipeline continues past discovery."""
    from perspicacite.pipeline.download.base import PaperDiscovery

    fake_disc = PaperDiscovery(
        doi="10.1/test",
        title="Test paper",
        authors=[],
        year=2023,
        abstract="Short abstract.",
        is_oa=False,
        pmcid="PMC12345",  # has PMCID → will try PMC
        arxiv_id=None,
        oa_url=None,
    )
    long_text = "Full text from PMC. " * 20  # 400 chars > 200

    with patch(
        "perspicacite.pipeline.download.unified.discover_paper_sources",
        new=AsyncMock(return_value=fake_disc),
    ), patch(
        "perspicacite.pipeline.download.unified.get_fulltext_from_pmc",
        new=AsyncMock(return_value=(long_text, [])),
    ) as mock_pmc, patch(
        "perspicacite.pipeline.download.unified._load_cached_references",
        return_value=[],
    ):
        result = await retrieve_paper_content("10.1/test", abstract_only=False)

    # PMC was attempted because abstract_only=False
    mock_pmc.assert_called_once()
    assert result.content_type == "structured"


# ── ingest_dois_into_kb respects ingest_mode ──────────────────────────────────


def _app_state(tmp_path: Path, ingest_mode: str = "abstract_only") -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            pdf_download=None,
            knowledge_base=SimpleNamespace(
                checkpoint_dir=tmp_path / "ckpt",
                log_dir=tmp_path / "logs",
                ingest_mode=ingest_mode,
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
async def test_ingest_dois_abstract_only_mode(tmp_path):
    """When app_state.config.knowledge_base.ingest_mode == 'abstract_only',
    ingest_dois_into_kb must pass abstract_only=True to retrieve_paper_content."""
    from perspicacite.pipeline.search_to_kb import ingest_dois_into_kb

    retrieve_calls: list[dict] = []

    async def fake_retrieve(doi, **kw):
        retrieve_calls.append({"doi": doi, "abstract_only": kw.get("abstract_only", False)})
        return SimpleNamespace(
            success=True,
            full_text=None,
            abstract="Short abstract for testing.",
            metadata={"title": "Test Paper", "authors": [], "year": 2023},
        )

    state = _app_state(tmp_path, ingest_mode="abstract_only")

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
        mock_dkb.return_value.add_papers = AsyncMock(return_value=1)

        await ingest_dois_into_kb(state, "kb1", ["10.1/a"])

    assert retrieve_calls, "retrieve_paper_content was never called"
    assert retrieve_calls[0]["abstract_only"] is True


@pytest.mark.asyncio
async def test_ingest_dois_abstract_counted_as_success(tmp_path):
    """A paper with abstract but no full_text must count as 'success' in
    abstract_only mode — it must NOT be in the 'failed' list and
    pdf_download['success'] must be 1."""
    from perspicacite.pipeline.search_to_kb import ingest_dois_into_kb

    async def fake_retrieve(doi, **kw):
        return SimpleNamespace(
            success=True,
            full_text=None,  # NO full text
            abstract="This is the abstract.",
            metadata={"title": "Abstract-only paper", "authors": [], "year": 2022},
        )

    state = _app_state(tmp_path, ingest_mode="abstract_only")

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
        mock_dkb.return_value.add_papers = AsyncMock(return_value=1)

        result = await ingest_dois_into_kb(state, "kb1", ["10.1/abstract-only"])

    failed_dois = [f["doi"] for f in result.get("failed", [])]
    assert "10.1/abstract-only" not in failed_dois
    assert result["pdf_download"]["success"] == 1
    assert result["pdf_download"]["failed"] == 0


@pytest.mark.asyncio
async def test_ingest_dois_persists_content_type(tmp_path):
    """The download result's content_type ('abstract'/'full_text'/'structured')
    must be carried onto the ingested Paper so downstream can distinguish
    abstract-only from full-text papers without counting chunks."""
    from perspicacite.pipeline.search_to_kb import ingest_dois_into_kb

    async def fake_retrieve(doi, **kw):
        return SimpleNamespace(
            success=True,
            full_text=None,
            abstract="This is the abstract.",
            metadata={"title": "Abstract-only paper", "authors": [], "year": 2022},
            content_type="abstract",
        )

    state = _app_state(tmp_path, ingest_mode="abstract_only")

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
        mock_dkb.return_value.add_papers = AsyncMock(return_value=1)

        await ingest_dois_into_kb(state, "kb1", ["10.1/abstract-only"])

    add_papers = mock_dkb.return_value.add_papers
    assert add_papers.called, "add_papers was never called"
    papers_arg = add_papers.call_args.args[0]
    assert len(papers_arg) == 1
    assert papers_arg[0].content_type == "abstract"
