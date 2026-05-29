"""Unit tests for the unified resolve_papers_pipeline helper."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from perspicacite.models.papers import Paper
from perspicacite.rag.resolve_papers import resolve_papers_pipeline


def _make_paper(title: str, abstract: str = "test abstract") -> Paper:
    return Paper(
        id=f"doi:10.1/{title}",
        title=title,
        abstract=abstract,
    )


# ---------------------------------------------------------------------------
# Test 1: aggregator → enrich → rerank all wired together
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_calls_aggregator_enrich_rerank():
    """Full pipeline: aggregator + enrich + rerank all get called."""
    papers = [_make_paper("paper A", "metabolomics abstract"), _make_paper("paper B", "chemistry abstract")]

    from perspicacite.search.screening import ScreenResult

    fake_screen_results = [
        ScreenResult(item={"_paper": papers[0], "title": "paper A", "abstract": "metabolomics abstract"}, score=0.9, kept=True),
        ScreenResult(item={"_paper": papers[1], "title": "paper B", "abstract": "chemistry abstract"}, score=0.5, kept=True),
    ]

    with patch(
        "perspicacite.rag.web_search.run_web_aggregator_search",
        AsyncMock(return_value=papers),
    ) as mock_agg, patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        AsyncMock(side_effect=lambda p, **kw: p),
    ) as mock_enrich, patch(
        "perspicacite.search.screening.screen_papers_rerank",
        AsyncMock(return_value=fake_screen_results),
    ) as mock_rerank:
        result = await resolve_papers_pipeline(
            query="metabolomics",
            databases=["openalex"],
            max_docs=5,
            app_state=None,
            enrich=True,
            rerank=True,
        )

    mock_agg.assert_called_once()
    mock_enrich.assert_called_once()
    mock_rerank.assert_called_once()
    assert len(result) == 2
    # rerank should produce paper A first (score 0.9 > 0.5)
    assert result[0].title == "paper A"


# ---------------------------------------------------------------------------
# Test 2: enrich=False, rerank=False skips those steps
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_skips_enrich_and_rerank_when_disabled():
    papers = [_make_paper("only paper")]

    mock_enrich = AsyncMock(side_effect=lambda p, **kw: p)
    mock_rerank = AsyncMock(return_value=[])

    with patch(
        "perspicacite.rag.web_search.run_web_aggregator_search",
        AsyncMock(return_value=papers),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        mock_enrich,
    ), patch(
        "perspicacite.search.screening.screen_papers_rerank",
        mock_rerank,
    ):
        result = await resolve_papers_pipeline(
            query="q",
            databases=None,
            max_docs=10,
            app_state=None,
            enrich=False,
            rerank=False,
        )

    mock_enrich.assert_not_called()
    mock_rerank.assert_not_called()
    assert result == papers


# ---------------------------------------------------------------------------
# Test 3: enrich failure is swallowed; papers still returned
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_continues_when_enrich_fails():
    papers = [_make_paper("paper X")]

    with patch(
        "perspicacite.rag.web_search.run_web_aggregator_search",
        AsyncMock(return_value=papers),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        AsyncMock(side_effect=RuntimeError("crossref down")),
    ), patch(
        "perspicacite.search.screening.screen_papers_rerank",
        # rerank skipped because only 1 paper (len <= 1)
        AsyncMock(return_value=[]),
    ):
        result = await resolve_papers_pipeline(
            query="test query",
            databases=None,
            max_docs=5,
            app_state=None,
            enrich=True,
            rerank=True,
        )

    # Despite enrich failure, papers should still be returned
    assert len(result) == 1
    assert result[0].title == "paper X"
