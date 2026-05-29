"""Cross-mode regression test: all callers that route through
resolve_papers_pipeline must produce consistent output from the same
upstream aggregator result.

This test confirms that basic._web_fallback_papers,
literature_survey._broad_search, and the MCP web_search tool all
invoke resolve_papers_pipeline (not diverging inline paths) and that
they all surface the same papers when the pipeline returns deterministic
output.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.models.papers import Author, Paper
from perspicacite.search.screening import ScreenResult


def _make_paper(n: int) -> Paper:
    return Paper(
        id=f"doi:10.1/paper{n}",
        title=f"Paper {n}",
        authors=[Author(name=f"Author {n}")],
        year=2024,
        doi=f"10.1/paper{n}",
        abstract=f"Abstract for paper {n} about metabolomics and mass spectrometry.",
    )


FAKE_PAPERS = [_make_paper(i) for i in range(1, 4)]  # 3 papers

FAKE_SCREEN = [
    ScreenResult(
        item={"_paper": FAKE_PAPERS[0], "title": FAKE_PAPERS[0].title, "abstract": FAKE_PAPERS[0].abstract},
        score=0.9, kept=True,
    ),
    ScreenResult(
        item={"_paper": FAKE_PAPERS[1], "title": FAKE_PAPERS[1].title, "abstract": FAKE_PAPERS[1].abstract},
        score=0.7, kept=True,
    ),
    ScreenResult(
        item={"_paper": FAKE_PAPERS[2], "title": FAKE_PAPERS[2].title, "abstract": FAKE_PAPERS[2].abstract},
        score=0.5, kept=True,
    ),
]


def _pipeline_patches():
    """Return context managers that intercept all three sub-steps."""
    return [
        patch(
            "perspicacite.rag.web_search.run_web_aggregator_search",
            AsyncMock(return_value=list(FAKE_PAPERS)),
        ),
        patch(
            "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
            AsyncMock(side_effect=lambda p, **kw: p),
        ),
        patch(
            "perspicacite.search.screening.screen_papers_rerank",
            AsyncMock(return_value=list(FAKE_SCREEN)),
        ),
    ]


@pytest.mark.asyncio
async def test_basic_web_fallback_uses_pipeline():
    """basic._web_fallback_papers must route through resolve_papers_pipeline."""
    from perspicacite.rag.modes.basic import _web_fallback_papers

    mock_pipeline = AsyncMock(return_value=list(FAKE_PAPERS))

    with patch("perspicacite.rag.resolve_papers.resolve_papers_pipeline", mock_pipeline):
        result = await _web_fallback_papers(
            query="metabolomics",
            databases=["openalex"],
            max_docs=5,
            app_state=None,
        )

    mock_pipeline.assert_called_once()
    call_kw = mock_pipeline.call_args.kwargs
    assert call_kw["query"] == "metabolomics"
    assert call_kw["enrich"] is True
    assert call_kw["rerank"] is True
    # Should return 3 candidate dicts (one per paper)
    assert len(result) == 3
    assert all(isinstance(c, dict) for c in result)
    assert result[0]["title"] == "Paper 1"


@pytest.mark.asyncio
async def test_literature_survey_broad_search_uses_pipeline():
    """literature_survey._broad_search must route through resolve_papers_pipeline."""
    from perspicacite.rag.modes.literature_survey import LiteratureSurveyRAGMode

    mock_pipeline = AsyncMock(return_value=list(FAKE_PAPERS))

    # Minimal mode instance — _broad_search only needs config for the
    # optimizer gate, which we'll stub out via global app_state.
    mode = LiteratureSurveyRAGMode.__new__(LiteratureSurveyRAGMode)
    mode.config = MagicMock()

    stub_state = MagicMock()
    stub_state.config = None  # disables optimizer path

    with patch("perspicacite.web.state.app_state", stub_state), patch(
        "perspicacite.rag.resolve_papers.resolve_papers_pipeline", mock_pipeline
    ):
        result = await mode._broad_search(
            query="metabolomics",
            databases=["openalex"],
        )

    mock_pipeline.assert_called_once()
    call_kw = mock_pipeline.call_args.kwargs
    assert call_kw["enrich"] is True
    assert call_kw["rerank"] is False  # survey skips rerank
    assert call_kw["optimize_query"] is False  # already optimised above
    assert result == FAKE_PAPERS


@pytest.mark.asyncio
async def test_mcp_web_search_uses_pipeline():
    """The MCP web_search tool must route through resolve_papers_pipeline."""
    from perspicacite.mcp.server import web_search

    mock_pipeline = AsyncMock(return_value=list(FAKE_PAPERS))

    with patch("perspicacite.rag.resolve_papers.resolve_papers_pipeline", mock_pipeline):
        out = await web_search(query="metabolomics", databases=["openalex"], max_results=5)

    mock_pipeline.assert_called_once()
    call_kw = mock_pipeline.call_args.kwargs
    assert call_kw["query"] == "metabolomics"
    assert call_kw["enrich"] is True
    assert call_kw["rerank"] is True

    data = json.loads(out)
    assert len(data["papers"]) == 3
    titles = {p["title"] for p in data["papers"]}
    assert titles == {"Paper 1", "Paper 2", "Paper 3"}
    assert "telemetry_summary" in data
