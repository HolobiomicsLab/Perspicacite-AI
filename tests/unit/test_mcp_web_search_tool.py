"""Unit tests for the new web_search MCP tool."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from perspicacite.mcp.server import web_search
from perspicacite.models.papers import Author, Paper
from perspicacite.search.screening import ScreenResult


@pytest.mark.asyncio
async def test_web_search_returns_serialised_papers():
    fake = [
        Paper(
            id="doi:10.1/x", title="Paper 1",
            authors=[Author(name="A. Author")],
            year=2024, doi="10.1/x", abstract="abs1",
        ),
        Paper(id="doi:10.1/y", title="Paper 2", doi="10.1/y"),
    ]
    # When enrich=True the pipeline calls enrich_papers, then screen_papers_rerank.
    # Patch both so we get deterministic results without loading ML models.
    fake_screen = [
        ScreenResult(item={"_paper": fake[0], "title": "Paper 1", "abstract": "abs1"}, score=0.9, kept=True),
        ScreenResult(item={"_paper": fake[1], "title": "Paper 2", "abstract": ""}, score=0.4, kept=True),
    ]
    with patch(
        "perspicacite.rag.web_search.run_web_aggregator_search",
        AsyncMock(return_value=fake),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        AsyncMock(side_effect=lambda p, **kw: p),
    ), patch(
        "perspicacite.search.screening.screen_papers_rerank",
        AsyncMock(return_value=fake_screen),
    ):
        out = await web_search(query="q", databases=["openalex"])
    data = json.loads(out)
    assert len(data["papers"]) == 2
    titles = {p["title"] for p in data["papers"]}
    assert "Paper 1" in titles
    assert data["papers"][0]["doi"] == "10.1/x"  # highest score lands first
    assert "telemetry_summary" in data


@pytest.mark.asyncio
async def test_web_search_skips_enrich_when_disabled():
    fake = [Paper(id="x", title="t")]
    mock_enrich = AsyncMock(side_effect=lambda p, **kw: p)
    with patch(
        "perspicacite.rag.web_search.run_web_aggregator_search",
        AsyncMock(return_value=fake),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        mock_enrich,
    ):
        await web_search(query="q", enrich=False)
    mock_enrich.assert_not_called()


@pytest.mark.asyncio
async def test_web_search_discovery_sources_from_typed_fields():
    """Regression test: discovery_sources and enrichment_sources must be read
    from the typed Paper fields, not from p.metadata['sources'] (which is
    permanently empty after Task 3.6 removed legacy metadata writes)."""
    fake = [
        Paper(
            id="doi:10.1/z",
            title="Typed Sources Paper",
            doi="10.1/z",
            discovery_sources=["openalex", "pubmed"],
            enrichment_sources=["crossref"],
        ),
    ]
    fake_screen = [
        ScreenResult(
            item={"_paper": fake[0], "title": "Typed Sources Paper", "abstract": ""},
            score=0.8,
            kept=True,
        ),
    ]
    with patch(
        "perspicacite.rag.web_search.run_web_aggregator_search",
        AsyncMock(return_value=fake),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        AsyncMock(side_effect=lambda p, **kw: p),
    ), patch(
        "perspicacite.search.screening.screen_papers_rerank",
        AsyncMock(return_value=fake_screen),
    ):
        out = await web_search(query="q", databases=["openalex"])
    data = json.loads(out)
    assert len(data["papers"]) == 1
    p = data["papers"][0]
    assert p["discovery_sources"] == ["openalex", "pubmed"], (
        "discovery_sources must come from Paper.discovery_sources, not metadata['sources']"
    )
    assert p["enrichment_sources"] == ["crossref"], (
        "enrichment_sources must come from Paper.enrichment_sources, not metadata['enrichment_sources']"
    )


@pytest.mark.asyncio
async def test_web_search_error_response():
    with patch(
        "perspicacite.rag.web_search.run_web_aggregator_search",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        out = await web_search(query="q")
    data = json.loads(out)
    assert "error" in data
    assert data["papers"] == []
