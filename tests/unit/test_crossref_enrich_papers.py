"""Unit tests for pipeline.enrichment.crossref_enrich.enrich_papers."""
from unittest.mock import patch

import pytest

from perspicacite.models.papers import Paper, PaperSource
from perspicacite.pipeline.enrichment.crossref_enrich import enrich_papers

_MODULE = "perspicacite.pipeline.enrichment.crossref_enrich"


async def _fake_canonicalize_fills_abstract(candidates, *, concurrency=None):
    for c in candidates:
        if c.get("doi") and not c.get("abstract"):
            c["abstract"] = "Crossref abstract text"
            c.setdefault("enrichment_sources", []).append("crossref")


async def _fake_canonicalize_noop(candidates, *, concurrency=None):
    pass


@pytest.mark.asyncio
async def test_enrich_papers_fills_missing_abstract():
    """Paper with DOI but no abstract gets Crossref's abstract."""
    p = Paper(
        id="doi:10.1234/x",
        title="Original Title",
        doi="10.1234/x",
        source=PaperSource.GOOGLE_SCHOLAR,
    )
    with patch(f"{_MODULE}.canonicalize_candidates", side_effect=_fake_canonicalize_fills_abstract):
        result = await enrich_papers([p])

    assert result[0].abstract == "Crossref abstract text"
    assert "crossref" in result[0].enrichment_sources


@pytest.mark.asyncio
async def test_enrich_papers_does_not_overwrite_existing_abstract():
    """If the Paper already has an abstract, Crossref's doesn't override."""
    p = Paper(
        id="doi:10.1234/x", title="Title", doi="10.1234/x",
        abstract="Original abstract from provider",
    )
    with patch(f"{_MODULE}.canonicalize_candidates", side_effect=_fake_canonicalize_fills_abstract):
        await enrich_papers([p])

    assert p.abstract == "Original abstract from provider"


@pytest.mark.asyncio
async def test_enrich_papers_skips_papers_without_doi():
    """No DOI → canonicalize runs but nothing to enrich → no enrichment_sources tag."""
    p = Paper(id="x", title="Untitled, no DOI")
    with patch(f"{_MODULE}.canonicalize_candidates", side_effect=_fake_canonicalize_noop):
        await enrich_papers([p])
    assert p.enrichment_sources == []


@pytest.mark.asyncio
async def test_enrich_papers_empty_list_short_circuits():
    """Empty list returns empty list, no HTTP."""
    assert await enrich_papers([]) == []
