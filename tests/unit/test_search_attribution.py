"""Unit tests for multi-provider dedup merge and attribution union (Issue 4).

Proves that DomainAwareAggregator correctly unions discovery_sources when two
providers return the same paper.

All I/O is faked — no network calls, no DB.
"""
from __future__ import annotations

import pytest

from perspicacite.models.papers import Paper, PaperSource
from perspicacite.search.domain_aggregator import DomainAwareAggregator

# ---------------------------------------------------------------------------
# Minimal async fake provider — mirrors the duck-type contract used by
# DomainAwareAggregator._call_provider():
#   - name: str
#   - domains: list[str]
#   - async search(query, max_results, year_min, year_max, **kwargs) -> list[Paper]
# ---------------------------------------------------------------------------

class _FakeProvider:
    def __init__(
        self,
        name: str,
        papers: list[Paper],
        domains: list[str] | None = None,
    ) -> None:
        self.name = name
        self.domains = domains if domains is not None else ["general"]
        self._papers = papers

    async def search(
        self,
        query: str,
        max_results: int = 25,
        year_min: int | None = None,
        year_max: int | None = None,
        **kwargs,
    ) -> list[Paper]:
        return list(self._papers)


def _paper(doi: str, title: str = "A Paper") -> Paper:
    return Paper(id=f"doi:{doi}", title=title, doi=doi, source=PaperSource.PUBMED)


def _paper_no_doi(title: str) -> Paper:
    return Paper(id=f"nodoi:{title[:20]}", title=title, source=PaperSource.PUBMED)


# ---------------------------------------------------------------------------
# Test 1 — Same DOI from two providers → discovery_sources contains both names
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_doi_discovery_sources_union_both_providers():
    """When two providers return the same DOI, the merged paper's
    discovery_sources must contain BOTH provider names, not just the first."""
    shared_doi = "10.1234/shared"
    pubmed = _FakeProvider("pubmed", [_paper(shared_doi)])
    dblp = _FakeProvider("dblp_sparql", [_paper(shared_doi)])

    agg = DomainAwareAggregator([pubmed, dblp], provider_timeout_s=5.0)
    results = await agg.search("test query", max_results=10)

    # Exactly one paper after dedup
    assert len(results) == 1
    sources = results[0].discovery_sources
    assert "pubmed" in sources, f"expected 'pubmed' in {sources}"
    assert "dblp_sparql" in sources, f"expected 'dblp_sparql' in {sources}"


# ---------------------------------------------------------------------------
# Test 3 — Different DOIs from two providers → two distinct papers returned
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_different_dois_no_spurious_dedup():
    """When two providers return papers with different DOIs, both papers must be
    present in results — the aggregator must NOT spuriously deduplicate them."""
    pubmed = _FakeProvider("pubmed", [_paper("10.1234/paper-A")])
    dblp = _FakeProvider("dblp_sparql", [_paper("10.5678/paper-B")])

    agg = DomainAwareAggregator([pubmed, dblp], provider_timeout_s=5.0)
    results = await agg.search("test query", max_results=10)

    assert len(results) == 2, f"expected 2 distinct papers, got {len(results)}"
    dois = {r.doi for r in results}
    assert "10.1234/paper-A" in dois
    assert "10.5678/paper-B" in dois


# ---------------------------------------------------------------------------
# Test 4 — No DOI, same title from two providers → discovery_sources union
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_doi_same_title_discovery_sources_union():
    """When two providers return papers with no DOI but the same title,
    the dedup-by-title path must still union discovery_sources for both providers."""
    shared_title = "Microbiome diversity in Antarctic soils"
    pubmed = _FakeProvider("pubmed", [_paper_no_doi(shared_title)])
    dblp = _FakeProvider("dblp_sparql", [_paper_no_doi(shared_title)])

    agg = DomainAwareAggregator([pubmed, dblp], provider_timeout_s=5.0)
    results = await agg.search("microbiome Antarctica", max_results=10)

    # Exactly one paper after title-based dedup
    assert len(results) == 1
    sources = results[0].discovery_sources
    assert "pubmed" in sources, f"expected 'pubmed' in {sources}"
    assert "dblp_sparql" in sources, f"expected 'dblp_sparql' in {sources}"


# ---------------------------------------------------------------------------
# Test 5 — discovery_sources not duplicated when single provider returns same DOI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_provider_discovery_source_not_duplicated():
    """A single provider should appear exactly once in discovery_sources,
    even if its own list contained the DOI already (guard against double-append)."""
    pubmed = _FakeProvider("pubmed", [_paper("10.1234/solo")])

    agg = DomainAwareAggregator([pubmed], provider_timeout_s=5.0)
    results = await agg.search("any", max_results=10)

    assert len(results) == 1
    sources = results[0].discovery_sources
    assert sources.count("pubmed") == 1, (
        f"'pubmed' appeared {sources.count('pubmed')} times in {sources}"
    )


# ---------------------------------------------------------------------------
# Test 6 — Ordering: discovery_sources preserves first-seen provider first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dedup_merge_preserves_first_provider_ordering():
    """The provider that was registered first in the aggregator's list should
    appear first in discovery_sources after a DOI-based dedup merge."""
    shared_doi = "10.1234/ordered"
    first = _FakeProvider("first_provider", [_paper(shared_doi)])
    second = _FakeProvider("second_provider", [_paper(shared_doi)])

    agg = DomainAwareAggregator([first, second], provider_timeout_s=5.0)
    results = await agg.search("any", max_results=10)

    assert len(results) == 1
    sources = results[0].discovery_sources
    assert sources[0] == "first_provider", (
        f"expected 'first_provider' at index 0, got {sources}"
    )
    assert sources[1] == "second_provider", (
        f"expected 'second_provider' at index 1, got {sources}"
    )
