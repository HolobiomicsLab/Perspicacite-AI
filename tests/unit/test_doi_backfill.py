"""Tests for ``perspicacite.pipeline.doi_backfill``.

Google Scholar hits arrive with a title and no DOI, and the ingest
filter drops them as ``no_doi``. The backfill step recovers a verified
DOI first. What matters here is not the resolver itself (covered by
``test_title_resolver.py``) but the bounding around it: only DOI-less
hits are looked up, the budget and concurrency caps hold, misses are
cached, and nothing is ever dropped silently.
"""
from __future__ import annotations

import asyncio

import pytest

from perspicacite.models.papers import Author, Paper, PaperSource
from perspicacite.pipeline import doi_backfill
from perspicacite.pipeline.doi_backfill import backfill_missing_dois


@pytest.fixture(autouse=True)
def _clean_cache():
    """The memo is per-process; keep tests independent of each other."""
    doi_backfill.clear_cache()
    yield
    doi_backfill.clear_cache()


def scholar_hit(title: str, *, doi: str | None = None, year: int = 2021) -> Paper:
    """A Google-Scholar-shaped hit: real title, no DOI unless given."""
    return Paper(
        id=doi or f"scholar:{abs(hash(title)) % 10**8}",
        title=title,
        authors=[Author(name="J Dupont"), Author(name="A Martin")],
        year=year,
        doi=doi,
        source=PaperSource.GOOGLE_SCHOLAR,
    )


class FakeResolver:
    """Stand-in for ``resolve_doi_from_title`` that records its calls."""

    def __init__(self, answers: dict[str, str | None] | None = None, delay: float = 0.0):
        self.answers = answers or {}
        self.delay = delay
        self.calls: list[str] = []
        self.concurrent = 0
        self.peak_concurrent = 0

    async def __call__(self, title, authors, year, *, http_client, enable_browser=False):
        self.calls.append(title)
        self.concurrent += 1
        self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.answers.get(title)
        finally:
            self.concurrent -= 1


@pytest.fixture
def resolver(monkeypatch):
    """Patch the resolver at its source module (imported lazily)."""
    fake = FakeResolver()
    monkeypatch.setattr(
        "perspicacite.pipeline.download.title_resolver.resolve_doi_from_title",
        fake,
    )
    return fake


async def test_papers_that_already_have_a_doi_are_never_looked_up(resolver):
    papers = [scholar_hit("Non-uniform sampling in fast 2D NMR", doi="10.1021/acs.1c00001")]

    out, stats = await backfill_missing_dois(papers)

    assert resolver.calls == []
    assert stats.missing_doi == 0
    assert stats.attempted == 0
    assert out == papers


async def test_resolved_doi_is_written_onto_a_copy(resolver):
    title = "Non-uniform sampling in fast 2D NMR spectroscopy"
    resolver.answers[title] = "10.1021/acs.analchem.1c00001"
    original = scholar_hit(title)

    out, stats = await backfill_missing_dois([original])

    assert stats.resolved == 1
    assert stats.unresolved == 0
    assert out[0].doi == "10.1021/acs.analchem.1c00001"
    # The placeholder id is upgraded so the hit looks like one the
    # provider itself had resolved.
    assert out[0].id == "10.1021/acs.analchem.1c00001"
    assert "title_resolver" in out[0].enrichment_sources
    # The aggregator's own object is left alone.
    assert original.doi is None
    assert original.id.startswith("scholar:")


async def test_unresolved_hit_is_counted_and_left_unchanged(resolver):
    paper = scholar_hit("A title no metadata API has ever indexed")

    out, stats = await backfill_missing_dois([paper])

    assert stats.attempted == 1
    assert stats.resolved == 0
    assert stats.unresolved == 1
    assert out[0].doi is None


async def test_budget_caps_lookups_and_reports_the_remainder(resolver):
    papers = [scholar_hit(f"Distinct paper title number {i}") for i in range(10)]

    _, stats = await backfill_missing_dois(papers, budget=3)

    assert len(resolver.calls) == 3
    assert stats.attempted == 3
    assert stats.over_budget == 7
    assert stats.missing_doi == 10


async def test_titles_too_short_to_match_are_not_attempted(resolver):
    papers = [scholar_hit("NMR"), scholar_hit("A perfectly usable long title here")]

    _, stats = await backfill_missing_dois(papers)

    assert stats.no_title == 1
    assert stats.attempted == 1
    assert resolver.calls == ["A perfectly usable long title here"]


async def test_concurrency_cap_is_respected(monkeypatch):
    fake = FakeResolver(delay=0.02)
    monkeypatch.setattr(
        "perspicacite.pipeline.download.title_resolver.resolve_doi_from_title",
        fake,
    )
    papers = [scholar_hit(f"Yet another distinct paper title {i}") for i in range(8)]

    await backfill_missing_dois(papers, concurrency=2)

    assert fake.peak_concurrent <= 2
    assert len(fake.calls) == 8


async def test_repeated_title_is_resolved_once_and_served_from_cache(resolver):
    title = "Fast 2D NMR with non-uniform sampling and compressed sensing"
    resolver.answers[title] = "10.1021/acs.analchem.1c00002"
    papers = [scholar_hit(title), scholar_hit(title)]

    out, stats = await backfill_missing_dois(papers)

    assert len(resolver.calls) == 1
    assert stats.cache_hits == 1
    assert stats.resolved == 2
    assert [p.doi for p in out] == ["10.1021/acs.analchem.1c00002"] * 2


async def test_misses_are_cached_too_so_they_are_not_re_paid(resolver):
    title = "A title that resolves to nothing at all, twice over"
    papers = [scholar_hit(title), scholar_hit(title)]

    _, stats = await backfill_missing_dois(papers)

    assert len(resolver.calls) == 1
    assert stats.cache_hits == 1
    assert stats.unresolved == 2


async def test_resolver_blowup_is_a_miss_not_a_crash(monkeypatch):
    async def exploding(*args, **kwargs):
        raise RuntimeError("crossref is down")

    monkeypatch.setattr(
        "perspicacite.pipeline.download.title_resolver.resolve_doi_from_title",
        exploding,
    )
    papers = [scholar_hit("A title whose lookup will raise an exception")]

    out, stats = await backfill_missing_dois(papers)

    assert stats.unresolved == 1
    assert out[0].doi is None


async def test_input_order_is_preserved_across_a_mixed_batch(resolver):
    resolver.answers["Second paper with a usable long title"] = "10.1000/second"
    papers = [
        scholar_hit("First paper already carrying a doi", doi="10.1000/first"),
        scholar_hit("Second paper with a usable long title"),
        scholar_hit("Third paper that will not resolve at all"),
    ]

    out, _ = await backfill_missing_dois(papers)

    assert [p.doi for p in out] == ["10.1000/first", "10.1000/second", None]


async def test_cache_evicts_oldest_entry_when_full(resolver, monkeypatch):
    monkeypatch.setattr(doi_backfill, "_CACHE_MAX_ENTRIES", 2)
    titles = [f"Distinctly worded paper title number {i}" for i in range(3)]

    await backfill_missing_dois([scholar_hit(t) for t in titles])

    assert len(doi_backfill._resolution_cache) == 2
