from __future__ import annotations

import pytest

from perspicacite.models.papers import Paper, PaperSource
from perspicacite.search.domain_aggregator import DomainAwareAggregator, ProviderHealthTracker


def _paper(doi: str, title: str = "Title") -> Paper:
    return Paper(id=doi, title=title, doi=doi, source=PaperSource.PUBMED)


class _Provider:
    def __init__(
        self,
        name: str,
        papers: list[Paper],
        domains: list[str] | None = None,
        tier: str = "reliable",
        retry: int = 0,
        fail: bool = False,
    ):
        self.name = name
        self.description = name
        self.domains = domains or ["general"]
        self.tier = tier
        self.retry = retry
        self._papers = papers
        self._fail = fail
        self.call_count = 0

    async def search(self, query, max_results=20, year_min=None, year_max=None, **kwargs):
        self.call_count += 1
        if self._fail:
            raise RuntimeError("provider failed")
        return self._papers


class _YearRecordingProvider:
    """Records the year window it was called with and returns papers of mixed years."""

    def __init__(self, papers: list[Paper]):
        self.name = "gen"
        self.description = "gen"
        self.domains = ["general"]
        self.tier = "reliable"
        self.retry = 0
        self._papers = papers
        self.seen_year_min: int | None = -1  # sentinel: never called
        self.seen_year_max: int | None = -1

    async def search(self, query, max_results=20, year_min=None, year_max=None, **kwargs):
        self.seen_year_min, self.seen_year_max = year_min, year_max
        return self._papers


@pytest.mark.asyncio
async def test_year_window_enforced_post_merge_not_forwarded():
    """A year_max keeps in-range + unknown-year papers, drops post-cutoff ones, and is
    NOT forwarded to providers (fetch on the fast no-year path, then filter)."""
    papers = [
        Paper(id="10.1/old", title="Old", doi="10.1/old", source=PaperSource.PUBMED, year=2018),
        Paper(id="10.1/new", title="New", doi="10.1/new", source=PaperSource.PUBMED, year=2024),
        Paper(id="10.1/none", title="NoYr", doi="10.1/none", source=PaperSource.PUBMED, year=None),
    ]
    p = _YearRecordingProvider(papers)
    agg = DomainAwareAggregator([p], provider_timeout_s=5.0)
    results = await agg.search("any query", year_max=2023)

    dois = {r.doi for r in results}
    assert dois == {"10.1/old", "10.1/none"}  # 2024 dropped; unknown-year kept
    # The year window is enforced locally, never pushed to the provider (avoids the
    # SciLEx per-year fan-out that returned zero results).
    assert p.seen_year_min is None and p.seen_year_max is None


@pytest.mark.asyncio
async def test_basic_routing_general_provider():
    p = _Provider("gen", [_paper("10.1/a")])
    agg = DomainAwareAggregator([p], provider_timeout_s=5.0)
    results = await agg.search("any query")
    assert len(results) == 1
    assert results[0].doi == "10.1/a"


@pytest.mark.asyncio
async def test_domain_provider_included_when_query_matches():
    bio = _Provider("bio", [_paper("10.1/bio")], domains=["biomedical"])
    phys = _Provider("phys", [_paper("10.1/phys")], domains=["physics"])
    agg = DomainAwareAggregator([bio, phys], provider_timeout_s=5.0)
    results = await agg.search("gene expression cancer")
    dois = {r.doi for r in results}
    assert "10.1/bio" in dois
    assert "10.1/phys" not in dois


@pytest.mark.asyncio
async def test_domain_provider_excluded_when_query_doesnt_match():
    phys = _Provider("phys", [_paper("10.1/phys")], domains=["physics"])
    agg = DomainAwareAggregator([phys], provider_timeout_s=5.0)
    results = await agg.search("gene expression cancer microbiome")
    assert results == []


@pytest.mark.asyncio
async def test_dedup_by_doi():
    p1 = _Provider("a", [_paper("10.1/dup"), _paper("10.1/unique")])
    p2 = _Provider("b", [_paper("10.1/dup")])
    agg = DomainAwareAggregator([p1, p2], provider_timeout_s=5.0)
    results = await agg.search("any")
    dois = [r.doi for r in results]
    assert dois.count("10.1/dup") == 1
    assert "10.1/unique" in dois


@pytest.mark.asyncio
async def test_dedup_by_title_when_no_doi():
    p1 = _Provider("a", [Paper(id="x", title="Same Title", source=PaperSource.PUBMED)])
    p2 = _Provider("b", [Paper(id="y", title="Same Title", source=PaperSource.PUBMED)])
    agg = DomainAwareAggregator([p1, p2], provider_timeout_s=5.0)
    results = await agg.search("any")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_failed_provider_returns_others():
    good = _Provider("good", [_paper("10.1/good")])
    bad = _Provider("bad", [], fail=True)
    agg = DomainAwareAggregator([good, bad], provider_timeout_s=5.0)
    results = await agg.search("any")
    assert len(results) == 1
    assert results[0].doi == "10.1/good"


@pytest.mark.asyncio
async def test_max_results_respected():
    papers = [_paper(f"10.1/{i}") for i in range(30)]
    p = _Provider("big", papers)
    agg = DomainAwareAggregator([p], provider_timeout_s=5.0)
    results = await agg.search("any", max_results=10)
    assert len(results) == 10


def test_circuit_breaker_trips_after_3_failures():
    tracker = ProviderHealthTracker()
    for _ in range(3):
        tracker.record_failure("prov")
    assert not tracker.is_available("prov")


def test_circuit_breaker_resets_on_success():
    tracker = ProviderHealthTracker()
    tracker.record_failure("prov")
    tracker.record_failure("prov")
    tracker.record_success("prov")
    tracker.record_failure("prov")  # counter reset, only 1 failure now
    assert tracker.is_available("prov")


@pytest.mark.asyncio
async def test_circuit_broken_provider_skipped():
    bad = _Provider("bad", [], fail=True)
    good = _Provider("good", [_paper("10.1/g")])
    agg = DomainAwareAggregator([bad, good], provider_timeout_s=5.0)
    # Trip the circuit manually
    for _ in range(3):
        agg._health.record_failure("bad")
    results = await agg.search("any")
    assert bad.call_count == 0
    assert len(results) == 1


def test_available_false_when_no_providers():
    agg = DomainAwareAggregator([])
    assert not agg.available


def test_available_true_when_providers_registered():
    p = _Provider("p", [])
    agg = DomainAwareAggregator([p])
    assert agg.available


@pytest.mark.asyncio
async def test_sources_attribution_populated():
    p1 = _Provider("provA", [_paper("10.1/x")])
    p2 = _Provider("provB", [_paper("10.1/x")])  # same DOI → dedup, source merged
    agg = DomainAwareAggregator([p1, p2], provider_timeout_s=5.0)
    results = await agg.search("any")
    assert len(results) == 1
    sources = results[0].discovery_sources
    assert "provA" in sources
    assert "provB" in sources


@pytest.mark.asyncio
async def test_sources_attribution_unique_papers():
    p1 = _Provider("provA", [_paper("10.1/a")])
    p2 = _Provider("provB", [_paper("10.1/b")])
    agg = DomainAwareAggregator([p1, p2], provider_timeout_s=5.0)
    results = await agg.search("any")
    assert len(results) == 2
    by_doi = {r.doi: r for r in results}
    assert by_doi["10.1/a"].discovery_sources == ["provA"]
    assert by_doi["10.1/b"].discovery_sources == ["provB"]


@pytest.mark.asyncio
async def test_retry_counts_as_one_circuit_failure():
    """A provider with retry=2 that fails all attempts counts as one failure, not three."""
    bad = _Provider("bad", [], retry=2, fail=True)
    agg = DomainAwareAggregator([bad], provider_timeout_s=5.0)
    await agg.search("any")
    # One logical failure recorded — circuit should NOT be tripped yet.
    assert agg._health.is_available("bad")
    # Two more logical failures (= 3 total) should trip it.
    await agg.search("any")
    await agg.search("any")
    assert not agg._health.is_available("bad")


@pytest.mark.asyncio
async def test_apis_kwarg_not_forwarded_to_non_scilex_provider():
    """apis kwarg must not reach providers other than scilex (they lack **kwargs handling)."""
    received_kwargs: dict = {}

    class StrictProvider(_Provider):
        async def search(self, query, max_results=20, year_min=None, year_max=None):  # no **kwargs
            received_kwargs.clear()
            return self._papers

    strict = StrictProvider("other", [_paper("10.1/x")])
    agg = DomainAwareAggregator([strict], provider_timeout_s=5.0)
    # Should not raise TypeError even though apis is passed and provider has no **kwargs.
    results = await agg.search("any", apis=["pubmed"])
    assert len(results) == 1


@pytest.mark.asyncio
async def test_collect_deadline_forwarded_to_scilex():
    """SciLEx gets a soft collection deadline (a fraction of its wait_for
    budget) so a slow backend returns partial results instead of nothing."""
    captured: dict = {}

    class CapturingScilex(_Provider):
        async def search(self, query, max_results=20, year_min=None, year_max=None, **kwargs):
            captured.update(kwargs)
            return self._papers

    sx = CapturingScilex("scilex", [_paper("10.1/x")], tier="flaky")
    agg = DomainAwareAggregator([sx], provider_timeout_s=10.0)
    await agg.search("any")
    # flaky tier => 10.0 * 2.25 = 22.5s wait_for; deadline is 0.8 of that.
    assert captured.get("collect_deadline_s") == pytest.approx(22.5 * 0.8)


@pytest.mark.asyncio
async def test_collect_deadline_not_forwarded_to_non_scilex():
    """Only SciLEx receives collect_deadline_s; other providers don't."""
    captured: dict = {}

    class Capturing(_Provider):
        async def search(self, query, max_results=20, year_min=None, year_max=None, **kwargs):
            captured.update(kwargs)
            return self._papers

    other = Capturing("europepmc", [_paper("10.1/x")])
    agg = DomainAwareAggregator([other], provider_timeout_s=10.0)
    await agg.search("any")
    assert "collect_deadline_s" not in captured
