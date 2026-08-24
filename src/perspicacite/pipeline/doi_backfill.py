"""Verified title → DOI backfill for search hits that carry no DOI.

Scrape-backed providers — Google Scholar first among them — return
cards with a title, an author line and a landing-page URL, but rarely a
DOI. ``search_to_kb``'s ingest filter requires one, so those hits were
counted under ``no_doi`` and silently never reached a KB: a Google
Scholar query could report ``candidates=0`` while having found eight
perfectly good papers.

This module closes that gap by running the existing
:func:`~perspicacite.pipeline.download.title_resolver.resolve_doi_from_title`
cascade over the DOI-less hits *before* the filter runs. Three
properties matter, in this order:

1. **Verified, never guessed.** Every DOI comes back from a metadata
   API that also returned the candidate's title, authors and year, and
   ``title_resolver`` accepts it only on author-token overlap + year
   ±1 + title Jaccard similarity. The Google-Scholar browser tier
   additionally confirms each scraped DOI through Crossref. A wrong
   DOI poisons a bibliography, which is worse than ingesting nothing —
   so a miss is always preferred to a loose match.
2. **Bounded.** Resolution costs one to five HTTP round-trips per
   paper. Callers get an attempt budget and a concurrency cap, only
   DOI-less hits are attempted, and a per-process memo remembers both
   hits *and* misses so repeated runs over overlapping result sets
   don't re-pay.
3. **Never silent.** Every outcome lands in :class:`BackfillStats`,
   which the caller surfaces in its report. Papers skipped because the
   budget ran out are counted separately from papers that genuinely
   could not be resolved.

The step is opt-in (``--resolve-missing-dois`` on the CLI): it changes
what enters a KB, so it should be a deliberate choice, not a default.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.doi_backfill")

DEFAULT_BUDGET = 25
DEFAULT_CONCURRENCY = 4

# Titles shorter than this can't be matched safely — ``title_resolver``
# short-circuits on them too. Counted as ``no_title`` rather than as a
# failed attempt so the success rate stays meaningful.
MIN_TITLE_CHARS = 10

# Per-process memo of title → DOI (``None`` = confirmed miss). Bounded
# because a long-lived server would otherwise grow it without limit.
_CACHE_MAX_ENTRIES = 2048
_resolution_cache: dict[tuple[str, int | None], str | None] = {}


@dataclass
class BackfillStats:
    """Outcome counts for one backfill run. Every DOI-less hit lands in
    exactly one of ``resolved`` / ``unresolved`` / ``no_title`` /
    ``over_budget``; ``cache_hits`` is a subset of the first two."""

    missing_doi: int = 0
    attempted: int = 0
    resolved: int = 0
    unresolved: int = 0
    no_title: int = 0
    over_budget: int = 0
    cache_hits: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def clear_cache() -> None:
    """Drop the per-process resolution memo. For tests and long runs."""
    _resolution_cache.clear()


def _paper_doi(paper: Any) -> str:
    return (getattr(paper, "doi", None) or "").strip()


def _paper_title(paper: Any) -> str:
    return (getattr(paper, "title", None) or "").strip()


def _author_names(paper: Any) -> list[str]:
    """Flatten a paper's authors to plain name strings.

    ``Paper.authors`` holds :class:`~perspicacite.models.papers.Author`
    objects, but search adapters and test doubles sometimes carry bare
    strings; ``title_resolver`` wants strings either way.
    """
    names: list[str] = []
    for author in getattr(paper, "authors", None) or []:
        name = getattr(author, "name", None) or (author if isinstance(author, str) else "")
        if name and str(name).strip():
            names.append(str(name).strip())
    return names


def _cache_key(title: str, year: Any) -> tuple[str, int | None]:
    normalized = " ".join(title.lower().split())
    try:
        year_int = int(str(year)[:4]) if year is not None else None
    except (TypeError, ValueError):
        year_int = None
    return normalized, year_int


def _cache_store(key: tuple[str, int | None], doi: str | None) -> None:
    """Insert into the memo, evicting the oldest entry when full."""
    if len(_resolution_cache) >= _CACHE_MAX_ENTRIES:
        del _resolution_cache[next(iter(_resolution_cache))]
    _resolution_cache[key] = doi


def _with_resolved_doi(paper: Any, doi: str) -> Any:
    """Return a copy of ``paper`` carrying ``doi``.

    The original is left untouched — callers hand us the aggregator's
    hit list and shouldn't see it mutate. ``id`` is upgraded too when
    it held a provider placeholder (``scholar:ab12…``), so the paper
    matches what the provider would have produced had it found the DOI
    itself. ``title_resolver`` is recorded as an enrichment source.
    """
    sources = list(getattr(paper, "enrichment_sources", None) or [])
    if "title_resolver" not in sources:
        sources.append("title_resolver")
    updates: dict[str, Any] = {"doi": doi, "enrichment_sources": sources}
    if not str(getattr(paper, "id", "") or "").startswith("10."):
        updates["id"] = doi
    if hasattr(paper, "model_copy"):
        return paper.model_copy(update=updates)
    import copy

    clone = copy.copy(paper)
    for key, value in updates.items():
        setattr(clone, key, value)
    return clone


async def _resolve_one(
    paper: Any,
    *,
    http_client: Any,
    enable_browser: bool,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, bool]:
    """Resolve one paper's DOI, consulting and filling the memo.

    Returns ``(doi_or_none, served_from_cache)``. Never raises: a
    resolver blow-up is recorded as a miss, because one flaky lookup
    must not abort ingest of the rest of the result set.
    """
    title = _paper_title(paper)
    year = getattr(paper, "year", None)
    key = _cache_key(title, year)
    if key in _resolution_cache:
        return _resolution_cache[key], True

    from perspicacite.pipeline.download.title_resolver import resolve_doi_from_title

    async with semaphore:
        try:
            doi = await resolve_doi_from_title(
                title,
                _author_names(paper),
                year,
                http_client=http_client,
                enable_browser=enable_browser,
            )
        except Exception as exc:
            logger.warning(
                "doi_backfill_resolver_error",
                title=title[:80],
                error=str(exc),
            )
            doi = None
    _cache_store(key, doi)
    return doi, False


def _split_by_budget(
    pending: list[tuple[int, Any]], budget: int
) -> tuple[list[tuple[int, Any]], int, int]:
    """Split DOI-less papers into ``(resolvable, no_title, over_budget)``.

    Papers whose title is too short to match safely are separated from
    those the budget simply couldn't reach — the two failures deserve
    different answers from the operator.
    """
    resolvable: list[tuple[int, Any]] = []
    no_title = over_budget = 0
    for index, paper in pending:
        if len(_paper_title(paper)) < MIN_TITLE_CHARS:
            no_title += 1
        elif len(resolvable) >= budget:
            over_budget += 1
        else:
            resolvable.append((index, paper))
    return resolvable, no_title, over_budget


async def backfill_missing_dois(
    papers: list[Any],
    *,
    budget: int = DEFAULT_BUDGET,
    concurrency: int = DEFAULT_CONCURRENCY,
    enable_browser: bool = False,
    http_client: Any | None = None,
) -> tuple[list[Any], BackfillStats]:
    """Fill in verified DOIs for the DOI-less papers in ``papers``.

    Args:
        papers: Search hits, in the order they should stay in. Papers
            that already carry a DOI are passed through untouched and
            cost nothing.
        budget: Max number of papers to look up in this run. Hits past
            the budget keep their empty DOI and are counted under
            ``over_budget`` — never dropped quietly.
        concurrency: Max simultaneous resolutions.
        enable_browser: Add ``title_resolver``'s headless-Chromium
            Google Scholar tier after the HTTP tiers. Slower and needs
            the ``browser`` extra; off by default.
        http_client: Optional ``httpx.AsyncClient`` to reuse. One is
            created and closed here when omitted.

    Returns:
        ``(papers, stats)`` — a new list in the input order with
        resolved DOIs filled in, and the :class:`BackfillStats` counts.
    """
    stats = BackfillStats()
    pending = [(i, p) for i, p in enumerate(papers) if not _paper_doi(p)]
    stats.missing_doi = len(pending)
    if not pending:
        return list(papers), stats

    resolvable, stats.no_title, stats.over_budget = _split_by_budget(pending, budget)
    stats.attempted = len(resolvable)

    if not resolvable:
        _log_outcome(stats, enable_browser)
        return list(papers), stats

    client, owns_client = _ensure_client(http_client)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    try:
        outcomes = await asyncio.gather(*[
            _resolve_one(
                paper,
                http_client=client,
                enable_browser=enable_browser,
                semaphore=semaphore,
            )
            for _, paper in resolvable
        ])
    finally:
        if owns_client:
            await client.aclose()

    out = list(papers)
    for (index, paper), (doi, from_cache) in zip(resolvable, outcomes, strict=True):
        stats.cache_hits += int(from_cache)
        if doi:
            out[index] = _with_resolved_doi(paper, doi)
            stats.resolved += 1
        else:
            stats.unresolved += 1
    _log_outcome(stats, enable_browser)
    return out, stats


def _ensure_client(http_client: Any | None) -> tuple[Any, bool]:
    """Return ``(client, we_created_it)``."""
    if http_client is not None:
        return http_client, False
    import httpx

    return httpx.AsyncClient(timeout=30.0, follow_redirects=True), True


def _log_outcome(stats: BackfillStats, enable_browser: bool) -> None:
    """Emit one structured line per run — the anti-silent-failure hook."""
    logger.info(
        "doi_backfill_done",
        missing_doi=stats.missing_doi,
        attempted=stats.attempted,
        resolved=stats.resolved,
        unresolved=stats.unresolved,
        no_title=stats.no_title,
        over_budget=stats.over_budget,
        cache_hits=stats.cache_hits,
        browser_tier=enable_browser,
    )
    if stats.over_budget:
        logger.warning(
            "doi_backfill_budget_exhausted",
            not_attempted=stats.over_budget,
            advice="raise --resolve-doi-budget to look up the remaining hits",
        )
