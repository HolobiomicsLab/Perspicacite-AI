"""Google Scholar search provider via headless Chromium.

Requires the ``[browser]`` optional dependency::

    uv pip install -e ".[browser]"
    playwright install chromium

The public API is the same as all other search providers:
``name``, ``domains``, ``tier``, ``retry`` class-level attributes and an
``async search(query, max_results, year_min, year_max)`` coroutine.

Playwright is imported lazily inside ``_render_and_extract_cards`` so the
module is importable even when the optional dep is absent.  Tests replace
``_render_and_extract_cards`` at the module level to avoid any browser.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
from typing import Any, ClassVar
from urllib.parse import quote

from perspicacite.logging import get_logger
from perspicacite.models.papers import Author, Paper, PaperSource

# Module-level sentinel returned by _render_and_extract_cards on CAPTCHA detection.
# Lets search() distinguish "CAPTCHA block" from "genuinely no results"
# via identity check (cards is _CAPTCHA_SENTINEL).
_CAPTCHA_SENTINEL: list[dict[str, str]] = []

logger = get_logger("perspicacite.search.google_scholar_playwright")

_SCHOLAR_BASE = "https://scholar.google.com/scholar"
_DOI_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"'>]+)")
# Generic DOI pattern that catches DOIs embedded in publisher landing-page
# URLs (Wiley, Springer, ACS, RSC, etc.) — e.g.
# https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202012345
# https://pubs.acs.org/doi/10.1021/acs.jnatprod.3c00468
_DOI_ANY_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>?#]+)")
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
_CITED_BY_RE = re.compile(r"^Cited by\s+(\d+)", re.IGNORECASE)


def _extract_citation_count(footer_text: str) -> int | None:
    """Parse 'Cited by N' from the ``.gs_fl`` footer text.

    Robust to varied whitespace and lead-text; returns None when no
    match (so Paper.citation_count stays None instead of 0, preserving
    the "unknown vs known-zero" distinction).
    """
    if not footer_text:
        return None
    m = _CITED_BY_RE.search(footer_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _build_scholar_url(
    query: str,
    year_min: int | None = None,
    year_max: int | None = None,
    start: int = 0,
) -> str:
    """Build a Google Scholar search URL."""
    params = f"q={quote(query)}"
    if year_min:
        params += f"&as_ylo={year_min}"
    if year_max:
        params += f"&as_yhi={year_max}"
    if start:
        params += f"&start={start}"
    return f"{_SCHOLAR_BASE}?{params}"


def _parse_meta_line(meta: str) -> tuple[str, str, int | None]:
    """Parse the Scholar ``gs_a`` metadata line.

    Input format:  "J Jumper, R Evans - Nature, 2021 - nature.com"
    Returns: (authors_str, venue_str, year_or_None)
    """
    parts = [p.strip() for p in meta.split(" - ")]
    authors = parts[0] if parts else ""
    venue = parts[1] if len(parts) > 1 else ""

    year: int | None = None
    m = _YEAR_RE.search(meta)
    if m:
        with contextlib.suppress(ValueError):
            year = int(m.group(0))

    return authors, venue, year


def _extract_doi_from_url(url: str) -> str | None:
    """Extract a bare DOI from a doi.org URL or publisher landing page.

    Tries doi.org URLs first (cleanest extraction), then falls back to
    any embedded DOI pattern in the URL path. This recovers DOIs from
    publisher links Google Scholar surfaces directly (Wiley, ACS, RSC,
    Springer, Frontiers, MDPI, …) without needing a separate Crossref
    title-search round-trip, which means more GS hits can be enriched
    by Crossref (the path that fills missing abstracts / journals).
    """
    if not url:
        return None
    m = _DOI_RE.match(url)
    if m:
        return m.group(1)
    # Fallback: scan the whole URL for an embedded DOI. We do a sanity
    # check on the captured suffix to avoid grabbing random query-string
    # junk that happens to look DOI-like.
    m = _DOI_ANY_RE.search(url)
    if m:
        doi = m.group(1)
        # Trim trailing punctuation that's clearly not part of a DOI.
        doi = doi.rstrip(").,;")
        return doi
    return None


async def _render_and_extract_cards(
    url: str,
    *,
    delay: float,
    headless: bool,
    user_agent: str,
) -> list[dict[str, str]]:
    """Launch Chromium, navigate to ``url``, return raw card dicts.

    Each dict has keys: ``title``, ``url``, ``meta``, ``snippet``.
    Returns ``[]`` when playwright is not installed or on any error.

    This function is the **single Playwright seam** — tests replace it
    with a sync or async mock that returns pre-built card lists.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "google_scholar_playwright_missing",
            hint="uv pip install -e '[browser]' && playwright install chromium",
        )
        return []

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=headless)
            try:
                ctx = await browser.new_context(user_agent=user_agent)
                page = await ctx.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                # Polite delay *after* the page loads
                await asyncio.sleep(delay)

                # CAPTCHA detection
                html = await page.content()
                if "captcha" in html.lower() or "unusual traffic" in html.lower():
                    logger.warning("google_scholar_captcha_detected", url=url[:100])
                    return _CAPTCHA_SENTINEL

                cards: list[dict[str, str]] = []
                for card_el in await page.query_selector_all(".gs_ri"):
                    # Title text (strip [PDF]/[HTML] prefixes Scholar adds)
                    title = ""
                    title_el = await card_el.query_selector(".gs_rt")
                    if title_el:
                        raw = (await title_el.inner_text()).strip()
                        title = re.sub(r"^\[(PDF|HTML|CITATION|BOOK)\]\s*", "", raw)

                    # Title link href (may contain doi.org or arxiv URL)
                    href = ""
                    link_el = await card_el.query_selector(".gs_rt a")
                    if link_el:
                        href = (await link_el.get_attribute("href")) or ""

                    # Author / venue / year line
                    meta = ""
                    meta_el = await card_el.query_selector(".gs_a")
                    if meta_el:
                        meta = (await meta_el.inner_text()).strip()

                    # Abstract snippet
                    snippet = ""
                    snip_el = await card_el.query_selector(".gs_rs")
                    if snip_el:
                        snippet = (await snip_el.inner_text()).strip()

                    # Footer (contains "Cited by N Related articles ...")
                    footer = ""
                    footer_el = await card_el.query_selector(".gs_fl")
                    if footer_el:
                        footer = (await footer_el.inner_text()).strip()

                    if title:
                        cards.append(
                            {
                                "title": title,
                                "url": href,
                                "meta": meta,
                                "snippet": snippet,
                                "footer": footer,
                            }
                        )
                return cards
            finally:
                await browser.close()
    except Exception as exc:
        logger.warning("google_scholar_render_failed", error=str(exc), url=url[:100])
        return []


class GoogleScholarPlaywrightProvider:
    """Google Scholar via headless Chromium.

    Implements the same protocol as EuropePMCSearchProvider,
    CORESearchProvider, etc. — drop it into DomainAwareAggregator.

    Uses ``tier = "flaky"`` so the aggregator gives it a 45-second
    timeout (2.25x 20 s default) and does not count a single failure
    as fatal.
    """

    name: ClassVar[str] = "google_scholar"
    description: ClassVar[str] = (
        "Google Scholar via headless Chromium (browser extra required)"
    )
    domains: ClassVar[list[str]] = ["general"]  # broad coverage across all domains
    tier: ClassVar[str] = "flaky"  # slow + rate-limited -> flaky tier (2.25x timeout)
    retry: ClassVar[int] = 0  # no retry; CAPTCHA risk on multiple attempts

    def __init__(
        self,
        *,
        delay_seconds: float = 2.0,
        headless: bool = True,
        user_agent: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        openrouter_fallback_enabled: bool = False,
        openrouter_api_key: str = "",
        openrouter_fallback_model: str = "deepseek/deepseek-chat",
        openrouter_fallback_domains: list[str] | None = None,
    ) -> None:
        self._delay = delay_seconds
        self._headless = headless
        self._user_agent = user_agent
        self._openrouter_enabled = openrouter_fallback_enabled
        self._openrouter_api_key = openrouter_api_key
        self._openrouter_model = openrouter_fallback_model
        self._openrouter_domains = openrouter_fallback_domains

    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        **_: Any,
    ) -> list[Paper]:
        # Preference order: OpenRouter Exa-style academic web search first
        # (when configured) — Google Scholar reliably CAPTCHA's headless
        # Chromium, so trying Playwright first usually wastes a slow
        # round-trip. We still fall through to Playwright if OpenRouter
        # returns nothing or isn't enabled, so behaviour stays "Scholar
        # answers something" rather than "OpenRouter failed, give up".
        if self._openrouter_enabled and self._openrouter_api_key:
            try:
                from perspicacite.search.openrouter_fallback import (
                    openrouter_academic_search,
                )
                papers = await openrouter_academic_search(
                    query,
                    api_key=self._openrouter_api_key,
                    model=self._openrouter_model,
                    max_results=max_results,
                    allowed_domains=self._openrouter_domains,
                )
                if papers:
                    logger.info(
                        "google_scholar_openrouter_primary_success",
                        count=len(papers),
                    )
                    return papers
                logger.info("google_scholar_openrouter_primary_empty")
            except Exception as exc:
                logger.warning(
                    "google_scholar_openrouter_primary_error", error=str(exc)
                )

        url = _build_scholar_url(query, year_min=year_min, year_max=year_max)
        try:
            cards = await _render_and_extract_cards(
                url,
                delay=self._delay,
                headless=self._headless,
                user_agent=self._user_agent,
            )
        except Exception as exc:
            logger.warning("google_scholar_search_error", error=str(exc))
            return []

        # CAPTCHA detected — already tried OpenRouter above (if enabled),
        # so the only remaining option is to give up on this provider.
        if cards is _CAPTCHA_SENTINEL:
            logger.warning(
                "google_scholar_captcha_and_no_fallback",
                openrouter_enabled=self._openrouter_enabled,
            )
            return []

        papers = []
        for card in cards[:max_results]:
            doi = _extract_doi_from_url(card.get("url", ""))
            authors_str, _venue, year = _parse_meta_line(card.get("meta", ""))

            # Build author list from comma-separated string
            authors: list[Author] = []
            for name in authors_str.split(","):
                name = name.strip()
                if name and len(name) > 1:
                    authors.append(Author(name=name))

            title = card.get("title") or "Untitled"
            paper_id = doi or "scholar:" + hashlib.sha256(title.encode()).hexdigest()[:8]
            citation_count = _extract_citation_count(card.get("footer", ""))

            papers.append(
                Paper(
                    id=paper_id,
                    title=title,
                    authors=authors,
                    year=year,
                    doi=doi,
                    abstract=card.get("snippet") or None,
                    citation_count=citation_count,
                    source=PaperSource.GOOGLE_SCHOLAR,
                    metadata={
                        "scholar_url": card.get("url", ""),
                        "sources": ["google_scholar"],
                    },
                )
            )
        logger.info(
            "google_scholar_search_done",
            query=query[:80],
            returned=len(papers),
        )
        return papers
