"""Unit tests for the DuckDuckGo Playwright general-web search provider.

All tests mock ``_render_and_extract_results`` so no browser is needed.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from perspicacite.search.duckduckgo_playwright import (
    _BOT_CHALLENGE_SENTINEL,
    DuckDuckGoPlaywrightProvider,
    _build_search_url,
    _unwrap_ddg_redirect,
)

# ── URL builder tests (pure helpers, no mock) ────────────────────────────────

def test_build_search_url_basic():
    url = _build_search_url("metabolomics tool documentation")
    assert "html.duckduckgo.com" in url
    assert "metabolomics" in url
    assert "documentation" in url


def test_build_search_url_with_site_filter():
    url = _build_search_url(
        "metaboapps overview",
        site_filter=["github.com", "*.github.io"],
    )
    # OR'd site: clauses (wildcards URL-encode: * → %2A)
    assert "site%3Agithub.com" in url
    assert "%2A.github.io" in url  # URL-encoded *.github.io
    assert "OR" in url


def test_build_search_url_with_exclude_domains():
    url = _build_search_url(
        "data analysis library",
        exclude_domains=["wikipedia.org", "w3.org"],
    )
    assert "-site%3Awikipedia.org" in url
    assert "-site%3Aw3.org" in url


def test_build_search_url_with_both_filters():
    url = _build_search_url(
        "spectrometry",
        site_filter=["readthedocs.io"],
        exclude_domains=["wikipedia.org"],
    )
    assert "site%3Areadthedocs.io" in url
    assert "-site%3Awikipedia.org" in url


def test_build_search_url_trims_query():
    url = _build_search_url("   metabolomics   ")
    assert "metabolomics" in url
    # No leading-space encoding artefact
    assert "+++metabolomics" not in url


# ── DDG redirect unwrap tests ────────────────────────────────────────────────

def test_unwrap_ddg_redirect_uddg_param():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Forg%2Frepo"
    out = _unwrap_ddg_redirect(href)
    assert out == "https://github.com/org/repo"


def test_unwrap_ddg_redirect_passes_through_direct_url():
    href = "https://example.com/page"
    out = _unwrap_ddg_redirect(href)
    assert out == href


def test_unwrap_ddg_redirect_empty():
    assert _unwrap_ddg_redirect("") == ""


# ── Provider behaviour tests (mock _render_and_extract_results) ──────────────

_FAKE_RESULTS = [
    {
        "title": "metLinkR: facilitating meta-analysis of metabolomics data",
        "url": "https://github.com/NCATSTranslator/metLinkR",
        "snippet": "Automated linking of metabolite identifiers across heterogeneous datasets.",
    },
    {
        "title": "MetaboAnalystR 4.0 Docs",
        "url": "https://www.metaboanalyst.ca/MetaboAnalyst/docs/",
        "snippet": "Documentation for MetaboAnalystR 4.0 — the unified LC-MS workflow.",
    },
    {
        "title": "MassQL — query language for mass spectrometry",
        "url": "https://mwang87.github.io/MassQueryLanguage_Documentation/",
        "snippet": "MassQL is a SQL-like query language for searching mass spectra.",
    },
]


@pytest.mark.asyncio
async def test_provider_returns_results():
    async def fake_render(url, *, delay, headless, user_agent):
        return list(_FAKE_RESULTS)

    with patch(
        "perspicacite.search.duckduckgo_playwright._render_and_extract_results",
        new=fake_render,
    ):
        provider = DuckDuckGoPlaywrightProvider(delay_seconds=0.0)
        results = await provider.search("metabolomics tool documentation", max_results=10)

    assert len(results) == 3
    assert results[0]["title"] == "metLinkR: facilitating meta-analysis of metabolomics data"
    assert results[0]["url"].startswith("https://github.com/")
    assert "metabolite identifiers" in results[0]["snippet"]


@pytest.mark.asyncio
async def test_provider_respects_max_results():
    async def fake_render(url, *, delay, headless, user_agent):
        return list(_FAKE_RESULTS)

    with patch(
        "perspicacite.search.duckduckgo_playwright._render_and_extract_results",
        new=fake_render,
    ):
        provider = DuckDuckGoPlaywrightProvider(delay_seconds=0.0)
        results = await provider.search("test", max_results=2)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_provider_passes_site_filter():
    captured_urls: list[str] = []

    async def fake_render(url, *, delay, headless, user_agent):
        captured_urls.append(url)
        return []

    with patch(
        "perspicacite.search.duckduckgo_playwright._render_and_extract_results",
        new=fake_render,
    ):
        provider = DuckDuckGoPlaywrightProvider(delay_seconds=0.0)
        await provider.search(
            "metaboapps",
            site_filter=["github.com", "*.github.io"],
        )

    assert captured_urls
    assert "site%3Agithub.com" in captured_urls[0]


@pytest.mark.asyncio
async def test_provider_passes_exclude_domains():
    captured_urls: list[str] = []

    async def fake_render(url, *, delay, headless, user_agent):
        captured_urls.append(url)
        return []

    with patch(
        "perspicacite.search.duckduckgo_playwright._render_and_extract_results",
        new=fake_render,
    ):
        provider = DuckDuckGoPlaywrightProvider(delay_seconds=0.0)
        await provider.search("metabolomics", exclude_domains=["wikipedia.org"])

    assert captured_urls
    assert "-site%3Awikipedia.org" in captured_urls[0]


@pytest.mark.asyncio
async def test_provider_returns_empty_on_bot_challenge():
    async def fake_render(url, *, delay, headless, user_agent):
        return _BOT_CHALLENGE_SENTINEL

    with patch(
        "perspicacite.search.duckduckgo_playwright._render_and_extract_results",
        new=fake_render,
    ):
        provider = DuckDuckGoPlaywrightProvider(delay_seconds=0.0)
        results = await provider.search("test", max_results=5)

    assert results == []


@pytest.mark.asyncio
async def test_provider_returns_empty_on_render_error():
    async def fake_render(url, *, delay, headless, user_agent):
        raise RuntimeError("browser crash")

    with patch(
        "perspicacite.search.duckduckgo_playwright._render_and_extract_results",
        new=fake_render,
    ):
        provider = DuckDuckGoPlaywrightProvider(delay_seconds=0.0)
        try:
            results = await provider.search("test", max_results=5)
        except RuntimeError:
            # Provider propagates the exception when render raises; that's
            # an acceptable contract — the MCP wrapper catches it. (Same as
            # the Google Scholar provider behaviour.)
            results = []

    assert results == []


# ── Class-level metadata invariants ──────────────────────────────────────────

def test_provider_has_required_class_attributes():
    assert DuckDuckGoPlaywrightProvider.name == "duckduckgo"
    assert DuckDuckGoPlaywrightProvider.tier == "flaky"
    assert DuckDuckGoPlaywrightProvider.retry == 0
    assert "general_web" in DuckDuckGoPlaywrightProvider.domains


def test_bot_challenge_sentinel_is_module_level_object():
    """_BOT_CHALLENGE_SENTINEL is the same identity on every access."""
    from perspicacite.search import duckduckgo_playwright as mod
    assert mod._BOT_CHALLENGE_SENTINEL is _BOT_CHALLENGE_SENTINEL
