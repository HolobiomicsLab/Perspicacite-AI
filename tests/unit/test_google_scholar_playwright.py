"""Unit tests for the Google Scholar Playwright provider.

All tests mock _render_and_extract_cards so no browser is needed.
"""
from __future__ import annotations

from unittest.mock import patch

from perspicacite.search.google_scholar_playwright import (
    GoogleScholarPlaywrightProvider,
    _build_scholar_url,
    _extract_doi_from_url,
    _parse_meta_line,
)

# ── Pure helper tests (no mock needed) ───────────────────────────────────────

def test_build_scholar_url_with_year_range():
    url = _build_scholar_url("alphafold protein", year_min=2020, year_max=2023)
    assert "as_ylo=2020" in url
    assert "as_yhi=2023" in url
    assert "scholar.google.com" in url


def test_build_scholar_url_without_years():
    url = _build_scholar_url("microbiome diversity")
    assert "as_ylo" not in url
    assert "as_yhi" not in url
    assert "scholar.google.com" in url


def test_build_scholar_url_pagination():
    url = _build_scholar_url("test", start=10)
    assert "start=10" in url


def test_parse_meta_line_full():
    authors, venue, year = _parse_meta_line(
        "J Jumper, R Evans, A Senior - Nature, 2021 - nature.com"
    )
    assert year == 2021
    assert "Jumper" in authors
    assert venue  # non-empty


def test_parse_meta_line_year_only():
    _, _, year = _parse_meta_line("Some Author - Some Journal - 2019")
    assert year == 2019


def test_parse_meta_line_no_year():
    _, _, year = _parse_meta_line("Some Author - Some Journal")
    assert year is None


def test_extract_doi_from_doi_url():
    doi = _extract_doi_from_url("https://doi.org/10.1038/s41587-020-00744-z")
    assert doi == "10.1038/s41587-020-00744-z"


def test_extract_doi_from_doi_url_http():
    doi = _extract_doi_from_url("http://dx.doi.org/10.1016/j.cell.2021.01.001")
    assert doi == "10.1016/j.cell.2021.01.001"


def test_extract_doi_from_non_doi_url_returns_none():
    assert _extract_doi_from_url("https://www.nature.com/articles/s41587") is None
    assert _extract_doi_from_url("") is None


def test_arxiv_url_now_yields_its_datacite_doi():
    """Deliberate reversal: arXiv URLs used to return None.

    They carry no DOI in the path, so every arXiv preprint Scholar
    returned was dropped downstream by the ``no_doi`` ingest filter.
    ``10.48550/arXiv.<id>`` is derivable from the id alone. See
    tests/unit/test_scholar_arxiv_doi.py.
    """
    assert _extract_doi_from_url("https://arxiv.org/abs/2204.12345") == (
        "10.48550/arXiv.2204.12345"
    )


# ── Provider behaviour tests (mock _render_and_extract_cards) ────────────────

_FAKE_CARDS = [
    {
        "title": "Deep Learning for Protein Structure",
        "url": "https://doi.org/10.1038/s41587-020-00744-z",
        "meta": "J Jumper, R Evans - Nature, 2021 - nature.com",
        "snippet": "We present AlphaFold2...",
    },
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762",
        "meta": "A Vaswani, N Shazeer - NeurIPS, 2017 - papers.nips.cc",
        "snippet": "The dominant sequence model...",
    },
]


async def test_provider_converts_cards_to_papers():
    async def fake_render(url, *, delay, headless, user_agent):
        return list(_FAKE_CARDS)

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ):
        provider = GoogleScholarPlaywrightProvider(delay_seconds=0.0)
        papers = await provider.search("protein structure prediction", max_results=10)

    assert len(papers) == 2
    p = papers[0]
    assert p.title == "Deep Learning for Protein Structure"
    assert p.doi == "10.1038/s41587-020-00744-z"
    assert p.year == 2021
    assert p.source.value == "google_scholar"


async def test_provider_respects_max_results():
    async def fake_render(url, *, delay, headless, user_agent):
        return list(_FAKE_CARDS)

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ):
        provider = GoogleScholarPlaywrightProvider(delay_seconds=0.0)
        papers = await provider.search("test", max_results=1)

    assert len(papers) <= 1


async def test_provider_returns_empty_on_render_error():
    async def fake_render(url, *, delay, headless, user_agent):
        raise RuntimeError("browser crash")

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ):
        provider = GoogleScholarPlaywrightProvider(delay_seconds=0.0)
        papers = await provider.search("test", max_results=5)

    assert papers == []


async def test_provider_passes_year_filters_to_url():
    captured_urls: list[str] = []

    async def fake_render(url, *, delay, headless, user_agent):
        captured_urls.append(url)
        return []

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ):
        provider = GoogleScholarPlaywrightProvider(delay_seconds=0.0)
        await provider.search("CRISPR", max_results=5, year_min=2020, year_max=2023)

    assert captured_urls
    assert "as_ylo=2020" in captured_urls[0]
    assert "as_yhi=2023" in captured_urls[0]


# ── CAPTCHA sentinel + OpenRouter fallback tests ──────────────────────────────

def test_captcha_sentinel_is_module_level_object():
    """_CAPTCHA_SENTINEL is the same object on every attribute access (module singleton)."""
    import perspicacite.search.google_scholar_playwright as _mod
    assert _mod._CAPTCHA_SENTINEL is _mod._CAPTCHA_SENTINEL
    assert isinstance(_mod._CAPTCHA_SENTINEL, list)


async def test_captcha_triggers_openrouter_fallback():
    """When _render_and_extract_cards returns _CAPTCHA_SENTINEL, search() calls fallback."""
    from perspicacite.models.papers import Paper, PaperSource
    from perspicacite.search.google_scholar_playwright import (
        _CAPTCHA_SENTINEL,
        GoogleScholarPlaywrightProvider,
    )

    fallback_paper = Paper(
        id="10.1/test",
        title="Fallback Paper",
        doi="10.1/test",
        source=PaperSource.OPENROUTER_WEB,
    )

    async def fake_render(url, *, delay, headless, user_agent):
        return _CAPTCHA_SENTINEL

    async def fake_openrouter(query, *, api_key, model, max_results, allowed_domains):
        return [fallback_paper]

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ), patch(
        "perspicacite.search.openrouter_fallback.openrouter_academic_search",
        new=fake_openrouter,
    ):
        provider = GoogleScholarPlaywrightProvider(
            delay_seconds=0.0,
            openrouter_fallback_enabled=True,
            openrouter_api_key="sk-test",
        )
        papers = await provider.search("CRISPR", max_results=5)

    assert len(papers) == 1
    assert papers[0].title == "Fallback Paper"
    assert papers[0].source == PaperSource.OPENROUTER_WEB


async def test_captcha_fallback_disabled_returns_empty():
    """When openrouter_fallback_enabled=False, CAPTCHA → [] without calling fallback."""
    from perspicacite.search.google_scholar_playwright import (
        _CAPTCHA_SENTINEL,
        GoogleScholarPlaywrightProvider,
    )

    fallback_called = []

    async def fake_render(url, *, delay, headless, user_agent):
        return _CAPTCHA_SENTINEL

    async def fake_openrouter(*a, **kw):
        fallback_called.append(True)
        return []

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ), patch(
        "perspicacite.search.openrouter_fallback.openrouter_academic_search",
        new=fake_openrouter,
    ):
        provider = GoogleScholarPlaywrightProvider(
            delay_seconds=0.0,
            openrouter_fallback_enabled=False,
        )
        papers = await provider.search("test", max_results=5)

    assert papers == []
    assert not fallback_called


async def test_captcha_fallback_passes_correct_args():
    """search() passes query, api_key, model, max_results, domains to fallback."""
    from perspicacite.search.google_scholar_playwright import (
        _CAPTCHA_SENTINEL,
        GoogleScholarPlaywrightProvider,
    )

    captured: dict = {}

    async def fake_render(url, *, delay, headless, user_agent):
        return _CAPTCHA_SENTINEL

    async def fake_openrouter(query, *, api_key, model, max_results, allowed_domains):
        captured.update(
            query=query, api_key=api_key, model=model,
            max_results=max_results, domains=allowed_domains,
        )
        return []

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ), patch(
        "perspicacite.search.openrouter_fallback.openrouter_academic_search",
        new=fake_openrouter,
    ):
        provider = GoogleScholarPlaywrightProvider(
            delay_seconds=0.0,
            openrouter_fallback_enabled=True,
            openrouter_api_key="sk-abc",
            openrouter_fallback_model="openai/gpt-4o-mini",
            openrouter_fallback_domains=["arxiv.org"],
        )
        await provider.search("deep learning", max_results=7)

    assert captured["query"] == "deep learning"
    assert captured["api_key"] == "sk-abc"
    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["max_results"] == 7
    assert captured["domains"] == ["arxiv.org"]


async def test_captcha_fallback_raises_returns_empty():
    """If the fallback itself raises (e.g. ImportError), search() still returns []."""
    from perspicacite.search.google_scholar_playwright import (
        _CAPTCHA_SENTINEL,
        GoogleScholarPlaywrightProvider,
    )

    async def fake_render(url, *, delay, headless, user_agent):
        return _CAPTCHA_SENTINEL

    async def fake_openrouter(*a, **kw):
        raise RuntimeError("fallback module broken")

    with patch(
        "perspicacite.search.google_scholar_playwright._render_and_extract_cards",
        new=fake_render,
    ), patch(
        "perspicacite.search.openrouter_fallback.openrouter_academic_search",
        new=fake_openrouter,
    ):
        provider = GoogleScholarPlaywrightProvider(
            delay_seconds=0.0,
            openrouter_fallback_enabled=True,
            openrouter_api_key="sk-test",
        )
        papers = await provider.search("test", max_results=5)

    assert papers == []
