"""Tests for the HTML landing-page capture fallback (Priority 3b)."""

import httpx
import pytest

from perspicacite.pipeline.download.html_capture import (
    _classify_tier,
    capture_landing_html,
)


def test_classify_tier_thresholds():
    """Three tiers driven by extracted-text character count."""
    assert _classify_tier(100) == "bibliographic_stub"
    assert _classify_tier(1_500) == "extended_abstract"
    assert _classify_tier(5_000) == "extended_abstract"
    assert _classify_tier(8_000) == "full_text_html"
    assert _classify_tier(80_000) == "full_text_html"


@pytest.mark.asyncio
async def test_capture_full_text_page(respx_mock, tmp_path):
    """A page with a long body is classified full_text_html."""
    body = (
        "<html><head><title>Test article</title></head>"
        "<body><h1>Body</h1>" + ("<p>some sentence about science</p>" * 400) +
        "</body></html>"
    )
    respx_mock.get("https://example.org/article/1").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/html"})
    )
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.x/y",
            landing_url="https://example.org/article/1",
            http_client=http,
            cache_dir=str(tmp_path),
        )
    assert cap is not None
    assert cap.tier == "full_text_html"
    assert cap.char_count > 8_000
    assert cap.path.exists()
    saved = cap.path.read_text()
    assert "Captured landing page" in saved
    assert "10.x/y" in saved


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 403])
async def test_challenge_is_reported_without_saving_a_snapshot(respx_mock, tmp_path, status):
    body = (
        '<html><title>Just a moment...</title>'
        '<script src="/cdn-cgi/challenge-platform/x"></script></html>'
    )
    url = "https://publisher.example/article"
    respx_mock.get(url).mock(return_value=httpx.Response(
        status, text=body, headers={"content-type": "text/html"},
    ))
    attempts = []
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.x/challenge", landing_url=url, title="Article title",
            abstract="An abstract available from discovery.",
            http_client=http, cache_dir=tmp_path, attempts=attempts,
        )
    assert cap is None
    assert not list(tmp_path.rglob("*.html"))
    assert attempts == [{
        "source": "landing_html", "status": "blocked", "url": url,
        "final_url": url, "marker": "cloudflare_challenge",
    }]


@pytest.mark.asyncio
async def test_capture_stub_with_abstract_splice(respx_mock, tmp_path):
    """A short paywall stub gets the OpenAlex abstract spliced in."""
    body = "<html><head><title>Paywalled</title></head><body><p>access required</p></body></html>"
    respx_mock.get("https://example.org/x").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/html"})
    )
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.x/p",
            landing_url="https://example.org/x",
            abstract="Authors describe a novel multi-agent framework that ...",
            http_client=http,
            cache_dir=str(tmp_path),
        )
    assert cap is not None
    assert cap.tier == "bibliographic_stub"
    saved = cap.path.read_text()
    assert "novel multi-agent framework" in saved
    assert "openalex-abstract" in saved


@pytest.mark.asyncio
async def test_capture_skips_non_html(respx_mock, tmp_path):
    """A PDF response (or any non-HTML content-type) should be skipped."""
    respx_mock.get("https://example.org/p").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4",
                                      headers={"content-type": "application/pdf"})
    )
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.x/q",
            landing_url="https://example.org/p",
            http_client=http,
            cache_dir=str(tmp_path),
        )
    assert cap is None


@pytest.mark.asyncio
async def test_capture_404_returns_none(respx_mock, tmp_path):
    """Don't blow up on 404; return None."""
    respx_mock.get("https://example.org/missing").mock(
        return_value=httpx.Response(404, text="Not found")
    )
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.x/m",
            landing_url="https://example.org/missing",
            http_client=http,
            cache_dir=str(tmp_path),
        )
    assert cap is None


@pytest.mark.asyncio
async def test_capture_uses_doi_redirect_when_no_landing(respx_mock, tmp_path):
    """When no landing_url is given but a DOI is, fall back to doi.org/<DOI>."""
    respx_mock.get("https://doi.org/10.x/r").mock(
        return_value=httpx.Response(200,
                                      text="<html><body><p>" + ("text " * 1000) + "</p></body></html>",
                                      headers={"content-type": "text/html"})
    )
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.x/r",
            landing_url=None,
            http_client=http,
            cache_dir=str(tmp_path),
        )
    assert cap is not None
    assert cap.tier == "extended_abstract"


@pytest.mark.asyncio
async def test_capture_builds_stub_when_live_fetch_403s(respx_mock, tmp_path):
    """Live-discovered (2026-05-16): Cloudflare-gated publishers
    (preprints.org, royalsocietypublishing.org, ACS, etc.) return 403
    to non-browser HTTP clients even with a realistic UA + cookie jar.
    When the live fetch fails but the caller has metadata (title +
    abstract from OpenAlex/Crossref), the capture function must
    synthesize a ``bibliographic_stub`` rather than silently returning
    None — that's the only way the push_to_zotero(attach_pdf=True)
    flow leaves *anything* in Zotero for Cloudflare-blocked items."""
    respx_mock.get("https://www.preprints.org/manuscript/abc").mock(
        return_value=httpx.Response(403, text="Forbidden by Cloudflare")
    )
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.20944/preprints.abc",
            landing_url="https://www.preprints.org/manuscript/abc",
            title="Building MCP-Native Hierarchical AI Scientist Ecosystems",
            abstract="This work introduces a framework for ...",
            http_client=http,
            cache_dir=str(tmp_path),
        )
    assert cap is not None
    assert cap.tier == "bibliographic_stub"
    saved = cap.path.read_text()
    assert "Building MCP-Native" in saved
    assert "framework for" in saved
    assert "HTTP 403" in saved  # reason field surfaces the cause


@pytest.mark.asyncio
async def test_capture_returns_none_when_403_and_no_metadata(respx_mock, tmp_path):
    """Symmetric to the stub-fallback test: if neither title nor
    abstract is supplied, there's nothing to stub — return None."""
    respx_mock.get("https://example.org/q").mock(
        return_value=httpx.Response(403, text="Forbidden")
    )
    async with httpx.AsyncClient() as http:
        cap = await capture_landing_html(
            doi="10.x/q",
            landing_url="https://example.org/q",
            http_client=http,
            cache_dir=str(tmp_path),
        )
    assert cap is None
