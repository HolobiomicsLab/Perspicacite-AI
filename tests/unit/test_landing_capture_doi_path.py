"""Public HTML full-text capture and its quality gate in the DOI pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from perspicacite.pipeline.download import unified
from perspicacite.pipeline.download.base import PaperDiscovery
from perspicacite.pipeline.download.html_capture import HtmlCapture


def _disc() -> PaperDiscovery:
    # No oa_url, no full text anywhere → pipeline reaches the fallback tail.
    return PaperDiscovery(
        doi="10.1038/paywalled",
        title="A Paywalled Nature Paper",
        abstract="A short abstract from discovery.",
        authors=["A. Researcher"],
        year=2026,
        is_oa=False,
    )


@pytest.mark.asyncio
async def test_landing_capture_used_when_enabled_and_pdf_missed(tmp_path):
    """A real 9,000-character HTML capture must reach the full-text tier."""
    body_text = "Research results. " * 500
    assert len(body_text) == 9_000
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(
            200, text=f"<html><body><p>{body_text}</p></body></html>",
            headers={"content-type": "text/html"},
        )
    with (
        patch(
            "perspicacite.pipeline.download.unified.discover_paper_sources",
            new_callable=AsyncMock,
            return_value=_disc(),
        ),
        patch(
            "perspicacite.pipeline.download.unified.get_fulltext_from_pmc",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch(
            "perspicacite.pipeline.download.unified.get_content_from_europepmc",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "perspicacite.pipeline.download.unified._try_pdf_sources",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("perspicacite.pipeline.download.discovery._CACHE_DIR", tmp_path),
    ):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await unified.retrieve_paper_content(
                "10.1038/paywalled",
                http_client=client,
                pdf_parser=object(),
                pdf_cache_dir=str(tmp_path),
                enable_landing_capture=True,
            )

    assert requests == ["https://doi.org/10.1038/paywalled"]
    assert result.success is True
    assert result.content_source == "landing_html"
    assert result.content_type == "full_text"
    assert result.full_text == body_text.strip()
    assert list((tmp_path / "html").glob("*.html"))


@pytest.mark.asyncio
async def test_landing_capture_not_used_when_disabled(tmp_path):
    """Default (flag off): the fallback must not run, and the pipeline
    degrades to abstract-only as before."""
    with (
        patch(
            "perspicacite.pipeline.download.unified.discover_paper_sources",
            new_callable=AsyncMock,
            return_value=_disc(),
        ),
        patch(
            "perspicacite.pipeline.download.unified.get_fulltext_from_pmc",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch(
            "perspicacite.pipeline.download.unified.get_content_from_europepmc",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "perspicacite.pipeline.download.unified._try_pdf_sources",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "perspicacite.pipeline.download.unified.capture_landing_html",
            new_callable=AsyncMock,
        ) as mock_cap,
        patch("perspicacite.pipeline.download.discovery._CACHE_DIR", tmp_path),
    ):
        result = await unified.retrieve_paper_content(
            "10.1038/paywalled",
            http_client=AsyncMock(),
            pdf_parser=object(),
        )

    mock_cap.assert_not_awaited()
    assert result.content_type == "abstract"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 403])
async def test_blocked_candidate_is_not_retried_as_landing_html(tmp_path, monkeypatch, status):
    disc = _disc()
    disc.oa_url = "https://publisher.example/article"
    disc.oa_candidates = [disc.oa_url]
    monkeypatch.setattr(unified, "discover_paper_sources", AsyncMock(return_value=disc))
    monkeypatch.setattr(unified, "get_fulltext_from_pmc", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(unified, "get_content_from_europepmc", AsyncMock(return_value=None))
    client = AsyncMock()
    client.get.return_value = httpx.Response(
        status, text='<html><div id="cf-chl-widget">Just a moment...</div></html>',
        headers={"content-type": "text/html"}, request=httpx.Request("GET", disc.oa_url),
    )

    result = await unified.retrieve_paper_content(
        disc.doi, http_client=client, pdf_parser=object(), pdf_cache_dir=str(tmp_path),
        enable_landing_capture=True,
    )

    assert result.content_type == "abstract"
    assert client.get.await_count == 1
    assert not list(tmp_path.rglob("*.html"))
    assert result.attempts[0]["status"] == "blocked"
    assert result.attempts[0]["marker"] == "cloudflare_challenge"


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", ["bibliographic_stub", "extended_abstract", "full_text"])
async def test_landing_capture_thin_result_falls_through_to_abstract(tmp_path, tier):
    """A stub/abstract-tier capture is no better than the abstract path, so it
    must not be accepted as full_text."""
    thin = HtmlCapture(
        path=Path(tmp_path) / "x.html",
        tier=tier,
        char_count=80,
        extracted_text="A Paywalled Nature Paper A short abstract.",
    )
    with (
        patch(
            "perspicacite.pipeline.download.unified.discover_paper_sources",
            new_callable=AsyncMock,
            return_value=_disc(),
        ),
        patch(
            "perspicacite.pipeline.download.unified.get_fulltext_from_pmc",
            new_callable=AsyncMock,
            return_value=(None, None),
        ),
        patch(
            "perspicacite.pipeline.download.unified.get_content_from_europepmc",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "perspicacite.pipeline.download.unified._try_pdf_sources",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "perspicacite.pipeline.download.unified.capture_landing_html",
            new_callable=AsyncMock,
            return_value=thin,
        ),
        patch("perspicacite.pipeline.download.discovery._CACHE_DIR", tmp_path),
    ):
        result = await unified.retrieve_paper_content(
            "10.1038/paywalled",
            http_client=AsyncMock(),
            pdf_parser=object(),
            enable_landing_capture=True,
        )

    assert result.content_type == "abstract"
