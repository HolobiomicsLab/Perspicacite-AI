"""Repository landing pages resolve one PDF hop with truthful attempt records."""

from unittest.mock import AsyncMock

import httpx
import pytest

from perspicacite.pipeline.download import unified
from perspicacite.pipeline.download.base import MIN_PLAUSIBLE_PDF_BYTES, PaperDiscovery
from perspicacite.pipeline.download.url_extractors import extract_repository_pdf_url

DOI = "10.1234/article"
LANDING_URL = "https://repository.test/items/article"
PDF_URL = "https://repository.test/files/article.pdf"
PDF_BYTES = b"%PDF-1.7\n" + b"x" * MIN_PLAUSIBLE_PDF_BYTES


def _response(url, body, content_type="text/html"):
    return httpx.Response(
        200,
        content=body,
        headers={"content-type": content_type},
        request=httpx.Request("GET", url),
    )


def _client(responses):
    async def get(url, **_kwargs):
        return responses[url]

    return AsyncMock(get=AsyncMock(side_effect=get))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "link, pdf_url",
    [
        ('<meta name="citation_pdf_url" content="../files/article.pdf">', PDF_URL),
        (
            '<meta name="citation_pdf_url" content="//cdn.test/article.pdf">',
            "https://cdn.test/article.pdf",
        ),
        (
            '<a href="/bitstream/123/1/article.pdf">Download</a>',
            "https://repository.test/bitstream/123/1/article.pdf",
        ),
        (
            '<a href="/bitstreams/abc-123/download">Download</a>',
            "https://repository.test/bitstreams/abc-123/download",
        ),
        ('<a href="/hal-123/document">Download</a>', "https://repository.test/hal-123/document"),
    ],
)
async def test_repository_landing_link_yields_pdf(link, pdf_url):
    client = _client(
        {
            LANDING_URL: _response(LANDING_URL, f"<html><head>{link}</head></html>"),
            pdf_url: _response(pdf_url, PDF_BYTES, "application/pdf"),
        }
    )
    discovery = PaperDiscovery(doi=DOI, oa_candidates=[LANDING_URL])

    result = await unified._try_pdf_sources(DOI, None, client, discovery)

    assert result == (PDF_BYTES, "repository_pdf")
    assert [call.args[0] for call in client.get.await_args_list] == [LANDING_URL, pdf_url]


@pytest.mark.asyncio
async def test_relative_pdf_url_uses_final_repository_redirect():
    handle = "http://hdl.handle.net/20.500.11850/123"
    client = _client(
        {
            handle: _response(
                LANDING_URL, '<meta name="citation_pdf_url" content="../files/article.pdf">'
            ),
            PDF_URL: _response(PDF_URL, PDF_BYTES, "application/pdf"),
        }
    )
    discovery = PaperDiscovery(doi=DOI, oa_candidates=[handle])

    result = await unified._try_pdf_sources(DOI, None, client, discovery)

    assert result == (PDF_BYTES, "repository_pdf")
    assert client.get.await_args_list[1].args[0] == PDF_URL


@pytest.mark.asyncio
async def test_landing_without_pdf_link_records_miss_and_is_not_refetched():
    client = _client({LANDING_URL: _response(LANDING_URL, "<html>No PDF link</html>")})
    discovery = PaperDiscovery(
        doi=DOI,
        oa_candidates=[LANDING_URL],
        unpaywall_pdf_url=LANDING_URL,
    )
    attempts = []

    result = await unified._try_pdf_sources(DOI, None, client, discovery, attempts=attempts)

    assert result is None
    assert client.get.await_count == 1
    assert attempts == [{"source": "publisher_oa_pdf", "status": "miss", "url": LANDING_URL}]


@pytest.mark.asyncio
async def test_resolution_stops_when_first_link_returns_more_html():
    next_page = "https://repository.test/second-page"
    client = _client(
        {
            LANDING_URL: _response(
                LANDING_URL, f'<meta name="citation_pdf_url" content="{next_page}">'
            ),
            next_page: _response(next_page, f'<meta name="citation_pdf_url" content="{PDF_URL}">'),
        }
    )
    discovery = PaperDiscovery(doi=DOI, oa_candidates=[LANDING_URL])
    attempts = []

    result = await unified._try_pdf_sources(DOI, None, client, discovery, attempts=attempts)

    assert result is None
    assert client.get.await_count == 2
    assert attempts[-1] == {"source": "repository_pdf", "status": "miss", "url": next_page}


@pytest.mark.asyncio
async def test_known_publisher_page_is_not_resolved_as_repository():
    publisher = "https://journals.publisher.test/article"
    candidate = "https://www.publisher.test/article"
    client = _client(
        {
            candidate: _response(
                candidate, '<meta name="citation_pdf_url" content="/article.pdf">'
            ),
            LANDING_URL: _response(
                LANDING_URL, f'<meta name="citation_pdf_url" content="{PDF_URL}">'
            ),
            PDF_URL: _response(PDF_URL, PDF_BYTES, "application/pdf"),
        }
    )
    discovery = PaperDiscovery(
        doi=DOI,
        publisher_url=publisher,
        oa_candidates=[candidate, LANDING_URL],
    )

    result = await unified._try_pdf_sources(DOI, None, client, discovery)

    assert result == (PDF_BYTES, "repository_pdf")
    assert client.get.await_count == 3


@pytest.mark.asyncio
async def test_paper_doi_redirect_identifies_publisher_host():
    resolver = f"https://doi.org/{DOI}"
    publisher = "https://publisher.test/article"
    client = _client(
        {
            resolver: _response(publisher, '<meta name="citation_pdf_url" content="/article.pdf">'),
        }
    )
    discovery = PaperDiscovery(doi=DOI, oa_candidates=[resolver])

    assert await unified._try_pdf_sources(DOI, None, client, discovery) is None
    assert client.get.await_count == 1


@pytest.mark.asyncio
async def test_repository_doi_redirect_can_resolve_a_pdf():
    resolver = "https://doi.org/10.3929/repository-copy"
    client = _client(
        {
            resolver: _response(
                LANDING_URL, '<meta name="citation_pdf_url" content="../files/article.pdf">'
            ),
            PDF_URL: _response(PDF_URL, PDF_BYTES, "application/pdf"),
        }
    )
    discovery = PaperDiscovery(
        doi=DOI,
        publisher_url="https://publisher.test/article",
        oa_candidates=[resolver],
    )

    assert await unified._try_pdf_sources(DOI, None, client, discovery) == (
        PDF_BYTES,
        "repository_pdf",
    )


@pytest.mark.parametrize("target", ["javascript:alert(1)", "file:///tmp/article.pdf", "#download"])
def test_non_http_and_self_links_are_ignored(target):
    html = f'<meta name="citation_pdf_url" content="{target}">'
    assert extract_repository_pdf_url(html, LANDING_URL) is None


@pytest.mark.asyncio
async def test_pdf_magic_wins_over_html_content_type():
    client = _client({PDF_URL: _response(PDF_URL, PDF_BYTES)})
    discovery = PaperDiscovery(doi=DOI, oa_candidates=[PDF_URL])

    assert await unified._try_pdf_sources(DOI, None, client, discovery) == (
        PDF_BYTES,
        "publisher_oa_pdf",
    )
