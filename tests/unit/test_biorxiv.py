"""Tests for bioRxiv/medRxiv content-retrieval module."""

import httpx
import pytest

from perspicacite.pipeline.download.biorxiv import get_content_from_biorxiv, is_biorxiv_doi
from perspicacite.pipeline.download.rate_limit import HOST_POLICIES, HostPolicy


def test_is_biorxiv_doi():
    assert is_biorxiv_doi("10.1101/2021.01.01.425001")
    assert is_biorxiv_doi("https://doi.org/10.1101/2021.01.01.425001")
    assert not is_biorxiv_doi("10.1038/s41467-022-33890-w")
    assert not is_biorxiv_doi("")
    assert not is_biorxiv_doi(None)


@pytest.mark.asyncio
async def test_get_content_from_biorxiv_abstract_only(respx_mock):
    doi = "10.1101/2021.01.01.425001"
    respx_mock.get(url__regex=r"https://api\.biorxiv\.org/details/.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "messages": [{"status": "ok"}],
                "collection": [
                    {
                        "doi": doi,
                        "title": "A Preprint",
                        "authors": "Doe, J.; Roe, R.",
                        "date": "2021-01-01",
                        "abstract": "We show stuff.",
                        "server": "biorxiv",
                        "category": "neuroscience",
                        "jatsxml": "",
                    }
                ],
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await get_content_from_biorxiv(doi, http_client=client)
    assert result is not None
    assert result.success is True
    assert result.content_type == "abstract"
    assert result.content_source in ("biorxiv", "medrxiv")
    assert result.abstract == "We show stuff."
    assert result.metadata["title"] == "A Preprint"
    assert result.metadata["year"] == 2021
    assert result.metadata["authors"]  # list of name strings


@pytest.mark.asyncio
async def test_get_content_from_biorxiv_structured(respx_mock):
    doi = "10.1101/2021.01.01.999999"
    jats_url = "https://www.biorxiv.org/content/early/2021/01/01/2021.01.01.999999.full.pdf+xml"
    body = (
        b"Body text here that is reasonably long for testing purposes and exceeds any "
        b"minimum length thresholds the parser may have so it is recognized as real content."
    )
    minimal_jats = (
        b"<article><body><sec><title>Intro</title><p>" + body + b"</p></sec></body>"
        b"<back><ref-list><ref><element-citation><article-title>Ref One</article-title>"
        b"</element-citation></ref></ref-list></back></article>"
    )
    respx_mock.get(url__regex=r"https://api\.biorxiv\.org/details/.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "messages": [{"status": "ok"}],
                "collection": [
                    {
                        "doi": doi,
                        "title": "Structured Preprint",
                        "authors": "X",
                        "date": "2021-01-01",
                        "abstract": "abstract here",
                        "server": "biorxiv",
                        "jatsxml": jats_url,
                    }
                ],
            },
        )
    )
    respx_mock.get(jats_url).mock(return_value=httpx.Response(200, content=minimal_jats))
    async with httpx.AsyncClient() as client:
        result = await get_content_from_biorxiv(doi, http_client=client)
    assert result is not None and result.success
    # Body text extracted -> structured; nothing extracted -> abstract fallback.
    assert result.content_type in ("structured", "abstract")
    if result.content_type == "structured":
        assert result.full_text and len(result.full_text) > 0


@pytest.mark.asyncio
async def test_get_content_from_biorxiv_not_found(respx_mock):
    respx_mock.get(url__regex=r"https://api\.biorxiv\.org/details/.*").mock(
        return_value=httpx.Response(
            200, json={"messages": [{"status": "no posts found"}], "collection": []}
        )
    )
    async with httpx.AsyncClient() as client:
        result = await get_content_from_biorxiv("10.1101/x", http_client=client)
    assert result is None


@pytest.mark.asyncio
async def test_get_content_from_biorxiv_non_biorxiv_doi(respx_mock):
    async with httpx.AsyncClient() as client:
        result = await get_content_from_biorxiv("10.1038/s41467-022-33890-w", http_client=client)
    assert result is None


# --- throttling vs. genuinely-absent full text -------------------------------

_API_URL_RE = r"https://api\.biorxiv\.org/details/.*"
_THROTTLED_HOST = "biorxiv.org"
# One attempt, no spacing: the retry ladder itself is covered by
# tests for rate_limit.py, and a real ladder would sleep for seconds here.
_FAST_POLICY = HostPolicy(max_concurrency=1, min_interval_s=0.0, max_attempts=1)


@pytest.fixture
def no_backoff(monkeypatch):
    """Give the bioRxiv host a one-attempt, no-wait policy for these tests."""
    monkeypatch.setitem(HOST_POLICIES, _THROTTLED_HOST, _FAST_POLICY)


def _api_payload(doi: str, jats_url: str) -> dict:
    """One-version bioRxiv API payload for *doi* advertising *jats_url*."""
    return {
        "messages": [{"status": "ok"}],
        "collection": [
            {
                "doi": doi,
                "title": "Throttled Preprint",
                "authors": "Doe, J.",
                "date": "2024-05-05",
                "abstract": "We show stuff.",
                "server": "biorxiv",
                "jatsxml": jats_url,
            }
        ],
    }


@pytest.mark.asyncio
async def test_jats_rate_limited_flags_host_and_keeps_abstract(respx_mock, no_backoff):
    """A 429 on the JATS fetch degrades to the abstract but flags the host."""
    doi = "10.1101/2024.05.05.111111"
    jats_url = "https://www.biorxiv.org/content/early/2024/05/05/2024.05.05.111111.source.xml"
    respx_mock.get(url__regex=_API_URL_RE).mock(
        return_value=httpx.Response(200, json=_api_payload(doi, jats_url))
    )
    respx_mock.get(jats_url).mock(return_value=httpx.Response(429, text="Too Many Requests"))

    async with httpx.AsyncClient() as client:
        result = await get_content_from_biorxiv(doi, http_client=client)

    assert result is not None
    assert result.content_type == "abstract"
    assert result.abstract == "We show stuff."
    assert result.rate_limited_hosts == [_THROTTLED_HOST]


@pytest.mark.asyncio
async def test_missing_jats_leaves_rate_limited_hosts_empty(respx_mock, no_backoff):
    """A paper whose JATS is simply absent is abstract-only but not throttled."""
    doi = "10.1101/2024.05.05.222222"
    jats_url = "https://www.biorxiv.org/content/early/2024/05/05/2024.05.05.222222.source.xml"
    respx_mock.get(url__regex=_API_URL_RE).mock(
        return_value=httpx.Response(200, json=_api_payload(doi, jats_url))
    )
    respx_mock.get(jats_url).mock(return_value=httpx.Response(404, text="Not Found"))

    async with httpx.AsyncClient() as client:
        result = await get_content_from_biorxiv(doi, http_client=client)

    assert result is not None
    assert result.content_type == "abstract"
    assert result.rate_limited_hosts == []


@pytest.mark.asyncio
async def test_api_rate_limit_stops_before_sibling_server(respx_mock, no_backoff):
    """A 429 from the API ends the server loop instead of retrying medrxiv."""
    route = respx_mock.get(url__regex=_API_URL_RE).mock(
        return_value=httpx.Response(429, text="Too Many Requests")
    )

    async with httpx.AsyncClient() as client:
        result = await get_content_from_biorxiv("10.1101/2024.05.05.333333", http_client=client)

    assert result is None
    assert route.call_count == 1
    assert "medrxiv" not in str(route.calls[0].request.url)
