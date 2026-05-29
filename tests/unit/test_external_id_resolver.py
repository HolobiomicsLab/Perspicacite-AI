"""Unit tests for arXiv → DOI and PMC → DOI resolvers.

Covers :mod:`perspicacite.pipeline.external_id_resolver`, the Task-5
follow-up that lets the skill-bundle linked-paper ingest auto-route
arXiv + PMC ids through the existing DOI ingest path instead of
silently surfacing them in ``linked_papers_skipped_non_doi``.

Network is fully stubbed via monkeypatching :meth:`httpx.AsyncClient.get`.
"""

from __future__ import annotations

import httpx
import pytest

from perspicacite.pipeline import external_id_resolver
from perspicacite.pipeline.external_id_resolver import (
    resolve_arxiv_to_doi,
    resolve_pmc_to_doi,
)

# ---------------------------------------------------------------------------
# arXiv → DOI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_arxiv_to_doi_happy_path(monkeypatch):
    """Title lookup + OpenAlex title.search → DOI extracted from Work."""

    async def fake_title(arxiv_id, client):
        return "Attention Is All You Need"

    monkeypatch.setattr(
        external_id_resolver, "resolve_arxiv_title", fake_title
    )

    async def fake_get(self, url, **kwargs):
        req = httpx.Request("GET", url)
        # Step 1: short-circuit lookup `/works/doi:10.48550/arxiv.<id>` → 404
        if "doi:10.48550/arxiv" in str(url).lower():
            return httpx.Response(404, json={}, request=req)
        # Step 2: title.search → 1 result with DOI URL
        params = kwargs.get("params") or {}
        assert "title.search" in (params.get("filter") or "")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "https://openalex.org/W2626778328",
                        "doi": "https://doi.org/10.1038/x",
                    }
                ]
            },
            request=req,
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    doi = await resolve_arxiv_to_doi("2204.12345")
    assert doi == "10.1038/x"


@pytest.mark.asyncio
async def test_resolve_arxiv_to_doi_arxiv_native_doi_short_circuit(
    monkeypatch,
):
    """OpenAlex /works/doi:10.48550/arxiv.<id> returns a Work →
    its DOI is returned directly (no title.search needed)."""
    calls: list[str] = []

    async def fake_get(self, url, **kwargs):
        calls.append(str(url))
        req = httpx.Request("GET", url)
        # Short-circuit: the arXiv-native DOI lookup returns a Work
        # whose DOI is preserved as-is.
        if "doi:10.48550/arxiv" in str(url).lower():
            return httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W3098425262",
                    "doi": "https://doi.org/10.48550/arxiv.2204.12345",
                },
                request=req,
            )
        # If the short-circuit didn't fire we don't get here.
        raise AssertionError("unexpected URL: " + str(url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    doi = await resolve_arxiv_to_doi("2204.12345")
    assert doi == "10.48550/arxiv.2204.12345"
    # Verifies we never had to fall back to the arXiv title API.
    assert len(calls) == 1
    assert "doi:10.48550/arxiv" in calls[0].lower()


@pytest.mark.asyncio
async def test_resolve_arxiv_to_doi_returns_none_on_title_miss(monkeypatch):
    """If the short-circuit misses AND resolve_arxiv_title returns None,
    the resolver returns None without hitting OpenAlex title.search."""

    async def fake_title(arxiv_id, client):
        return None

    monkeypatch.setattr(
        external_id_resolver, "resolve_arxiv_title", fake_title
    )

    async def fake_get(self, url, **kwargs):
        req = httpx.Request("GET", url)
        # Short-circuit miss; afterwards the resolver should bail.
        if "doi:10.48550/arxiv" in str(url).lower():
            return httpx.Response(404, json={}, request=req)
        raise AssertionError(
            "title.search must not be invoked after title miss: " + str(url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await resolve_arxiv_to_doi("2204.12345") is None


@pytest.mark.asyncio
async def test_resolve_arxiv_to_doi_returns_none_on_openalex_miss(
    monkeypatch,
):
    """OpenAlex title.search returns an empty result set → None."""

    async def fake_title(arxiv_id, client):
        return "A nonexistent paper title"

    monkeypatch.setattr(
        external_id_resolver, "resolve_arxiv_title", fake_title
    )

    async def fake_get(self, url, **kwargs):
        req = httpx.Request("GET", url)
        if "doi:10.48550/arxiv" in str(url).lower():
            return httpx.Response(404, json={}, request=req)
        return httpx.Response(200, json={"results": []}, request=req)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await resolve_arxiv_to_doi("2204.12345") is None


@pytest.mark.asyncio
async def test_resolve_arxiv_to_doi_handles_network_error(monkeypatch):
    """``httpx.HTTPError`` anywhere → resolver returns None (caught + logged)."""

    async def fake_title(arxiv_id, client):
        return "Attention Is All You Need"

    monkeypatch.setattr(
        external_id_resolver, "resolve_arxiv_title", fake_title
    )

    async def fake_get(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await resolve_arxiv_to_doi("2204.12345") is None


# ---------------------------------------------------------------------------
# PMC → DOI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_pmc_to_doi_happy_path(monkeypatch):
    """NCBI idconv returns a single record with a DOI → resolver returns it."""

    async def fake_get(self, url, **kwargs):
        req = httpx.Request("GET", url)
        assert "idconv" in str(url)
        params = kwargs.get("params") or {}
        assert params.get("ids") == "PMC9123456"
        assert params.get("format") == "json"
        return httpx.Response(
            200,
            json={
                "records": [
                    {"pmcid": "PMC9123456", "doi": "10.1093/x"}
                ]
            },
            request=req,
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    doi = await resolve_pmc_to_doi("PMC9123456")
    assert doi == "10.1093/x"


@pytest.mark.asyncio
async def test_resolve_pmc_to_doi_no_doi_in_response(monkeypatch):
    """Record present but no ``doi`` key → returns None."""

    async def fake_get(self, url, **kwargs):
        req = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={"records": [{"pmcid": "PMC9123456"}]},
            request=req,
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await resolve_pmc_to_doi("PMC9123456") is None


@pytest.mark.asyncio
async def test_resolve_pmc_to_doi_handles_network_error(monkeypatch):
    """``httpx.HTTPError`` → resolver returns None."""

    async def fake_get(self, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    assert await resolve_pmc_to_doi("PMC9123456") is None


# ---------------------------------------------------------------------------
# DOI normalisation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolved_doi_is_normalised(monkeypatch):
    """Uppercase prefix arriving from upstream → lowercased on return.

    Exercises both resolvers — same `_normalize_doi` helper is reused.
    """
    # arXiv path: OpenAlex short-circuit returns an uppercase-prefix DOI.
    async def fake_get_arxiv(self, url, **kwargs):
        req = httpx.Request("GET", url)
        if "doi:10.48550/arxiv" in str(url).lower():
            return httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1038/UPPER-suffix",
                },
                request=req,
            )
        raise AssertionError("unexpected URL: " + str(url))

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get_arxiv)
    arxiv_doi = await resolve_arxiv_to_doi("2204.12345")
    # Prefix lowercased; suffix preserved per the DOI standard.
    assert arxiv_doi == "10.1038/UPPER-suffix"

    # PMC path: idconv returns uppercase-prefix DOI in `doi` field.
    async def fake_get_pmc(self, url, **kwargs):
        req = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "records": [
                    {"pmcid": "PMC9", "doi": "10.1093/MIXED-suffix"}
                ]
            },
            request=req,
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get_pmc)
    pmc_doi = await resolve_pmc_to_doi("PMC9")
    assert pmc_doi == "10.1093/MIXED-suffix"
