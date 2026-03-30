#!/usr/bin/env python3
"""Live (non-mocked) tests for publisher PDF/content download APIs.

These tests issue real HTTP requests. They are marked ``live``; run them with::

    pytest tests/test_publisher_api_live.py -m live -v -s

Or run a single test::

    pytest tests/test_publisher_api_live.py::TestLivePublisherAPIs::test_live_arxiv_pdf -v -s

**Credentials** are read from the environment first, then from ``config.yml`` via
``load_config().pdf_download`` when unset.

Environment variables (set as needed):

- ``UNPAYWALL_EMAIL`` — optional; used when exercising ``get_pdf_with_fallback`` with arXiv.
- ``WILEY_TDM_TOKEN`` + ``WILEY_TEST_DOI`` — Wiley TDM (DOI must be one your token can access).
- ``ELSEVIER_API_KEY`` + ``ELSEVIER_TEST_DOI`` — ScienceDirect content API.
- ``AAAS_API_KEY`` + ``AAAS_TEST_DOI`` — Science family PDF.
- ``RSC_API_KEY`` (optional) + ``RSC_TEST_DOI`` — RSC PDF (**required** for the RSC live test).
- ``SPRINGER_API_KEY`` + ``SPRINGER_TEST_DOI`` — Springer / Nature PDF via API.
- ``ACS_TEST_DOI`` — **required** for the ACS live test (use a 10.1021/ article your network can fetch).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _setting(env_name: str, config_attr: str) -> str | None:
    v = os.getenv(env_name)
    if v:
        return v.strip() or None
    try:
        from perspicacite.config.loader import load_config

        cfg = load_config()
        return getattr(cfg.pdf_download, config_attr, None)
    except Exception:
        return None


def _is_pdf(data: bytes) -> bool:
    return bool(data) and data[:4] == b"%PDF" and len(data) > 2000


@pytest.mark.live
@pytest.mark.integration
class TestLivePublisherAPIs:
    """Real network calls to publisher modules and fallback orchestrator."""

    @pytest.mark.asyncio
    async def test_live_arxiv_pdf(self) -> None:
        """arXiv: no API key; should return a PDF for a standard arXiv DOI."""
        from perspicacite.pipeline.download.arxiv import download_from_arxiv

        doi = os.getenv("ARXIV_TEST_DOI", "10.48550/arXiv.1706.03762")
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await download_from_arxiv(doi=doi, http_client=http_client)
        assert pdf is not None, "arXiv download returned None"
        assert _is_pdf(pdf), f"Expected PDF magic and size; got {len(pdf or b'')} bytes"

    @pytest.mark.asyncio
    async def test_live_arxiv_via_get_pdf_with_fallback(self) -> None:
        """Fallback path: Unpaywall then arXiv branch for an arXiv DOI."""
        from perspicacite.pipeline.download import get_pdf_with_fallback

        email = _setting("UNPAYWALL_EMAIL", "unpaywall_email")
        doi = os.getenv("ARXIV_TEST_DOI", "10.48550/arXiv.1706.03762")
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await get_pdf_with_fallback(
                doi,
                http_client=http_client,
                unpaywall_email=email,
            )
        assert pdf is not None
        assert _is_pdf(pdf)

    @pytest.mark.asyncio
    async def test_live_wiley_tdm(self) -> None:
        from perspicacite.pipeline.download.wiley import download_from_wiley_tdm

        token = _setting("WILEY_TDM_TOKEN", "wiley_tdm_token")
        doi = os.getenv("WILEY_TEST_DOI")
        if not token or not doi:
            pytest.skip("Set WILEY_TDM_TOKEN and WILEY_TEST_DOI (DOI your token can access)")

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await download_from_wiley_tdm(doi, token, http_client=http_client)
        assert pdf is not None, "Wiley TDM returned None (403/404 or non-PDF)"
        assert _is_pdf(pdf), "Wiley response was not a valid PDF"

    @pytest.mark.asyncio
    async def test_live_elsevier_content(self) -> None:
        from perspicacite.pipeline.download.elsevier import get_content_from_elsevier

        key = _setting("ELSEVIER_API_KEY", "elsevier_api_key")
        doi = os.getenv("ELSEVIER_TEST_DOI")
        if not key:
            pytest.skip("Set ELSEVIER_API_KEY for live Elsevier test")
        if not doi:
            pytest.skip("Set ELSEVIER_TEST_DOI (a DOI your API key is entitled to)")

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            result = await get_content_from_elsevier(doi, key, http_client=http_client)
        assert result.success, f"Elsevier API failed: {result.error}"
        assert result.content and len(result.content.strip()) > 200, (
            "Elsevier returned empty or trivial content"
        )

    @pytest.mark.asyncio
    async def test_live_aaas_pdf(self) -> None:
        from perspicacite.pipeline.download.aaas import download_from_aaas

        key = _setting("AAAS_API_KEY", "aaas_api_key")
        doi = os.getenv("AAAS_TEST_DOI")
        if not key or not doi:
            pytest.skip("Set AAAS_API_KEY and AAAS_TEST_DOI (a DOI you can access)")

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await download_from_aaas(doi, api_key=key, http_client=http_client)
        assert pdf is not None, "AAAS download returned None"
        assert _is_pdf(pdf)

    @pytest.mark.asyncio
    async def test_live_rsc_pdf(self) -> None:
        from perspicacite.pipeline.download.rsc import download_from_rsc

        doi = os.getenv("RSC_TEST_DOI")
        if not doi:
            pytest.skip("Set RSC_TEST_DOI to a DOI you can retrieve (OA, API key, or institutional IP)")
        key = _setting("RSC_API_KEY", "rsc_api_key")
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await download_from_rsc(doi, api_key=key, http_client=http_client)
        assert pdf is not None, (
            "RSC download failed (try an open-access DOI, institutional network, or RSC_API_KEY)"
        )
        assert _is_pdf(pdf)

    @pytest.mark.asyncio
    async def test_live_springer_pdf(self) -> None:
        from perspicacite.pipeline.download.springer import download_from_springer

        key = _setting("SPRINGER_API_KEY", "springer_api_key")
        doi = os.getenv("SPRINGER_TEST_DOI")
        if not key or not doi:
            pytest.skip("Set SPRINGER_API_KEY and SPRINGER_TEST_DOI (DOI you are entitled to)")

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await download_from_springer(doi, api_key=key, http_client=http_client)
        assert pdf is not None
        assert _is_pdf(pdf)

    @pytest.mark.asyncio
    async def test_live_acs_pdf(self) -> None:
        from perspicacite.pipeline.download.acs import download_from_acs

        doi = os.getenv("ACS_TEST_DOI")
        if not doi:
            pytest.skip(
                "Set ACS_TEST_DOI to a 10.1021/ article PDF your environment can download (often needs IP access)"
            )
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await download_from_acs(doi, http_client=http_client)
        assert pdf is not None, (
            "ACS PDF download failed (set ACS_TEST_DOI to a known OA 10.1021/ article)"
        )
        assert _is_pdf(pdf)

    @pytest.mark.asyncio
    async def test_live_get_pdf_with_fallback_publisher_chain_wiley(self) -> None:
        """End-to-end fallback with Wiley token (requires DOI your token can access)."""
        from perspicacite.pipeline.download import get_pdf_with_fallback

        token = _setting("WILEY_TDM_TOKEN", "wiley_tdm_token")
        doi = os.getenv("WILEY_TEST_DOI")
        if not token or not doi:
            pytest.skip("Set WILEY_TDM_TOKEN and WILEY_TEST_DOI")

        email = _setting("UNPAYWALL_EMAIL", "unpaywall_email")
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http_client:
            pdf = await get_pdf_with_fallback(
                doi,
                http_client=http_client,
                unpaywall_email=email,
                wiley_tdm_token=token,
            )
        assert pdf is not None
        assert _is_pdf(pdf)
