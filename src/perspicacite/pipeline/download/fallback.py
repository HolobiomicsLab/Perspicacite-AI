"""Main fallback orchestrator for PDF/content download.

Tries multiple sources in order:
1. Unpaywall (open access)
2. arXiv (open access, no API key needed)
3. Publisher APIs (Wiley, Elsevier, AAAS, ACS, RSC, etc.)
4. Alternative endpoints (e.g., Sci-Hub)
"""

from typing import Any

import httpx

from perspicacite.logging import get_logger
from .base import logger, DownloadResult, ContentResult
from .unpaywall import download_from_unpaywall
from .arxiv import download_from_arxiv, is_arxiv_doi, is_arxiv_url
from .wiley import download_from_wiley_tdm
from .elsevier import get_content_from_elsevier
from .aaas import download_from_aaas, is_aaas_doi
from .acs import download_from_acs, is_acs_doi
from .rsc import download_from_rsc, is_rsc_doi
from .springer import download_from_springer, is_springer_doi
from .alternative import download_from_alternative_endpoint


async def get_pdf_with_fallback(
    doi: str,
    url: str | None = None,
    alternative_endpoint: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    unpaywall_email: str | None = None,
    wiley_tdm_token: str | None = None,
    aaas_api_key: str | None = None,
    rsc_api_key: str | None = None,
    springer_api_key: str | None = None,
) -> bytes | None:
    """
    Get PDF for DOI, trying multiple sources in order.

    Sources (in order):
    1. Unpaywall (open access)
    2. arXiv (open access, no API key)
    3. ACS (check if OA or institutional access)
    4. RSC (check if OA or institutional access)
    5. AAAS/Science (check if OA or institutional access)
    6. Springer Nature (check if OA or institutional access)
    7. Wiley TDM API (if token provided)
    8. Alternative endpoint (e.g., Sci-Hub)

    Args:
        doi: DOI to lookup
        url: Optional URL (may be arXiv URL)
        alternative_endpoint: Optional alternative endpoint URL
        http_client: Optional HTTP client
        unpaywall_email: Email for Unpaywall API
        wiley_tdm_token: Wiley TDM API token for institutional access
        aaas_api_key: AAAS API key
        rsc_api_key: RSC API key

    Returns:
        PDF bytes or None if not found from any source
    """
    client = http_client or httpx.AsyncClient(timeout=30.0)
    should_close = http_client is None

    try:
        # 1. Try Unpaywall first (open access)
        pdf_bytes = await download_from_unpaywall(doi, client, unpaywall_email)
        if pdf_bytes:
            return pdf_bytes

        # 2. Try arXiv (open access, no API key needed)
        if (doi and (is_arxiv_doi(doi) or is_arxiv_url(doi))) or (url and is_arxiv_url(url)):
            logger.info("pdf_download_trying_arxiv", doi=doi)
            pdf_bytes = await download_from_arxiv(doi=doi, url=url, http_client=client)
            if pdf_bytes:
                return pdf_bytes

        # 3. Try ACS if it's an ACS DOI
        if doi and is_acs_doi(doi):
            logger.info("pdf_download_trying_acs", doi=doi)
            pdf_bytes = await download_from_acs(doi, client)
            if pdf_bytes:
                return pdf_bytes

        # 4. Try RSC if it's an RSC DOI
        if doi and is_rsc_doi(doi):
            logger.info("pdf_download_trying_rsc", doi=doi)
            pdf_bytes = await download_from_rsc(doi, rsc_api_key, client)
            if pdf_bytes:
                return pdf_bytes

        # 5. Try AAAS/Science if it's an AAAS DOI
        if doi and is_aaas_doi(doi):
            logger.info("pdf_download_trying_aaas", doi=doi)
            pdf_bytes = await download_from_aaas(doi, aaas_api_key, client)
            if pdf_bytes:
                return pdf_bytes

        # 6. Try Springer if it's a Springer DOI
        if doi and is_springer_doi(doi):
            logger.info("pdf_download_trying_springer", doi=doi)
            pdf_bytes = await download_from_springer(doi, springer_api_key, client)
            if pdf_bytes:
                return pdf_bytes

        # 7. Try Wiley TDM API if token is available
        if wiley_tdm_token:
            logger.info("pdf_download_trying_wiley", doi=doi)
            pdf_bytes = await download_from_wiley_tdm(doi, wiley_tdm_token, client)
            if pdf_bytes:
                return pdf_bytes

        # 7. Try alternative endpoint if provided
        if alternative_endpoint:
            logger.info("pdf_download_trying_alternative", doi=doi, endpoint=alternative_endpoint)
            pdf_bytes = await download_from_alternative_endpoint(doi, alternative_endpoint, client)
            if pdf_bytes:
                return pdf_bytes

        logger.warning("pdf_download_not_found", doi=doi)
        return None

    finally:
        if should_close:
            await client.aclose()


async def get_content_with_fallback(
    doi: str,
    url: str | None = None,
    alternative_endpoint: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    unpaywall_email: str | None = None,
    wiley_tdm_token: str | None = None,
    elsevier_api_key: str | None = None,
    aaas_api_key: str | None = None,
    rsc_api_key: str | None = None,
    springer_api_key: str | None = None,
) -> ContentResult:
    """
    Get content (PDF or text) for DOI, trying multiple sources.

    This function is more flexible than get_pdf_with_fallback as it can
    return text content from Elsevier API when PDF is not available.

    Sources (in order):
    1. Unpaywall (PDF)
    2. arXiv (PDF)
    3. ACS (PDF)
    4. RSC (PDF)
    5. AAAS/Science (PDF)
    6. Springer Nature (PDF)
    7. Wiley TDM API (PDF, if token provided)
    8. Elsevier API (text, if key provided)
    9. Alternative endpoint (PDF)

    Args:
        doi: DOI to lookup
        url: Optional URL (may be arXiv URL)
        alternative_endpoint: Optional alternative endpoint URL
        http_client: Optional HTTP client
        unpaywall_email: Email for Unpaywall API
        wiley_tdm_token: Wiley TDM API token
        elsevier_api_key: Elsevier API key
        aaas_api_key: AAAS API key
        rsc_api_key: RSC API key
        springer_api_key: Springer API key

    Returns:
        ContentResult with content and metadata
    """
    client = http_client or httpx.AsyncClient(timeout=30.0)
    should_close = http_client is None

    try:
        # 1. Try Unpaywall first (PDF)
        pdf_bytes = await download_from_unpaywall(doi, client, unpaywall_email)
        if pdf_bytes:
            return ContentResult(
                success=True,
                content=pdf_bytes,
                content_type="pdf",
                source="unpaywall",
            )

        # 2. Try arXiv (PDF)
        if (doi and (is_arxiv_doi(doi) or is_arxiv_url(doi))) or (url and is_arxiv_url(url)):
            logger.info("content_download_trying_arxiv", doi=doi)
            pdf_bytes = await download_from_arxiv(doi=doi, url=url, http_client=client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="arxiv",
                )

        # 3. Try ACS (PDF)
        if doi and is_acs_doi(doi):
            logger.info("content_download_trying_acs", doi=doi)
            pdf_bytes = await download_from_acs(doi, client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="acs",
                )

        # 4. Try RSC (PDF)
        if doi and is_rsc_doi(doi):
            logger.info("content_download_trying_rsc", doi=doi)
            pdf_bytes = await download_from_rsc(doi, rsc_api_key, client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="rsc",
                )

        # 5. Try AAAS (PDF)
        if doi and is_aaas_doi(doi):
            logger.info("content_download_trying_aaas", doi=doi)
            pdf_bytes = await download_from_aaas(doi, aaas_api_key, client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="aaas",
                )

        # 6. Try Springer (PDF)
        if doi and is_springer_doi(doi):
            logger.info("content_download_trying_springer", doi=doi)
            pdf_bytes = await download_from_springer(doi, springer_api_key, client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="springer",
                )

        # 7. Try Wiley TDM API if token is available (PDF)
        if wiley_tdm_token:
            pdf_bytes = await download_from_wiley_tdm(doi, wiley_tdm_token, client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="wiley_tdm",
                )

        # 7. Try Elsevier API if key is available (text)
        if elsevier_api_key:
            result = await get_content_from_elsevier(doi, elsevier_api_key, client)
            if result.success:
                return result

        # 8. Try alternative endpoint (PDF)
        if alternative_endpoint:
            pdf_bytes = await download_from_alternative_endpoint(doi, alternative_endpoint, client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="alternative_endpoint",
                )

        # Nothing worked
        logger.warning("content_download_not_found", doi=doi)
        return ContentResult(
            success=False,
            content=None,
            content_type="unknown",
            source="none",
            error="Content not found from any source",
        )

    finally:
        if should_close:
            await client.aclose()
