"""Main fallback orchestrator for PDF/content download.

Tries multiple sources in order:
1. Unpaywall (open access; needs contact email)
2. arXiv (open access, no API key)
3. Publisher-specific direct / API routes (ACS, RSC, AAAS, Springer, …)
4. OpenAlex OA PDF URLs (no key; optional ``OPENALEX_MAILTO`` / ``UNPAYWALL_EMAIL``)
5. Europe PMC PDF when the work is in PMC OA (no key)
6. Wiley ``/doi/pdf/`` without TDM token (OA or institutional IP; 10.1002/ DOIs)
7. Wiley TDM API (if token provided)
8. Alternative endpoints (e.g., Sci-Hub)
"""

from typing import Any

import httpx

from perspicacite.logging import get_logger
from .base import logger, DownloadResult, ContentResult
from .unpaywall import download_from_unpaywall
from .arxiv import download_from_arxiv, is_arxiv_doi, is_arxiv_url
from .wiley import download_from_wiley_tdm, download_from_wiley_direct
from .openalex_oa import download_pdf_from_openalex_oa
from .europepmc import download_pdf_from_europepmc, get_fulltext_from_europepmc
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

    PREFERRED FOR: Simple PDF download when you just need the document bytes.
    
    NOTE: For structure-aware chunking or when PDF parsing fails, consider using
    get_content_with_fallback() instead. It can return structured XML/text from
    Elsevier API which preserves document structure (sections, headings, etc.)
    better than raw PDF text extraction.

    Sources (in order):
    1. Unpaywall (open access)
    2. arXiv (open access, no API key)
    3. ACS / RSC / AAAS / Springer (direct or keyed)
    4. OpenAlex OA PDF (no key)
    5. Europe PMC PDF (no key, PMC OA subset)
    6. Wiley direct ``/doi/pdf/`` for typical Wiley DOIs (no TDM token)
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

        # 3. OpenAlex OA (fast, no key — try before paywalled publishers)
        logger.info("pdf_download_trying_openalex_oa", doi=doi)
        pdf_bytes = await download_pdf_from_openalex_oa(doi, client)
        if pdf_bytes:
            return pdf_bytes

        # 4. Europe PMC (OA subset, no key — definitive for PMC papers)
        logger.info("pdf_download_trying_europepmc", doi=doi)
        pdf_bytes = await download_pdf_from_europepmc(doi, client)
        if pdf_bytes:
            return pdf_bytes

        # 5. Try ACS if it's an ACS DOI (may require institutional access)
        if doi and is_acs_doi(doi):
            logger.info("pdf_download_trying_acs", doi=doi)
            pdf_bytes = await download_from_acs(doi, client)
            if pdf_bytes:
                return pdf_bytes

        # 6. Try RSC if it's an RSC DOI (may require institutional access)
        if doi and is_rsc_doi(doi):
            logger.info("pdf_download_trying_rsc", doi=doi)
            pdf_bytes = await download_from_rsc(doi, rsc_api_key, client)
            if pdf_bytes:
                return pdf_bytes

        # 7. Try AAAS/Science if it's an AAAS DOI (may require institutional access)
        if doi and is_aaas_doi(doi):
            logger.info("pdf_download_trying_aaas", doi=doi)
            pdf_bytes = await download_from_aaas(doi, aaas_api_key, client)
            if pdf_bytes:
                return pdf_bytes

        # 8. Try Springer if it's a Springer DOI (may require institutional access)
        if doi and is_springer_doi(doi):
            logger.info("pdf_download_trying_springer", doi=doi)
            pdf_bytes = await download_from_springer(doi, springer_api_key, client)
            if pdf_bytes:
                return pdf_bytes

        # Typical Wiley Online Library DOI prefix — try direct PDF (OA / campus IP)
        if doi and doi.lower().startswith("10.1002/"):
            logger.info("pdf_download_trying_wiley_direct", doi=doi)
            pdf_bytes = await download_from_wiley_direct(doi, client)
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

    PREFERRED FOR: Structure-aware chunking and text extraction
    
    Unlike get_pdf_with_fallback which returns raw PDF bytes, this function
    returns a ContentResult that may contain:
    - PDF bytes (from open access sources)
    - Structured XML/text (from Elsevier API with full document structure)
    
    The structured text from Elsevier is particularly valuable for:
    - Structure-aware chunking (preserving sections, headings, paragraphs)
    - Better semantic understanding (knowing if text is from abstract vs methods)
    - Avoiding PDF parsing errors
    
    Sources (in order):
    1. Unpaywall (PDF)
    2. arXiv (PDF)
    3. ACS / RSC / AAAS / Springer (PDF)
    4. OpenAlex OA PDF
    5. Europe PMC PDF
    6. Wiley direct PDF (10.1002/…)
    7. Wiley TDM API (if token provided)
    8. Elsevier API (structured XML/text, if key provided)
    9. Alternative endpoint (PDF)

    Args:
        doi: DOI to lookup
        url: Optional URL (may be arXiv URL)
        alternative_endpoint: Optional alternative endpoint URL
        http_client: Optional HTTP client
        unpaywall_email: Email for Unpaywall API
        wiley_tdm_token: Wiley TDM API token
        elsevier_api_key: Elsevier API key (for structured text)
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

        logger.info("content_download_trying_openalex_oa", doi=doi)
        pdf_bytes = await download_pdf_from_openalex_oa(doi, client)
        if pdf_bytes:
            return ContentResult(
                success=True,
                content=pdf_bytes,
                content_type="pdf",
                source="openalex_oa",
            )

        logger.info("content_download_trying_europepmc", doi=doi)
        eu_text, eu_sections = await get_fulltext_from_europepmc(doi, client)
        if eu_text:
            return ContentResult(
                success=True,
                content=eu_text,
                content_type="text",
                source="europepmc",
                metadata={"sections": eu_sections} if eu_sections else None,
            )

        if doi and doi.lower().startswith("10.1002/"):
            logger.info("content_download_trying_wiley_direct", doi=doi)
            pdf_bytes = await download_from_wiley_direct(doi, client)
            if pdf_bytes:
                return ContentResult(
                    success=True,
                    content=pdf_bytes,
                    content_type="pdf",
                    source="wiley_direct",
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
