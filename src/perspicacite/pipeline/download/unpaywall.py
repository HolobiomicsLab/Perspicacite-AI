"""Unpaywall API for open access PDFs.

Register at: https://unpaywall.org/products/api
"""

import os

import httpx

from perspicacite.logging import get_logger
from .base import logger, PDFDownloader


async def get_open_access_url(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
    email: str | None = None,
) -> str | None:
    """
    Query Unpaywall for open access PDF URL.

    Args:
        doi: DOI to lookup
        http_client: Optional HTTP client
        email: Email for Unpaywall API (required). Uses UNPAYWALL_EMAIL env var or config.

    Returns:
        OA PDF URL or None
    """
    # Get email from parameter, env var, or default
    if not email:
        email = os.getenv("UNPAYWALL_EMAIL")
    if not email:
        logger.error("unpaywall_no_email", doi=doi)
        return None

    client = http_client or httpx.AsyncClient(timeout=10.0)
    should_close = http_client is None

    try:
        # Unpaywall API (no key required, but email is required)
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        response = await client.get(url)
        response.raise_for_status()

        data = response.json()

        # Check for best OA location
        if data.get("is_oa") and data.get("best_oa_location"):
            pdf_url = data["best_oa_location"].get("pdf_url")
            if pdf_url:
                logger.info("unpaywall_found", doi=doi, url=pdf_url)
                return pdf_url

        logger.info("unpaywall_no_oa", doi=doi)
        return None

    except Exception as e:
        logger.error("unpaywall_error", doi=doi, error=str(e))
        return None

    finally:
        if should_close:
            await client.aclose()


async def download_from_unpaywall(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
    email: str | None = None,
) -> bytes | None:
    """Download PDF from Unpaywall open access URL.
    
    Args:
        doi: DOI to lookup
        http_client: Optional HTTP client
        email: Email for Unpaywall API
        
    Returns:
        PDF bytes or None
    """
    oa_url = await get_open_access_url(doi, http_client, email)
    if not oa_url:
        return None
    
    logger.info("unpaywall_downloading", doi=doi, url=oa_url)
    downloader = PDFDownloader()
    return await downloader.download(oa_url, http_client)
