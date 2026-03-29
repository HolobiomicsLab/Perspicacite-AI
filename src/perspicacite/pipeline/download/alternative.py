"""Alternative endpoints (e.g., Sci-Hub mirrors).

WARNING: Using alternative endpoints may violate terms of service or
copyright laws in your jurisdiction. Use at your own risk.
"""

from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from perspicacite.logging import get_logger
from .base import logger


async def download_from_alternative_endpoint(
    doi: str,
    base_url: str,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """
    Try to get PDF from an alternative endpoint (e.g., Sci-Hub).

    The alternative endpoint should accept a DOI appended to the base URL
    and return an HTML page containing a PDF link or embed.

    Args:
        doi: DOI to lookup
        base_url: Base URL of alternative endpoint (e.g., "https://example.com/")
        http_client: Optional HTTP client

    Returns:
        PDF bytes or None if not found
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None

    try:
        # Build URL: base_url + doi
        if not base_url.endswith("/"):
            base_url += "/"
        url = urljoin(base_url, doi)

        logger.info("alternative_endpoint_attempt", doi=doi, url=url)

        # Fetch the HTML page
        response = await client.get(url)
        response.raise_for_status()

        # Parse HTML to find PDF links
        soup = BeautifulSoup(response.text, "html.parser")

        # Look for PDF in <embed> tags (most common in Sci-Hub-like sites)
        embeds = soup.find_all("embed", type="application/pdf")
        for embed in embeds:
            src = embed.get("src")
            if src:
                pdf_url = src if src.startswith(("http:", "https:")) else urljoin(url, src)
                logger.info("alternative_endpoint_pdf_found", source="embed", url=pdf_url)
                pdf_response = await client.get(pdf_url)
                pdf_response.raise_for_status()
                return pdf_response.content

        # Look for PDF in <iframe> tags
        iframes = soup.find_all("iframe")
        for iframe in iframes:
            src = iframe.get("src")
            if src and ".pdf" in src.lower():
                pdf_url = src if src.startswith(("http:", "https:")) else urljoin(url, src)
                logger.info("alternative_endpoint_pdf_found", source="iframe", url=pdf_url)
                pdf_response = await client.get(pdf_url)
                pdf_response.raise_for_status()
                return pdf_response.content

        # Look for PDF links in <a> tags
        links = soup.find_all("a", href=True)
        for link in links:
            href = link["href"]
            if href.endswith(".pdf"):
                pdf_url = href if href.startswith(("http:", "https:")) else urljoin(url, href)
                logger.info("alternative_endpoint_pdf_found", source="link", url=pdf_url)
                pdf_response = await client.get(pdf_url)
                pdf_response.raise_for_status()
                return pdf_response.content

        logger.warning("alternative_endpoint_no_pdf", doi=doi, url=url)
        return None

    except httpx.HTTPStatusError as e:
        logger.error(
            "alternative_endpoint_http_error",
            doi=doi,
            url=base_url,
            status=e.response.status_code,
        )
        return None
    except Exception as e:
        logger.error(
            "alternative_endpoint_error",
            doi=doi,
            url=base_url,
            error=str(e),
        )
        return None
    finally:
        if should_close:
            await client.aclose()
