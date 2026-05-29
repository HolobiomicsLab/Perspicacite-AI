"""Alternative endpoint downloader for user-maintained PDF sources.

This module fetches a PDF from a base URL the user controls — typically a
private or institutional repository: a campus proxy, an on-prem PDF cache,
an internal aggregator of pre-cleared PDFs, or similar. The endpoint is
expected to accept ``<base>/<doi>`` and return either an HTML page with
PDF links/embeds or the PDF bytes directly.

The endpoint URL is user-provided and empty by default. Users are
responsible for the legality of any endpoint they configure.
"""

from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from .base import logger


async def download_from_alternative_endpoint(
    doi: str,
    base_url: str,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Fetch a PDF from a user-configured private/institutional endpoint.

    The endpoint should accept ``<base_url>/<doi>`` and return either
    the PDF bytes inline or an HTML page that embeds/links the PDF
    (``<embed>``, ``<iframe>``, or ``<a href="*.pdf">``).

    Args:
        doi: DOI to fetch.
        base_url: Base URL of the user's endpoint (e.g.,
            ``https://pdfs.internal.example.org/``).
        http_client: Optional shared HTTP client.

    Returns:
        PDF bytes, or ``None`` if not found / unreachable.
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

        # Look for PDF in <embed> tags
        embeds = soup.find_all("embed", type="application/pdf")
        for embed in embeds:
            src_raw = embed.get("src")
            if src_raw:
                src: str = str(src_raw)  # bibtexparser values type as str|AttributeValueList
                pdf_url = src if src.startswith(("http:", "https:")) else urljoin(url, src)
                logger.info("alternative_endpoint_pdf_found", source="embed", url=pdf_url)
                pdf_response = await client.get(pdf_url)
                pdf_response.raise_for_status()
                return pdf_response.content

        # Look for PDF in <iframe> tags
        iframes = soup.find_all("iframe")
        for iframe in iframes:
            src_raw = iframe.get("src")
            if src_raw:
                src = str(src_raw)  # bibtexparser values type as str|AttributeValueList
                if ".pdf" in src.lower():
                    pdf_url = src if src.startswith(("http:", "https:")) else urljoin(url, src)
                    logger.info("alternative_endpoint_pdf_found", source="iframe", url=pdf_url)
                    pdf_response = await client.get(pdf_url)
                    pdf_response.raise_for_status()
                    return pdf_response.content

        # Look for PDF links in <a> tags
        links = soup.find_all("a", href=True)
        for link in links:
            href: str = str(link["href"])  # bibtexparser values type as str|AttributeValueList
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
