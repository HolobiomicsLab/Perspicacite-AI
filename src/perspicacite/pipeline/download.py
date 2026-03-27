"""Content download utilities with retry logic."""

from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download")


class PDFDownloader:
    """Download PDFs from URLs with retry logic."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def download(self, url: str, http_client: httpx.AsyncClient | None = None) -> bytes | None:
        """
        Download PDF from URL.

        Args:
            url: PDF URL
            http_client: Optional HTTP client

        Returns:
            PDF bytes or None if download failed
        """
        client = http_client or httpx.AsyncClient(timeout=self.timeout)
        should_close = http_client is None

        try:
            logger.info("pdf_download_start", url=url)

            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Check if content is PDF
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                logger.warning(
                    "pdf_download_not_pdf",
                    url=url,
                    content_type=content_type,
                )

            pdf_bytes = response.content

            logger.info(
                "pdf_download_success",
                url=url,
                size_bytes=len(pdf_bytes),
            )

            return pdf_bytes

        except httpx.HTTPStatusError as e:
            logger.error(
                "pdf_download_http_error",
                url=url,
                status=e.response.status_code,
            )
            return None
        except Exception as e:
            logger.error("pdf_download_error", url=url, error=str(e))
            return None

        finally:
            if should_close:
                await client.aclose()

    async def download_and_parse(
        self,
        url: str,
        pdf_parser: Any,
        http_client: httpx.AsyncClient | None = None,
    ) -> str | None:
        """
        Download PDF and extract text.

        Args:
            url: PDF URL
            pdf_parser: PDFParser instance
            http_client: Optional HTTP client

        Returns:
            Extracted text or None if failed
        """
        pdf_bytes = await self.download(url, http_client)
        if not pdf_bytes:
            return None

        try:
            parsed = await pdf_parser.parse(pdf_bytes)
            return parsed.text
        except Exception as e:
            logger.error("pdf_parse_error", url=url, error=str(e))
            return None


async def get_open_access_url(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> str | None:
    """
    Query Unpaywall for open access PDF URL.

    Args:
        doi: DOI to lookup
        http_client: Optional HTTP client

    Returns:
        OA PDF URL or None
    """
    client = http_client or httpx.AsyncClient(timeout=10.0)
    should_close = http_client is None

    try:
        # Unpaywall API (no key required for basic use)
        url = f"https://api.unpaywall.org/v2/{doi}?email=user@example.com"
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


async def get_pdf_from_alternative_endpoint(
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
    from bs4 import BeautifulSoup
    
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
            url=url if 'url' in locals() else base_url,
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


async def get_pdf_with_fallback(
    doi: str,
    alternative_endpoint: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """
    Get PDF for DOI, trying Unpaywall first, then alternative endpoint.
    
    Args:
        doi: DOI to lookup
        alternative_endpoint: Optional alternative endpoint URL
        http_client: Optional HTTP client
        
    Returns:
        PDF bytes or None if not found
    """
    client = http_client or httpx.AsyncClient(timeout=30.0)
    should_close = http_client is None
    
    try:
        # Try Unpaywall first
        oa_url = await get_open_access_url(doi, client)
        if oa_url:
            logger.info("pdf_download_using_unpaywall", doi=doi, url=oa_url)
            downloader = PDFDownloader()
            pdf_bytes = await downloader.download(oa_url, client)
            if pdf_bytes:
                return pdf_bytes
        
        # Try alternative endpoint if provided
        if alternative_endpoint:
            logger.info("pdf_download_trying_alternative", doi=doi, endpoint=alternative_endpoint)
            pdf_bytes = await get_pdf_from_alternative_endpoint(doi, alternative_endpoint, client)
            if pdf_bytes:
                return pdf_bytes
        
        logger.warning("pdf_download_not_found", doi=doi)
        return None
        
    finally:
        if should_close:
            await client.aclose()
