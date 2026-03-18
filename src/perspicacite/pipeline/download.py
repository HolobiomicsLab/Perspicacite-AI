"""Content download utilities with retry logic."""

from pathlib import Path
from typing import Any

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
