"""Base classes and utilities for download modules."""

from dataclasses import dataclass
from typing import Any

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download")


@dataclass
class DownloadResult:
    """Result of a PDF download attempt."""
    success: bool
    content: bytes | None
    source: str  # e.g., "unpaywall", "wiley", "alternative"
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ContentResult:
    """Result of a content download attempt (text/XML)."""
    success: bool
    content: str | None
    content_type: str  # "pdf", "text", "xml"
    source: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PaperDiscovery:
    """Result of DOI source discovery via OpenAlex + Unpaywall."""

    doi: str
    pmcid: str | None = None
    arxiv_id: str | None = None
    oa_url: str | None = None
    abstract: str | None = None
    title: str | None = None
    is_oa: bool = False
    work_type: str | None = None  # "article", "preprint", etc.
    unpaywall_pdf_url: str | None = None


@dataclass
class PaperContent:
    """Unified result from retrieve_paper_content().

    content_type values:
      - "structured": full text with sections + references (JATS XML, HTML)
      - "full_text": full text from PDF extraction (no structure)
      - "abstract": abstract only (no full text available)
      - "none": no content found
    """

    success: bool
    doi: str
    content_type: str  # "structured" | "full_text" | "abstract" | "none"
    full_text: str | None = None
    sections: dict[str, str] | None = None
    references: list[dict] | None = None
    abstract: str | None = None
    content_source: str = "none"  # "pmc", "arxiv_html", "publisher_pdf", etc.
    metadata: dict[str, Any] | None = None


class PDFDownloader:
    """Generic PDF downloader with retry logic."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries

    async def download(
        self,
        url: str,
        http_client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes | None:
        """Download PDF from URL."""
        client = http_client or httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True
        )
        should_close = http_client is None
        # Browser-like UA prevents NCBI PMC / Europe PMC from serving
        # HTML landing pages instead of actual PDFs.
        merged = {
            "User-Agent": "Mozilla/5.0 (compatible; Perspicacite/2.0)",
            **(headers or {}),
        }

        try:
            logger.info("pdf_download_start", url=url)

            response = await client.get(url, headers=merged, follow_redirects=True)
            response.raise_for_status()

            # Check if content is PDF
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                if not response.content.startswith(b"%PDF"):
                    logger.warning(
                        "pdf_download_not_pdf",
                        url=url,
                        content_type=content_type,
                    )
                    return None

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
