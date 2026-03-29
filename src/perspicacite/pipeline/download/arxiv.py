"""arXiv PDF download.

arXiv provides free, open access to all papers. No API key required.
Paper URLs can be converted from /abs/ to /pdf/ format.

Website: https://arxiv.org/
API Docs: https://info.arxiv.org/help/api/index.html
"""

import httpx

from perspicacite.logging import get_logger
from .base import logger


def is_arxiv_doi(doi: str) -> bool:
    """Check if DOI is an arXiv DOI."""
    if not doi:
        return False
    doi_lower = doi.lower()
    return (
        doi_lower.startswith("10.48550/") or
        "arxiv" in doi_lower
    )


def is_arxiv_url(url: str) -> bool:
    """Check if URL is an arXiv URL."""
    if not url:
        return False
    return "arxiv.org" in url.lower()


def get_arxiv_id_from_doi(doi: str) -> str | None:
    """Extract arXiv ID from DOI.
    
    Handles formats:
    - 10.48550/arXiv.2101.12345
    - 10.48550/arXiv:2101.12345
    - arXiv.2101.12345
    - arXiv:2101.12345
    """
    if not doi:
        return None
    
    doi_lower = doi.lower()
    
    # Handle 10.48550/arXiv.xxxxx format
    if "arxiv" in doi_lower:
        # Extract after arxiv. or arxiv:
        for prefix in ["arxiv.", "arxiv:", "/arxiv.", "/arxiv:"]:
            if prefix in doi_lower:
                parts = doi_lower.split(prefix, 1)
                if len(parts) > 1:
                    return parts[1].strip()
    
    return None


def convert_abs_to_pdf_url(url: str) -> str | None:
    """Convert arXiv abstract URL to PDF URL.
    
    Examples:
    - https://arxiv.org/abs/2101.12345 -> https://arxiv.org/pdf/2101.12345
    - https://arxiv.org/abs/2101.12345v2 -> https://arxiv.org/pdf/2101.12345v2
    """
    if not url:
        return None
    
    url_lower = url.lower()
    
    # Handle /abs/ URLs
    if "/abs/" in url_lower:
        arxiv_id = url.split("/abs/")[-1].split("?")[0].split("#")[0]
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}"
    
    return None


async def download_from_arxiv(
    identifier: str | None = None,
    doi: str | None = None,
    url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Download PDF from arXiv.
    
    Args:
        identifier: Direct arXiv ID (e.g., "2101.12345")
        doi: DOI that might be an arXiv DOI
        url: URL that might be an arXiv URL
        http_client: Optional HTTP client
        
    Returns:
        PDF bytes or None
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None
    
    pdf_url = None
    arxiv_id = None
    
    try:
        # Determine arXiv ID from various inputs
        if identifier:
            arxiv_id = identifier
            logger.info("arxiv_using_identifier", arxiv_id=arxiv_id)
        elif doi and is_arxiv_doi(doi):
            arxiv_id = get_arxiv_id_from_doi(doi)
            logger.info("arxiv_extracted_from_doi", doi=doi, arxiv_id=arxiv_id)
        elif url and is_arxiv_url(url):
            pdf_url = convert_abs_to_pdf_url(url)
            if pdf_url:
                logger.info("arxiv_converted_url", original_url=url, pdf_url=pdf_url)
        
        # Build PDF URL from arXiv ID
        if arxiv_id and not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            logger.info("arxiv_pdf_url", arxiv_id=arxiv_id, url=pdf_url)
        
        if not pdf_url:
            logger.debug("arxiv_no_valid_identifier")
            return None
        
        # Download PDF
        logger.info("arxiv_downloading", url=pdf_url)
        response = await client.get(pdf_url)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type or response.content.startswith(b"%PDF"):
            logger.info("arxiv_success", size_bytes=len(response.content))
            return response.content
        else:
            logger.warning("arxiv_not_pdf", content_type=content_type)
            return None
            
    except httpx.HTTPStatusError as e:
        logger.error(
            "arxiv_http_error",
            status=e.response.status_code,
            url=pdf_url if pdf_url else "unknown",
        )
        return None
    except Exception as e:
        logger.error("arxiv_error", error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()
