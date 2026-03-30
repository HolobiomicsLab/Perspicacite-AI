"""ACS (American Chemical Society) Publications.

Supports all ACS journals (JACS, ACS Nano, etc.)

Access typically requires institutional subscription.
Some articles are open access (ACS AuthorChoice, ACS Central Science).

Website: https://pubs.acs.org/

This uses the public PDF URL pattern ``https://pubs.acs.org/doi/pdf/{article_id}``
(website delivery, not a separate documented API contract like Elsevier’s).
"""

import httpx

from perspicacite.logging import get_logger
from .base import logger


def is_acs_doi(doi: str) -> bool:
    """Check if DOI belongs to ACS."""
    if not doi:
        return False
    return doi.lower().startswith("10.1021/")


def extract_acs_article_id(doi: str) -> str | None:
    """Extract ACS article ID from DOI.
    
    Example: 10.1021/jacs.9b12345 -> jacs.9b12345
    """
    if not doi:
        return None
    
    # Remove DOI prefix
    prefixes = [
        "https://doi.org/10.1021/",
        "http://doi.org/10.1021/",
        "10.1021/",
    ]
    
    for prefix in prefixes:
        if doi.lower().startswith(prefix.lower()):
            return doi[len(prefix):]
    
    return None


async def download_from_acs(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Download PDF from ACS.
    
    Strategy:
    1. Try direct PDF link (may work with institutional IP or open access)
    2. ACS AuthorChoice articles are open access
    
    Args:
        doi: DOI to download
        http_client: Optional HTTP client
        
    Returns:
        PDF bytes or None
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None
    
    try:
        article_id = extract_acs_article_id(doi)
        if not article_id:
            logger.warning("acs_invalid_doi", doi=doi)
            return None
        
        logger.info("acs_attempt", doi=doi, article_id=article_id)
        
        # ACS PDF URL format
        pdf_url = f"https://pubs.acs.org/doi/pdf/{article_id}"
        
        logger.info("acs_downloading", url=pdf_url)
        response = await client.get(pdf_url)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type or response.content.startswith(b"%PDF"):
            logger.info("acs_success", doi=doi, size_bytes=len(response.content))
            return response.content
        else:
            # Check if we got an HTML error/access denied page
            if b"<html" in response.content[:1000].lower():
                logger.warning("acs_access_denied_or_not_found", doi=doi)
            else:
                logger.warning("acs_not_pdf", doi=doi, content_type=content_type)
            return None
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.warning("acs_access_denied", doi=doi)
        elif e.response.status_code == 404:
            logger.warning("acs_not_found", doi=doi)
        else:
            logger.error(
                "acs_http_error",
                doi=doi,
                status=e.response.status_code,
            )
        return None
    except Exception as e:
        logger.error("acs_error", doi=doi, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()


async def check_acs_open_access(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Check if ACS article is open access.
    
    Args:
        doi: DOI to check
        http_client: Optional HTTP client
        
    Returns:
        True if article appears to be open access
    """
    client = http_client or httpx.AsyncClient(timeout=10.0)
    should_close = http_client is None
    
    try:
        article_id = extract_acs_article_id(doi)
        if not article_id:
            return False
        
        url = f"https://pubs.acs.org/doi/{article_id}"
        response = await client.get(url)
        
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Check for OA indicators
        oa_indicators = [
            "open access",
            "authorchoice",
            "acs central science",  # This journal is fully OA
            "free access",
            "this article is licensed under",
        ]
        
        for indicator in oa_indicators:
            if indicator in html:
                logger.info("acs_detected_oa", doi=doi, indicator=indicator)
                return True
        
        return False
        
    except Exception as e:
        logger.error("acs_oa_check_error", doi=doi, error=str(e))
        return False
    finally:
        if should_close:
            await client.aclose()
