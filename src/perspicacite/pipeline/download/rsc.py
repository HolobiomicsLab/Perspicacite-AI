"""RSC (Royal Society of Chemistry) Publications.

Supports all RSC journals (Chemical Science, ChemComm, etc.)

Some articles are open access (RSC Gold, Chemical Science is fully OA).
Access to others requires institutional subscription.

Website: https://pubs.rsc.org/
Developer portal: https://developer.rsc.org/ (register for API keys and confirm
endpoint paths; RSC may change API versions.)

When an API key is set we call ``GET https://api.rsc.org/articles/{doi}/pdf``
with header ``apikey`` as used in common RSC API examples—validate against your
contract. The direct ``pubs.rsc.org`` PDF URL is a **website** pattern, not
necessarily stable for automation; prefer the official API when you have a key.
"""

import httpx

from perspicacite.logging import get_logger
from .base import logger


def is_rsc_doi(doi: str) -> bool:
    """Check if DOI belongs to RSC."""
    if not doi:
        return False
    return doi.lower().startswith("10.1039/")


async def download_from_rsc(
    doi: str,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Download PDF from RSC.
    
    Strategy:
    1. Try RSC API if key provided
    2. Try direct PDF link (may work with institutional IP or open access)
    
    Args:
        doi: DOI to download
        api_key: RSC API key (optional, for enhanced access)
        http_client: Optional HTTP client
        
    Returns:
        PDF bytes or None
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None
    
    try:
        # Clean DOI
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        logger.info("rsc_attempt", doi=clean_doi)
        
        # Try RSC API if key provided
        if api_key:
            url = f"https://api.rsc.org/articles/{clean_doi}/pdf"
            headers = {
                "apikey": api_key,
                "User-Agent": "Perspicacite/2.0",
            }
            
            logger.info("rsc_trying_api", doi=clean_doi)
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" in content_type or response.content.startswith(b"%PDF"):
                    logger.info("rsc_api_success", doi=clean_doi)
                    return response.content
        
        # Try direct PDF link
        # RSC PDF URLs follow pattern: https://pubs.rsc.org/en/content/articlepdf/YYYY/JN/JNXXXXXXA
        # But we can use the DOI resolver which redirects appropriately
        pdf_url = f"https://pubs.rsc.org/en/content/articlepdf/{clean_doi}"
        
        logger.info("rsc_trying_direct", url=pdf_url)
        response = await client.get(pdf_url)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type or response.content.startswith(b"%PDF"):
            logger.info("rsc_direct_success", doi=clean_doi, size_bytes=len(response.content))
            return response.content
        else:
            logger.warning("rsc_not_pdf", content_type=content_type)
            return None
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.warning("rsc_access_denied", doi=doi)
        elif e.response.status_code == 404:
            logger.warning("rsc_not_found", doi=doi)
        else:
            logger.error(
                "rsc_http_error",
                doi=doi,
                status=e.response.status_code,
            )
        return None
    except Exception as e:
        logger.error("rsc_error", doi=doi, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()


async def check_rsc_open_access(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Check if RSC article is open access.
    
    Args:
        doi: DOI to check
        http_client: Optional HTTP client
        
    Returns:
        True if article appears to be open access
    """
    client = http_client or httpx.AsyncClient(timeout=10.0)
    should_close = http_client is None
    
    try:
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        url = f"https://pubs.rsc.org/en/content/articlelanding/{clean_doi}"
        
        response = await client.get(url)
        
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Check for OA indicators
        oa_indicators = [
            "open access",
            "gold open access",
            "free access",
            "creative commons",
            "cc-by",
        ]
        
        for indicator in oa_indicators:
            if indicator in html:
                logger.info("rsc_detected_oa", doi=doi, indicator=indicator)
                return True
        
        return False
        
    except Exception as e:
        logger.error("rsc_oa_check_error", doi=doi, error=str(e))
        return False
    finally:
        if should_close:
            await client.aclose()
