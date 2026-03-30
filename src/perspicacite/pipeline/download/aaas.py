"""AAAS / Science family journals — PDF via website URLs.

There is no well-documented public REST API in the same style as Elsevier or
Wiley TDM. This module uses ``https://www.science.org/doi/pdf/{doi}``, which is
the same PDF route as the browser. An optional ``Authorization: Bearer …``
header is sent when ``api_key`` is set; that pattern is **not** verified against
published AAAS API documentation—confirm with AAAS / institutional licensing if
you rely on it.

Licensing and permissions: https://www.science.org/content/page/about-science-licenses-and-permissions
"""

import httpx

from perspicacite.logging import get_logger
from .base import logger


def is_aaas_doi(doi: str) -> bool:
    """Check if DOI belongs to AAAS/Science journals."""
    if not doi:
        return False
    doi_lower = doi.lower()
    # Science journals typically have DOIs like 10.1126/...
    return doi_lower.startswith("10.1126/")


async def download_from_aaas(
    doi: str,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Download PDF from AAAS/Science.
    
    Strategy:
    1. Try institutional API if key provided
    2. Try direct DOI resolution (may work if IP has access)
    
    Args:
        doi: DOI to download
        api_key: AAAS API key (optional, for institutional access)
        http_client: Optional HTTP client
        
    Returns:
        PDF bytes or None
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None
    
    try:
        # Clean DOI
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        logger.info("aaas_attempt", doi=clean_doi)
        
        # Try API if key provided
        if api_key:
            # AAAS doesn't have a public PDF API like Wiley
            # But we can try with authorization headers
            url = f"https://www.science.org/doi/pdf/{clean_doi}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Perspicacite/2.0",
            }
            
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" in content_type or response.content.startswith(b"%PDF"):
                    logger.info("aaas_api_success", doi=clean_doi)
                    return response.content
        
        # Try direct PDF link (may work with institutional IP or open access)
        pdf_url = f"https://www.science.org/doi/pdf/{clean_doi}"
        logger.info("aaas_trying_direct", url=pdf_url)
        
        response = await client.get(pdf_url)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type or response.content.startswith(b"%PDF"):
            logger.info("aaas_direct_success", doi=clean_doi)
            return response.content
        else:
            logger.warning("aaas_not_pdf", content_type=content_type)
            return None
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.warning("aaas_access_denied", doi=doi)
        else:
            logger.error(
                "aaas_http_error",
                doi=doi,
                status=e.response.status_code,
            )
        return None
    except Exception as e:
        logger.error("aaas_error", doi=doi, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()


async def get_aaas_metadata(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> dict | None:
    """Get metadata for AAAS article (useful for checking access).
    
    Args:
        doi: DOI to lookup
        http_client: Optional HTTP client
        
    Returns:
        Metadata dict or None
    """
    client = http_client or httpx.AsyncClient(timeout=10.0)
    should_close = http_client is None
    
    try:
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        url = f"https://www.science.org/doi/{clean_doi}"
        
        response = await client.get(url)
        response.raise_for_status()
        
        # Parse basic metadata from HTML (simplified)
        html = response.text
        is_oa = "Open Access" in html or "oa-icon" in html
        
        return {
            "doi": clean_doi,
            "is_open_access": is_oa,
            "url": url,
        }
        
    except Exception as e:
        logger.error("aaas_metadata_error", doi=doi, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()
