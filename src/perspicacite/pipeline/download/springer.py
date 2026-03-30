"""Springer Nature API.

Supports Springer journals, Nature journals, BioMed Central, etc.

API access requires a key. Some content is open access.
Register at: https://dev.springernature.com/

Meta API (metadata + PDF URLs when present) is documented at:
https://dev.springernature.com/docs/api-endpoints/meta-api/

We use ``GET https://api.springernature.com/meta/v2/json`` with ``q=doi:...``
and ``api_key=...`` as in the official examples (HTTPS required).

Note: PDF URLs from metadata may still require institutional access to fetch.
``link.springer.com/content/pdf/{doi}.pdf`` matches many Springer articles;
Nature / BMC may use other host patterns—fallback direct URL can fail for those.
"""

import httpx

from perspicacite.logging import get_logger
from .base import logger


def is_springer_doi(doi: str) -> bool:
    """Check if DOI belongs to Springer Nature."""
    if not doi:
        return False
    doi_lower = doi.lower()
    # Springer uses 10.1007/, Nature uses 10.1038/
    return (
        doi_lower.startswith("10.1007/") or
        doi_lower.startswith("10.1038/") or
        doi_lower.startswith("10.1186/")  # BioMed Central
    )


def get_springer_journal_from_doi(doi: str) -> str | None:
    """Extract journal identifier from Springer DOI.
    
    Examples:
    - 10.1007/s00216-020-12345 -> s00216 (Analytical and Bioanalytical Chemistry)
    - 10.1038/s41586-020-1234 -> nature
    """
    if not doi:
        return None
    
    doi_lower = doi.lower()
    
    # Nature journals (10.1038/)
    if doi_lower.startswith("10.1038/"):
        return "nature"
    
    # BioMed Central (10.1186/)
    if doi_lower.startswith("10.1186/"):
        return "biomed-central"
    
    # Springer journals (10.1007/)
    if doi_lower.startswith("10.1007/"):
        # Extract journal code (first part after 10.1007/)
        parts = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").split("/")
        if len(parts) >= 2:
            journal_part = parts[1]
            # Handle format like s00216-020-12345
            if "-" in journal_part:
                return journal_part.split("-")[0]
            return journal_part
    
    return None


async def download_from_springer(
    doi: str,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Download PDF from Springer Nature.
    
    Strategy:
    1. Try Springer API if key provided (get PDF URL)
    2. Try direct PDF link (may work with institutional IP or OA)
    
    Args:
        doi: DOI to download
        api_key: Springer Nature API key (optional)
        http_client: Optional HTTP client
        
    Returns:
        PDF bytes or None
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None
    
    try:
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        logger.info("springer_attempt", doi=clean_doi)
        
        # Try API if key provided to get metadata and PDF URL
        if api_key:
            api_url = (
                f"https://api.springernature.com/meta/v2/json"
                f"?q=doi:{clean_doi}&api_key={api_key}"
            )
            
            logger.info("springer_trying_api", doi=clean_doi)
            response = await client.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])
                
                if records:
                    record = records[0]
                    # Look for open access PDF URL
                    urls = record.get("url", [])
                    for url_obj in urls:
                        if url_obj.get("format") == "pdf":
                            pdf_url = url_obj.get("value")
                            if pdf_url:
                                logger.info("springer_api_found_pdf", doi=clean_doi, url=pdf_url)
                                pdf_response = await client.get(pdf_url)
                                if pdf_response.status_code == 200:
                                    content_type = pdf_response.headers.get("content-type", "").lower()
                                    if "pdf" in content_type or pdf_response.content.startswith(b"%PDF"):
                                        logger.info("springer_api_success", doi=clean_doi)
                                        return pdf_response.content
        
        # Try direct PDF link
        # Springer PDF URLs: https://link.springer.com/content/pdf/{doi}.pdf
        pdf_url = f"https://link.springer.com/content/pdf/{clean_doi}.pdf"
        
        logger.info("springer_trying_direct", url=pdf_url)
        response = await client.get(pdf_url)
        response.raise_for_status()
        
        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type or response.content.startswith(b"%PDF"):
            logger.info("springer_direct_success", doi=clean_doi, size_bytes=len(response.content))
            return response.content
        else:
            # Check if we got an HTML error page
            if b"<html" in response.content[:1000].lower():
                logger.warning("springer_access_denied_or_not_found", doi=clean_doi)
            else:
                logger.warning("springer_not_pdf", doi=clean_doi, content_type=content_type)
            return None
            
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.warning("springer_access_denied", doi=doi)
        elif e.response.status_code == 404:
            logger.warning("springer_not_found", doi=doi)
        else:
            logger.error(
                "springer_http_error",
                doi=doi,
                status=e.response.status_code,
            )
        return None
    except Exception as e:
        logger.error("springer_error", doi=doi, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()


async def check_springer_open_access(
    doi: str,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> bool:
    """Check if Springer article is open access.
    
    Args:
        doi: DOI to check
        api_key: Springer API key (optional but recommended)
        http_client: Optional HTTP client
        
    Returns:
        True if article appears to be open access
    """
    client = http_client or httpx.AsyncClient(timeout=10.0)
    should_close = http_client is None
    
    try:
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        
        # Try API if key provided
        if api_key:
            api_url = (
                f"https://api.springernature.com/meta/v2/json"
                f"?q=doi:{clean_doi}&api_key={api_key}"
            )
            response = await client.get(api_url)
            
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])
                
                if records:
                    record = records[0]
                    # Check for open access flags
                    if record.get("openaccess") == "true":
                        logger.info("springer_api_detected_oa", doi=doi)
                        return True
        
        # Fallback: check landing page
        url = f"https://link.springer.com/article/{clean_doi}"
        response = await client.get(url)
        
        if response.status_code != 200:
            return False
        
        html = response.text.lower()
        
        # Check for OA indicators
        oa_indicators = [
            "open access",
            "open choice",
            "creative commons",
            "cc-by",
            "free full text",
        ]
        
        for indicator in oa_indicators:
            if indicator in html:
                logger.info("springer_detected_oa", doi=doi, indicator=indicator)
                return True
        
        return False
        
    except Exception as e:
        logger.error("springer_oa_check_error", doi=doi, error=str(e))
        return False
    finally:
        if should_close:
            await client.aclose()


async def get_springer_metadata(
    doi: str,
    api_key: str,
    http_client: httpx.AsyncClient | None = None,
) -> dict | None:
    """Get metadata for Springer article via API.
    
    Args:
        doi: DOI to lookup
        api_key: Springer API key (required)
        http_client: Optional HTTP client
        
    Returns:
        Metadata dict or None
    """
    if not api_key:
        logger.warning("springer_metadata_no_api_key", doi=doi)
        return None
    
    client = http_client or httpx.AsyncClient(timeout=10.0)
    should_close = http_client is None
    
    try:
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        api_url = (
            f"https://api.springernature.com/meta/v2/json"
            f"?q=doi:{clean_doi}&api_key={api_key}"
        )
        
        response = await client.get(api_url)
        response.raise_for_status()
        
        data = response.json()
        records = data.get("records", [])
        
        if records:
            return records[0]
        return None
        
    except Exception as e:
        logger.error("springer_metadata_error", doi=doi, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()
