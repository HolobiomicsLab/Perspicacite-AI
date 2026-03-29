"""Wiley TDM (Text and Data Mining) API.

Register at: https://developer.wiley.com/api/wiley-text-and-data-mining-tdm/

Note: Requires a Wiley TDM Client Token. This is different from a regular
Wiley API key - you need to specifically register for TDM access.
"""

import httpx

from perspicacite.logging import get_logger
from .base import logger


async def download_from_wiley_tdm(
    doi: str,
    api_token: str,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """
    Download PDF from Wiley TDM (Text and Data Mining) API.

    Args:
        doi: DOI to download
        api_token: Wiley TDM API client token
        http_client: Optional HTTP client

    Returns:
        PDF bytes or None if download failed
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None

    try:
        # Clean DOI - remove prefix if present
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{clean_doi}"

        logger.info("wiley_tdm_attempt", doi=doi, url=url)

        headers = {
            "Wiley-TDM-Client-Token": api_token,
            "User-Agent": "Perspicacite/2.0",
        }

        response = await client.get(url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        # Check if we got a PDF
        if "pdf" in content_type or response.content.startswith(b"%PDF"):
            logger.info("wiley_tdm_success", doi=doi, size_bytes=len(response.content))
            return response.content
        else:
            logger.warning("wiley_tdm_not_pdf", doi=doi, content_type=content_type)
            return None

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.warning("wiley_tdm_not_entitled", doi=doi)
        else:
            logger.error(
                "wiley_tdm_http_error",
                doi=doi,
                status=e.response.status_code,
            )
        return None
    except Exception as e:
        logger.error("wiley_tdm_error", doi=doi, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()
