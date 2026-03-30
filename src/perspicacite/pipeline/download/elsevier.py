"""Elsevier ScienceDirect Article (Full Text) Retrieval API.

Official documentation:
https://dev.elsevier.com/documentation/ArticleRetrievalAPI.wadl

We call ``GET https://api.elsevier.com/content/article/doi/{doi}`` with
``X-ELS-APIKey`` and ``Accept: text/xml``. The API defaults to ``view=META``;
we pass ``view=FULL`` so responses include full-text XML when entitled.

Optional headers (not set here) from the same spec: ``Authorization`` (OAuth),
``X-ELS-Authtoken``, ``X-ELS-Insttoken`` for user/institution entitlements.

``Accept: application/pdf`` is also supported by Elsevier for direct PDF bytes.
"""

from typing import Any

import httpx

from perspicacite.logging import get_logger
from .base import logger, ContentResult


async def get_content_from_elsevier(
    doi: str,
    api_key: str,
    http_client: httpx.AsyncClient | None = None,
) -> ContentResult:
    """
    Get article content from Elsevier ScienceDirect API.

    Args:
        doi: DOI to lookup
        api_key: Elsevier API key
        http_client: Optional HTTP client

    Returns:
        ContentResult with extracted text
    """
    client = http_client or httpx.AsyncClient(timeout=30.0)
    should_close = http_client is None

    try:
        # view=FULL required for full article body; default view is META only.
        # See Article Retrieval API query param "view" (META | FULL | ...).
        url = f"https://api.elsevier.com/content/article/doi/{doi}?view=FULL"

        logger.info("elsevier_api_attempt", doi=doi, url=url)

        headers = {
            "X-ELS-APIKey": api_key,
            "User-Agent": "Perspicacite/2.0",
            "Accept": "text/xml",
        }

        response = await client.get(url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "xml" in content_type:
            return await _parse_elsevier_xml(response.text, doi)
        else:
            # Plain text response
            logger.info("elsevier_api_success_text", doi=doi)
            return ContentResult(
                success=True,
                content=response.text,
                content_type="text",
                source="elsevier_api",
            )

    except httpx.HTTPStatusError as e:
        error_msg = None
        if e.response.status_code == 403:
            error_msg = "Not entitled to content"
            logger.warning("elsevier_api_not_entitled", doi=doi)
        elif e.response.status_code == 400:
            error_msg = "Invalid DOI"
            logger.warning("elsevier_api_invalid_doi", doi=doi)
        else:
            error_msg = f"HTTP {e.response.status_code}"
            logger.error("elsevier_api_http_error", doi=doi, status=e.response.status_code)
        
        return ContentResult(
            success=False,
            content=None,
            content_type="text",
            source="elsevier_api",
            error=error_msg,
        )
    except Exception as e:
        logger.error("elsevier_api_error", doi=doi, error=str(e))
        return ContentResult(
            success=False,
            content=None,
            content_type="text",
            source="elsevier_api",
            error=str(e),
        )
    finally:
        if should_close:
            await client.aclose()


async def _parse_elsevier_xml(xml_text: str, doi: str) -> ContentResult:
    """Parse Elsevier XML response and extract text content."""
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_text)

        extracted_text = []
        body_found = False

        # Try to find body and extract paragraphs
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag == 'body':
                body_found = True
                for para_elem in elem.iter():
                    para_tag = para_elem.tag.split('}')[-1] if '}' in para_elem.tag else para_elem.tag
                    if 'para' in para_tag.lower() and para_elem.text:
                        text = para_elem.text.strip()
                        if text and len(text) > 10:
                            extracted_text.append(text)
                break

        # Fallback: if no body found, look for paragraphs anywhere
        if not body_found or len(extracted_text) < 3:
            for elem in root.iter():
                tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if 'para' in tag.lower() and elem.text:
                    text = elem.text.strip()
                    if text and len(text) > 10:
                        extracted_text.append(text)

        if extracted_text:
            content_text = " ".join(extracted_text)
            logger.info(
                "elsevier_api_success",
                doi=doi,
                content_length=len(content_text),
                paragraphs=len(extracted_text),
            )
            return ContentResult(
                success=True,
                content=content_text,
                content_type="text",
                source="elsevier_api",
            )
        else:
            logger.warning("elsevier_api_no_content_extracted", doi=doi)
            return ContentResult(
                success=False,
                content=None,
                content_type="text",
                source="elsevier_api",
                error="No content extracted from XML",
            )

    except ET.ParseError as e:
        logger.error("elsevier_api_xml_parse_error", doi=doi, error=str(e))
        return ContentResult(
            success=False,
            content=None,
            content_type="text",
            source="elsevier_api",
            error=f"XML parse error: {e}",
        )
