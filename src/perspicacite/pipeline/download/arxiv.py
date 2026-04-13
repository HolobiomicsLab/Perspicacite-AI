"""arXiv PDF download.

arXiv provides free, open access to all papers. No API key required.
Paper URLs can be converted from /abs/ to /pdf/ format.

Website: https://arxiv.org/
API Docs: https://info.arxiv.org/help/api/index.html
"""

import re

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


# Regex for arXiv URL patterns:
# /abs/2604.06788, /html/2604.06788v1, /pdf/2604.06788, /format/2604.06788
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|html|pdf|format)/(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def get_arxiv_id_from_url(url: str) -> str | None:
    """Extract arXiv ID from any arXiv URL (/abs/, /html/, /pdf/, /format/)."""
    if not url:
        return None
    m = _ARXIV_URL_RE.search(url)
    return m.group(1) if m else None


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


async def fetch_arxiv_html(
    arxiv_id: str,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[str | None, dict[str, str] | None, dict[str, str | None] | None]:
    """Fetch and parse arXiv HTML version of a paper.

    arXiv serves HTML at https://arxiv.org/html/{id} for many papers.
    Returns structured text with sections when available.

    Args:
        arxiv_id: arXiv identifier (e.g., "2401.12345")
        http_client: Optional HTTP client

    Returns:
        (full_text, sections, page_title) or (None, None, None) if unavailable.
        page_title is extracted from the HTML <title> tag or first <h1>.
    """
    client = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    should_close = http_client is None

    try:
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        logger.info("arxiv_html_attempt", arxiv_id=arxiv_id, url=html_url)
        response = await client.get(html_url)
        if response.status_code != 200:
            logger.info(
                "arxiv_html_not_available",
                arxiv_id=arxiv_id,
                status=response.status_code,
            )
            return None, None, None

        html = response.text

        try:
            from bs4 import BeautifulSoup, Tag, NavigableString
        except ImportError:
            # Fallback: just extract raw text
            text = response.text
            if len(text) > 500:
                return text, None, None
            return None, None, None

        soup = BeautifulSoup(html, "html.parser")

        # Extract page title from <title> or first <h1> before decomposing elements
        page_title: str | None = None
        title_tag = soup.find("title")
        if title_tag:
            raw_title = title_tag.get_text(strip=True)
            # arXiv <title> often has format "Title [arXiv:ID]" or "Title - arXiv"
            for sep in (" [arXiv:", " - arXiv", " | arXiv"):
                if sep in raw_title:
                    raw_title = raw_title[: raw_title.index(sep)].strip()
            if raw_title:
                page_title = raw_title
        if not page_title:
            h1 = soup.find("h1", class_="title") or soup.find("h1")
            if h1:
                h1_text = h1.get_text(strip=True)
                # Strip "Title:" prefix that arXiv sometimes adds
                if h1_text.lower().startswith("title:"):
                    h1_text = h1_text[6:].strip()
                if len(h1_text) > 5:
                    page_title = h1_text

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        # Find main content area
        main = soup.find("article") or soup.find("main") or soup.find("body") or soup

        # Extract sections from h2/h3 headers and section elements
        sections: dict[str, str] = {}
        current_title = "Preamble"
        current_parts: list[str] = []

        for child in main.children:
            if isinstance(child, NavigableString):
                text = child.strip()
                if text:
                    current_parts.append(text)
                continue
            if not isinstance(child, Tag):
                continue

            tag_name = (child.name or "").lower()

            if tag_name in ("h1", "h2", "h3"):
                # Flush previous section
                if current_parts:
                    sections[current_title] = "\n".join(current_parts)
                    current_parts = []
                current_title = child.get_text(strip=True) or current_title
            elif tag_name == "section":
                sec_header = child.find(["h1", "h2", "h3"])
                sec_title = (
                    sec_header.get_text(strip=True) if sec_header else "Section"
                )
                sec_text = child.get_text(separator="\n", strip=True)
                if sec_text and len(sec_text) > 50:
                    sections[sec_title] = sec_text
            else:
                text = child.get_text(strip=True)
                if text:
                    current_parts.append(text)

        # Flush last section
        if current_parts:
            sections[current_title] = "\n".join(current_parts)

        # Full text
        full_text = main.get_text(separator="\n", strip=True)

        if full_text and len(full_text) > 200:
            logger.info(
                "arxiv_html_success",
                arxiv_id=arxiv_id,
                text_length=len(full_text),
                sections=len(sections),
            )
            return full_text, sections if sections else None, page_title

        logger.info("arxiv_html_too_short", arxiv_id=arxiv_id)
        return None, None, None

    except Exception as e:
        logger.error("arxiv_html_error", arxiv_id=arxiv_id, error=str(e))
        return None, None, None
    finally:
        if should_close:
            await client.aclose()
