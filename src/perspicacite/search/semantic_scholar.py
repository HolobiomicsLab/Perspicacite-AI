"""Direct Semantic Scholar API client for paper-by-ID lookup.

Unlike the SciLEx adapter (which does keyword search), this module uses
Semantic Scholar's paper retrieval endpoint for direct lookups by DOI,
arXiv ID, PMID, or Semantic Scholar corpus ID.

API docs: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_retrieval
"""

import os
import re

import httpx

from perspicacite.logging import get_logger
from perspicacite.models.papers import Author, Paper, PaperSource

logger = get_logger("perspicacite.search.semantic_scholar")

# Fields to request from the Semantic Scholar API
_S2_FIELDS = (
    "title,abstract,authors,year,externalIds,"
    "citationCount,venue,openAccessPdf,url"
)

# Patterns for normalizing paper IDs
_DOI_RE = re.compile(r"^10\.\d{4,9}/")
_ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
_OLD_ARXIV_RE = re.compile(r"^[a-z-]+/\d{7}$", re.IGNORECASE)
_PMID_RE = re.compile(r"^\d{7,8}$")


def _get_api_key() -> str | None:
    """Resolve Semantic Scholar API key from environment or config.yml.

    Priority: env vars, then config.yml (pdf_download.semantic_scholar_api_key).
    """
    # 1. Environment variables
    key = (
        os.environ.get("SCILEX_SEMANTIC_SCHOLAR_API_KEY")
        or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    )
    if key:
        return key

    # 2. config.yml
    try:
        from perspicacite.config.loader import load_config
        config = load_config()
        return config.pdf_download.semantic_scholar_api_key
    except Exception:
        return None


def normalize_paper_id(raw_id: str) -> str:
    """Normalize a paper identifier for the Semantic Scholar API.

    The /paper/{paper_id} endpoint requires:
    - DOIs prefixed with ``DOI:``
    - arXiv IDs prefixed with ``ArXiv:``
    - PMIDs prefixed with ``PMID:``
    - S2 corpus IDs prefixed with ``CorpusId:``
    - Plain URLs passed as-is

    Returns the normalized identifier string.
    """
    if not raw_id:
        return raw_id

    s = raw_id.strip()

    # Already has a known prefix — return as-is
    for prefix in ("DOI:", "ArXiv:", "PMID:", "CorpusId:", "https://", "http://"):
        if s.startswith(prefix):
            return s

    # DOI: starts with 10.XXXX/
    if _DOI_RE.match(s):
        return f"DOI:{s}"

    # Modern arXiv ID: YYYY.NNNNN or YYYY.NNNNNvN
    if _ARXIV_RE.match(s):
        return f"ArXiv:{s}"

    # Old-style arXiv ID: arch-ive/NNNNNNN
    if _OLD_ARXIV_RE.match(s):
        return f"ArXiv:{s}"

    # PMID: pure 7-8 digit number (but could be ambiguous, only if clearly a PMID)
    # We don't auto-detect PMIDs to avoid false positives with other numeric IDs.
    # Callers should prefix PMIDs explicitly: "PMID:12345678"

    # Default: pass through (could be S2 paper ID or something else)
    return s


def _map_s2_response(data: dict) -> Paper:
    """Map a Semantic Scholar API response to a Paper model."""
    ext_ids = data.get("externalIds") or {}

    # Extract DOI
    doi = ext_ids.get("DOI")

    # Build paper ID from DOI or S2 paperId
    paper_id = f"doi:{doi}" if doi else data.get("paperId", "unknown")

    # Authors
    authors = []
    for a in data.get("authors") or []:
        name = a.get("name", "").strip()
        if name:
            authors.append(Author(name=name))

    # Year
    year = data.get("year")
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    # Open access PDF
    pdf_url = None
    oa_pdf = data.get("openAccessPdf")
    if isinstance(oa_pdf, dict):
        pdf_url = oa_pdf.get("url")

    return Paper(
        id=paper_id,
        title=data.get("title") or "Untitled",
        authors=authors,
        abstract=data.get("abstract"),
        year=year,
        journal=data.get("venue") or None,
        doi=doi,
        pmid=ext_ids.get("PubMed"),
        url=data.get("url"),
        pdf_url=pdf_url,
        citation_count=data.get("citationCount"),
        source=PaperSource.WEB_SEARCH,
        metadata={
            "s2_paper_id": data.get("paperId"),
            "s2_arxiv_id": ext_ids.get("ArXiv"),
            "s2_corpus_id": ext_ids.get("CorpusId"),
        },
    )


async def lookup_paper(
    paper_id: str,
    http_client: httpx.AsyncClient | None = None,
) -> Paper | None:
    """Look up a single paper by DOI, arXiv ID, PMID, or S2 ID.

    Args:
        paper_id: Paper identifier (DOI, arXiv ID, PMID, S2 ID, or URL).
            DOIs are auto-prefixed with ``DOI:``, arXiv IDs with ``ArXiv:``.
        http_client: Optional reusable HTTP client.

    Returns:
        Paper model or None if the paper was not found or the request failed.
    """
    normalized = normalize_paper_id(paper_id)
    if not normalized:
        return None

    client = http_client or httpx.AsyncClient(timeout=15.0)
    should_close = http_client is None

    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{normalized}"
        headers = {}
        api_key = _get_api_key()
        if api_key:
            headers["x-api-key"] = api_key

        response = await client.get(
            url, params={"fields": _S2_FIELDS}, headers=headers,
        )

        if response.status_code == 404:
            logger.debug(f"S2 paper not found: {normalized}")
            return None

        if response.status_code == 429:
            logger.warning(f"S2 rate limited for: {normalized}")
            return None

        response.raise_for_status()
        data = response.json()

        if not data or not data.get("paperId"):
            return None

        paper = _map_s2_response(data)
        logger.info(
            f"S2 lookup success: {paper.title[:60]}",
            s2_id=data.get("paperId"),
            doi=paper.doi,
        )
        return paper

    except Exception as e:
        logger.error(f"S2 lookup error for {normalized}: {e}")
        return None

    finally:
        if should_close:
            await client.aclose()
