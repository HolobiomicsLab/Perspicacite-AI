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
    except Exception as exc:
        logger.debug("s2_api_key_config_load_failed", error=str(exc))
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
        source=PaperSource.SEMANTIC_SCHOLAR,
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


def _ss_record_to_oa_like_work(record: dict, *, key: str) -> dict | None:
    """Map an S2 references/citations record to an OpenAlex-like work dict.

    ``key`` is ``citedPaper`` for /references or ``citingPaper`` for /citations.
    Returns None for malformed records.
    """
    paper = record.get(key) or {}
    if not paper:
        return None
    ext_ids = paper.get("externalIds") or {}
    doi = ext_ids.get("DOI")
    arxiv_id = ext_ids.get("ArXiv")

    # Authors → OpenAlex authorships shape so _paper_from_oa_work picks them up
    authorships = []
    for a in paper.get("authors") or []:
        name = (a or {}).get("name", "").strip()
        if name:
            authorships.append({"author": {"display_name": name}})

    # Journal: _paper_from_oa_work reads primary_location.source.display_name
    venue = paper.get("venue") or None
    primary_location: dict = {}
    if venue:
        primary_location = {"source": {"display_name": venue}}

    # NOTE: abstract_inverted_index is None — S2 gives us a plain
    # abstract string, but _paper_from_oa_work's _reconstruct_abstract
    # only consumes inverted indexes. The plain abstract is preserved
    # under "abstract" as a fallback for downstream readers; ExpansionHit
    # tolerates abstract being None from this path (OpenAlex sometimes
    # also returns null inverted indexes).

    return {
        # Stable OA-shaped id from S2 paperId when there's no DOI
        "id": f"https://openalex.org/W_S2_{paper.get('paperId', '')}",
        "doi": (f"https://doi.org/{doi}" if doi else None),
        "title": paper.get("title") or "Untitled",
        "display_name": paper.get("title") or "Untitled",
        "publication_year": paper.get("year"),
        "cited_by_count": paper.get("citationCount"),
        "abstract_inverted_index": None,
        "abstract": paper.get("abstract"),
        "authorships": authorships,
        "primary_location": primary_location,
        # Diagnostic / future-use payload (not consumed by _paper_from_oa_work):
        "metadata": {
            "arxiv_id": arxiv_id,
            "s2_paper_id": paper.get("paperId"),
            "s2_corpus_id": paper.get("corpusId"),
            "ss_is_influential": record.get("isInfluential", False),
        },
    }


_SS_GRAPH_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_SS_REF_CIT_FIELDS = (
    "title,abstract,authors,year,externalIds,citationCount,venue"
)


async def _ss_fetch_graph(
    paper_id: str,
    endpoint: str,           # "references" or "citations"
    *,
    limit: int,
    http_client: httpx.AsyncClient | None,
) -> list[dict]:
    """Shared HTTP path for /references and /citations.

    Returns adapted records (OpenAlex-like dicts). On 4xx / 5xx / network
    error, logs and returns [] — the caller (snowball_expand) treats SS
    failure as a no-op enrichment, not an error.
    """
    normalized = normalize_paper_id(paper_id)
    if not normalized:
        return []

    clamped_limit = max(1, min(int(limit), 1000))

    client = http_client or httpx.AsyncClient(timeout=15.0)
    should_close = http_client is None
    try:
        url = f"{_SS_GRAPH_BASE}/{normalized}/{endpoint}"
        headers: dict[str, str] = {}
        api_key = _get_api_key()
        if api_key:
            headers["x-api-key"] = api_key

        params: dict[str, str | int] = {"fields": _SS_REF_CIT_FIELDS, "limit": clamped_limit}
        response = await client.get(url, params=params, headers=headers)

        # If a configured API key is rejected (403/401), retry without it
        # — SS allows public access at a lower rate, which is fine for one
        # snowball expansion.
        if api_key and response.status_code in (401, 403):
            logger.info(
                "snowball_ss_key_rejected_retrying_unauth",
                paper_id=normalized, endpoint=endpoint, status=response.status_code,
            )
            response = await client.get(url, params=params, headers={})

        if response.status_code == 404:
            logger.info("snowball_ss_paper_not_found", paper_id=normalized, endpoint=endpoint)
            return []
        if response.status_code == 429:
            logger.warning("snowball_ss_rate_limited", paper_id=normalized, endpoint=endpoint)
            return []
        if response.status_code >= 400:
            logger.warning(
                "snowball_ss_error",
                paper_id=normalized,
                endpoint=endpoint,
                status=response.status_code,
            )
            return []

        body = response.json() or {}
        records = body.get("data") or []
        key = "citedPaper" if endpoint == "references" else "citingPaper"
        out: list[dict] = []
        for rec in records:
            mapped = _ss_record_to_oa_like_work(rec, key=key)
            if mapped is not None:
                out.append(mapped)
        return out

    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "snowball_ss_error",
            paper_id=normalized,
            endpoint=endpoint,
            error=str(exc),
        )
        return []

    finally:
        if should_close:
            await client.aclose()


async def fetch_ss_references(
    paper_id: str,
    *,
    limit: int = 100,
    http_client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Papers that ``paper_id`` cites (backward direction).

    Returns OpenAlex-like work dicts consumed by
    ``perspicacite.pipeline.snowball._paper_from_oa_work``. Returns [] on
    any SS-side failure (404 / 429 / 5xx / network).
    """
    return await _ss_fetch_graph(
        paper_id, "references", limit=limit, http_client=http_client,
    )


async def fetch_ss_citations(
    paper_id: str,
    *,
    limit: int = 100,
    http_client: httpx.AsyncClient | None = None,
) -> list[dict]:
    """Papers that cite ``paper_id`` (forward direction).

    Same shape and failure semantics as ``fetch_ss_references``.
    """
    return await _ss_fetch_graph(
        paper_id, "citations", limit=limit, http_client=http_client,
    )
