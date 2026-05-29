"""bioRxiv / medRxiv content retrieval.

Uses the bioRxiv public API to fetch preprint metadata and, when available,
the JATS XML full text.

API reference:
  GET https://api.biorxiv.org/details/{server}/{doi}
  server ∈ {biorxiv, medrxiv}
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

import httpx

from perspicacite.logging import get_logger
from perspicacite.pipeline.download.base import PaperContent
from perspicacite.pipeline.download.pmc import (
    _extract_references_from_xml,
    _extract_sections_from_xml,
    _extract_text_from_xml,
)

logger = get_logger("perspicacite.pipeline.download.biorxiv")

_BIORXIV_API_BASE = "https://api.biorxiv.org/details"
_DOI_PREFIX_RE = re.compile(r"10\.1101/", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_biorxiv_doi(doi: str | None) -> bool:
    """Return True iff *doi* looks like a bioRxiv / medRxiv DOI (prefix 10.1101/).

    Handles bare DOIs and https://doi.org/... prefixed forms.
    Returns False for empty or None input.
    """
    if not doi:
        return False
    return bool(_DOI_PREFIX_RE.search(doi))


def _normalize_doi(doi: str) -> str:
    """Strip common URL prefixes from a DOI string."""
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
    return doi.strip()


def _parse_authors(authors_str: str) -> list[str]:
    """Parse a bioRxiv author string (semicolon or ' and ' separated) into a list."""
    if not authors_str:
        return []
    # Split on semicolons first, then on ' and ' within remaining tokens
    parts: list[str] = []
    for chunk in authors_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Some records use " and " inside or between semicolons
        for sub in re.split(r"\band\b", chunk, flags=re.IGNORECASE):
            sub = sub.strip().strip(",").strip()
            if sub:
                parts.append(sub)
    return parts


async def _fetch_biorxiv_record(doi: str, http_client: httpx.AsyncClient) -> dict[str, Any] | None:
    """Query the bioRxiv API for *doi*, trying biorxiv then medrxiv.

    Returns the *last* record in the collection (most recent version),
    or None if not found.
    """
    for server in ("biorxiv", "medrxiv"):
        url = f"{_BIORXIV_API_BASE}/{server}/{doi}"
        try:
            response = await http_client.get(url, follow_redirects=True)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning(
                "biorxiv_api_request_failed",
                url=url,
                error=str(exc),
            )
            continue

        collection: list[dict[str, Any]] = data.get("collection") or []
        if collection:
            logger.info(
                "biorxiv_api_hit",
                server=server,
                doi=doi,
                versions=len(collection),
            )
            return collection[-1]  # most recent version

    return None


async def get_content_from_biorxiv(
    doi: str,
    http_client: httpx.AsyncClient,
    **_: Any,
) -> PaperContent | None:
    """Fetch content for a bioRxiv / medRxiv preprint.

    Returns:
        PaperContent with content_type "structured" or "abstract",
        or None if the DOI is not a bioRxiv DOI or the preprint is not found.
    """
    if not is_biorxiv_doi(doi):
        return None

    norm_doi = _normalize_doi(doi)

    record = await _fetch_biorxiv_record(norm_doi, http_client)
    if record is None:
        logger.info("biorxiv_not_found", doi=norm_doi)
        return None

    # ---- Build metadata ------------------------------------------------
    server_field: str = (record.get("server") or "biorxiv").lower()
    content_source = "medrxiv" if "med" in server_field else "biorxiv"

    date_str: str = record.get("date") or ""
    year: int | None = None
    if date_str and len(date_str) >= 4:
        with contextlib.suppress(ValueError):
            year = int(date_str[:4])

    authors_raw: str = record.get("authors") or ""
    authors_list = _parse_authors(authors_raw)

    metadata: dict[str, Any] = {
        "doi": norm_doi,
        "title": record.get("title") or "",
        "authors": authors_list,
        "year": year,
        "journal": content_source,
        "category": record.get("category") or "",
        "is_oa": True,
        "work_type": "preprint",
    }

    abstract: str | None = record.get("abstract") or None

    # ---- Try JATS XML full text ----------------------------------------
    jats_url: str = (record.get("jatsxml") or "").strip()
    if jats_url:
        try:
            jats_response = await http_client.get(jats_url, follow_redirects=True)
            jats_response.raise_for_status()
            xml_bytes: bytes = jats_response.content

            full_text = _extract_text_from_xml(xml_bytes)
            if full_text:
                sections = _extract_sections_from_xml(xml_bytes)
                references = _extract_references_from_xml(xml_bytes)
                logger.info(
                    "biorxiv_jats_structured",
                    doi=norm_doi,
                    text_length=len(full_text),
                )
                return PaperContent(
                    success=True,
                    doi=norm_doi,
                    content_type="structured",
                    content_source=content_source,
                    full_text=full_text,
                    sections=sections,
                    references=references,
                    abstract=abstract,
                    metadata=metadata,
                )
        except Exception as exc:
            logger.warning(
                "biorxiv_jats_fetch_failed",
                doi=norm_doi,
                jats_url=jats_url,
                error=str(exc),
            )
        # Fall through to abstract

    # ---- Abstract fallback ---------------------------------------------
    if abstract:
        logger.info("biorxiv_abstract_only", doi=norm_doi)
        return PaperContent(
            success=True,
            doi=norm_doi,
            content_type="abstract",
            content_source=content_source,
            abstract=abstract,
            metadata=metadata,
        )

    logger.warning("biorxiv_no_content", doi=norm_doi)
    return None
