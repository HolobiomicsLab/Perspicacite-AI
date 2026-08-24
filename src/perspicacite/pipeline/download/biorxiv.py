"""bioRxiv / medRxiv content retrieval.

Uses the bioRxiv public API to fetch preprint metadata and, when available,
the JATS XML full text.

Both the API and the JATS files sit behind one throttled host, so every
request goes through ``polite_get``. When a 429 is what cost us the full
text, the abstract-only result carries the throttled host in
``PaperContent.rate_limited_hosts`` so a later pass can retry it; a paper
that simply has no JATS leaves that list empty.

API reference:
  GET https://api.biorxiv.org/details/{server}/{doi}
  server in {biorxiv, medrxiv}
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from perspicacite.logging import get_logger
from perspicacite.pipeline.download.base import (
    CONTENT_TYPE_ABSTRACT,
    CONTENT_TYPE_STRUCTURED,
    PaperContent,
)
from perspicacite.pipeline.download.pmc import (
    _extract_references_from_xml,
    _extract_sections_from_xml,
    _extract_text_from_xml,
)
from perspicacite.pipeline.download.rate_limit import RateLimited, polite_get

if TYPE_CHECKING:
    import httpx

logger = get_logger("perspicacite.pipeline.download.biorxiv")

_BIORXIV_API_BASE = "https://api.biorxiv.org/details"
_DOI_PREFIX_RE = re.compile(r"10\.1101/", re.IGNORECASE)

# The two API servers, tried in order. They share one host budget, so a 429
# on the first is a reason to stop, not to move on to the second.
_API_SERVERS = ("biorxiv", "medrxiv")
_BIORXIV_SOURCE = "biorxiv"
_MEDRXIV_SOURCE = "medrxiv"
# Digits in the year field of the ISO-8601 date the API returns ("YYYY-MM-DD").
_YEAR_DIGITS = 4

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


def _parse_year(date_str: str) -> int | None:
    """Publication year from a bioRxiv date string, None when unparseable."""
    if len(date_str) < _YEAR_DIGITS:
        return None
    try:
        return int(date_str[:_YEAR_DIGITS])
    except ValueError:
        return None


def _latest_record(data: dict[str, Any], server: str, doi: str) -> dict[str, Any] | None:
    """Most recent version in an API payload, or None when it holds no record."""
    collection: list[dict[str, Any]] = data.get("collection") or []
    if not collection:
        return None
    logger.info("biorxiv_api_hit", server=server, doi=doi, versions=len(collection))
    return collection[-1]  # most recent version


async def _fetch_biorxiv_record(doi: str, http_client: httpx.AsyncClient) -> dict[str, Any] | None:
    """Query the bioRxiv API for *doi*, trying biorxiv then medrxiv.

    Returns the most recent version record, or None when neither server has
    it or the shared host throttled us.
    """
    for server in _API_SERVERS:
        url = f"{_BIORXIV_API_BASE}/{server}/{doi}"
        try:
            response = await polite_get(http_client, url, follow_redirects=True)
            response.raise_for_status()
            data = response.json()
        except RateLimited as exc:
            # Sibling server shares the host budget; asking now deepens it.
            logger.warning(
                "biorxiv_api_rate_limited", url=url, host=exc.host, attempts=exc.attempts
            )
            break
        except Exception as exc:
            logger.warning("biorxiv_api_request_failed", url=url, error=str(exc))
            continue
        record = _latest_record(data, server, doi)
        if record is not None:
            return record
    return None


async def _fetch_jats_xml(
    jats_url: str, http_client: httpx.AsyncClient
) -> tuple[bytes | None, str | None]:
    """Fetch the JATS XML at *jats_url*.

    Returns (xml bytes, throttled host). The host is set only when the fetch
    was refused with a retryable throttling status, so callers can tell
    "the host would not serve it" from "there is nothing to serve".
    """
    if not jats_url:
        return None, None
    try:
        response = await polite_get(http_client, jats_url, follow_redirects=True)
        response.raise_for_status()
        return response.content, None
    except RateLimited as exc:
        logger.warning(
            "biorxiv_jats_rate_limited", jats_url=jats_url, host=exc.host, attempts=exc.attempts
        )
        return None, exc.host
    except Exception as exc:
        logger.warning("biorxiv_jats_fetch_failed", jats_url=jats_url, error=str(exc))
        return None, None


def _build_metadata(record: dict[str, Any], doi: str) -> dict[str, Any]:
    """Normalised metadata dict for an API *record* retrieved under *doi*."""
    server_field: str = (record.get("server") or _BIORXIV_SOURCE).lower()
    return {
        "doi": doi,
        "title": record.get("title") or "",
        "authors": _parse_authors(record.get("authors") or ""),
        "year": _parse_year(record.get("date") or ""),
        "journal": _MEDRXIV_SOURCE if "med" in server_field else _BIORXIV_SOURCE,
        "category": record.get("category") or "",
        "is_oa": True,
        "work_type": "preprint",
    }


def _extract_body_text(xml_bytes: bytes, doi: str) -> str | None:
    """Body text of JATS *xml_bytes*, None when it has none or is malformed."""
    try:
        return _extract_text_from_xml(xml_bytes)
    except Exception as exc:  # malformed XML degrades, never propagates
        logger.warning("biorxiv_jats_parse_failed", doi=doi, error=str(exc))
        return None


def _build_structured_content(
    xml_bytes: bytes, metadata: dict[str, Any], abstract: str | None
) -> PaperContent | None:
    """Structured PaperContent from JATS *xml_bytes*, None without body text.

    *metadata* supplies the DOI and the content source; *abstract* is
    carried through unchanged.
    """
    full_text = _extract_body_text(xml_bytes, metadata["doi"])
    if not full_text:
        return None
    logger.info("biorxiv_jats_structured", doi=metadata["doi"], text_length=len(full_text))
    return PaperContent(
        success=True,
        doi=metadata["doi"],
        content_type=CONTENT_TYPE_STRUCTURED,
        content_source=metadata["journal"],
        full_text=full_text,
        sections=_extract_sections_from_xml(xml_bytes),
        references=_extract_references_from_xml(xml_bytes),
        abstract=abstract,
        metadata=metadata,
    )


def _build_abstract_content(
    abstract: str, metadata: dict[str, Any], throttled_host: str | None
) -> PaperContent:
    """Abstract-only PaperContent, flagged when throttling caused the degradation.

    *throttled_host* is the host that refused the full text with a 429, or
    None when no full text existed to begin with. It becomes
    ``rate_limited_hosts``, which therefore stays empty for a paper that
    genuinely has no full text and is not worth re-offering.
    """
    logger.info("biorxiv_abstract_only", doi=metadata["doi"], rate_limited_host=throttled_host)
    return PaperContent(
        success=True,
        doi=metadata["doi"],
        content_type=CONTENT_TYPE_ABSTRACT,
        content_source=metadata["journal"],
        abstract=abstract,
        metadata=metadata,
        rate_limited_hosts=[throttled_host] if throttled_host else [],
    )


async def get_content_from_biorxiv(
    doi: str,
    http_client: httpx.AsyncClient,
    **_: Any,
) -> PaperContent | None:
    """Fetch content for a bioRxiv / medRxiv preprint.

    Returns:
        PaperContent with content_type "structured" or "abstract" (the
        latter carrying rate_limited_hosts when a 429 cost us the full
        text), or None if the DOI is not a bioRxiv DOI or nothing was found.
    """
    if not is_biorxiv_doi(doi):
        return None
    norm_doi = _normalize_doi(doi)
    record = await _fetch_biorxiv_record(norm_doi, http_client)
    if record is None:
        logger.info("biorxiv_not_found", doi=norm_doi)
        return None

    metadata = _build_metadata(record, norm_doi)
    abstract: str | None = record.get("abstract") or None
    xml_bytes, throttled_host = await _fetch_jats_xml(
        (record.get("jatsxml") or "").strip(), http_client
    )
    if xml_bytes:
        structured = _build_structured_content(xml_bytes, metadata, abstract)
        if structured is not None:
            return structured
    if abstract:
        return _build_abstract_content(abstract, metadata, throttled_host)

    logger.warning("biorxiv_no_content", doi=norm_doi)
    return None


if __name__ == "__main__":
    SMOKE_RECORD = {
        "title": "A Preprint",
        "authors": "Doe, J.; Roe, R. and Poe, P.",
        "date": "2021-01-01",
        "server": "medRxiv",
        "category": "epidemiology",
    }
    # _extract_text_from_xml drops anything under 200 chars, so pad the body.
    SMOKE_BODY = b"Body sentence long enough to clear the extractor floor. " * 5
    SMOKE_XML = b"<article><body><sec><p>" + SMOKE_BODY + b"</p></sec></body></article>"

    assert is_biorxiv_doi("https://doi.org/10.1101/2021.01.01.425001")
    assert not is_biorxiv_doi("10.1038/s41467-022-33890-w")
    assert _normalize_doi("https://doi.org/10.1101/x ") == "10.1101/x"
    assert _parse_authors("Doe, J.; Roe, R.") == ["Doe, J.", "Roe, R."]
    assert _parse_year("2021-01-01") == 2021
    assert _parse_year("n/a") is None

    smoke_meta = _build_metadata(SMOKE_RECORD, "10.1101/smoke")
    assert smoke_meta["journal"] == _MEDRXIV_SOURCE
    assert len(smoke_meta["authors"]) == 3

    structured = _build_structured_content(SMOKE_XML, smoke_meta, "abs")
    assert structured is not None and structured.content_type == CONTENT_TYPE_STRUCTURED
    assert _build_structured_content(b"not xml at all", smoke_meta, "abs") is None

    throttled = _build_abstract_content("abs", smoke_meta, "biorxiv.org")
    assert throttled.rate_limited_hosts == ["biorxiv.org"]
    assert throttled.content_type == CONTENT_TYPE_ABSTRACT
    assert _build_abstract_content("abs", smoke_meta, None).rate_limited_hosts == []
    print("biorxiv.py smoke checks passed")
