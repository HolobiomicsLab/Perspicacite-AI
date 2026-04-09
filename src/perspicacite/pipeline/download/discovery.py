"""Source discovery for a DOI via OpenAlex and Unpaywall.

Queries OpenAlex first (richer metadata, optional mailto for polite pool),
then Unpaywall (requires email) to fill gaps. Results are cached to disk.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from perspicacite.logging import get_logger
from .base import PaperDiscovery

logger = get_logger("perspicacite.pipeline.download.discovery")

_CACHE_DIR = Path("./data/papers")


def _discovery_cache_path(doi: str) -> Path:
    safe = doi.replace("/", "_").replace(":", "-")
    return _CACHE_DIR / f"{safe}_discovery.json"


def _read_discovery_cache(doi: str) -> PaperDiscovery | None:
    path = _discovery_cache_path(doi)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return PaperDiscovery(**data)
    except Exception:
        return None


def _write_discovery_cache(disc: PaperDiscovery) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _discovery_cache_path(disc.doi)
    path.write_text(
        json.dumps(
            {
                "doi": disc.doi,
                "pmcid": disc.pmcid,
                "arxiv_id": disc.arxiv_id,
                "oa_url": disc.oa_url,
                "abstract": disc.abstract,
                "title": disc.title,
                "is_oa": disc.is_oa,
                "work_type": disc.work_type,
                "unpaywall_pdf_url": disc.unpaywall_pdf_url,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _invert_abstract(inv_idx: dict) -> str | None:
    """Convert OpenAlex inverted abstract index to plain text."""
    if not inv_idx:
        return None
    word_positions: list[tuple[int, str]] = []
    for word, positions in inv_idx.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions) if word_positions else None


def _extract_arxiv_id_from_openalex(work: dict) -> str | None:
    """Try to find an arXiv ID in OpenAlex work data."""
    sids = work.get("ids") or {}
    # OpenAlex may store arXiv ID directly
    for key in ("arxiv", "arxiv_id"):
        val = sids.get(key)
        if val:
            # Strip URL prefix if present
            return val.rsplit("/", 1)[-1] if "/" in val else val

    # Check if DOI itself is an arXiv DOI
    doi = work.get("doi") or ""
    if "arxiv" in doi.lower():
        from .arxiv import get_arxiv_id_from_doi

        return get_arxiv_id_from_doi(doi)
    return None


def _extract_pmcid_from_unpaywall_locations(
    oa_locations: list[dict],
) -> str | None:
    """Extract PMCID from Unpaywall OA locations that point to PMC."""
    for loc in oa_locations or []:
        url = loc.get("url") or loc.get("url_for_landing_page") or ""
        # Match patterns like pmc.ncbi.nlm.nih.gov/articles/PMC12345
        m = re.search(r"PMC\d+", url)
        if m:
            return m.group(0)
    return None


async def discover_paper_sources(
    doi: str,
    http_client: httpx.AsyncClient,
    unpaywall_email: str | None = None,
) -> PaperDiscovery:
    """Discover available sources and metadata for a DOI.

    Queries OpenAlex then optionally Unpaywall to learn:
    PMCID, arXiv ID, OA URL, abstract, title, work type.

    Always returns a PaperDiscovery (never raises). Fields are None
    if services are unreachable.
    """
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    disc = PaperDiscovery(doi=clean)

    # 1. Check disk cache
    cached = _read_discovery_cache(clean)
    if cached is not None:
        logger.info("discovery_cache_hit", doi=clean)
        return cached

    # 2. OpenAlex
    oa_ok = False
    try:
        mailto = os.getenv("OPENALEX_MAILTO") or os.getenv("UNPAYWALL_EMAIL") or ""
        params = {"mailto": mailto} if mailto else None
        api_url = f"https://api.openalex.org/works/doi:{clean}"
        logger.info("discovery_openalex_lookup", doi=clean)
        r = await http_client.get(api_url, params=params)
        if r.status_code == 200:
            work = r.json()
            oa_ok = True
            disc.title = work.get("title")
            disc.is_oa = (work.get("open_access") or {}).get("is_oa", False)
            disc.work_type = work.get("type")
            # Authors
            authorships = work.get("authorships") or []
            disc.authors = [
                (a.get("author") or {}).get("display_name")
                for a in authorships
                if (a.get("author") or {}).get("display_name")
            ] or None
            # Year
            py = work.get("publication_year")
            if py:
                disc.year = int(py)

            ids = work.get("ids") or {}
            # OpenAlex may return PMCID with or without "PMC" prefix
            pmcid = ids.get("pmcid")
            if pmcid and not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"
            disc.pmcid = pmcid
            disc.arxiv_id = _extract_arxiv_id_from_openalex(work)

            # OA URL
            oa_info = work.get("open_access") or {}
            disc.oa_url = oa_info.get("oa_url")

            # Reconstruct abstract from inverted index
            raw_abstract = work.get("abstract_inverted_index")
            if raw_abstract and isinstance(raw_abstract, dict):
                disc.abstract = _invert_abstract(raw_abstract)
    except Exception as e:
        logger.info("discovery_openalex_failed", doi=clean, error=str(e))

    # 3. Unpaywall (fills gaps OpenAlex left)
    email = unpaywall_email or os.getenv("UNPAYWALL_EMAIL")
    if email:
        try:
            url = f"https://api.unpaywall.org/v2/{clean}?email={email}"
            logger.info("discovery_unpaywall_lookup", doi=clean)
            r = await http_client.get(url)
            if r.status_code == 200:
                data = r.json()
                # PMCID from OA locations (Unpaywall doesn't have it directly)
                if not disc.pmcid:
                    disc.pmcid = _extract_pmcid_from_unpaywall_locations(
                        data.get("oa_locations")
                    )
                if not disc.is_oa:
                    disc.is_oa = data.get("is_oa", False)
                # Abstract (Unpaywall usually doesn't have it but just in case)
                if not disc.abstract:
                    disc.abstract = data.get("abstract")
                # PDF URL
                best = data.get("best_oa_location") or {}
                pdf_url = best.get("url_for_pdf") or best.get("url")
                if pdf_url and not disc.oa_url:
                    disc.oa_url = pdf_url
                disc.unpaywall_pdf_url = pdf_url
        except Exception as e:
            logger.info("discovery_unpaywall_failed", doi=clean, error=str(e))

    # 4. Cache if we learned anything useful
    if oa_ok or disc.pmcid or disc.arxiv_id or disc.abstract:
        _write_discovery_cache(disc)

    return disc
