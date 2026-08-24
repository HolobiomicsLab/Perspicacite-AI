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
from .openalex_oa import _collect_pdf_candidate_urls

logger = get_logger("perspicacite.pipeline.download.discovery")

_CACHE_DIR = Path("./data/papers")


_ARXIV_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
}

# arXiv puts the id in the url path: /abs/2310.11511v1, /pdf/2310.11511, and
# the pre-2007 form /abs/math.GT/0309136. Source: arxiv.org identifier scheme.
ARXIV_URL_ID_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/"
    r"(?P<arxiv_id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)

# PMC article urls, both hosts and both id spellings. The current form is
# .../articles/PMC6940144; the legacy form drops the prefix
# (.../pmc/articles/6940144). Sources: NCBI and Europe PMC url schemes.
PMC_ARTICLE_URL_PATTERN = re.compile(
    r"(?:ncbi\.nlm\.nih\.gov|europepmc\.org)/[^\s?]*articles/(?:PMC)?(?P<digits>\d+)",
    re.IGNORECASE,
)

# Unpaywall location url fields, pdf first. url_for_landing_page never points
# at a file, so it is only consulted when mining ids out of a url.
UNPAYWALL_PDF_URL_KEYS = ("url_for_pdf", "url")
UNPAYWALL_ALL_URL_KEYS = ("url_for_pdf", "url", "url_for_landing_page")

# OpenAlex location objects and the url fields they carry.
OPENALEX_NAMED_LOCATION_KEYS = ("best_oa_location", "primary_location")
OPENALEX_URL_KEYS = ("pdf_url", "landing_page_url")


async def _enrich_from_arxiv_atom(
    arxiv_id: str,
    disc: PaperDiscovery,
    http_client: httpx.AsyncClient,
) -> None:
    """Fill missing title/authors/year on ``disc`` from the arXiv Atom API.

    Mutates ``disc`` in place. Only fills fields that are currently empty —
    OpenAlex data takes precedence over arXiv data.

    arXiv export API: https://export.arxiv.org/api/query?id_list=<id>
    Returns Atom XML. The id_list endpoint accepts arXiv IDs (without
    version suffix). For a single id, the feed contains exactly one
    ``<entry>`` with ``<title>``, ``<author><name>...``, ``<published>``.
    """
    import xml.etree.ElementTree as ET

    bare = arxiv_id.split("v")[0] if re.match(r".*v\d+$", arxiv_id) else arxiv_id
    url = f"https://export.arxiv.org/api/query?id_list={bare}"
    logger.info("discovery_arxiv_atom_lookup", arxiv_id=bare)
    r = await http_client.get(url)
    if r.status_code != 200:
        return
    root = ET.fromstring(r.text)
    entry = root.find("atom:entry", _ARXIV_ATOM_NS)
    if entry is None:
        return

    title_el = entry.find("atom:title", _ARXIV_ATOM_NS)
    if title_el is not None and title_el.text and not disc.title:
        disc.title = " ".join(title_el.text.split())

    if not disc.authors:
        names = []
        for author_el in entry.findall("atom:author", _ARXIV_ATOM_NS):
            name_el = author_el.find("atom:name", _ARXIV_ATOM_NS)
            if name_el is not None and name_el.text:
                names.append(name_el.text.strip())
        if names:
            disc.authors = names

    if disc.year is None:
        published_el = entry.find("atom:published", _ARXIV_ATOM_NS)
        if published_el is not None and published_el.text:
            m = re.match(r"^(\d{4})", published_el.text)
            if m:
                disc.year = int(m.group(1))


def _title_word_jaccard(a: str | None, b: str | None) -> float:
    """Jaccard similarity over lowercased word sets, stop-word-light.

    Used by the F-32 cross-check to decide whether OpenAlex's title is
    plausibly the same paper as arXiv's. 1.0 = identical sets, 0.0 = disjoint.
    """
    if not a or not b:
        return 0.0
    def _toks(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


async def _cross_check_arxiv_title(
    arxiv_id: str,
    disc: PaperDiscovery,
    http_client: httpx.AsyncClient,
    *,
    similarity_threshold: float = 0.4,
) -> None:
    """F-32: for arXiv DOIs, compare ``disc.title`` against arXiv's canonical
    title. When they're wildly different (word-Jaccard < threshold), assume
    the upstream metadata is corrupted and overwrite with the arXiv version.

    OpenAlex has been observed to mis-attribute arXiv records (e.g. 2310.11511
    titled 'CareerX' but authored by the Self-RAG team). The arXiv API is
    the source of truth for arXiv preprints, so when it disagrees with the
    aggregator, the aggregator loses.

    No-op when ``disc.title`` was empty (the regular enrich path already
    handled that).
    """
    import xml.etree.ElementTree as ET

    if not disc.title:
        return  # nothing to compare against; the enrich path will fill it
    bare = arxiv_id.split("v")[0] if re.match(r".*v\d+$", arxiv_id) else arxiv_id
    url = f"https://export.arxiv.org/api/query?id_list={bare}"
    try:
        r = await http_client.get(url)
        if r.status_code != 200:
            return
        root = ET.fromstring(r.text)
        entry = root.find("atom:entry", _ARXIV_ATOM_NS)
        if entry is None:
            return
        title_el = entry.find("atom:title", _ARXIV_ATOM_NS)
        if title_el is None or not title_el.text:
            return
        arxiv_title = " ".join(title_el.text.split())
    except Exception as e:
        logger.info("discovery_arxiv_xcheck_error", arxiv_id=bare, error=str(e))
        return

    sim = _title_word_jaccard(disc.title, arxiv_title)
    if sim < similarity_threshold:
        logger.warning(
            "discovery_title_mismatch_correcting",
            arxiv_id=bare,
            openalex_title=disc.title[:200],
            arxiv_title=arxiv_title[:200],
            similarity=round(sim, 3),
        )
        disc.title = arxiv_title


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
                "oa_candidates": disc.oa_candidates,
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


def _location_urls(loc: object, keys: tuple[str, ...]) -> list[str]:
    """Return one location's non-empty urls, in `keys` order.

    `loc` is an untrusted payload fragment; a non-dict yields an empty list
    rather than raising.
    """
    if not isinstance(loc, dict):
        return []
    return [str(loc[key]) for key in keys if loc.get(key)]


def _dedupe_urls(urls: list[str]) -> list[str]:
    """Return `urls` without duplicates or non-http entries, order preserved."""
    seen: set[str] = set()
    ranked: list[str] = []
    for url in urls:
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        ranked.append(url)
    return ranked


def _openalex_locations(work: dict) -> list[dict]:
    """Return every location object an OpenAlex work carries.

    The `locations` array comes first, then best_oa_location and
    primary_location. Non-dict entries are dropped.
    """
    listed = [loc for loc in (work.get("locations") or []) if isinstance(loc, dict)]
    named = [work.get(key) for key in OPENALEX_NAMED_LOCATION_KEYS]
    return listed + [loc for loc in named if isinstance(loc, dict)]


def _openalex_location_urls(work: dict) -> list[str]:
    """Return every pdf and landing-page url named by a work's locations."""
    return [
        url
        for loc in _openalex_locations(work)
        for url in _location_urls(loc, OPENALEX_URL_KEYS)
    ]


def _arxiv_id_from_openalex_ids(ids: dict) -> str | None:
    """Return the arXiv id stored directly in an OpenAlex `ids` block.

    None means the block named no arXiv id, not that the work has none.
    """
    for key in ("arxiv", "arxiv_id"):
        val = ids.get(key)
        if val:
            return str(val).rsplit("/", 1)[-1] if "/" in str(val) else str(val)
    return None


def _arxiv_id_from_location_urls(work: dict) -> str | None:
    """Return the arXiv id named by any of a work's location urls, else None."""
    for url in _openalex_location_urls(work):
        match = ARXIV_URL_ID_PATTERN.search(url)
        if match:
            return match.group("arxiv_id")
    return None


def _extract_arxiv_id_from_openalex(work: dict) -> str | None:
    """Return an arXiv id for an OpenAlex work, or None when it names none.

    Looks in `ids` first, then the DOI, then every location url — OpenAlex
    frequently carries the arXiv record only as a location, with no arxiv
    key at all. None is "no arXiv id found", never "not an arXiv paper".
    """
    direct = _arxiv_id_from_openalex_ids(work.get("ids") or {})
    if direct:
        return direct
    doi = work.get("doi") or ""
    if "arxiv" in doi.lower():
        from .arxiv import get_arxiv_id_from_doi

        return get_arxiv_id_from_doi(doi)
    return _arxiv_id_from_location_urls(work)


def _extract_pmcid_from_unpaywall_locations(
    oa_locations: list[dict] | None,
) -> str | None:
    """Return the PMCID named by any Unpaywall OA location, else None.

    Scans url_for_pdf, url and url_for_landing_page of every location and
    accepts both the PMC-prefixed and the legacy bare-digit article path, on
    either PMC host. The id is always returned PMC-prefixed. None means no
    location pointed at PMC, not that the paper has no PMCID.
    """
    for loc in oa_locations or []:
        for url in _location_urls(loc, UNPAYWALL_ALL_URL_KEYS):
            match = PMC_ARTICLE_URL_PATTERN.search(url)
            if match:
                return f"PMC{match.group('digits')}"
    return None


def _collect_openalex_oa_candidates(work: dict) -> list[str]:
    """Return an OpenAlex work's OA pdf urls, best first, deduped.

    Ranking: locations[].pdf_url, then best_oa_location.pdf_url, then
    primary_location.pdf_url, then open_access.oa_url gated on is_oa.
    Delegates to openalex_oa._collect_pdf_candidate_urls — imported, not
    mirrored, so discovery and the downloader can never rank differently.
    An empty list means the payload named no OA url.
    """
    return _collect_pdf_candidate_urls(work)


def _collect_unpaywall_oa_candidates(data: dict) -> list[str]:
    """Return an Unpaywall response's OA urls, best first, deduped.

    Ranking: best_oa_location.url_for_pdf, best_oa_location.url, then each
    remaining oa_locations entry's url_for_pdf then url. Empty and non-http
    values are skipped. An empty list means the payload named no OA url.
    """
    ordered = _location_urls(data.get("best_oa_location"), UNPAYWALL_PDF_URL_KEYS)
    for loc in data.get("oa_locations") or []:
        ordered.extend(_location_urls(loc, UNPAYWALL_PDF_URL_KEYS))
    return _dedupe_urls(ordered)


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

    # 1. Check disk cache. If cached but missing authors/year for an arXiv
    # paper (a known weakness of older cache entries — OpenAlex didn't have
    # them populated yet at write time), opportunistically enrich from the
    # arXiv Atom API and re-save.
    cached = _read_discovery_cache(clean)
    if cached is not None:
        from .arxiv import get_arxiv_id_from_doi, is_arxiv_doi
        needs_enrich = is_arxiv_doi(clean) and (
            not cached.authors or cached.year is None or not cached.title
        )
        arxiv_id_for_cache = (
            cached.arxiv_id
            or (get_arxiv_id_from_doi(clean) if is_arxiv_doi(clean) else None)
        )
        if needs_enrich and arxiv_id_for_cache:
            try:
                await _enrich_from_arxiv_atom(
                    arxiv_id_for_cache, cached, http_client,
                )
                if not cached.arxiv_id:
                    cached.arxiv_id = arxiv_id_for_cache
                _write_discovery_cache(cached)
                logger.info("discovery_cache_enriched_arxiv", doi=clean)
            except Exception as e:
                logger.info(
                    "discovery_cache_enrich_failed",
                    doi=clean,
                    error=str(e),
                )
        # F-32: also re-run the title cross-check on cache hits so corrupted
        # OpenAlex titles cached by older versions get corrected on the next
        # access (no need to wipe the cache dir manually).
        if arxiv_id_for_cache and cached.title:
            try:
                before = cached.title
                await _cross_check_arxiv_title(
                    arxiv_id_for_cache, cached, http_client,
                )
                if cached.title != before:
                    _write_discovery_cache(cached)
            except Exception as e:
                logger.info(
                    "discovery_cache_xcheck_failed",
                    doi=clean,
                    error=str(e),
                )
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
            disc.oa_candidates = _collect_openalex_oa_candidates(work)
            disc.license = oa_info.get("license")

            # Reconstruct abstract from inverted index
            raw_abstract = work.get("abstract_inverted_index")
            if raw_abstract and isinstance(raw_abstract, dict):
                disc.abstract = _invert_abstract(raw_abstract)
    except Exception as e:
        logger.info("discovery_openalex_failed", doi=clean, error=str(e))

    # 2b. arXiv export API (fills gaps for very-new arXiv papers that
    # OpenAlex hasn't fully indexed yet — title may be present but
    # authorships and publication_year often missing for the first weeks).
    arxiv_id_for_fallback = disc.arxiv_id
    if not arxiv_id_for_fallback:
        from .arxiv import get_arxiv_id_from_doi, is_arxiv_doi
        if is_arxiv_doi(clean):
            arxiv_id_for_fallback = get_arxiv_id_from_doi(clean)
    if arxiv_id_for_fallback and (not disc.authors or disc.year is None or not disc.title):
        try:
            await _enrich_from_arxiv_atom(arxiv_id_for_fallback, disc, http_client)
            if not disc.arxiv_id:
                disc.arxiv_id = arxiv_id_for_fallback
        except Exception as e:
            logger.info("discovery_arxiv_atom_failed", arxiv_id=arxiv_id_for_fallback, error=str(e))

    # F-32: for arXiv DOIs, cross-check the title against arXiv's canonical
    # entry — corrupted OpenAlex records (e.g. 2310.11511 → 'CareerX') get
    # overwritten with the arXiv version. Only runs when OpenAlex gave us a
    # title (so we have something to validate).
    if arxiv_id_for_fallback and disc.title and oa_ok:
        try:
            await _cross_check_arxiv_title(arxiv_id_for_fallback, disc, http_client)
        except Exception as e:
            logger.info("discovery_arxiv_xcheck_failed",
                        arxiv_id=arxiv_id_for_fallback, error=str(e))

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
                disc.oa_candidates = _dedupe_urls(
                    disc.oa_candidates + _collect_unpaywall_oa_candidates(data)
                )
                if not disc.license:
                    disc.license = best.get("license")
        except Exception as e:
            logger.info("discovery_unpaywall_failed", doi=clean, error=str(e))

    # 4. Cache if we learned anything useful
    if oa_ok or disc.pmcid or disc.arxiv_id or disc.abstract:
        _write_discovery_cache(disc)

    return disc


if __name__ == "__main__":
    # Offline smoke check: pure payload parsing, no network, no clients.
    _work = {
        "locations": [{"landing_page_url": "https://arxiv.org/abs/2310.11511v1"}],
        "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/pdf/2310.11511"},
    }
    assert _extract_arxiv_id_from_openalex(_work) == "2310.11511v1"
    assert _collect_openalex_oa_candidates(_work) == [
        "https://arxiv.org/pdf/2310.11511"
    ]
    assert _extract_arxiv_id_from_openalex({"ids": {}}) is None

    _legacy = [{"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/6940144/"}]
    assert _extract_pmcid_from_unpaywall_locations(_legacy) == "PMC6940144"
    _elsewhere = [{"url": "https://example.org/articles/6940144"}]
    assert _extract_pmcid_from_unpaywall_locations(_elsewhere) is None

    _unpaywall = {
        "best_oa_location": {"url": "https://repo.example.org/landing"},
        "oa_locations": [{"url_for_pdf": "https://repo.example.org/file.pdf"}],
    }
    assert _collect_unpaywall_oa_candidates(_unpaywall) == [
        "https://repo.example.org/landing",
        "https://repo.example.org/file.pdf",
    ]
    assert _collect_unpaywall_oa_candidates({}) == []
    print("discovery smoke ok")
