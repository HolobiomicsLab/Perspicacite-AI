"""Europe PMC — full-text via PMCID using PMC AWS Open Data.

Resolution: Europe PMC search API maps DOI → PMCID.
Content: PMC AWS Open Data (S3) provides direct access to JATS XML and
plain text — free, no login, no API key.

  https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
  https://pmc-oa-opendata.s3.amazonaws.com/PMC3041641.1/PMC3041641.1.xml

Fallback order after PMCID resolution:
1. S3 JATS XML — structured sections, best quality
2. S3 plain text — guaranteed content if XML is missing
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download.europepmc")

# ---------------------------------------------------------------------------
# File cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("./data/papers")


def _cache_path(pmcid: str) -> Path:
    return _CACHE_DIR / f"{pmcid}.txt"


def _cache_sections_path(pmcid: str) -> Path:
    return _CACHE_DIR / f"{pmcid}_sections.json"


def _read_cache(pmcid: str) -> tuple[str | None, dict[str, str] | None]:
    """Return cached text + sections if available."""
    tp = _cache_path(pmcid)
    if not tp.exists():
        return None, None
    text = tp.read_text(encoding="utf-8")
    if len(text.strip()) < 200:
        return None, None
    sp = _cache_sections_path(pmcid)
    sections = None
    if sp.exists():
        try:
            sections = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            pass
    logger.info("pmc_cache_hit", pmcid=pmcid, text_length=len(text))
    return text, sections


def _write_cache(pmcid: str, text: str, sections: dict[str, str] | None) -> None:
    """Persist text + sections to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(pmcid).write_text(text, encoding="utf-8")
    if sections:
        _cache_sections_path(pmcid).write_text(
            json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    logger.info("pmc_cache_write", pmcid=pmcid, text_length=len(text))

# ---------------------------------------------------------------------------
# S3 URL templates
# ---------------------------------------------------------------------------

_S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"


def _s3_xml_url(pmcid: str) -> str:
    return f"{_S3_BASE}/{pmcid}.1/{pmcid}.1.xml"


def _s3_txt_url(pmcid: str) -> str:
    return f"{_S3_BASE}/{pmcid}.1/{pmcid}.1.txt"


# ---------------------------------------------------------------------------
# XML text extraction (JATS/NLM format)
# ---------------------------------------------------------------------------

_BODY_XPATHS = [".//article/body", ".//body", ".//pmc-articleset//body"]


def _extract_text_from_xml(xml_bytes: bytes) -> str | None:
    """Extract body text from JATS/NLM full-text XML."""
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return None

    body = None
    for xpath in _BODY_XPATHS:
        body = root.find(xpath)
        if body is not None:
            break

    if body is None:
        return None

    pieces: list[str] = []

    def _walk(el: ElementTree.Element):
        tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if tag in ("title", "label"):
            pieces.append("\n\n")
        if el.text:
            pieces.append(el.text)
        for child in el:
            _walk(child)
            if child.tail:
                pieces.append(child.tail)
        if tag in ("p", "sec", "section"):
            pieces.append("\n\n")

    _walk(body)
    import re
    text = "".join(pieces).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text if len(text) > 200 else None


def _extract_sections_from_xml(xml_bytes: bytes) -> dict[str, str] | None:
    """Extract structured sections from JATS/NLM full-text XML."""
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return None

    body = None
    for xpath in _BODY_XPATHS:
        body = root.find(xpath)
        if body is not None:
            break

    if body is None:
        return None

    ns = ""
    if "}" in root.tag:
        ns = root.tag.split("}")[0] + "}"

    def _find(el: ElementTree.Element, local: str):
        hit = el.find(f"{ns}{local}") if ns else None
        if hit is None:
            hit = el.find(local)
        if hit is None:
            for child in el:
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag == local:
                    hit = child
                    break
        return hit

    sections: dict[str, str] = {}

    def _walk_sections(el: ElementTree.Element, prefix: str = ""):
        for sec in el:
            stag = sec.tag.split("}")[-1] if "}" in sec.tag else sec.tag
            if stag != "sec":
                continue

            title_el = _find(sec, "title")
            sec_title = (title_el.text or "").strip() if title_el is not None else "Unknown Section"
            full_title = f"{prefix} > {sec_title}" if prefix else sec_title

            paras: list[str] = []
            for child in sec:
                ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if ctag == "p":
                    ptext = "".join(child.itertext()).strip()
                    if ptext:
                        paras.append(ptext)
                elif ctag not in ("sec", "title", "fig", "table-wrap", "xref", "ref-list"):
                    ptext = "".join(child.itertext()).strip()
                    if ptext:
                        paras.append(ptext)

            if paras:
                sections[full_title] = "\n\n".join(paras)

            _walk_sections(sec, full_title)

    _walk_sections(body)

    # Extract abstract
    abstract_el = None
    for xpath in (".//article/abstract", ".//abstract"):
        abstract_el = root.find(xpath)
        if abstract_el is not None:
            break
    if abstract_el is not None:
        abstract_text = "".join(abstract_el.itertext()).strip()
        if abstract_text:
            sections["Abstract"] = abstract_text

    return sections if sections else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def _resolve_pmcid(
    doi: str,
    client: httpx.AsyncClient,
) -> str | None:
    """Resolve a DOI to a PMCID via Europe PMC search API."""
    search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": f"DOI:{doi}",
        "format": "json",
        "resultType": "core",
        "pageSize": "1",
    }
    logger.info("europepmc_search", doi=doi)
    r = await client.get(search_url, params=params)
    r.raise_for_status()
    data = r.json()
    results = data.get("resultList", {}).get("result") or []
    if not results:
        logger.info("europepmc_no_hit", doi=doi)
        return None

    hit = results[0] if isinstance(results[0], dict) else None
    if not hit:
        return None

    pmcid = hit.get("pmcid")
    if not pmcid or not str(pmcid).upper().startswith("PMC"):
        logger.info("europepmc_no_pmcid", doi=doi)
        return None

    return str(pmcid).strip()


async def get_fulltext_from_europepmc(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[str | None, dict[str, str] | None]:
    """Fetch full text when the article is in PMC.

    Uses PMC AWS Open Data (S3) for direct JATS XML / plain text access.
    DOI → PMCID resolution via Europe PMC search API.

    Returns:
        ``(full_text, sections)`` or ``(None, None)`` if unavailable.
        *sections* is a dict of section-title → text.
    """
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean:
        return None, None

    client = http_client or httpx.AsyncClient(timeout=45.0, follow_redirects=True)
    should_close = http_client is None

    try:
        # Step 1: Resolve DOI → PMCID
        pmcid = await _resolve_pmcid(clean, client)
        if not pmcid:
            return None, None

        # Step 2: Check disk cache
        cached_text, cached_sections = _read_cache(pmcid)
        if cached_text:
            return cached_text, cached_sections

        # Step 3: Try S3 JATS XML (best quality, structured sections)
        xml_url = _s3_xml_url(pmcid)
        logger.info("pmc_s3_try_xml", doi=clean, pmcid=pmcid, url=xml_url)
        try:
            r_xml = await client.get(xml_url, headers={"Accept": "application/xml"})
            if r_xml.status_code == 200 and r_xml.content:
                sections = _extract_sections_from_xml(r_xml.content)
                text = _extract_text_from_xml(r_xml.content)
                if text and len(text) > 200:
                    logger.info(
                        "pmc_s3_xml_success",
                        doi=clean,
                        pmcid=pmcid,
                        text_length=len(text),
                        sections=len(sections) if sections else 0,
                    )
                    _write_cache(pmcid, text, sections)
                    return text, sections
        except Exception as e:
            logger.info("pmc_s3_xml_failed", doi=clean, error=str(e))

        # Step 4: Fallback — S3 plain text
        txt_url = _s3_txt_url(pmcid)
        logger.info("pmc_s3_try_txt", doi=clean, pmcid=pmcid, url=txt_url)
        try:
            r_txt = await client.get(txt_url)
            if r_txt.status_code == 200 and r_txt.text:
                text = r_txt.text.strip()
                if text and len(text) > 200:
                    logger.info(
                        "pmc_s3_txt_success",
                        doi=clean,
                        pmcid=pmcid,
                        text_length=len(text),
                    )
                    _write_cache(pmcid, text, None)
                    return text, None
        except Exception as e:
            logger.info("pmc_s3_txt_failed", doi=clean, error=str(e))

        logger.warning("pmc_s3_failed", doi=clean, pmcid=pmcid)
        return None, None

    except Exception as e:
        logger.error("europepmc_error", doi=clean, error=str(e))
        return None, None
    finally:
        if should_close:
            await client.aclose()
