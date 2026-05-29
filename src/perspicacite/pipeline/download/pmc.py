"""PMC Open Access — full-text and references via PMC AWS Open Data.

DOI → PMCID resolution via Europe PMC search API.
Content: PMC AWS Open Data (S3) provides direct access to JATS XML and
plain text — free, no login, no API key.

  https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
  https://pmc-oa-opendata.s3.amazonaws.com/PMC3041641.1/PMC3041641.1.xml

Fallback order after PMCID resolution:
1. S3 JATS XML — structured sections + references, best quality (OA subset only)
2. S3 plain text — guaranteed content if XML is missing (OA subset only)
3. NCBI efetch JATS XML — serves PMCIDs outside the OA bulk subset (author
   manuscripts, "restricted-by pmc") for individual retrieval
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from xml.etree import ElementTree

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download.pmc")

# ---------------------------------------------------------------------------
# File cache
# ---------------------------------------------------------------------------

_CACHE_DIR = Path("./data/papers")


def _cache_path(pmcid: str) -> Path:
    return _CACHE_DIR / f"{pmcid}.txt"


def _cache_sections_path(pmcid: str) -> Path:
    return _CACHE_DIR / f"{pmcid}_sections.json"


def _cache_refs_path(pmcid: str) -> Path:
    return _CACHE_DIR / f"{pmcid}_refs.json"


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
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            sections = json.loads(sp.read_text(encoding="utf-8"))
    logger.info("pmc_cache_hit", pmcid=pmcid, text_length=len(text))
    return text, sections


def _write_cache(
    pmcid: str,
    text: str,
    sections: dict[str, str] | None,
    refs: list[dict] | None = None,
    doi: str | None = None,
) -> None:
    """Persist text + sections + refs to disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(pmcid).write_text(text, encoding="utf-8")
    if sections:
        _cache_sections_path(pmcid).write_text(
            json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if refs:
        refs_json = json.dumps(refs, ensure_ascii=False, indent=2)
        # Write by PMCID
        _cache_refs_path(pmcid).write_text(refs_json, encoding="utf-8")
        # Also write by DOI so unified pipeline can find it
        if doi:
            clean_doi = doi.strip().lower()
            for prefix in ("https://doi.org/", "http://doi.org/"):
                if clean_doi.startswith(prefix):
                    clean_doi = clean_doi[len(prefix):]
            doi_refs_path = _CACHE_DIR / f"{clean_doi.replace('/', '_')}_refs.json"
            doi_refs_path.write_text(refs_json, encoding="utf-8")
    logger.info("pmc_cache_write", pmcid=pmcid, text_length=len(text))

# ---------------------------------------------------------------------------
# S3 URL templates
# ---------------------------------------------------------------------------

_S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"


def _s3_xml_url(pmcid: str) -> str:
    return f"{_S3_BASE}/{pmcid}.1/{pmcid}.1.xml"


def _s3_txt_url(pmcid: str) -> str:
    return f"{_S3_BASE}/{pmcid}.1/{pmcid}.1.txt"


_EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _efetch_xml_url(pmcid: str) -> str:
    """NCBI efetch JATS XML URL for a PMCID.

    Unlike the OA S3 bucket, efetch serves the full JATS for individual
    retrieval even when the article is outside the OA bulk subset (author
    manuscripts, "restricted-by pmc"). Includes ``tool`` and an optional
    ``NCBI_API_KEY`` for polite, higher-quota access.
    """
    import os

    numeric = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid
    url = f"{_EFETCH_BASE}?db=pmc&id={numeric}&rettype=xml&tool=perspicacite"
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        url += f"&api_key={api_key}"
    return url


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
                if ctag == "p" or ctag not in ("sec", "title", "fig", "table-wrap", "xref", "ref-list"):
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


def _all_text(el) -> str:
    """Recursively collect all text content from an XML element."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        parts.append(_all_text(child))
        if child.tail:
            parts.append(child.tail)
    return " ".join(p.strip() for p in parts if p.strip())


def _extract_figures_from_xml(xml_bytes: bytes) -> list[dict]:
    """Extract <fig> and <table-wrap> elements from JATS/NLM XML.

    Returns a list of dicts with keys:
        fig_id, label, caption, fig_type, page (None for JATS)
    """
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return []

    body = None
    for xpath in _BODY_XPATHS:
        body = root.find(xpath)
        if body is not None:
            break

    if body is None:
        return []

    out: list[dict] = []
    index = 0
    for el in body.iter():
        raw_tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
        if raw_tag not in ("fig", "table-wrap"):
            continue

        fig_type = "table" if raw_tag == "table-wrap" else el.attrib.get("fig-type") or "figure"

        fig_id = el.attrib.get("id") or f"{raw_tag}_{index}"

        label_el = None
        caption_el = None
        for child in el:
            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if ctag == "label" and label_el is None:
                label_el = child
            elif ctag == "caption" and caption_el is None:
                caption_el = child

        label = (label_el.text or "").strip() if label_el is not None else ""
        caption = _all_text(caption_el)[:500] if caption_el is not None else ""

        if not label and not caption:
            index += 1
            continue

        out.append({
            "fig_id": fig_id,
            "label": label or None,
            "caption": caption or None,
            "fig_type": fig_type,
            "page": None,
        })
        index += 1

    return out


def _extract_supplementary_from_xml(xml_bytes: bytes, pmcid: str) -> list[dict] | None:
    """Extract supplementary-material entries from a PMC JATS XML.

    JATS marks SI files with ``<supplementary-material>`` (or
    ``<inline-supplementary-material>``) elements that carry an ``xlink:href``
    pointing to the file. PMC mirrors these at
    ``https://www.ncbi.nlm.nih.gov/pmc/articles/<pmcid>/bin/<href>``.

    Returns a list of dicts with keys:
        - id: the JATS ``id`` attribute when present
        - label: e.g. "Supplementary Figure 1"
        - caption: textual caption / description
        - href: relative filename inside the PMC bin/ folder
        - url: absolute fetchable URL (PMC bin/ mirror)
        - mime_type: from JATS ``mimetype``/``mime-subtype`` when present

    Returns None when the XML has no supplementary blocks (vast majority
    of papers).
    """
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return None

    ns_x = "{http://www.w3.org/1999/xlink}"
    out: list[dict] = []

    def _tag(el: ElementTree.Element) -> str:
        return el.tag.split("}")[-1] if "}" in el.tag else el.tag

    def _walk(el: ElementTree.Element) -> None:
        for child in el:
            if _tag(child) in ("supplementary-material", "inline-supplementary-material"):
                # Prefer the href on the block itself; fall back to child
                # <media>, <graphic>, <ext-link> (some papers put it there).
                href = (
                    child.attrib.get(f"{ns_x}href")
                    or child.attrib.get("href")
                    or ""
                ).strip()
                mime = (
                    child.attrib.get("mimetype")
                    or child.attrib.get("mime-subtype")
                    or ""
                )
                label = ""
                caption = ""
                for sub in child:
                    stag = _tag(sub)
                    if stag == "label":
                        label = "".join(sub.itertext()).strip()
                    elif stag == "caption":
                        caption = " ".join(
                            "".join(p.itertext()).strip()
                            for p in sub
                        ).strip() or "".join(sub.itertext()).strip()
                    elif stag in ("media", "graphic", "self-uri", "ext-link") and not href:
                        href = (
                            sub.attrib.get(f"{ns_x}href")
                            or sub.attrib.get("href")
                            or ""
                        ).strip()
                        if not mime:
                            mime = (
                                sub.attrib.get("mimetype")
                                or sub.attrib.get("mime-subtype")
                                or ""
                            )
                # SI file URL composition. Prefer the PMC OA S3 bucket
                # (no bot-gating, public read) because if we got the JATS
                # XML from S3 the SI is there too. Fall back to the gated
                # pmc.ncbi.nlm.nih.gov web URL for papers not in OA bulk.
                # External href (http://...) passes through unchanged.
                alt_urls: list[str] = []
                if href.startswith("http://") or href.startswith("https://"):
                    url = href
                else:
                    # Primary: S3 OA bucket (version 1, matches what
                    # _s3_xml_url uses for the JATS fetch).
                    url = f"{_S3_BASE}/{pmcid}.1/{href}"
                    # Fallback: gated web URL on pmc.ncbi.nlm.nih.gov.
                    numeric_id = pmcid[3:] if pmcid.upper().startswith("PMC") else pmcid
                    alt_urls.append(
                        f"https://pmc.ncbi.nlm.nih.gov/articles/instance/{numeric_id}/bin/{href}"
                    )
                out.append({
                    "id": child.attrib.get("id"),
                    "label": label or None,
                    "caption": caption or None,
                    "href": href or None,
                    "url": url,
                    "alt_urls": alt_urls or None,
                    "mime_type": mime or None,
                })
            else:
                _walk(child)

    _walk(root)
    return out or None


def _extract_references_from_xml(xml_bytes: bytes) -> list[dict] | None:
    """Extract references from JATS XML <ref-list>.

    Returns a list of dicts with keys: doi, title, authors, year, journal, text.
    """
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return None

    ns = ""
    if "}" in root.tag:
        ns = root.tag.split("}")[0] + "}"

    # Find <ref-list> — can be under <back> or directly under <article>
    ref_list = root.find(f"{ns}back/{ns}ref-list")
    if ref_list is None:
        ref_list = root.find(f".//{ns}ref-list")
    if ref_list is None:
        return None

    refs: list[dict] = []
    for ref_el in ref_list.findall(f"{ns}ref"):
        entry: dict = {}

        # Try <mixed-citation> first (richer), then <element-citation>
        citation = ref_el.find(f"{ns}mixed-citation")
        if citation is None:
            citation = ref_el.find(f"{ns}element-citation")
        if citation is None:
            # Fallback: grab all text from the ref element
            text = "".join(ref_el.itertext()).strip()
            if text:
                entry["text"] = text
                refs.append(entry)
            continue

        # Extract DOI from <ext-link> or <pub-id>
        for ext in citation.findall(f".//{ns}ext-link"):
            href = ext.get("xlink:href", ext.get("href", ""))
            if not href:
                href = (ext.text or "").strip()
            if "doi.org/" in href:
                entry["doi"] = href.split("doi.org/")[-1]
                break
        if "doi" not in entry:
            for pub_id in citation.findall(f".//{ns}pub-id"):
                if (pub_id.get("pub-id-type") or "") == "doi" and pub_id.text:
                    entry["doi"] = pub_id.text.strip()
                    break

        # Extract title from <article-title>
        title_el = citation.find(f"{ns}article-title")
        if title_el is not None:
            entry["title"] = "".join(title_el.itertext()).strip()

        # Extract source/journal
        source_el = citation.find(f"{ns}source")
        if source_el is not None and source_el.text:
            entry["journal"] = source_el.text.strip()

        # Extract year
        year_el = citation.find(f"{ns}year")
        if year_el is not None and year_el.text:
            entry["year"] = year_el.text.strip()

        # Extract authors
        authors: list[str] = []
        for name_el in citation.findall(f".//{ns}name"):
            surname = name_el.find(f"{ns}surname")
            given = name_el.find(f"{ns}given-names")
            parts = []
            if given is not None and given.text:
                parts.append(given.text.strip())
            if surname is not None and surname.text:
                parts.append(surname.text.strip())
            if parts:
                authors.append(" ".join(parts))
        if authors:
            entry["authors"] = authors

        # Fallback full text of the citation
        if "title" not in entry:
            text = "".join(citation.itertext()).strip()
            if text:
                entry["text"] = text

        if entry:
            refs.append(entry)

    return refs if refs else None


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


async def get_supplementary_from_pmc(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> list[dict] | None:
    """List Supplementary Information files attached to a PMC OA paper.

    Returns a list of {"id", "label", "caption", "href", "url", "mime_type"}
    dicts pointing at the PMC bin/ mirror, or ``None`` when the paper isn't
    on PMC OA or has no SI blocks in its JATS XML.

    The list is metadata only — fetching the actual file bytes is left to
    the caller (typical pattern: per-file httpx.get keyed off ``url``).

    Most high-impact papers have most of their data in SI; this is the
    most reliable source for that data (publisher SI URLs vary
    per-publisher; PMC normalizes them at a predictable path).
    """
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean:
        return None
    client = http_client or httpx.AsyncClient(timeout=45.0, follow_redirects=True)
    should_close = http_client is None
    try:
        pmcid = await _resolve_pmcid(clean, client)
        if not pmcid:
            return None
        r_xml = await client.get(_s3_xml_url(pmcid), headers={"Accept": "application/xml"})
        if r_xml.status_code != 200 or not r_xml.content:
            return None
        items = _extract_supplementary_from_xml(r_xml.content, pmcid)
        if items:
            logger.info("pmc_supplementary_found", doi=clean, pmcid=pmcid, count=len(items))
        return items
    except Exception as e:
        logger.info("pmc_supplementary_failed", doi=clean, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()


async def get_fulltext_from_pmc(
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
                    refs = _extract_references_from_xml(r_xml.content)
                    logger.info(
                        "pmc_s3_xml_success",
                        doi=clean,
                        pmcid=pmcid,
                        text_length=len(text),
                        sections=len(sections) if sections else 0,
                        references=len(refs) if refs else 0,
                    )
                    _write_cache(pmcid, text, sections, refs, doi=clean)
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

        # Step 5: Fallback — NCBI efetch JATS XML.
        # The OA S3 bucket only holds the OA bulk subset; PMCIDs outside it
        # (author manuscripts, "restricted-by pmc") 404 there. efetch still
        # serves the full JATS for individual retrieval, in the same format
        # the extractors above already understand.
        efetch_url = _efetch_xml_url(pmcid)
        logger.info("pmc_efetch_try_xml", doi=clean, pmcid=pmcid, url=efetch_url)
        try:
            r_ef = await client.get(efetch_url, headers={"Accept": "application/xml"})
            if r_ef.status_code == 200 and r_ef.content:
                sections = _extract_sections_from_xml(r_ef.content)
                text = _extract_text_from_xml(r_ef.content)
                if text and len(text) > 200:
                    refs = _extract_references_from_xml(r_ef.content)
                    logger.info(
                        "pmc_efetch_xml_success",
                        doi=clean,
                        pmcid=pmcid,
                        text_length=len(text),
                        sections=len(sections) if sections else 0,
                        references=len(refs) if refs else 0,
                    )
                    _write_cache(pmcid, text, sections, refs, doi=clean)
                    return text, sections
        except Exception as e:
            logger.info("pmc_efetch_xml_failed", doi=clean, error=str(e))

        logger.warning("pmc_s3_failed", doi=clean, pmcid=pmcid)
        return None, None

    except Exception as e:
        logger.error("europepmc_error", doi=clean, error=str(e))
        return None, None
    finally:
        if should_close:
            await client.aclose()
