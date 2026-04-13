"""Unified paper content retrieval pipeline.

Priority flow:
  1. DISCOVERY     -- OpenAlex + Unpaywall → metadata, PMCID, arXiv ID, OA URLs
  2. ALTERNATIVE   -- User-configured endpoint (if set)
  3. STRUCTURED    -- PMC JATS XML, then arXiv HTML (sections + references)
  4. PDF TEXT      -- Publisher OA, arXiv PDF, Unpaywall, publisher APIs
  5. ABSTRACT      -- From discovery metadata
  6. DISCARD       -- No content available
"""

from __future__ import annotations

from typing import Any

import httpx

from perspicacite.logging import get_logger
from .base import PaperContent, PaperDiscovery, PDFDownloader
from .discovery import discover_paper_sources
from .pmc import get_fulltext_from_pmc
from .arxiv import (
    download_from_arxiv,
    fetch_arxiv_html,
    get_arxiv_id_from_doi,
    is_arxiv_doi,
    is_arxiv_url,
)
from .openalex_oa import download_pdf_from_openalex_oa
from .acs import download_from_acs, is_acs_doi
from .rsc import download_from_rsc, is_rsc_doi
from .aaas import download_from_aaas, is_aaas_doi
from .springer import download_from_springer, is_springer_doi
from .wiley import download_from_wiley_tdm, download_from_wiley_direct
from .elsevier import get_content_from_elsevier
from .alternative import download_from_alternative_endpoint

logger = get_logger("perspicacite.pipeline.download.unified")


def _none_result(doi: str) -> PaperContent:
    return PaperContent(
        success=False,
        doi=doi,
        content_type="none",
        content_source="none",
    )


async def _parse_pdf_bytes(pdf_bytes: bytes, pdf_parser: Any) -> str | None:
    """Extract text from PDF bytes using the provided parser."""
    if not pdf_bytes or len(pdf_bytes) < 1000:
        return None
    if not pdf_bytes[:4] == b"%PDF":
        # Non-PDF bytes (e.g. text encoded as bytes)
        text = pdf_bytes.decode("utf-8", errors="replace")
        return text if len(text.strip()) > 200 else None
    parsed = await pdf_parser.parse(pdf_bytes)
    text = parsed.text if parsed else None
    if text and len(text.strip()) > 200:
        return text
    return None


async def retrieve_paper_content(
    doi: str,
    *,
    url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    pdf_parser: Any = None,
    alternative_endpoint: str | None = None,
    unpaywall_email: str | None = None,
    wiley_tdm_token: str | None = None,
    elsevier_api_key: str | None = None,
    aaas_api_key: str | None = None,
    rsc_api_key: str | None = None,
    springer_api_key: str | None = None,
) -> PaperContent:
    """Retrieve paper content using the unified priority pipeline.

    Steps:
      1. DISCOVERY: OpenAlex then Unpaywall
      2. STRUCTURED full text: PMC JATS XML, then arXiv HTML
      3. PDF full text: OA PDF, arXiv, Unpaywall, publisher APIs, alternative
      4. ABSTRACT only: from discovery
      5. DISCARD: no content

    Args:
        doi: Paper DOI.
        url: Optional paper URL (may help arXiv detection).
        http_client: Optional httpx.AsyncClient for connection reuse.
        pdf_parser: Optional PDFParser for PDF text extraction.
            If None, PDF sources are skipped.
        alternative_endpoint: Optional alternative endpoint URL.
        unpaywall_email: Email for Unpaywall API.
        wiley_tdm_token: Wiley TDM API token.
        elsevier_api_key: Elsevier API key.
        aaas_api_key: AAAS API key.
        rsc_api_key: RSC API key.
        springer_api_key: Springer API key.

    Returns:
        PaperContent with the best available content.
    """
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean:
        return _none_result(doi)

    client = http_client or httpx.AsyncClient(
        timeout=60.0, follow_redirects=True
    )
    should_close = http_client is None

    try:
        # ── STEP 1: DISCOVERY ──────────────────────────────────────────
        disc = await discover_paper_sources(clean, client, unpaywall_email)
        logger.info(
            "unified_discovery_complete",
            doi=clean,
            pmcid=disc.pmcid,
            arxiv_id=disc.arxiv_id,
            is_oa=disc.is_oa,
            has_abstract=disc.abstract is not None,
        )

        # ── STEP 2: ALTERNATIVE ENDPOINT (before structured/PDF) ──────
        if alternative_endpoint and pdf_parser is not None:
            alt_pdf = await download_from_alternative_endpoint(
                clean, alternative_endpoint, client
            )
            if alt_pdf:
                text = await _parse_pdf_bytes(alt_pdf, pdf_parser)
                if text:
                    return PaperContent(
                        success=True,
                        doi=clean,
                        content_type="full_text",
                        full_text=text,
                        abstract=disc.abstract,
                        content_source="alternative",
                        metadata={
                            "title": disc.title,
                            "authors": disc.authors,
                            "year": disc.year,
                            "doi": clean,
                        },
                    )

        # ── STEP 3: STRUCTURED FULL TEXT ────────────────────────────────

        # 2a. PMC JATS XML (sections + references)
        if disc.pmcid:
            text, sections = await get_fulltext_from_pmc(clean, client)
            if text and len(text.strip()) > 200:
                refs = _load_cached_references(clean)
                return PaperContent(
                    success=True,
                    doi=clean,
                    content_type="structured",
                    full_text=text,
                    sections=sections,
                    references=refs,
                    abstract=disc.abstract,
                    content_source="pmc",
                    metadata={"pmcid": disc.pmcid, "title": disc.title},
                )

        # 2b. arXiv HTML
        arxiv_id = disc.arxiv_id
        if not arxiv_id and is_arxiv_doi(clean):
            arxiv_id = get_arxiv_id_from_doi(clean)
        if not arxiv_id and url and is_arxiv_url(url) and "/abs/" in url:
            arxiv_id = url.split("/abs/")[-1].split("?")[0].split("#")[0]

        if arxiv_id:
            html_text, html_sections, _html_title = await fetch_arxiv_html(arxiv_id, client)
            if html_text and len(html_text.strip()) > 200:
                return PaperContent(
                    success=True,
                    doi=clean,
                    content_type="structured" if html_sections else "full_text",
                    full_text=html_text,
                    sections=html_sections,
                    abstract=disc.abstract,
                    content_source="arxiv_html",
                    metadata={"arxiv_id": arxiv_id, "title": disc.title},
                )

        # ── STEP 3: PDF FULL TEXT ───────────────────────────────────────
        if pdf_parser is not None:
            pdf_result = await _try_pdf_sources(
                clean,
                url,
                client,
                disc,
                unpaywall_email=unpaywall_email,
                wiley_tdm_token=wiley_tdm_token,
                aaas_api_key=aaas_api_key,
                rsc_api_key=rsc_api_key,
                springer_api_key=springer_api_key,
            )
            if pdf_result:
                pdf_bytes, source_label = pdf_result
                text = await _parse_pdf_bytes(pdf_bytes, pdf_parser)
                if text:
                    return PaperContent(
                        success=True,
                        doi=clean,
                        content_type="full_text",
                        full_text=text,
                        abstract=disc.abstract,
                        content_source=source_label,
                        metadata={"title": disc.title},
                    )

        # Elsevier API (structured text, not PDF)
        if elsevier_api_key:
            result = await get_content_from_elsevier(clean, elsevier_api_key, client)
            if result.success and result.content:
                return PaperContent(
                    success=True,
                    doi=clean,
                    content_type="full_text",
                    full_text=result.content,
                    abstract=disc.abstract,
                    content_source="elsevier",
                    metadata={"title": disc.title},
                )

        # ── STEP 4: ABSTRACT ONLY ───────────────────────────────────────
        if disc.abstract and len(disc.abstract.strip()) > 20:
            return PaperContent(
                success=True,
                doi=clean,
                content_type="abstract",
                abstract=disc.abstract,
                content_source="openalex" if disc.title else "unknown",
                metadata={"title": disc.title, "is_oa": disc.is_oa},
            )

        # ── STEP 5: DISCARD ─────────────────────────────────────────────
        logger.warning("unified_no_content", doi=clean)
        return _none_result(clean)

    except Exception as e:
        logger.error("unified_pipeline_error", doi=clean, error=str(e))
        return _none_result(clean)
    finally:
        if should_close:
            await client.aclose()


async def _try_pdf_sources(
    doi: str,
    url: str | None,
    client: httpx.AsyncClient,
    disc: PaperDiscovery,
    *,
    unpaywall_email: str | None = None,
    wiley_tdm_token: str | None = None,
    aaas_api_key: str | None = None,
    rsc_api_key: str | None = None,
    springer_api_key: str | None = None,
) -> tuple[bytes, str] | None:
    """Try PDF sources in priority order. Returns (bytes, source_label) or None."""

    # 3a. Publisher OA PDF via discovery OA URL
    if disc.oa_url:
        downloader = PDFDownloader()
        data = await downloader.download(disc.oa_url, http_client=client)
        if data and len(data) > 1000:
            return data, "publisher_oa_pdf"

    # 3b. arXiv PDF
    if disc.arxiv_id or is_arxiv_doi(doi) or (url and is_arxiv_url(url)):
        pdf = await download_from_arxiv(doi=doi, url=url, http_client=client)
        if pdf:
            return pdf, "arxiv_pdf"

    # 3c. Unpaywall PDF URL
    if disc.unpaywall_pdf_url:
        downloader = PDFDownloader()
        data = await downloader.download(disc.unpaywall_pdf_url, http_client=client)
        if data and len(data) > 1000:
            return data, "unpaywall_pdf"

    # 3d. OpenAlex OA PDF
    pdf = await download_pdf_from_openalex_oa(doi, client)
    if pdf:
        return pdf, "openalex_oa_pdf"

    # 3e. Publisher-specific APIs
    if is_acs_doi(doi):
        pdf = await download_from_acs(doi, client)
        if pdf:
            return pdf, "acs_pdf"

    if is_rsc_doi(doi):
        pdf = await download_from_rsc(doi, rsc_api_key, client)
        if pdf:
            return pdf, "rsc_pdf"

    if is_aaas_doi(doi):
        pdf = await download_from_aaas(doi, aaas_api_key, client)
        if pdf:
            return pdf, "aaas_pdf"

    if is_springer_doi(doi):
        pdf = await download_from_springer(doi, springer_api_key, client)
        if pdf:
            return pdf, "springer_pdf"

    if doi.lower().startswith("10.1002/"):
        pdf = await download_from_wiley_direct(doi, client)
        if pdf:
            return pdf, "wiley_pdf"

    if wiley_tdm_token:
        pdf = await download_from_wiley_tdm(doi, wiley_tdm_token, client)
        if pdf:
            return pdf, "wiley_tdm_pdf"

    return None


def _load_cached_references(doi: str) -> list[dict] | None:
    """Load cached references from the sections JSON file."""
    import json
    from pathlib import Path

    clean_doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if clean_doi.startswith(prefix):
            clean_doi = clean_doi[len(prefix):]

    cache_dir = Path("./data/papers")
    if not cache_dir.exists():
        return None

    refs_file = cache_dir / f"{clean_doi.replace('/', '_')}_refs.json"
    if refs_file.exists():
        try:
            return json.loads(refs_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            pass
    return None
