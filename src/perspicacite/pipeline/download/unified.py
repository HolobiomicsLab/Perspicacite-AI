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

from .aaas import download_from_aaas, is_aaas_doi
from .acs import download_from_acs, is_acs_doi
from .alternative import download_from_alternative_endpoint
from .arxiv import (
    download_from_arxiv,
    fetch_arxiv_html,
    get_arxiv_id_from_doi,
    is_arxiv_doi,
    is_arxiv_url,
)
from .base import PaperContent, PaperDiscovery, PDFDownloader
from .biorxiv import get_content_from_biorxiv, is_biorxiv_doi
from .discovery import discover_paper_sources
from .elsevier import get_content_from_elsevier
from .europepmc import get_content_from_europepmc
from .html_capture import capture_landing_html
from .openalex_oa import download_pdf_from_openalex_oa
from .pmc import get_fulltext_from_pmc
from .rsc import download_from_rsc, is_rsc_doi
from .springer import download_from_springer, is_springer_doi
from .wiley import download_from_wiley_direct, download_from_wiley_tdm

logger = get_logger("perspicacite.pipeline.download.unified")


def _none_result(doi: str) -> PaperContent:
    return PaperContent(
        success=False,
        doi=doi,
        content_type="none",
        content_source="none",
    )


def _metadata_from_discovery(
    disc: PaperDiscovery,
    doi: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a uniform metadata dict from a PaperDiscovery result.

    Every PaperContent return site uses this so that downstream consumers
    (orchestrator, web app) get authors/year/title/ids in one consistent shape.
    """
    md: dict[str, Any] = {
        "doi": doi,
        "title": disc.title,
        "authors": disc.authors,
        "year": disc.year,
        "is_oa": disc.is_oa,
        "work_type": disc.work_type,
        "license": disc.license,
    }
    if disc.arxiv_id:
        md["arxiv_id"] = disc.arxiv_id
    if disc.pmcid:
        md["pmcid"] = disc.pmcid
    if getattr(disc, "journal", None):
        md["journal"] = disc.journal
    if extra:
        md.update(extra)
    return md


async def _parse_pdf_bytes(pdf_bytes: bytes, pdf_parser: Any) -> str | None:
    """Extract text from PDF bytes using the provided parser."""
    if not pdf_bytes or len(pdf_bytes) < 1000:
        return None
    if pdf_bytes[:4] != b"%PDF":
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
    cookies_path: str | None = None,
    cookie_domains: list[str] | None = None,
    pdf_cache_dir: str | None = None,
    abstract_only: bool = False,
    enable_landing_capture: bool = False,
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

    biorxiv_abstract_fallback: PaperContent | None = None
    # Per-step audit trail surfaced on the final PaperContent.attempts so
    # the caller can tell *why* the pipeline produced no content (vs.
    # the previous silent "no content" reason).
    attempts: list[dict[str, Any]] = []

    if http_client is not None:
        client = http_client
        should_close = False
    else:
        # When we own the client we can attach the cookie jar. Caller-supplied
        # clients are responsible for their own cookies (see
        # build_authenticated_client below).
        client_kwargs: dict[str, Any] = {"timeout": 60.0, "follow_redirects": True}
        if cookies_path:
            from perspicacite.pipeline.download.cookies import build_cookie_jar
            jar = build_cookie_jar(cookies_path)
            if jar is not None:
                client_kwargs["cookies"] = jar
        client = httpx.AsyncClient(**client_kwargs)
        should_close = True

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

        # ── Crossref gap-fill (cheap; never overwrites discovery values) ──
        if any(
            getattr(disc, f, None) in (None, "", [])
            for f in ("title", "authors", "year", "abstract")
        ):
            try:
                from .crossref import enrich_from_crossref

                base_meta = {
                    "title": disc.title,
                    "authors": disc.authors,
                    "year": disc.year,
                    "abstract": disc.abstract,
                    "journal": getattr(disc, "journal", None),
                }
                patch = await enrich_from_crossref(
                    clean, http_client=client, base_metadata=base_meta, mailto=unpaywall_email
                )
                if patch.get("title") and not disc.title:
                    disc.title = patch["title"]
                if patch.get("authors") and not disc.authors:
                    disc.authors = patch["authors"]
                if patch.get("year") and not disc.year:
                    disc.year = patch["year"]
                if patch.get("abstract") and not disc.abstract:
                    disc.abstract = patch["abstract"]
                if patch.get("journal") and not getattr(disc, "journal", None):
                    disc.journal = patch["journal"]
            except Exception as e:
                logger.warning("crossref_enrich_skipped", doi=clean, error=str(e))

        # ── ABSTRACT-ONLY FAST PATH ────────────────────────────────────────
        if abstract_only:
            if disc.abstract:
                return PaperContent(
                    success=True,
                    doi=clean,
                    content_type="abstract",
                    full_text=None,
                    abstract=disc.abstract,
                    content_source="discovery",
                    metadata=_metadata_from_discovery(disc, clean),
                )
            return _none_result(doi)

        # ── STEP 2: STRUCTURED FULL TEXT ────────────────────────────────

        # 2a. PMC JATS XML (sections + references).
        # Run regardless of disc.pmcid: get_fulltext_from_pmc resolves its own
        # PMCID via Europe PMC, which catches papers that are in PMC but whose
        # PMCID OpenAlex/Unpaywall discovery did not surface. It short-circuits
        # cheaply when the DOI is not in PMC.
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
                metadata=_metadata_from_discovery(disc, clean),
            )

        # 2a-bis. Europe PMC fullTextXML (broader OA coverage)
        epmc = await get_content_from_europepmc(
            doi=clean,
            pmid=None,  # PaperDiscovery has no pmid field; only DOI+PMCID resolution
            pmcid=disc.pmcid,
            http_client=client,
        )
        epmc_text = (epmc.full_text or "") if epmc is not None else ""
        if epmc is not None and epmc.success and len(epmc_text.strip()) > 200:
            # Preserve discovery-derived metadata
            return PaperContent(
                success=True,
                doi=clean,
                content_type="structured",
                full_text=epmc.full_text,
                sections=epmc.sections,
                references=epmc.references,
                abstract=disc.abstract,
                content_source="europepmc",
                metadata=_metadata_from_discovery(disc, clean, epmc.metadata),
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
                    metadata=_metadata_from_discovery(disc, clean, {"arxiv_id": arxiv_id}),
                )

        # bioRxiv / medRxiv preprints
        if is_biorxiv_doi(clean):
            br = await get_content_from_biorxiv(clean, http_client=client)
            if br is not None and br.success:
                if br.content_type == "structured":
                    return br
                if br.content_type == "abstract":
                    biorxiv_abstract_fallback = br

        # ── STEP 3: PDF FULL TEXT ───────────────────────────────────────
        if pdf_parser is not None:
            # Cache hit: serve bytes from disk and skip every network
            # downloader. Provenance label says "pdf_cache" so the
            # caller can tell.
            cached_bytes: bytes | None = None
            if pdf_cache_dir:
                from perspicacite.pipeline.download.pdf_cache import (
                    get_cached_pdf,
                )
                cached_bytes = get_cached_pdf(clean, pdf_cache_dir)
            if cached_bytes is not None:
                pdf_result = (cached_bytes, "pdf_cache")
            else:
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
                    attempts=attempts,
                )
                if pdf_result and pdf_cache_dir:
                    # Persist the winning bytes so the next ingest is free.
                    from perspicacite.pipeline.download.pdf_cache import (
                        store_pdf,
                    )
                    store_pdf(
                        clean, pdf_result[0], pdf_cache_dir,
                        source=pdf_result[1],
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
                        metadata=_metadata_from_discovery(disc, clean),
                    )

        # Elsevier API (structured text, not PDF)
        if elsevier_api_key:
            result = await get_content_from_elsevier(clean, elsevier_api_key, client)
            if result.success and result.content:
                pc = PaperContent(
                    success=True,
                    doi=clean,
                    content_type="full_text",
                    full_text=result.content,
                    abstract=disc.abstract,
                    content_source="elsevier",
                    metadata=_metadata_from_discovery(disc, clean),
                )
                pc.attempts.extend(attempts)
                return pc
            attempts.append({
                "source": "elsevier",
                "status": "error" if result.error else "miss",
                **({"error": result.error} if result.error else {}),
            })
        elif clean.lower().startswith(("10.1016/", "10.1006/", "10.1053/")):
            attempts.append({"source": "elsevier", "status": "skip", "reason": "no_api_key"})

        # ── STEP 3b: ALTERNATIVE ENDPOINT (last-resort PDF fallback) ────
        # User-configured private/institutional repository. Demoted to
        # the very bottom of the PDF chain so it only fires when every
        # OA path (PMC, Europe PMC, arXiv HTML, biorxiv JATS) and every
        # publisher PDF tier has missed. Useful for paywalled papers
        # the user has rights to via an institutional cache, without
        # competing with structured-text sources for OA content.
        if alternative_endpoint and pdf_parser is not None:
            alt_pdf = await download_from_alternative_endpoint(
                clean, alternative_endpoint, client,
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
                        metadata=_metadata_from_discovery(disc, clean),
                    )

        # ── STEP 3c: COOKIE-AUTHENTICATED LANDING CAPTURE ───────────────
        # Last-resort full text from the publisher landing page, fetched
        # through the (cookie-authenticated) client. Brings the DOI path to
        # parity with ingest_url for paywalled papers the user has cookie
        # access to. Opt-in (enabled when cookies are configured) and only
        # accepted at the full_text tier — thinner captures fall through to
        # the abstract path below.
        if enable_landing_capture:
            try:
                cap = await capture_landing_html(
                    doi=clean,
                    landing_url=disc.oa_url,
                    abstract=disc.abstract or "",
                    title=disc.title or "",
                    http_client=client,
                    cache_dir=pdf_cache_dir,
                )
            except Exception as e:
                logger.info("landing_capture_failed", doi=clean, error=str(e))
                cap = None
            if cap is not None and cap.tier == "full_text" and cap.extracted_text:
                pc = PaperContent(
                    success=True,
                    doi=clean,
                    content_type="full_text",
                    full_text=cap.extracted_text,
                    abstract=disc.abstract,
                    content_source="landing_html",
                    metadata=_metadata_from_discovery(disc, clean),
                )
                pc.attempts.extend(attempts)
                return pc

        # ── STEP 4: ABSTRACT ONLY ───────────────────────────────────────
        if disc.abstract and len(disc.abstract.strip()) > 20:
            pc = PaperContent(
                success=True,
                doi=clean,
                content_type="abstract",
                abstract=disc.abstract,
                content_source="openalex" if disc.title else "unknown",
                metadata=_metadata_from_discovery(disc, clean),
            )
            # F-30: surface the per-tier attempts trail even on successful
            # abstract-only degradation so operators can see which publisher
            # paths were skipped or missed before we settled for the abstract.
            pc.attempts.extend(attempts)
            return pc

        # ── STEP 4b: bioRxiv abstract fallback (when discovery had none) ──
        if biorxiv_abstract_fallback is not None:
            # Same F-30 fix for the bioRxiv-only fallback path
            if hasattr(biorxiv_abstract_fallback, "attempts"):
                biorxiv_abstract_fallback.attempts.extend(attempts)
            return biorxiv_abstract_fallback

        # ── STEP 5: DISCARD ─────────────────────────────────────────────
        logger.warning("unified_no_content", doi=clean, attempts=len(attempts))
        pc = PaperContent(
            success=False,
            doi=clean,
            content_type="none",
            content_source="none",
            metadata=_metadata_from_discovery(disc, clean),
        )
        pc.attempts.extend(attempts)
        return pc

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
    attempts: list[dict[str, Any]] | None = None,
) -> tuple[bytes, str] | None:
    """Try PDF sources in priority order. Returns (bytes, source_label) or None.

    When ``attempts`` is provided, each tier appends a {source,status,...}
    record so the caller can surface why nothing worked.
    """

    def _record(src: str, status: str, **extra: Any) -> None:
        if attempts is None:
            return
        rec: dict[str, Any] = {"source": src, "status": status}
        rec.update(extra)
        attempts.append(rec)

    # 3a. Publisher OA PDF via discovery OA URL
    if disc.oa_url:
        downloader = PDFDownloader()
        data = await downloader.download(disc.oa_url, http_client=client)
        if data and len(data) > 1000:
            return data, "publisher_oa_pdf"
        _record("publisher_oa_pdf", "miss", url=disc.oa_url)

    # 3b. arXiv PDF
    if disc.arxiv_id or is_arxiv_doi(doi) or (url and is_arxiv_url(url)):
        pdf = await download_from_arxiv(doi=doi, url=url, http_client=client)
        if pdf:
            return pdf, "arxiv_pdf"
        _record("arxiv_pdf", "miss")

    # 3c. Unpaywall PDF URL
    if disc.unpaywall_pdf_url:
        downloader = PDFDownloader()
        data = await downloader.download(disc.unpaywall_pdf_url, http_client=client)
        if data and len(data) > 1000:
            return data, "unpaywall_pdf"
        _record("unpaywall_pdf", "miss", url=disc.unpaywall_pdf_url)

    # 3d. OpenAlex OA PDF
    pdf = await download_pdf_from_openalex_oa(doi, client)
    if pdf:
        return pdf, "openalex_oa_pdf"
    _record("openalex_oa_pdf", "miss")

    # 3e. Publisher-specific APIs
    if is_acs_doi(doi):
        pdf = await download_from_acs(doi, client)
        if pdf:
            return pdf, "acs_pdf"
        _record("acs_pdf", "miss")

    if is_rsc_doi(doi):
        if not rsc_api_key:
            _record("rsc_pdf", "skip", reason="no_api_key")
        else:
            pdf = await download_from_rsc(doi, rsc_api_key, client)
            if pdf:
                return pdf, "rsc_pdf"
            _record("rsc_pdf", "miss")

    if is_aaas_doi(doi):
        if not aaas_api_key:
            _record("aaas_pdf", "skip", reason="no_api_key")
        else:
            pdf = await download_from_aaas(doi, aaas_api_key, client)
            if pdf:
                return pdf, "aaas_pdf"
            _record("aaas_pdf", "miss")

    if is_springer_doi(doi):
        if not springer_api_key:
            _record("springer_pdf", "skip", reason="no_api_key")
        else:
            pdf = await download_from_springer(doi, springer_api_key, client)
            if pdf:
                return pdf, "springer_pdf"
            _record("springer_pdf", "miss",
                    hint="API key present but no PDF returned — check entitlement or DOI type")

    if doi.lower().startswith("10.1002/"):
        pdf = await download_from_wiley_direct(doi, client)
        if pdf:
            return pdf, "wiley_pdf"
        _record("wiley_pdf", "miss")

    if wiley_tdm_token:
        pdf = await download_from_wiley_tdm(doi, wiley_tdm_token, client)
        if pdf:
            return pdf, "wiley_tdm_pdf"
        _record("wiley_tdm_pdf", "miss")

    return None


async def download_paper_pdf(
    doi: str,
    *,
    url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    unpaywall_email: str | None = None,
    wiley_tdm_token: str | None = None,
    aaas_api_key: str | None = None,
    rsc_api_key: str | None = None,
    springer_api_key: str | None = None,
    pdf_cache_dir: str | None = None,
) -> tuple[bytes, str] | None:
    """Download a PDF for ``doi``, irrespective of structured-text availability.

    Used by ``push_to_zotero(attach_pdf=True)`` to ensure an actual PDF
    binary lands in cache. The unified pipeline normally returns
    structured HTML (e.g. arXiv) before reaching the PDF tier, so a
    separate PDF-only fetch is needed when the caller specifically
    wants the PDF artifact.

    Discovers OA URLs via OpenAlex/Unpaywall, then tries each PDF
    source in the same priority order as ``_try_pdf_sources``. Caches
    the winning bytes when ``pdf_cache_dir`` is set.

    Returns ``(bytes, source_label)`` on success; ``None`` if no PDF
    can be found across any route.
    """
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean:
        return None

    if http_client is not None:
        client = http_client
        should_close = False
    else:
        client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        should_close = True

    try:
        if pdf_cache_dir:
            from perspicacite.pipeline.download.pdf_cache import get_cached_pdf
            cached = get_cached_pdf(clean, pdf_cache_dir)
            if cached is not None:
                return cached, "pdf_cache"

        disc = await discover_paper_sources(clean, client, unpaywall_email)
        result = await _try_pdf_sources(
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
        if result and pdf_cache_dir:
            from perspicacite.pipeline.download.pdf_cache import store_pdf
            store_pdf(clean, result[0], pdf_cache_dir, source=result[1])
        return result
    finally:
        if should_close:
            await client.aclose()


def _load_cached_references(doi: str) -> list[dict] | None:
    """Load cached references from the sections JSON file."""
    import json
    from pathlib import Path

    clean_doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if clean_doi.startswith(prefix):
            clean_doi = clean_doi[len(prefix) :]

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
