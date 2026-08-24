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
from .base import (
    CONTENT_TYPE_ABSTRACT,
    CONTENT_TYPE_FULL_TEXT,
    CONTENT_TYPE_NONE,
    CONTENT_TYPE_STRUCTURED,
    MIN_EXTRACTED_TEXT_CHARS,
    MIN_PLAUSIBLE_PDF_BYTES,
    PaperContent,
    PaperDiscovery,
    PDFDownloader,
    is_abstract_only,
)
from .biorxiv import get_content_from_biorxiv, is_biorxiv_doi
from .cookies import looks_like_paywall_html
from .discovery import discover_paper_sources
from .elsevier import get_content_from_elsevier
from .europepmc import get_content_from_europepmc
from .html_capture import capture_landing_html
from .interstitial import PDF_MAGIC, interstitial_marker
from .pmc import get_fulltext_from_pmc
from .rate_limit import RateLimited
from .rsc import download_from_rsc, is_rsc_doi
from .springer import download_from_springer, is_springer_doi
from .wiley import download_from_wiley_direct, download_from_wiley_tdm

logger = get_logger("perspicacite.pipeline.download.unified")

# Timeout for a client this module owns; the value the pipeline used
# inline before it was named.
OWNED_CLIENT_TIMEOUT_S = 60.0
# Below this an abstract is a stub ("n/a", a copyright line), not content.
MIN_PLAUSIBLE_ABSTRACT_CHARS = 20
# _parse_pdf_bytes inspects bytes with no request context, so the
# interstitial host check is skipped and only the body is matched.
NO_RESPONSE_URL = ""
NO_RESPONSE_CONTENT_TYPE = ""

# ``attempts`` statuses. "miss" means we asked and there was nothing;
# "rate_limited" means the host refused to answer, which is not evidence
# of absence and is the one status worth retrying later.
STATUS_MISS = "miss"
STATUS_SKIP = "skip"
STATUS_ERROR = "error"
STATUS_RATE_LIMITED = "rate_limited"


def _none_result(doi: str) -> PaperContent:
    """Empty PaperContent for a DOI nothing could be retrieved for."""
    return PaperContent(
        success=False,
        doi=doi,
        content_type=CONTENT_TYPE_NONE,
        content_source="none",
    )


def _merge_hosts(existing: list[str], extra: list[str]) -> list[str]:
    """Union of two host lists, preserving first-seen order."""
    merged = list(existing)
    for host in extra:
        if host not in merged:
            merged.append(host)
    return merged


def _throttled_hosts(attempts: list[dict[str, Any]]) -> list[str]:
    """Hosts an `attempts` trail recorded as throttled, first-seen order."""
    named = [
        str(a["host"])
        for a in attempts
        if a.get("status") == STATUS_RATE_LIMITED and a.get("host")
    ]
    return _merge_hosts([], named)


def _finalize(
    pc: PaperContent,
    attempts: list[dict[str, Any]],
    extra_hosts: list[str],
) -> PaperContent:
    """Attach the audit trail and throttling signal to `pc`, then return it.

    Inputs: the PaperContent about to be returned, the per-tier `attempts`
    trail, and `extra_hosts` observed throttling us outside that trail.
    Returns the same object, so callers can ``return _finalize(...)``.
    """
    pc.attempts.extend(attempts)
    hosts = _merge_hosts(_throttled_hosts(attempts), extra_hosts)
    pc.rate_limited_hosts = _merge_hosts(pc.rate_limited_hosts, hosts)
    return pc


def _record_biorxiv_throttling(
    br: PaperContent, attempts: list[dict[str, Any]]
) -> list[str]:
    """Add the bioRxiv result's throttled hosts to `attempts`; return them.

    Inputs: the PaperContent bioRxiv produced and the audit trail to append
    to. Returns the host list so the caller can carry it onto its own
    result — an empty list means bioRxiv was not throttled.
    """
    for host in br.rate_limited_hosts:
        attempts.append(
            {"source": br.content_source, "status": STATUS_RATE_LIMITED, "host": host}
        )
    return list(br.rate_limited_hosts)


def _build_owned_client(cookies_path: str | None) -> httpx.AsyncClient:
    """Build a client this module owns, carrying the configured cookie jar.

    Inputs: `cookies_path` to a Netscape cookies.txt, or None. Returns a
    client the caller must close. Caller-supplied clients are responsible
    for their own cookies (see build_authenticated_client).
    """
    client_kwargs: dict[str, Any] = {
        "timeout": OWNED_CLIENT_TIMEOUT_S,
        "follow_redirects": True,
    }
    if cookies_path:
        from perspicacite.pipeline.download.cookies import build_cookie_jar
        jar = build_cookie_jar(cookies_path)
        if jar is not None:
            client_kwargs["cookies"] = jar
    return httpx.AsyncClient(**client_kwargs)


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


def _is_non_document_body(body: bytes) -> bool:
    """Whether `body` is a bot wall or a paywall page rather than a document.

    Inputs: the raw bytes a downloader returned. Returns True only when a
    signal actually fired; the PDF magic number settles the question first,
    so a real PDF is never rejected by an HTML token in its metadata.
    """
    if body.startswith(PDF_MAGIC):
        return False
    marker = interstitial_marker(NO_RESPONSE_URL, NO_RESPONSE_CONTENT_TYPE, body)
    if marker:
        logger.warning("parse_body_interstitial", marker=marker, size_bytes=len(body))
        return True
    if looks_like_paywall_html(body):
        logger.warning("parse_body_is_html", size_bytes=len(body))
        return True
    return False


async def _parse_pdf_bytes(pdf_bytes: bytes, pdf_parser: Any) -> str | None:
    """Extract text from downloaded bytes using the provided parser.

    Inputs: `pdf_bytes` as returned by a downloader and a `pdf_parser`
    exposing ``parse``. Returns the extracted text, or None when the body
    is implausibly small, a bot wall, a paywall page, or yields too few
    characters to be body text.
    """
    if not pdf_bytes or len(pdf_bytes) < MIN_PLAUSIBLE_PDF_BYTES:
        return None
    if _is_non_document_body(pdf_bytes):
        return None
    if not pdf_bytes.startswith(PDF_MAGIC):
        # Non-PDF bytes (e.g. text encoded as bytes)
        text = pdf_bytes.decode("utf-8", errors="replace")
        return text if len(text.strip()) > MIN_EXTRACTED_TEXT_CHARS else None
    parsed = await pdf_parser.parse(pdf_bytes)
    text = parsed.text if parsed else None
    return text if text and len(text.strip()) > MIN_EXTRACTED_TEXT_CHARS else None


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
        PaperContent with the best available content. ``success`` reports
        that *something* was retrieved, not that full text was: use
        has_full_text(content_type) / is_abstract_only(content_type) to
        tell an abstract-only degradation from a full-text hit. A
        non-empty ``rate_limited_hosts`` means the outcome is a throttling
        artefact and the DOI is worth re-offering later.
    """
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean:
        return _none_result(doi)

    biorxiv_abstract_fallback: PaperContent | None = None
    # Per-step audit trail surfaced on the final PaperContent.attempts so
    # the caller can tell *why* the pipeline produced no content (vs.
    # the previous silent "no content" reason).
    attempts: list[dict[str, Any]] = []
    # Hosts that throttled us outside the PDF-tier attempts trail.
    throttled_hosts: list[str] = []

    # Bound before the try so the finally can never reference an unbound
    # name when client construction itself fails.
    client: httpx.AsyncClient | None = None
    should_close = False

    try:
        if http_client is not None:
            client = http_client
        else:
            client = _build_owned_client(cookies_path)
            should_close = True

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
                # success=True means "the abstract was retrieved", never
                # "full text was retrieved": callers separate the two with
                # is_abstract_only(content_type), not with success.
                return PaperContent(
                    success=True,
                    doi=clean,
                    content_type=CONTENT_TYPE_ABSTRACT,
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
        if text and len(text.strip()) > MIN_EXTRACTED_TEXT_CHARS:
            refs = _load_cached_references(clean)
            return PaperContent(
                success=True,
                doi=clean,
                content_type=CONTENT_TYPE_STRUCTURED,
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
        if epmc is not None and epmc.success and len(epmc_text.strip()) > MIN_EXTRACTED_TEXT_CHARS:
            # Preserve discovery-derived metadata
            return PaperContent(
                success=True,
                doi=clean,
                content_type=CONTENT_TYPE_STRUCTURED,
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
            if html_text and len(html_text.strip()) > MIN_EXTRACTED_TEXT_CHARS:
                return PaperContent(
                    success=True,
                    doi=clean,
                    content_type=(
                        CONTENT_TYPE_STRUCTURED if html_sections else CONTENT_TYPE_FULL_TEXT
                    ),
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
                throttled_hosts = _merge_hosts(
                    throttled_hosts, _record_biorxiv_throttling(br, attempts)
                )
                if br.content_type == CONTENT_TYPE_STRUCTURED:
                    return br
                if is_abstract_only(br.content_type):
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
                    return _finalize(
                        PaperContent(
                            success=True,
                            doi=clean,
                            content_type=CONTENT_TYPE_FULL_TEXT,
                            full_text=text,
                            abstract=disc.abstract,
                            content_source=source_label,
                            metadata=_metadata_from_discovery(disc, clean),
                        ),
                        attempts,
                        throttled_hosts,
                    )

        # Elsevier API (structured text, not PDF)
        if elsevier_api_key:
            result = await get_content_from_elsevier(clean, elsevier_api_key, client)
            if result.success and result.content:
                pc = PaperContent(
                    success=True,
                    doi=clean,
                    content_type=CONTENT_TYPE_FULL_TEXT,
                    full_text=result.content,
                    abstract=disc.abstract,
                    content_source="elsevier",
                    metadata=_metadata_from_discovery(disc, clean),
                )
                return _finalize(pc, attempts, throttled_hosts)
            attempts.append({
                "source": "elsevier",
                "status": STATUS_ERROR if result.error else STATUS_MISS,
                **({"error": result.error} if result.error else {}),
            })
        elif clean.lower().startswith(("10.1016/", "10.1006/", "10.1053/")):
            attempts.append(
                {"source": "elsevier", "status": STATUS_SKIP, "reason": "no_api_key"}
            )

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
                    return _finalize(
                        PaperContent(
                            success=True,
                            doi=clean,
                            content_type=CONTENT_TYPE_FULL_TEXT,
                            full_text=text,
                            abstract=disc.abstract,
                            content_source="alternative",
                            metadata=_metadata_from_discovery(disc, clean),
                        ),
                        attempts,
                        throttled_hosts,
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
                    content_type=CONTENT_TYPE_FULL_TEXT,
                    full_text=cap.extracted_text,
                    abstract=disc.abstract,
                    content_source="landing_html",
                    metadata=_metadata_from_discovery(disc, clean),
                )
                return _finalize(pc, attempts, throttled_hosts)

        # ── STEP 4: ABSTRACT ONLY ───────────────────────────────────────
        if disc.abstract and len(disc.abstract.strip()) > MIN_PLAUSIBLE_ABSTRACT_CHARS:
            # success=True reports a successful *abstract* retrieval. Callers
            # that need body text must ask has_full_text(content_type); the
            # abstract tier is a degradation, not a full-text hit.
            pc = PaperContent(
                success=True,
                doi=clean,
                content_type=CONTENT_TYPE_ABSTRACT,
                abstract=disc.abstract,
                content_source="openalex" if disc.title else "unknown",
                metadata=_metadata_from_discovery(disc, clean),
            )
            # F-30: surface the per-tier attempts trail even on successful
            # abstract-only degradation so operators can see which publisher
            # paths were skipped or missed before we settled for the abstract.
            return _finalize(pc, attempts, throttled_hosts)

        # ── STEP 4b: bioRxiv abstract fallback (when discovery had none) ──
        if biorxiv_abstract_fallback is not None:
            # Same F-30 fix for the bioRxiv-only fallback path
            return _finalize(biorxiv_abstract_fallback, attempts, throttled_hosts)

        # ── STEP 5: DISCARD ─────────────────────────────────────────────
        logger.warning("unified_no_content", doi=clean, attempts=len(attempts))
        pc = PaperContent(
            success=False,
            doi=clean,
            content_type=CONTENT_TYPE_NONE,
            content_source="none",
            metadata=_metadata_from_discovery(disc, clean),
        )
        return _finalize(pc, attempts, throttled_hosts)

    except Exception as e:
        logger.error("unified_pipeline_error", doi=clean, error=str(e))
        return _none_result(clean)
    finally:
        if should_close and client is not None:
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
    record so the caller can surface why nothing worked. A tier whose host
    throttled us is recorded with STATUS_RATE_LIMITED and its host, which
    is how the caller learns the miss is not evidence of absence.
    """

    def _record(src: str, status: str, **extra: Any) -> None:
        if attempts is None:
            return
        rec: dict[str, Any] = {"source": src, "status": status}
        rec.update(extra)
        attempts.append(rec)

    async def _fetch(src: str, pdf_url: str) -> bytes | None:
        """Download `pdf_url` for tier `src`; record a miss or throttling."""
        try:
            data = await PDFDownloader().download(pdf_url, http_client=client)
        except RateLimited as exc:
            _record(src, STATUS_RATE_LIMITED, url=pdf_url, host=exc.host)
            return None
        if data and len(data) >= MIN_PLAUSIBLE_PDF_BYTES:
            return data
        _record(src, STATUS_MISS, url=pdf_url)
        return None

    # 3a. Publisher OA PDFs from discovery, best candidate first. The
    # single oa_url is the fallback for a discovery that predates the
    # ranked list, so no route regresses.
    for oa_url in disc.oa_candidates or ([disc.oa_url] if disc.oa_url else []):
        data = await _fetch("publisher_oa_pdf", oa_url)
        if data:
            return data, "publisher_oa_pdf"

    # 3b. arXiv PDF
    if disc.arxiv_id or is_arxiv_doi(doi) or (url and is_arxiv_url(url)):
        pdf = await download_from_arxiv(doi=doi, url=url, http_client=client)
        if pdf:
            return pdf, "arxiv_pdf"
        _record("arxiv_pdf", STATUS_MISS)

    # 3c. Unpaywall PDF URL
    if disc.unpaywall_pdf_url:
        data = await _fetch("unpaywall_pdf", disc.unpaywall_pdf_url)
        if data:
            return data, "unpaywall_pdf"

    # 3d. Publisher-specific APIs. The former OpenAlex OA tier is gone:
    # discovery already ranks OpenAlex's OA urls into oa_candidates, so
    # re-fetching the same work here only cost a second API round trip.
    if is_acs_doi(doi):
        pdf = await download_from_acs(doi, client)
        if pdf:
            return pdf, "acs_pdf"
        _record("acs_pdf", STATUS_MISS)

    if is_rsc_doi(doi):
        if not rsc_api_key:
            _record("rsc_pdf", STATUS_SKIP, reason="no_api_key")
        else:
            pdf = await download_from_rsc(doi, rsc_api_key, client)
            if pdf:
                return pdf, "rsc_pdf"
            _record("rsc_pdf", STATUS_MISS)

    if is_aaas_doi(doi):
        if not aaas_api_key:
            _record("aaas_pdf", STATUS_SKIP, reason="no_api_key")
        else:
            pdf = await download_from_aaas(doi, aaas_api_key, client)
            if pdf:
                return pdf, "aaas_pdf"
            _record("aaas_pdf", STATUS_MISS)

    if is_springer_doi(doi):
        if not springer_api_key:
            _record("springer_pdf", STATUS_SKIP, reason="no_api_key")
        else:
            pdf = await download_from_springer(doi, springer_api_key, client)
            if pdf:
                return pdf, "springer_pdf"
            _record("springer_pdf", STATUS_MISS,
                    hint="API key present but no PDF returned — check entitlement or DOI type")

    if doi.lower().startswith("10.1002/"):
        pdf = await download_from_wiley_direct(doi, client)
        if pdf:
            return pdf, "wiley_pdf"
        _record("wiley_pdf", STATUS_MISS)

    if wiley_tdm_token:
        pdf = await download_from_wiley_tdm(doi, wiley_tdm_token, client)
        if pdf:
            return pdf, "wiley_tdm_pdf"
        _record("wiley_tdm_pdf", STATUS_MISS)

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


if __name__ == "__main__":
    import asyncio

    # Offline smoke checks: no network, no client, no chroma_db.
    SMOKE_PDF = b"%PDF-1.7 " + b"x" * MIN_PLAUSIBLE_PDF_BYTES
    SMOKE_HTML = b"<!DOCTYPE html><html><body>Access options</body></html>" + b" " * 4096
    SMOKE_WALL = b"<html>cloudPMC-viewer-pow</html>" + b" " * 4096
    SMOKE_TEXT = ("word " * MIN_PLAUSIBLE_PDF_BYTES).encode()

    class _StubParser:
        """Parser stub returning a fixed text, so no PDF engine is loaded."""

        class _Parsed:
            text = "body " * MIN_EXTRACTED_TEXT_CHARS

        async def parse(self, _data: bytes) -> _StubParser._Parsed:
            """Return the fixed parse result for any bytes."""
            return self._Parsed()

    async def _check_parse() -> None:
        """A real PDF parses; a paywall page and a bot wall do not."""
        parser = _StubParser()
        assert await _parse_pdf_bytes(SMOKE_PDF, parser)
        assert await _parse_pdf_bytes(SMOKE_HTML, parser) is None
        assert await _parse_pdf_bytes(SMOKE_WALL, parser) is None
        assert await _parse_pdf_bytes(SMOKE_TEXT, parser)
        assert await _parse_pdf_bytes(b"%PDF-tiny", parser) is None

    assert not _is_non_document_body(SMOKE_PDF)
    assert _is_non_document_body(SMOKE_HTML)
    assert _is_non_document_body(SMOKE_WALL)
    assert not _is_non_document_body(SMOKE_TEXT)
    assert _merge_hosts(["a"], ["a", "b"]) == ["a", "b"]
    assert _throttled_hosts([{"status": STATUS_MISS, "host": "a.org"}]) == []
    assert _throttled_hosts([{"status": STATUS_RATE_LIMITED, "host": "a.org"}]) == ["a.org"]
    smoke_pc = _finalize(
        _none_result("10.0/x"),
        [{"source": "s", "status": STATUS_RATE_LIMITED, "host": "a.org"}],
        ["b.org"],
    )
    assert smoke_pc.rate_limited_hosts == ["a.org", "b.org"]
    assert len(smoke_pc.attempts) == 1
    assert smoke_pc.content_type == CONTENT_TYPE_NONE
    asyncio.run(_check_parse())
    print("unified.py smoke checks passed")
