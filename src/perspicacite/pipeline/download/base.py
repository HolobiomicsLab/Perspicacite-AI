"""Base classes and utilities for download modules."""

from dataclasses import dataclass, field
from typing import Any

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download")

# Values of ``PaperContent.content_type``. The strings are persisted into
# ChromaDB chunk metadata by rag/dynamic_kb.py, which also selects the
# embedding model from them, so they are a storage format: name them here,
# never rename the values.
CONTENT_TYPE_STRUCTURED = "structured"
CONTENT_TYPE_FULL_TEXT = "full_text"
CONTENT_TYPE_ABSTRACT = "abstract"
CONTENT_TYPE_NONE = "none"
FULL_TEXT_TIERS = (CONTENT_TYPE_STRUCTURED, CONTENT_TYPE_FULL_TEXT)

# A response smaller than this is never a real article PDF; publishers and
# bot walls answer with a few hundred bytes of HTML or a stub. 1024 is the
# floor pdf_cache has always enforced — this constant unifies the checks on
# that value rather than tightening it, so no PDF that used to be accepted
# starts being rejected.
MIN_PLAUSIBLE_PDF_BYTES = 1024
# Below this, an extraction produced page furniture rather than body text.
MIN_EXTRACTED_TEXT_CHARS = 200


def is_abstract_only(content_type: str | None) -> bool:
    """Whether `content_type` names an abstract-only result.

    A predicate rather than an equality test so callers stay correct if the
    vocabulary ever gains a second abstract-shaped tier.
    """
    return content_type == CONTENT_TYPE_ABSTRACT


def has_full_text(content_type: str | None) -> bool:
    """Whether `content_type` names a result carrying body text."""
    return content_type in FULL_TEXT_TIERS


@dataclass
class DownloadResult:
    """Result of a PDF download attempt."""

    success: bool
    content: bytes | None
    source: str  # e.g., "unpaywall", "wiley", "alternative"
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ContentResult:
    """Result of a content download attempt (text/XML)."""

    success: bool
    content: str | None
    content_type: str  # "pdf", "text", "xml"
    source: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PaperDiscovery:
    """Result of DOI source discovery via OpenAlex + Unpaywall."""

    doi: str
    pmcid: str | None = None
    arxiv_id: str | None = None
    oa_url: str | None = None
    # Every open-access PDF url discovery found, best first. ``oa_url``
    # stays the legacy single value so older callers keep working.
    oa_candidates: list[str] = field(default_factory=list)
    abstract: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    is_oa: bool = False
    work_type: str | None = None  # "article", "preprint", etc.
    unpaywall_pdf_url: str | None = None
    journal: str | None = None
    license: str | None = None  # OA license id, e.g. "cc-by"


@dataclass
class PaperContent:
    """Unified result from retrieve_paper_content().

    content_type values:
      - "structured": full text with sections + references (JATS XML, HTML)
      - "full_text": full text from PDF extraction (no structure)
      - "abstract": abstract only (no full text available)
      - "none": no content found

    attempts: ordered list of pipeline-step diagnostics, one per source
        actually tried. Each entry has at minimum a ``source`` label and
        a ``status`` ("miss" | "error" | "skip" | "hit"). Errors carry an
        ``error`` field. The caller can surface this in failure messages
        so an operator can tell whether the failure was config (API key
        missing) or content (genuinely not available).

    rate_limited_hosts: hosts that answered HTTP 429 while this paper was
        being retrieved. Non-empty means the outcome is a throttling
        artefact, not a property of the paper, and the DOI should be
        re-offered on a later pass.
    """

    success: bool
    doi: str
    content_type: str  # "structured" | "full_text" | "abstract" | "none"
    full_text: str | None = None
    sections: dict[str, str] | None = None
    references: list[dict] | None = None
    abstract: str | None = None
    content_source: str = "none"  # "pmc", "arxiv_html", "publisher_pdf", etc.
    metadata: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    # Hosts that answered 429 during this retrieval. A non-empty list
    # means the result is degraded by throttling and worth retrying.
    rate_limited_hosts: list[str] = field(default_factory=list)

    def record_attempt(
        self, source: str, status: str, *, error: str | None = None, **extra: Any,
    ) -> None:
        entry: dict[str, Any] = {"source": source, "status": status}
        if error:
            entry["error"] = error
        if extra:
            entry.update(extra)
        self.attempts.append(entry)


class PDFDownloader:
    """Generic PDF downloader with retry logic.

    Optional **cookie jar**: when ``cookies_path`` is set, the
    Netscape-format ``cookies.txt`` (exported from a browser logged into
    a library proxy / publisher) is attached to outgoing requests whose
    host matches ``cookie_domains``. This is the server-side equivalent
    of how the Zotero Connector browser extension grabs paywalled
    PDFs — the user does the actual SSO/proxy login in their browser
    and re-exports the cookie jar; Perspicacité just replays it.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        *,
        cookies_path: str | None = None,
        cookie_domains: list[str] | None = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cookies_path = cookies_path
        self.cookie_domains = list(cookie_domains or [])

    def _matches_cookie_domains(self, url: str) -> bool:
        """True when this URL's host matches the configured allowlist
        (or the allowlist is empty, meaning attach to everything)."""
        if not self.cookie_domains:
            return True
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        return any(d.lower() in host for d in self.cookie_domains)

    def _load_cookie_jar(self) -> Any:
        """Load Netscape-format cookies.txt. Returns an http.cookiejar
        compatible jar or None on failure / missing file.

        Also runs a freshness check against ``cookie_domains`` and logs
        a warning per stale domain — the most common cause of paywalled
        PDFs silently returning HTML is an expired institutional cookie.
        """
        if not self.cookies_path:
            return None
        try:
            from http.cookiejar import MozillaCookieJar
            from pathlib import Path

            from perspicacite.pipeline.download.cookies import (
                check_cookie_freshness_for_domains,
            )
            p = Path(self.cookies_path).expanduser()
            if not p.exists():
                logger.warning("pdf_cookies_path_missing", path=str(p))
                return None
            jar = MozillaCookieJar(str(p))
            jar.load(ignore_discard=True, ignore_expires=True)
            logger.info("pdf_cookies_loaded", path=str(p), count=len(jar))
            # Surface stale-domain warnings once at load time. We only
            # warn for domains that look broken — quiet on healthy ones.
            warnings = check_cookie_freshness_for_domains(
                jar, self.cookie_domains
            )
            for w in warnings:
                if w.status == "ok":
                    continue
                logger.warning(
                    "pdf_cookies_stale",
                    domain=w.domain,
                    status=w.status,
                    matched_hosts=w.matched_hosts,
                    advice=w.advice,
                )
            return jar
        except Exception as e:
            logger.warning("pdf_cookies_load_failed", error=str(e))
            return None

    async def download(
        self,
        url: str,
        http_client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes | None:
        """Download a PDF from `url`.

        Inputs: `url` to fetch, an optional caller-owned `http_client`
        (whose own transport policy then wins), optional extra `headers`.

        Returns the PDF bytes, or None when the response was an HTTP
        error, a bot-wall interstitial, an implausibly small body, or a
        non-PDF page. Raises ``RateLimited`` when the host kept
        throttling us, so a caller can tell throttling apart from a
        document that is genuinely unavailable.
        """
        # Imported here to keep this module's header untouched; both are
        # leaf modules of this package, so there is no import cycle.
        from perspicacite.pipeline.download.interstitial import interstitial_marker
        from perspicacite.pipeline.download.rate_limit import RateLimited, polite_get

        # Build a client. When the caller supplied one, respect it
        # (cookies from this jar can be patched into the request);
        # otherwise build one carrying the configured cookie jar.
        cookie_jar = None
        if http_client is None and self.cookies_path:
            cookie_jar = self._load_cookie_jar()
        if http_client is None:
            client_kwargs: dict[str, Any] = {
                "timeout": self.timeout,
                "follow_redirects": True,
                # max_retries is the connect-level budget; retries on an
                # HTTP status belong to the host limiter in polite_get.
                "transport": httpx.AsyncHTTPTransport(retries=self.max_retries),
            }
            if cookie_jar is not None and self._matches_cookie_domains(url):
                client_kwargs["cookies"] = cookie_jar
            client = httpx.AsyncClient(**client_kwargs)
        else:
            client = http_client
        should_close = http_client is None
        # Browser-like UA prevents NCBI PMC / Europe PMC from serving
        # HTML landing pages instead of actual PDFs.
        merged = {
            "User-Agent": "Mozilla/5.0 (compatible; Perspicacite/2.0)",
            **(headers or {}),
        }

        try:
            logger.info("pdf_download_start", url=url)

            response = await polite_get(
                client, url, headers=merged, follow_redirects=True,
            )
            response.raise_for_status()

            # A bot wall answers 200 with a challenge page; storing it
            # would look exactly like a successful download.
            marker = interstitial_marker(
                str(response.url),
                response.headers.get("content-type", ""),
                response.content,
            )
            if marker:
                logger.warning("pdf_download_interstitial", url=url, marker=marker)
                return None

            size_bytes = len(response.content)
            if size_bytes < MIN_PLAUSIBLE_PDF_BYTES:
                logger.warning(
                    "pdf_download_too_small",
                    url=url,
                    size_bytes=size_bytes,
                    minimum_bytes=MIN_PLAUSIBLE_PDF_BYTES,
                )
                return None

            # A wrong content-type on a .pdf url is common, so the magic
            # bytes are the safety net rather than the header alone.
            content_type = response.headers.get("content-type", "").lower()
            declared_pdf = "pdf" in content_type or url.lower().endswith(".pdf")
            if declared_pdf or response.content.startswith(b"%PDF"):
                logger.info("pdf_download_success", url=url, size_bytes=size_bytes)
                return response.content

            # HTML back from a domain we hold cookies for: the cookie has
            # very likely expired. Distinct event so the user sees the
            # right fix instead of just "PDF not found".
            from perspicacite.pipeline.download.cookies import looks_like_paywall_html
            cookie_gated = bool(
                self.cookies_path
                and self.cookie_domains
                and self._matches_cookie_domains(url)
                and looks_like_paywall_html(response.content)
            )
            if cookie_gated:
                logger.warning(
                    "pdf_cookie_likely_expired",
                    url=url,
                    content_type=content_type,
                    advice=(
                        "Publisher returned HTML on a cookie-gated "
                        "domain. Re-run `perspicacite "
                        "import-browser-cookies` to refresh."
                    ),
                )
                return None
            logger.warning("pdf_download_not_pdf", url=url, content_type=content_type)
            return None

        except RateLimited as e:
            # Re-raised: a throttled host is not evidence of absence.
            logger.warning(
                "pdf_download_rate_limited",
                url=url,
                host=e.host,
                attempts=e.attempts,
            )
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                "pdf_download_http_error",
                url=url,
                status=e.response.status_code,
            )
            return None
        except Exception as e:
            logger.error("pdf_download_error", url=url, error=str(e))
            return None
        finally:
            if should_close:
                await client.aclose()


if __name__ == "__main__":
    assert is_abstract_only(CONTENT_TYPE_ABSTRACT)
    assert not is_abstract_only(CONTENT_TYPE_FULL_TEXT)
    assert has_full_text(CONTENT_TYPE_STRUCTURED)
    assert not has_full_text(CONTENT_TYPE_NONE)
    assert PaperDiscovery(doi="10.0/x").oa_candidates == []
    assert PaperContent(
        success=False, doi="10.0/x", content_type=CONTENT_TYPE_NONE
    ).rate_limited_hosts == []
    # download() delegates its retry budget; no client is built here.
    SMOKE_RETRIES = 2  # any value != the constructor default
    assert PDFDownloader(max_retries=SMOKE_RETRIES).max_retries == SMOKE_RETRIES
    print("base.py smoke checks passed")
