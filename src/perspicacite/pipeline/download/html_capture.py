"""HTML fallback capture (Priority 3b in the content-acquisition plan).

When the PDF route fails for a paper, fall back to capturing the publisher
landing page as a self-contained HTML snapshot. Worse than a PDF, better
than nothing — unlocks the body for open-abstract pages, gives the KB
chunker something to ingest, and gives Zotero an attachment to surface.

Three quality tiers are reported back so callers can weight chunks
appropriately:

- ``full_text_html``: page contains the article body (PMC HTML, BMC,
  MDPI, OA pages with body in HTML).
- ``extended_abstract``: page has abstract + section headers + figure
  captions but body is paywalled.
- ``bibliographic_stub``: page has only title, authors, DOI, abstract.

The module is intentionally dependency-light — uses the same stdlib HTML
parser the rest of the codebase relies on. A future enhancement can
plug Trafilatura in behind a feature flag for higher-quality main-content
extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download.html_capture")


# Length thresholds (chars of extracted text body) for the three tiers.
# Tuned against a Mimosa-AI sample of 8 publisher landing pages where the
# body sizes ranged from ~600 chars (Nature paywalled stub) to ~80k chars
# (PMC full-text HTML).
_TIER_FULLTEXT_MIN_CHARS = 8_000
_TIER_ABSTRACT_MIN_CHARS = 1_500


@dataclass(frozen=True)
class HtmlCapture:
    """One captured landing-page snapshot.

    ``path`` is where the HTML file was written. ``tier`` reports the
    extraction quality (see module docstring). ``char_count`` is the
    extracted-text character count — used both by the tier classifier
    and by downstream chunkers.
    """

    path: Path
    tier: str
    char_count: int
    extracted_title: str | None = None
    extracted_text: str | None = None


class _TextExtractor(HTMLParser):
    """Collect visible text, drop <script>/<style>/<nav>/<header>/<footer> blocks."""

    _SKIP: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "nav", "header", "footer", "aside", "form"}
    )

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._depth_skip = 0
        self.title: str | None = None
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP:
            self._depth_skip += 1
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP and self._depth_skip > 0:
            self._depth_skip -= 1
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._depth_skip > 0:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title and self.title is None:
            self.title = text
        self._chunks.append(text)

    def get_text(self) -> str:
        return " ".join("".join(c + " " for c in self._chunks).split()).strip()


def _classify_tier(char_count: int) -> str:
    if char_count >= _TIER_FULLTEXT_MIN_CHARS:
        return "full_text_html"
    if char_count >= _TIER_ABSTRACT_MIN_CHARS:
        return "extended_abstract"
    return "bibliographic_stub"


def _build_snapshot_html(
    *,
    title: str,
    doi: str,
    landing_url: str,
    body_html: str,
    extracted_text: str,
    tier: str,
) -> str:
    """Wrap the raw page HTML in a small banner so the snapshot is
    self-identifying when opened from Zotero's attachment list."""
    safe_title = re.sub(r"[<>&]", "", title or doi or "Captured page")
    return (
        "<!DOCTYPE html>\n<html><head>"
        "<meta charset='utf-8'>"
        f"<title>{safe_title}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:60em;margin:2em auto;padding:0 1em;color:#222;}"
        ".banner{background:#fff3cd;border:1px solid #ffeeba;padding:1em;margin-bottom:1em;border-radius:4px;}"
        ".banner small{color:#664d03;}"
        "</style>"
        "</head><body>"
        "<div class='banner'>"
        f"<strong>Captured landing page</strong> — {safe_title}<br>"
        f"<small>tier: <code>{tier}</code> · chars: {len(extracted_text):,} · "
        f"source: <a href='{landing_url}'>{landing_url}</a> · "
        f"doi: <code>{doi}</code></small>"
        "</div>"
        f"{body_html}"
        "</body></html>"
    )


def _doi_slug(doi: str, fallback: str = "") -> str:
    """Filesystem-safe slug derived from the DOI, with a URL fallback.

    For URL-route items (no DOI), every entry would otherwise collide
    on ``no_doi.html`` — a problem when ``push_to_zotero`` processes
    multiple URL-only items in one call. Pass the landing URL as
    ``fallback`` and the slug becomes URL-derived instead.
    """
    if doi:
        return re.sub(r"[^a-zA-Z0-9.]+", "_", doi.lower())[:120]
    if fallback:
        return re.sub(r"[^a-zA-Z0-9.]+", "_", fallback.lower())[:120]
    return "no_doi"


def _build_stub_html(
    *, doi: str, title: str, abstract: str, landing_url: str, reason: str,
) -> tuple[str, int]:
    """Synthesize a bibliographic-stub HTML when the live landing-page
    fetch is blocked (Cloudflare 403, paywall, etc.). Returns the HTML
    body and the extracted-text char count for tier classification."""
    safe_title = re.sub(r"[<>&]", "", title or doi or "Bibliographic stub")
    body_html = (
        f"<h1>{safe_title}</h1>"
        f"<p><strong>DOI:</strong> <code>{doi}</code></p>"
        + (f"<p><strong>Landing URL:</strong> <a href='{landing_url}'>{landing_url}</a></p>"
           if landing_url else "")
        + (f"<section><h2>Abstract</h2><p>{abstract}</p></section>"
           if abstract else "<p><em>No abstract available from OpenAlex/Crossref.</em></p>")
        + f"<p><small>This is a synthesized stub — the live publisher page was "
          f"not reachable ({reason}). Source metadata: OpenAlex + Crossref via "
          f"Perspicacité unified discovery.</small></p>"
    )
    chars = len(safe_title) + len(abstract or "")
    return body_html, chars


async def capture_landing_html(
    *,
    doi: str,
    landing_url: str | None,
    abstract: str = "",
    title: str = "",
    http_client: httpx.AsyncClient,
    cache_dir: str | Path | None,
    timeout_s: float = 15.0,
) -> HtmlCapture | None:
    """Capture the publisher landing page as an HTML snapshot.

    Used as a Priority 3b fallback when the PDF route fails. The captured
    snapshot is written to ``<cache_dir>/html/<doi-slug>.html`` and
    classified into one of three tiers by extracted body length.

    When the live publisher page is unreachable (4xx, network error,
    non-HTML response) but the caller supplied at least a ``title`` or
    ``abstract`` from upstream discovery, a synthesized
    ``bibliographic_stub`` is written instead. This is the
    "worse than HTML, better than nothing" tier — gives the user a
    Zotero attachment with whatever metadata we could gather, even when
    Cloudflare blocks the live fetch.

    Returns ``None`` only when nothing usable is available — no URL, no
    title, and no abstract.
    """
    out_dir = (
        Path(cache_dir).expanduser() / "html"
        if cache_dir else Path.home() / ".cache" / "perspicacite" / "html"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    url = landing_url or (f"https://doi.org/{doi}" if doi else "")
    if not url and not (title or abstract):
        logger.debug("html_capture_no_url_and_no_meta", doi=doi)
        return None

    # Publisher landing pages (preprints.org, royalsocietypublishing.org,
    # nature.com, doi.org redirect targets) gate non-browser UAs with
    # HTTP 403. Send a realistic Chrome/Mac UA — matches what the rest
    # of the pipeline uses (see pipeline/download/supplementary.py).
    # Even with a browser UA + cookie jar, Cloudflare-protected publishers
    # (ACS, AAAS, RSC) return 403 to non-browser clients — we still fall
    # back to a synthesized stub from upstream metadata in that case.
    fetch_failure_reason: str | None = None
    body_html = ""
    if url:
        try:
            r = await http_client.get(
                url, timeout=timeout_s, follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    "Accept": (
                        "text/html,application/xhtml+xml,application/xml;q=0.9,"
                        "image/avif,image/webp,*/*;q=0.8"
                    ),
                },
            )
        except httpx.HTTPError as exc:
            logger.warning("html_capture_fetch_error", url=url, error=str(exc))
            fetch_failure_reason = f"network error: {exc.__class__.__name__}"
        else:
            if r.status_code >= 400:
                logger.info("html_capture_status_error", url=url, status=r.status_code)
                fetch_failure_reason = f"HTTP {r.status_code}"
            elif "html" not in r.headers.get("content-type", "").lower():
                logger.debug("html_capture_not_html", url=url,
                             content_type=r.headers.get("content-type"))
                fetch_failure_reason = (
                    f"non-HTML response ({r.headers.get('content-type','?')})"
                )
            elif not (r.text or "").strip():
                fetch_failure_reason = "empty body"
            else:
                body_html = r.text
    else:
        fetch_failure_reason = "no landing URL"

    # Live HTML capture path
    if not fetch_failure_reason:
        ext = _TextExtractor()
        try:
            ext.feed(body_html)
        except Exception as exc:
            logger.warning("html_capture_parse_error", url=url, error=str(exc))
            fetch_failure_reason = f"parse error: {exc}"
        else:
            text = ext.get_text()
            if not text:
                fetch_failure_reason = "no extractable text"

    # Stub fallback when the live fetch failed but we have metadata.
    if fetch_failure_reason:
        if not (title or abstract):
            logger.info(
                "html_capture_stub_skipped_no_meta",
                doi=doi, url=url, reason=fetch_failure_reason,
            )
            return None
        body_html, stub_chars = _build_stub_html(
            doi=doi, title=title, abstract=abstract,
            landing_url=url, reason=fetch_failure_reason,
        )
        text = (title or "") + " " + (abstract or "")
        ext_title = title or doi or url
        tier = "bibliographic_stub"
        extracted_title = ext_title
        logger.info(
            "html_capture_stub_built",
            doi=doi, reason=fetch_failure_reason, chars=stub_chars,
        )
    else:
        tier = _classify_tier(len(text))
        extracted_title = ext.title or title or doi or url

    # When the live page is a paywalled stub but the caller supplied an
    # abstract from OpenAlex/Crossref discovery, splice it in so the
    # KB chunker has *something* to retrieve. (The synthesized-stub path
    # already embeds the abstract in body_html.)
    augmented_body = body_html
    if (
        not fetch_failure_reason
        and tier == "bibliographic_stub"
        and abstract and abstract.strip() not in text
    ):
        augmented_body = body_html + (
            "<section data-source='openalex-abstract'>"
            "<h2>Abstract (from OpenAlex/Crossref discovery)</h2>"
            f"<p>{abstract}</p></section>"
        )

    snapshot = _build_snapshot_html(
        title=extracted_title,
        doi=doi,
        landing_url=url,
        body_html=augmented_body,
        extracted_text=text,
        tier=tier,
    )

    dest = out_dir / f"{_doi_slug(doi, fallback=landing_url or '')}.html"
    dest.write_text(snapshot, encoding="utf-8")
    logger.info(
        "html_capture_saved",
        doi=doi, tier=tier, chars=len(text), path=str(dest),
    )
    return HtmlCapture(
        path=dest,
        tier=tier,
        char_count=len(text),
        extracted_title=extracted_title,
        extracted_text=text,
    )
