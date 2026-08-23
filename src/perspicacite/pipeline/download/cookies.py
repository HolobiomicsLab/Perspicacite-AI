"""Cookie-jar freshness inspection.

Institutional / paywall PDF fetch relies on a Netscape ``cookies.txt``
exported from the user's logged-in browser. Those cookies expire — some
in days, some in months — and the failure mode (cookie silently expired,
publisher serves the paywall HTML in place of the PDF) is confusing.

This module gives us:

- :func:`scan_cookie_freshness` — per-host summary of expired / expiring /
  session-only / fresh cookies, computed from the ``http.cookiejar``
  ``expires`` field. Pure inspection, no network.
- :func:`check_cookie_freshness_for_domains` — given the user's
  ``pdf_download.cookie_domains`` allowlist, produce a list of warnings
  ready to print (CLI) or log (downloader startup).

The downloader calls the second helper once at jar-load time and
surfaces stale entries via :mod:`perspicacite.logging`. The CLI
``check-cookies`` subcommand prints the same report for the user.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

# 7 days = soon enough that we should nag the user to re-import
EXPIRY_SOON_SECONDS = 7 * 24 * 3600


@dataclass
class HostCookieSummary:
    """One row of the freshness report.

    ``fresh_max_expiry`` is None when this host only has expired or
    session cookies; otherwise it's the latest still-valid Unix
    timestamp. We use it both to rank "is this host healthy" and to
    show a human-readable expiry date.
    """

    host: str
    total: int = 0
    expired: int = 0
    expiring_soon: int = 0
    session: int = 0
    fresh_max_expiry: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.fresh_max_expiry is not None:
            d["fresh_max_expiry_iso"] = datetime.fromtimestamp(
                self.fresh_max_expiry, tz=UTC
            ).isoformat()
        return d


def scan_cookie_freshness(jar: Iterable[Any]) -> dict[str, HostCookieSummary]:
    """Categorize every cookie in ``jar`` by host and expiry state.

    ``jar`` is any iterable of cookie objects with ``.domain`` and
    ``.expires`` (None for session cookies); :class:`http.cookiejar.Cookie`
    fits, which is what :class:`http.cookiejar.MozillaCookieJar` yields.
    """
    now = int(time.time())
    soon = now + EXPIRY_SOON_SECONDS
    by_host: dict[str, HostCookieSummary] = {}
    for c in jar:
        host = (getattr(c, "domain", "") or "").lstrip(".").lower()
        if not host:
            continue
        entry = by_host.setdefault(host, HostCookieSummary(host=host))
        entry.total += 1
        exp = getattr(c, "expires", None)
        if exp is None or getattr(c, "discard", False):
            entry.session += 1
            continue
        try:
            exp_i = int(exp)
        except (TypeError, ValueError):
            entry.session += 1
            continue
        if exp_i < now:
            entry.expired += 1
        elif exp_i < soon:
            entry.expiring_soon += 1
            if entry.fresh_max_expiry is None or exp_i > entry.fresh_max_expiry:
                entry.fresh_max_expiry = exp_i
        else:
            if entry.fresh_max_expiry is None or exp_i > entry.fresh_max_expiry:
                entry.fresh_max_expiry = exp_i
    return by_host


@dataclass
class CookieDomainWarning:
    """One entry in the freshness report for a configured domain.

    ``status`` is one of: ``"no_cookies"``, ``"all_expired"``,
    ``"expiring_soon"``, ``"ok"``.
    """

    domain: str
    status: str
    matched_hosts: int = 0
    soonest_expiry: int | None = None
    advice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.soonest_expiry is not None:
            d["soonest_expiry_iso"] = datetime.fromtimestamp(
                self.soonest_expiry, tz=UTC
            ).isoformat()
        return d


def check_cookie_freshness_for_domains(
    jar: Iterable[Any],
    cookie_domains: list[str],
) -> list[CookieDomainWarning]:
    """For each ``cookie_domains`` substring, classify its overall health.

    Returns one :class:`CookieDomainWarning` per configured domain.
    Callers typically filter to ``status != "ok"`` for the warning
    surface and show all entries in the ``check-cookies`` report.
    """
    if not cookie_domains:
        return []
    summary = scan_cookie_freshness(jar)
    results: list[CookieDomainWarning] = []
    now = int(time.time())
    soon = now + EXPIRY_SOON_SECONDS
    for d in cookie_domains:
        d_lower = d.lower()
        matching = [s for h, s in summary.items() if d_lower in h]
        if not matching:
            results.append(
                CookieDomainWarning(
                    domain=d,
                    status="no_cookies",
                    matched_hosts=0,
                    advice=(
                        "No cookies captured for this domain. The browser "
                        "may not be logged in to it, or the host substring "
                        "doesn't match. Re-run `perspicacite "
                        "import-browser-cookies`."
                    ),
                )
            )
            continue
        fresh_expiries = [
            s.fresh_max_expiry for s in matching if s.fresh_max_expiry is not None
        ]
        if not fresh_expiries:
            results.append(
                CookieDomainWarning(
                    domain=d,
                    status="all_expired",
                    matched_hosts=len(matching),
                    advice=(
                        "All cookies for this domain are expired or "
                        "session-only. Re-run `perspicacite "
                        "import-browser-cookies` after logging in again."
                    ),
                )
            )
            continue
        soonest = min(fresh_expiries)
        if soonest < soon:
            results.append(
                CookieDomainWarning(
                    domain=d,
                    status="expiring_soon",
                    matched_hosts=len(matching),
                    soonest_expiry=soonest,
                    advice=(
                        "Cookies expire within 7 days. Re-run "
                        "`perspicacite import-browser-cookies` soon."
                    ),
                )
            )
        else:
            results.append(
                CookieDomainWarning(
                    domain=d,
                    status="ok",
                    matched_hosts=len(matching),
                    soonest_expiry=soonest,
                )
            )
    return results


def build_cookie_jar(cookies_path: str) -> Any:
    """Load a Netscape ``cookies.txt`` and return the jar, or ``None`` on failure.

    Centralizes the load+log so :class:`PDFDownloader`, the top-level
    :func:`retrieve_paper_content`, and any caller that builds its own
    ``httpx.AsyncClient`` all attach cookies the same way.
    """
    from http.cookiejar import MozillaCookieJar
    from pathlib import Path

    from perspicacite.logging import get_logger as _gl
    _logger = _gl("perspicacite.pipeline.download.cookies")
    try:
        p = Path(cookies_path).expanduser()
        if not p.exists():
            _logger.warning("pdf_cookies_path_missing", path=str(p))
            return None
        jar = MozillaCookieJar(str(p))
        jar.load(ignore_discard=True, ignore_expires=True)
        _logger.info("pdf_cookies_loaded", path=str(p), count=len(jar))
        return jar
    except Exception as e:
        _logger.warning("pdf_cookies_load_failed", error=str(e))
        return None


def build_authenticated_client(
    *,
    cookies_path: str | None,
    timeout: float = 120.0,
) -> Any:
    """Return an ``httpx.AsyncClient`` with the cookie jar attached when
    ``cookies_path`` is set. Callers that pass their own client into
    :func:`retrieve_paper_content` should construct it via this helper
    so paywalled-publisher requests inherit the session cookies.
    """
    import httpx
    kwargs: dict[str, Any] = {"timeout": timeout, "follow_redirects": True}
    if cookies_path:
        jar = build_cookie_jar(cookies_path)
        if jar is not None:
            kwargs["cookies"] = jar
    return httpx.AsyncClient(**kwargs)


def looks_like_paywall_html(content: bytes, *, head: int = 2048) -> bool:
    """Cheap heuristic: did the publisher return HTML instead of a PDF?

    Used in the downloader to distinguish "PDF body" from "HTML landing
    page" — the latter is the canonical symptom of an expired or missing
    cookie on a paywalled article. We only need to be right enough to
    tell the user "your cookies probably need refreshing".
    """
    if not content:
        return False
    head_b = content[:head].lower()
    return b"<html" in head_b or b"<!doctype html" in head_b


# ---------------------------------------------------------------------------
# Live validation
# ---------------------------------------------------------------------------
# An unexpired cookie is not a working cookie. A session can be revoked
# server-side, or the institution's proxy can drop the entitlement, while
# the jar still reports months of remaining life. The only way to know is
# to ask a publisher for something it refuses anonymously.


@dataclass
class CookieProbeResult:
    """Outcome of one live fetch against a publisher.

    ``status`` is one of: ``"authenticated"``, ``"paywalled"``,
    ``"no_cookies"``, ``"blocked"``, ``"error"``.
    """

    domain: str
    status: str
    http_status: int | None = None
    content_type: str | None = None
    bytes_received: int = 0
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def jar_has_live_cookie(jar: Iterable[Any], domain: str) -> bool:
    """True when the jar holds at least one unexpired cookie for ``domain``."""
    now = int(time.time())
    for cookie in jar:
        host = (getattr(cookie, "domain", "") or "").lower()
        if domain.lower() not in host:
            continue
        expires = getattr(cookie, "expires", None)
        if expires is None or int(expires) > now:
            return True
    return False


def classify_probe_response(
    status_code: int, content_type: str | None, body: bytes,
) -> tuple[str, str | None]:
    """Map a publisher's response to a probe status and a reason.

    PDF bytes mean the session was accepted. HTML means the publisher
    served its paywall or login page instead, which is what a dead
    session looks like from the outside.
    """
    if status_code in (401, 403):
        return "blocked", f"HTTP {status_code}"
    if status_code >= 400:
        return "error", f"HTTP {status_code}"
    if body.startswith(b"%PDF"):
        return "authenticated", None
    if looks_like_paywall_html(body) or "html" in (content_type or "").lower():
        return "paywalled", "publisher returned HTML, not a PDF"
    return "error", f"unexpected content-type {content_type!r}"


async def _probe_one(client: Any, domain: str, url: str) -> CookieProbeResult:
    """One authenticated fetch, classified. Never raises."""
    try:
        resp = await client.get(url)
    except Exception as exc:
        return CookieProbeResult(domain=domain, status="error", detail=str(exc)[:200])
    body = resp.content or b""
    ctype = resp.headers.get("content-type")
    status, detail = classify_probe_response(resp.status_code, ctype, body)
    return CookieProbeResult(
        domain=domain, status=status, http_status=resp.status_code,
        content_type=ctype, bytes_received=len(body), detail=detail,
    )


async def probe_cookie_urls(
    *,
    cookies_path: str | None,
    probe_urls: dict[str, str],
    timeout: float = 30.0,
) -> list[CookieProbeResult]:
    """Fetch one known **paywalled** article PDF per domain, with the jar.

    ``probe_urls`` maps a cookie domain to a direct PDF URL for an article
    that publisher refuses anonymously. Pointing an entry at an open-access
    article makes that probe pass whatever the jar contains, so the report
    would state the opposite of the truth — pick closed articles.
    """
    if not probe_urls:
        return []
    jar = build_cookie_jar(cookies_path) if cookies_path else None
    results: list[CookieProbeResult] = []
    async with build_authenticated_client(cookies_path=cookies_path, timeout=timeout) as client:
        for domain, url in sorted(probe_urls.items()):
            if jar is None or not jar_has_live_cookie(jar, domain):
                results.append(CookieProbeResult(
                    domain=domain, status="no_cookies",
                    detail="no unexpired cookie for this domain",
                ))
                continue
            results.append(await _probe_one(client, domain, url))
    return results
