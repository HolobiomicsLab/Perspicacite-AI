# src/perspicacite/integrations/zotero_license.py
"""License classification for Zotero items.

Resolution chain: Crossref → OpenAlex → Zotero tags → heuristic.
Results cached in-memory per DOI with 7-day TTL.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.integrations.zotero_license")

_PERMISSIVE_SPDX_PREFIXES = (
    "CC0",
    "CC-BY-4", "CC-BY-3", "CC-BY-2", "CC-BY-1",
    "CC-BY-SA",
    "MIT",
    "Apache-2.0",
    "BSD-",
    "ISC",
)
_CLOSED_SPDX_PATTERNS = ("CC-BY-NC", "CC-BY-ND")

_CC_URL_SLUG_RE = re.compile(
    r"creativecommons\.org/licenses/([a-z0-9\-]+)/([0-9.]+)/",
    re.I,
)
_CC_PD_RE = re.compile(r"creativecommons\.org/publicdomain/zero/", re.I)

_KNOWN_OPEN_TAGS = {"open-access", "open access", "cc-by", "cc0"}
_KNOWN_CLOSED_TAGS = {"closed", "paywalled", "restricted"}

_SEVEN_DAYS = 7 * 24 * 3600


@dataclass
class LicenseInfo:
    spdx: str | None
    classification: str   # "permissive" | "closed" | "unknown"
    policy: str           # "verbatim" | "reflavor"
    source: str           # "crossref" | "openalex" | "zotero_tag" | "heuristic" | "unknown"


def _url_to_spdx(url: str) -> str | None:
    """Best-effort conversion of a license URL to an SPDX identifier."""
    if _CC_PD_RE.search(url):
        return "CC0-1.0"
    m = _CC_URL_SLUG_RE.search(url)
    if m:
        slug = m.group(1).upper()
        ver = m.group(2)
        if slug == "BY":
            return f"CC-BY-{ver}"
        return f"CC-{slug}-{ver}"
    if "opensource.org/licenses/MIT" in url or url.lower().endswith("/mit"):
        return "MIT"
    if "apache.org/licenses/LICENSE-2.0" in url:
        return "Apache-2.0"
    return None


class LicenseClassifier:
    """Classify a paper's license from DOI or Zotero item metadata."""

    def __init__(self) -> None:
        # {doi: (LicenseInfo, expire_timestamp)}
        self._cache: dict[str, tuple[LicenseInfo, float]] = {}

    # ------------------------------------------------------------------
    # Public sync helpers (pure, no I/O — for testing and tag resolution)
    # ------------------------------------------------------------------

    def classify_spdx(self, spdx: str | None) -> LicenseInfo:
        """Classify a known SPDX identifier string."""
        if not spdx:
            return LicenseInfo(spdx=None, classification="unknown", policy="reflavor", source="unknown")
        upper = spdx.upper()
        for pat in _CLOSED_SPDX_PATTERNS:
            if upper.startswith(pat):
                return LicenseInfo(spdx=spdx, classification="closed", policy="reflavor", source="spdx")
        for prefix in _PERMISSIVE_SPDX_PREFIXES:
            if upper.startswith(prefix.upper()):
                return LicenseInfo(spdx=spdx, classification="permissive", policy="verbatim", source="spdx")
        return LicenseInfo(spdx=spdx, classification="unknown", policy="reflavor", source="unknown")

    def classify_url(self, url: str) -> LicenseInfo:
        """Classify from a license URL (e.g. from Crossref)."""
        spdx = _url_to_spdx(url)
        info = self.classify_spdx(spdx)
        info.source = "url"
        return info

    def classify_zotero_tags(self, zotero_item: dict) -> LicenseInfo | None:
        """Return LicenseInfo if any known license tag is present; else None."""
        tags = [
            (t.get("tag") or "").lower().strip()
            for t in ((zotero_item.get("data") or {}).get("tags") or [])
        ]
        for tag in tags:
            # Check restrictive CC-BY variants BEFORE the generic cc-by check
            if any(tag.startswith(r) for r in ("cc-by-nc", "cc-by-nd")):
                return LicenseInfo(spdx=None, classification="closed", policy="reflavor", source="zotero_tag")
            if tag in _KNOWN_OPEN_TAGS or tag.startswith("cc-by"):
                return LicenseInfo(spdx=None, classification="permissive", policy="verbatim", source="zotero_tag")
            if tag in _KNOWN_CLOSED_TAGS:
                return LicenseInfo(spdx=None, classification="closed", policy="reflavor", source="zotero_tag")
        return None

    def heuristic(self, *, is_oa: bool) -> LicenseInfo:
        """Low-confidence guess when no explicit license is known."""
        if is_oa:
            return LicenseInfo(spdx=None, classification="permissive", policy="verbatim", source="heuristic")
        return LicenseInfo(spdx=None, classification="unknown", policy="reflavor", source="heuristic")

    def get_cached(self, doi: str) -> LicenseInfo | None:
        entry = self._cache.get(doi)
        if entry is None:
            return None
        info, expires = entry
        if time.time() > expires:
            del self._cache[doi]
            return None
        return info

    def _store(self, doi: str, info: LicenseInfo) -> LicenseInfo:
        self._cache[doi] = (info, time.time() + _SEVEN_DAYS)
        return info

    # ------------------------------------------------------------------
    # Async resolution (I/O)
    # ------------------------------------------------------------------

    async def classify(
        self,
        doi: str,
        *,
        zotero_item: dict | None = None,
        http_client: httpx.AsyncClient | None = None,
        unpaywall_email: str = "perspicacite@example.com",
    ) -> LicenseInfo:
        """Full resolution chain. Returns cached result when available."""
        cached = self.get_cached(doi)
        if cached is not None:
            return cached

        client = http_client or httpx.AsyncClient()
        close_client = http_client is None

        try:
            # 1. Crossref
            info = await self._from_crossref(doi, client)
            if info is not None:
                return self._store(doi, info)

            # 2. OpenAlex
            info, is_oa = await self._from_openalex(doi, client)
            if info is not None:
                return self._store(doi, info)

            # 3. Zotero tags
            if zotero_item is not None:
                info = self.classify_zotero_tags(zotero_item)
                if info is not None:
                    return self._store(doi, info)

            # 4. Heuristic
            return self._store(doi, self.heuristic(is_oa=is_oa or False))

        except Exception:
            # Never crash the caller over a license lookup failure.
            return LicenseInfo(spdx=None, classification="unknown", policy="reflavor", source="unknown")
        finally:
            if close_client:
                await client.aclose()

    async def _from_crossref(
        self, doi: str, client: httpx.AsyncClient
    ) -> LicenseInfo | None:
        try:
            r = await client.get(
                f"https://api.crossref.org/works/{doi}",
                timeout=8.0,
                headers={"User-Agent": "Perspicacite/1.0 (mailto:perspicacite@example.com)"},
            )
            if r.status_code != 200:
                return None
            licenses = (r.json().get("message") or {}).get("license") or []
            for lic in licenses:
                url = lic.get("URL") or ""
                if url:
                    info = self.classify_url(url)
                    info.source = "crossref"
                    if info.classification != "unknown":
                        return info
        except Exception as exc:
            logger.debug("crossref_license_lookup_failed", doi=doi, error=str(exc))
        return None

    async def _from_openalex(
        self, doi: str, client: httpx.AsyncClient
    ) -> tuple[LicenseInfo | None, bool]:
        """Returns (LicenseInfo | None, is_oa: bool)."""
        try:
            r = await client.get(
                f"https://api.openalex.org/works/https://doi.org/{doi}",
                timeout=8.0,
            )
            if r.status_code != 200:
                return None, False
            body = r.json()
            oa = body.get("open_access") or {}
            is_oa = bool(oa.get("is_oa"))
            best_loc = body.get("best_oa_location") or {}
            lic = best_loc.get("license") or ""
            if lic:
                info = self.classify_spdx(lic)
                info.source = "openalex"
                if info.classification != "unknown":
                    return info, is_oa
            return None, is_oa
        except Exception as exc:
            logger.debug("openalex_license_lookup_failed", doi=doi, error=str(exc))
            return None, False
