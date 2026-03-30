"""OpenAlex — resolve OA PDF URLs by DOI (no API key).

OpenAlex documents polite use of ``mailto=`` on requests:
https://docs.openalex.org/how-to-use/rate-limits-and-authentication

We fetch ``/works/doi:{doi}`` and try ``pdf_url`` fields from locations, then
``best_oa_location``, ``primary_location``, then ``open_access.oa_url`` (may be
a landing page; we only accept responses that look like PDFs).
"""

from __future__ import annotations

import os

import httpx

from perspicacite.logging import get_logger
from .base import PDFDownloader

logger = get_logger("perspicacite.pipeline.download.openalex_oa")


def _mailto() -> str | None:
    return (
        os.getenv("OPENALEX_MAILTO")
        or os.getenv("UNPAYWALL_EMAIL")
        or None
    )


def _collect_pdf_candidate_urls(work: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(u: str | None) -> None:
        if not u or not u.startswith("http"):
            return
        if u in seen:
            return
        seen.add(u)
        out.append(u)

    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            add(loc.get("pdf_url"))

    for key in ("best_oa_location", "primary_location"):
        loc = work.get(key)
        if isinstance(loc, dict):
            add(loc.get("pdf_url"))

    oa = work.get("open_access") or {}
    if oa.get("is_oa") and oa.get("oa_url"):
        add(oa["oa_url"])

    return out


async def download_pdf_from_openalex_oa(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
    mailto: str | None = None,
) -> bytes | None:
    """If OpenAlex lists an OA PDF (or printable) URL for this DOI, download it."""
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean:
        return None

    client = http_client or httpx.AsyncClient(timeout=45.0, follow_redirects=True)
    should_close = http_client is None
    mail = (mailto or _mailto() or "").strip()

    try:
        params = {}
        if mail:
            params["mailto"] = mail

        api_url = f"https://api.openalex.org/works/doi:{clean}"
        logger.info("openalex_oa_lookup", doi=clean, url=api_url)

        response = await client.get(api_url, params=params or None)
        if response.status_code == 404:
            logger.info("openalex_oa_not_found", doi=clean)
            return None
        response.raise_for_status()
        work = response.json()

        if not (work.get("open_access") or {}).get("is_oa"):
            logger.info("openalex_oa_not_oa", doi=clean)
            return None

        candidates = _collect_pdf_candidate_urls(work)
        if not candidates:
            logger.info("openalex_oa_no_pdf_url", doi=clean)
            return None

        downloader = PDFDownloader()
        for pdf_url in candidates:
            logger.info("openalex_oa_try_download", doi=clean, url=pdf_url[:120])
            data = await downloader.download(pdf_url, http_client=client)
            if data and data[:4] == b"%PDF" and len(data) > 1000:
                logger.info("openalex_oa_success", doi=clean, size_bytes=len(data))
                return data

        logger.warning("openalex_oa_no_valid_pdf", doi=clean, tried=len(candidates))
        return None

    except httpx.HTTPStatusError as e:
        logger.warning(
            "openalex_oa_http_error",
            doi=clean,
            status=e.response.status_code,
        )
        return None
    except Exception as e:
        logger.error("openalex_oa_error", doi=clean, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()
