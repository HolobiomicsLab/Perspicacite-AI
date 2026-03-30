"""Europe PMC — OA full-text PDF via PMCID (no API key).

Search: https://www.ebi.ac.uk/europepmc/webservices/rest/
We query by DOI; if a ``pmcid`` is returned we try NCBI / Europe PMC PDF routes.

PDFs apply to the open-access subset; see Europe PMC usage terms.
"""

from __future__ import annotations

import httpx

from perspicacite.logging import get_logger
from .base import PDFDownloader

logger = get_logger("perspicacite.pipeline.download.europepmc")


async def download_pdf_from_europepmc(
    doi: str,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Download OA PDF when the article is in PMC and openly accessible."""
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not clean:
        return None

    client = http_client or httpx.AsyncClient(timeout=45.0, follow_redirects=True)
    should_close = http_client is None

    try:
        search_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": f"DOI:{clean}",
            "format": "json",
            "resultType": "core",
            "pageSize": "1",
        }
        logger.info("europepmc_search", doi=clean)
        r = await client.get(search_url, params=params)
        r.raise_for_status()
        data = r.json()
        results = data.get("resultList", {}).get("result") or []
        if not results:
            logger.info("europepmc_no_hit", doi=clean)
            return None

        hit = results[0] if isinstance(results[0], dict) else None
        if not hit:
            return None

        pmcid = hit.get("pmcid")
        if not pmcid or not str(pmcid).upper().startswith("PMC"):
            logger.info("europepmc_no_pmcid", doi=clean)
            return None

        pmc_full = str(pmcid).strip()

        pdf_urls = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_full}/pdf/",
            f"https://europepmc.org/articles/{pmc_full}?pdf=render",
        ]

        downloader = PDFDownloader()
        for pdf_url in pdf_urls:
            logger.info("europepmc_try_pdf", doi=clean, url=pdf_url)
            data_bytes = await downloader.download(pdf_url, http_client=client)
            if data_bytes and data_bytes[:4] == b"%PDF" and len(data_bytes) > 1000:
                logger.info("europepmc_success", doi=clean, size_bytes=len(data_bytes))
                return data_bytes

        logger.warning("europepmc_pdf_failed", doi=clean, pmcid=pmc_full)
        return None

    except Exception as e:
        logger.error("europepmc_error", doi=clean, error=str(e))
        return None
    finally:
        if should_close:
            await client.aclose()
