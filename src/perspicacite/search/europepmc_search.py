"""Europe PMC REST API search provider."""

from __future__ import annotations

import contextlib
from typing import Any, ClassVar

import httpx

from perspicacite.logging import get_logger
from perspicacite.models.papers import Author, Paper, PaperSource

logger = get_logger("perspicacite.search.europepmc")

_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class EuropePMCSearchProvider:
    """Searches Europe PMC via their free REST API."""

    name: ClassVar[str] = "europepmc"
    description: ClassVar[str] = "Europe PMC biomedical literature search (free REST API)"
    domains: ClassVar[list[str]] = ["biomedical"]
    tier: ClassVar[str] = "reliable"
    retry: ClassVar[int] = 0

    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        **_: Any,
    ) -> list[Paper]:
        q = query
        if year_min or year_max:
            y_min = year_min or 1800
            y_max = year_max or 2100
            q = f"({q}) AND (FIRST_PDATE:[{y_min}-01-01 TO {y_max}-12-31])"

        params: dict[str, str | int] = {
            "query": q,
            "resultType": "core",
            "pageSize": min(max_results, 100),
            "format": "json",
            "cursorMark": "*",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                resp = await client.get(_BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("europepmc_search_error", error=str(exc), query=query[:80])
                return []

        papers: list[Paper] = []
        for item in (data.get("resultList") or {}).get("result") or []:
            doi = item.get("doi") or None
            pmid = item.get("pmid") or None
            paper_id = doi or (f"pmid:{pmid}" if pmid else f"epmc:{item.get('id', 'unknown')}")

            authors: list[Author] = []
            for name in (item.get("authorString") or "").split(","):
                name = name.strip()
                if name:
                    authors.append(Author(name=name))

            year: int | None = None
            with contextlib.suppress(KeyError, ValueError, TypeError):
                year = int(item["pubYear"])

            papers.append(
                Paper(
                    id=paper_id,
                    title=item.get("title") or "Untitled",
                    authors=authors,
                    year=year,
                    doi=doi,
                    pmid=pmid,
                    abstract=item.get("abstractText"),
                    journal=item.get("journalTitle"),
                    source=PaperSource.EUROPE_PMC,
                    metadata={"epmc_id": item.get("id"), "is_oa": item.get("isOpenAccess") == "Y"},
                )
            )

        logger.info("europepmc_search", query=query[:80], results=len(papers))
        return papers
