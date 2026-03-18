"""DOI resolution utilities."""

import httpx

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper, PaperSource, Author

logger = get_logger("perspicacite.search.doi")


async def resolve_doi(doi: str, http_client: httpx.AsyncClient | None = None) -> Paper | None:
    """
    Resolve DOI to paper metadata via CrossRef API.

    Args:
        doi: DOI to resolve
        http_client: Optional HTTP client

    Returns:
        Paper model or None if resolution failed
    """
    client = http_client or httpx.AsyncClient()
    should_close = http_client is None

    try:
        url = f"https://api.crossref.org/works/{doi}"
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()

        data = response.json()["message"]

        # Extract authors
        authors = []
        for author_data in data.get("author", []):
            given = author_data.get("given", "")
            family = author_data.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(Author(name=name, given=given, family=family))

        # Extract year
        year = None
        published = data.get("published-print") or data.get("published-online")
        if published and "date-parts" in published:
            try:
                year = published["date-parts"][0][0]
            except (IndexError, TypeError):
                pass

        return Paper(
            id=f"doi:{doi}",
            title=data.get("title", ["Untitled"])[0],
            authors=authors,
            abstract=None,  # CrossRef doesn't provide abstracts
            year=year,
            journal=data.get("container-title", [None])[0],
            doi=doi,
            url=data.get("URL"),
            citation_count=data.get("is-referenced-by-count"),
            source=PaperSource.WEB_SEARCH,
        )

    except Exception as e:
        logger.error("doi_resolution_error", doi=doi, error=str(e))
        return None

    finally:
        if should_close:
            await client.aclose()


async def resolve_dois_batch(
    dois: list[str],
    http_client: httpx.AsyncClient | None = None,
) -> list[Paper]:
    """
    Resolve multiple DOIs in batch.

    Args:
        dois: List of DOIs
        http_client: Optional HTTP client

    Returns:
        List of successfully resolved papers
    """
    import asyncio

    client = http_client or httpx.AsyncClient()

    try:
        tasks = [resolve_doi(doi, client) for doi in dois]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        papers = []
        for result in results:
            if isinstance(result, Paper):
                papers.append(result)
            elif isinstance(result, Exception):
                logger.warning("doi_batch_resolution_error", error=str(result))

        return papers

    finally:
        if http_client is None:
            await client.aclose()
