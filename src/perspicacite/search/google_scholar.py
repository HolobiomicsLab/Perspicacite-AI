"""Google Scholar search fallback.

Simple fallback when SciLEx is not available.
Note: This uses a basic approach. For production, consider using:
- scholarly library (unofficial, may break)
- SerpAPI (paid, reliable)
- Direct scraping (fragile, against ToS)
"""

from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper, PaperSource, Author

logger = get_logger("perspicacite.search.google_scholar")


class GoogleScholarSearch:
    """
    Simple Google Scholar search.

    This is a basic implementation. For production use,
    consider using a proper API or service.
    """

    async def search(
        self,
        query: str,
        max_results: int = 10,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> list[Paper]:
        """
        Search Google Scholar.

        Args:
            query: Search query
            max_results: Maximum results
            year_min: Minimum year
            year_max: Maximum year

        Returns:
            List of papers (placeholder - returns empty)
        """
        logger.warning(
            "google_scholar_not_implemented",
            query=query,
            message="Google Scholar search requires additional setup (scholarly library or SerpAPI)",
        )
        return []


class SearchAggregator:
    """
    Aggregates multiple search providers.

    Tries SciLEx first, falls back to Google Scholar.
    """

    def __init__(self):
        self.scilex = None
        self.fallback = GoogleScholarSearch()

        # Try to import SciLEx
        try:
            from perspicacite.search.scilex_adapter import SciLExAdapter
            self.scilex = SciLExAdapter()
            logger.info("search_aggregator_scilex_available")
        except ImportError:
            logger.warning("search_aggregator_scilex_unavailable")

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[Paper]:
        """
        Search using best available provider.

        Args:
            query: Search query
            max_results: Maximum results
            **kwargs: Additional search parameters

        Returns:
            List of papers
        """
        papers = []

        # Try SciLEx first
        if self.scilex:
            try:
                papers = await self.scilex.search(query, max_results=max_results, **kwargs)
                if papers:
                    logger.info("search_scilex_success", results=len(papers))
                    return papers
            except Exception as e:
                logger.error("search_scilex_error", error=str(e))

        # Fallback to Google Scholar
        try:
            papers = await self.fallback.search(query, max_results=max_results, **kwargs)
            if papers:
                logger.info("search_fallback_success", results=len(papers))
        except Exception as e:
            logger.error("search_fallback_error", error=str(e))

        return papers
