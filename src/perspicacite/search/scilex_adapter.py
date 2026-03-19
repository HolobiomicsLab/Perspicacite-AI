"""SciLEx adapter for Perspicacité v2.

This module provides an adapter to use SciLEx as a literature search provider.
SciLEx is installed as a dependency and imported directly.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper, PaperSource

logger = get_logger("perspicacite.search.scilex")


class SciLExAdapter:
    """
    Adapter to use SciLEx as a search provider.

    Integration approach: Library import with adapter pattern.
    SciLEx is installed as a dependency (pip install -e ../SciLEx).
    This adapter:
    1. Prepares SciLEx config files in a temp directory
    2. Calls SciLEx collection functions programmatically
    3. Maps SciLEx dict/DataFrame output to Perspicacite Paper models
    4. Wraps sync calls in asyncio.to_thread()

    Fallback: If SciLEx is not installed, gracefully degrades to built-in search.
    """

    def __init__(self, api_config: dict[str, Any] | None = None):
        """
        Initialize SciLEx adapter.

        Args:
            api_config: Optional API configuration dict with API keys
        """
        self._scilex_available = self._check_scilex()
        self.api_config = api_config or {}

    def _check_scilex(self) -> bool:
        """Check if SciLEx is installed."""
        try:
            import scilex
            return True
        except ImportError:
            logger.warning("scilex_not_available")
            return False

    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        apis: list[str] | None = None,
    ) -> list[Paper]:
        """
        Search academic databases via SciLEx.

        Args:
            query: Search query
            max_results: Maximum results per API
            year_min: Minimum publication year
            year_max: Maximum publication year
            apis: List of APIs to use (default: semantic_scholar, openalex, pubmed)

        Returns:
            List of Paper models
        """
        if not self._scilex_available:
            logger.warning("scilex_not_available_fallback")
            return await self._fallback_search(query, max_results)

        return await asyncio.to_thread(
            self._scilex_search_sync,
            query,
            max_results,
            year_min,
            year_max,
            apis,
        )

    def _scilex_search_sync(
        self,
        query: str,
        max_results: int,
        year_min: int | None,
        year_max: int | None,
        apis: list[str] | None,
    ) -> list[Paper]:
        """
        Synchronous SciLEx collection.

        This runs in a thread pool to not block the event loop.
        """
        from scilex.crawlers.collector_collection import CollectCollection
        try:
            from scilex.crawlers.aggregate import deduplicate, convertToZoteroFormat
        except ImportError:
            # Fallback if functions not available
            from scilex.crawlers.aggregate import deduplicate
            convertToZoteroFormat = None  # Will handle below

        # Default APIs if not specified
        if apis is None:
            apis = ["semantic_scholar", "openalex", "pubmed"]

        # Create temp config
        with tempfile.TemporaryDirectory() as tmpdir:
            main_config = {
                "collect_name": "perspicacite_search",
                "output_dir": tmpdir,
                "keywords": [[query], []],  # SciLEx format: list of keyword groups
                "apis": apis,
                "years": [year_min or 1900, year_max or 2100],
                "fields": [],
                "collect_type": "references",
                "zotero": False,
                "zotero_id": "",
                "zotero_key": "",
            }

            # Build API config
            api_config = self._build_api_config(apis)

            try:
                # Initialize collection
                collector = CollectCollection(main_config, api_config)

                # Run collection
                logger.info("scilex_collection_start", query=query, apis=apis)
                collector.run_collection()

                # Load and process results
                import pandas as pd

                results_file = Path(tmpdir) / "perspicacite_search" / "all_data.csv"
                if not results_file.exists():
                    logger.warning("scilex_no_results", query=query)
                    return []

                df = pd.read_csv(results_file)

                # Deduplicate
                df_deduped = deduplicate(df)

                # Convert to Zotero format for consistent output (if available)
                if convertToZoteroFormat:
                    df_zotero = convertToZoteroFormat(df_deduped)
                else:
                    df_zotero = df_deduped

                # Convert to Paper models
                papers = self._map_scilex_to_papers(df_zotero)

                logger.info(
                    "scilex_collection_complete",
                    query=query,
                    found=len(papers),
                )

                return papers[:max_results]

            except Exception as e:
                logger.error("scilex_collection_error", error=str(e))
                # Re-raise so caller can handle fallback
                raise

    def _build_api_config(self, apis: list[str]) -> dict[str, Any]:
        """Build API configuration dict for SciLEx."""
        config = {}

        for api in apis:
            api_upper = api.upper()
            env_key = f"SCILEX_{api_upper}_API_KEY"
            api_key = os.environ.get(env_key) or self.api_config.get(api, {}).get("api_key")

            if api_key:
                config[api_upper] = {"api_key": api_key}
            else:
                config[api_upper] = {}

        return config

    def _map_scilex_to_papers(self, df: Any) -> list[Paper]:
        """
        Map SciLEx DataFrame to Perspicacite Paper models.

        Args:
            df: SciLEx DataFrame (in Zotero format)

        Returns:
            List of Paper models
        """
        papers = []

        for _, row in df.iterrows():
            try:
                paper = self._map_single_record(row)
                papers.append(paper)
            except Exception as e:
                logger.warning("scilex_map_error", error=str(e))
                continue

        return papers

    def _map_single_record(self, row: Any) -> Paper:
        """Map a single SciLEx record to Paper model."""
        # Extract authors
        authors = []
        author_field = row.get("author", "")
        if isinstance(author_field, str) and author_field:
            for author_str in author_field.split("; "):
                author_str = author_str.strip()
                if not author_str:
                    continue

                # Try to parse "Last, First" format
                if "," in author_str:
                    parts = author_str.split(",", 1)
                    family = parts[0].strip()
                    given = parts[1].strip() if len(parts) > 1 else None
                    name = f"{given} {family}" if given else family
                else:
                    # "First Last" format
                    parts = author_str.rsplit(" ", 1)
                    if len(parts) == 2:
                        family = parts[1]
                        given = parts[0]
                        name = author_str
                    else:
                        name = author_str
                        family = None
                        given = None

                from perspicacite.models.papers import Author

                authors.append(Author(name=name, given=given, family=family))

        # Extract year
        year = None
        date_field = row.get("date", "")
        if isinstance(date_field, str) and date_field:
            try:
                year = int(date_field.split("-")[0])
            except (ValueError, IndexError):
                pass

        # Generate ID from DOI or PMID
        doi = row.get("DOI", "")
        pmid = row.get("pmid", "")
        url = row.get("url", "")

        if doi:
            paper_id = f"doi:{doi}"
        elif pmid:
            paper_id = f"pmid:{pmid}"
        elif url:
            paper_id = url
        else:
            import hashlib

            title = row.get("title", "")
            paper_id = f"generated:{hashlib.md5(title.encode()).hexdigest()[:12]}"

        # Get abstract
        abstract = row.get("abstractNote", "") or row.get("abstract", "")

        # Get PDF URL
        pdf_url = row.get("file", "")
        if isinstance(pdf_url, str) and pdf_url.startswith("/"):
            # Local file path - skip
            pdf_url = None

        # Get citation count
        citation_count = None
        if "citation_count" in row:
            try:
                citation_count = int(row["citation_count"])
            except (ValueError, TypeError):
                pass

        return Paper(
            id=paper_id,
            title=row.get("title", "Untitled"),
            authors=authors,
            abstract=abstract or None,
            year=year,
            journal=row.get("publicationTitle") or row.get("journal"),
            doi=doi or None,
            pmid=str(pmid) if pmid else None,
            url=url or None,
            pdf_url=pdf_url,
            citation_count=citation_count,
            source=PaperSource.SCILEX,
            keywords=row.get("tags", "").split(", ") if row.get("tags") else [],
            metadata={
                "scilex_api": row.get("api_source", "unknown"),
                "scilex_date_added": row.get("dateAdded", ""),
            },
        )

    async def _fallback_search(self, query: str, max_results: int) -> list[Paper]:
        """Fallback search when SciLEx is not available."""
        logger.info("fallback_search", query=query)
        # Return empty list - caller should use alternative search
        return []


class SciLExSearchProvider:
    """
    High-level search provider interface for SciLEx.

    This provides a simpler interface for the RAG engine to use.
    """

    def __init__(self, adapter: SciLExAdapter | None = None):
        self.adapter = adapter or SciLExAdapter()

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[Paper]:
        """Search for papers."""
        return await self.adapter.search(query, max_results=max_results, **kwargs)

    @property
    def name(self) -> str:
        return "scilex"

    @property
    def description(self) -> str:
        return "Search academic databases via SciLEx (10+ APIs)"
