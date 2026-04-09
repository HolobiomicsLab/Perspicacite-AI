"""SciLEx adapter for Perspicacité v2.

This module provides an adapter to use SciLEx as a literature search provider.
Uses SciLEx's collection, then manually aggregates and converts to Papers.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper, PaperSource

logger = get_logger("perspicacite.search.scilex")


class SciLExAdapter:
    """Adapter to use SciLEx as a search provider."""

    def __init__(self, api_config: dict[str, Any] | None = None):
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
        article_type: str | None = None,
    ) -> list[Paper]:
        """Search academic databases via SciLEx."""
        if not self._scilex_available:
            logger.warning("scilex_not_available_fallback")
            return []

        return await asyncio.to_thread(
            self._scilex_search_sync,
            query,
            max_results,
            year_min,
            year_max,
            apis,
            article_type,
        )

    def _scilex_search_sync(
        self,
        query: str,
        max_results: int,
        year_min: int | None,
        year_max: int | None,
        apis: list[str] | None,
        article_type: str | None = None,
    ) -> list[Paper]:
        """Synchronous SciLEx search."""
        from scilex.crawlers.collector_collection import CollectCollection
        from scilex.crawlers.aggregate import (
            OpenAlextoZoteroFormat,
            SemanticScholartoZoteroFormat,
            ArxivtoZoteroFormat,
            PubMedtoZoteroFormat,
            IEEEtoZoteroFormat,
            SpringertoZoteroFormat,
            DBLPtoZoteroFormat,
            deduplicate,
        )

        # API name mapping
        api_name_map = {
            "semantic_scholar": "SemanticScholar",
            "openalex": "OpenAlex",
            "pubmed": "PubMed",
            "arxiv": "Arxiv",
            "ieee": "IEEE",
            "springer": "Springer",
            "dblp": "DBLP",
        }

        # Format converters
        api_converters = {
            "OpenAlex": OpenAlextoZoteroFormat,
            "SemanticScholar": SemanticScholartoZoteroFormat,
            "Arxiv": ArxivtoZoteroFormat,
            "PubMed": PubMedtoZoteroFormat,
            "IEEE": IEEEtoZoteroFormat,
            "Springer": SpringertoZoteroFormat,
            "DBLP": DBLPtoZoteroFormat,
        }

        # Default APIs
        if apis is None:
            apis = ["semantic_scholar", "openalex", "pubmed"]

        # Use year range if provided; default to last 3 years
        if year_min and year_max:
            years = [year_min, year_max]
        elif year_max:
            years = [year_max, year_max]
        elif year_min:
            years = [year_min, year_min]
        else:
            current_year = datetime.now().year
            years = [current_year - 3, current_year]
        capitalized_apis = [api_name_map.get(a, a) for a in apis]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure SciLEx
            main_config = {
                "collect_name": "perspicacite_search",
                "output_dir": tmpdir,
                "keywords": [[query], []],
                "apis": capitalized_apis,
                "years": years,
                "fields": [],
                "collect_type": "references",
                "zotero": False,
                "zotero_id": "",
                "zotero_key": "",
            }

            api_config = self._build_api_config(apis)

            try:
                # Phase 1: Collect
                logger.info("scilex_collection_start", query=query, apis=apis)
                collector = CollectCollection(main_config, api_config)

                queries_by_api = collector.queryCompositor()

                for api_name, queries in queries_by_api.items():
                    api_collect_list = []
                    for idx, query_dict in enumerate(queries):
                        query_dict["id_collect"] = idx
                        query_dict["total_art"] = 0
                        query_dict["last_page"] = 0
                        query_dict["coll_art"] = 0
                        query_dict["state"] = 0
                        query_dict["max_articles_per_query"] = max_results * 2
                        api_collect_list.append({"query": query_dict, "api": api_name})

                    try:
                        logger.info(f"Collecting from {api_name}...")
                        collector.run_job_collects(api_collect_list)
                        logger.info(f"Successfully collected from {api_name}")
                    except Exception as api_error:
                        logger.warning(f"API {api_name} failed: {api_error}")
                        continue

                # Phase 2: Manual aggregation
                logger.info("scilex_aggregation_start")
                repo_path = Path(tmpdir) / "perspicacite_search"

                all_records = []

                # Walk through API directories
                for api_dir in repo_path.iterdir():
                    if not api_dir.is_dir():
                        continue
                    if api_dir.name in ["config_used.yml", "citation_cache.db"]:
                        continue

                    api_name = api_dir.name
                    converter = api_converters.get(api_name)

                    if not converter:
                        logger.warning(f"No converter for API: {api_name}")
                        continue

                    # Process each query directory
                    for query_dir in api_dir.iterdir():
                        if not query_dir.is_dir():
                            continue

                        # Process each result file
                        for result_file in query_dir.iterdir():
                            if not result_file.is_file():
                                continue

                            try:
                                with open(result_file) as f:
                                    data = json.load(f)

                                # Handle different response formats
                                if isinstance(data, dict) and "results" in data:
                                    papers_list = data["results"]
                                elif isinstance(data, list):
                                    papers_list = data
                                else:
                                    continue

                                # Convert each paper to Zotero format
                                for paper_data in papers_list:
                                    try:
                                        zotero_record = converter(paper_data)
                                        zotero_record["archive"] = api_name
                                        all_records.append(zotero_record)
                                    except Exception as conv_error:
                                        logger.debug(f"Conversion error: {conv_error}")
                                        continue

                            except Exception as e:
                                logger.debug(f"Failed to read {result_file}: {e}")
                                continue

                logger.info(f"scilex_collected_records", count=len(all_records))

                if not all_records:
                    logger.warning("scilex_no_results", query=query)
                    return []

                # Create DataFrame
                df = pd.DataFrame(all_records)

                # Deduplicate
                try:
                    df_deduped = deduplicate(df)
                    logger.info(f"scilex_deduplicated", before=len(df), after=len(df_deduped))
                except Exception as e:
                    logger.debug(f"Deduplication error: {e}")
                    df_deduped = df

                # Convert to Paper models
                papers = self._map_dataframe_to_papers(df_deduped)

                # Post-filter by article_type
                if article_type:
                    papers = self._filter_by_article_type(papers, article_type)

                logger.info("scilex_collection_complete", query=query, found=len(papers))
                return papers[:max_results]

            except Exception as e:
                logger.error("scilex_collection_error", error=str(e))
                raise

    def _filter_by_article_type(self, papers: list[Paper], article_type: str) -> list[Paper]:
        """Post-filter papers by article type.

        Uses metadata.type (from SciLex itemType) when available, falls back to
        title/journal keyword heuristics.
        """
        type_lower = article_type.lower().strip()

        # Map user-friendly type names to Zotero itemType values
        type_map = {
            "review": ["journalarticle"],  # Reviews are journalArticle + keyword match
            "article": ["journalarticle"],
            "conference": ["conferencepaper", "proceedings-article"],
            "preprint": ["preprint", "manuscript"],
        }
        match_types = type_map.get(type_lower, [type_lower])

        # Keyword heuristics for reviews
        review_kw = {"review", "reviews", "survey", "surveys", "systematic review"}

        filtered = []
        for p in papers:
            meta_type = (p.metadata.get("type") or "").lower()

            # For "review": match journalArticle + review keyword
            if type_lower == "review":
                if any(t in meta_type for t in match_types):
                    title_lower = (p.title or "").lower()
                    journal_lower = (p.journal or "").lower()
                    if any(kw in title_lower for kw in review_kw) or any(
                        kw in journal_lower for kw in review_kw
                    ):
                        filtered.append(p)
                continue

            # For other types: match by itemType
            if any(t in meta_type for t in match_types):
                filtered.append(p)

        return filtered

    def _build_api_config(self, apis: list[str]) -> dict[str, Any]:
        """Build API configuration dict for SciLEx.
        
        Note: SciLEx uses capitalized API names (e.g., "SemanticScholar") as keys,
        not uppercase env var names.
        """
        # Map lowercase API names to SciLEx capitalized names
        api_name_map = {
            "semantic_scholar": "SemanticScholar",
            "openalex": "OpenAlex",
            "pubmed": "PubMed",
            "arxiv": "Arxiv",
            "ieee": "IEEE",
            "springer": "Springer",
            "dblp": "DBLP",
        }
        
        config = {}
        for api in apis:
            api_upper = api.upper()
            scilex_api_name = api_name_map.get(api, api)
            
            # Try multiple env var naming conventions
            env_key_prefixed = f"SCILEX_{api_upper}_API_KEY"
            env_key_direct = f"{api_upper}_API_KEY"
            
            api_key = (
                os.environ.get(env_key_prefixed) or
                os.environ.get(env_key_direct) or
                self.api_config.get(api, {}).get("api_key")
            )
            
            if api_key:
                config[scilex_api_name] = {"api_key": api_key}
                # Log only first/last 4 chars of key for security
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
                logger.info(f"API key configured for {scilex_api_name}", key_mask=masked_key, key_length=len(api_key))
            else:
                config[scilex_api_name] = {}
                logger.debug(f"No API key found for {scilex_api_name}", checked_vars=[env_key_prefixed, env_key_direct])
        return config

    def _map_dataframe_to_papers(self, df: pd.DataFrame) -> list[Paper]:
        """Map SciLEx aggregated DataFrame to Paper models."""
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
        """Map a single SciLEx record (Zotero format) to Paper model."""
        from perspicacite.models.papers import Author

        def safe_str(value, default=""):
            if value is None:
                return default
            if isinstance(value, float) and pd.isna(value):
                return default
            return str(value) if value else default

        # Extract fields from Zotero format
        title = safe_str(row.get("title"), "Untitled")
        abstract = safe_str(row.get("abstractNote")) or safe_str(row.get("abstract"))

        # Parse authors (semicolon-separated in Zotero format: "Last1, First1; Last2, First2")
        authors = []
        author_field = safe_str(row.get("authors")) or safe_str(row.get("author"))
        if author_field:
            # Split by semicolon (handle both "; " and ";" separators)
            for author_str in author_field.replace("; ", ";").split(";"):
                author_str = author_str.strip()
                if not author_str:
                    continue
                # Zotero format is "Last, First" 
                if "," in author_str:
                    parts = author_str.split(",", 1)
                    family = parts[0].strip()
                    given = parts[1].strip() if len(parts) > 1 else None
                    name = f"{given} {family}" if given else family
                else:
                    # Fallback: try to parse "First Last"
                    parts = author_str.rsplit(" ", 1)
                    if len(parts) == 2:
                        given = parts[0]
                        family = parts[1]
                        name = author_str
                    else:
                        name = author_str
                        given = None
                        family = None
                authors.append(Author(name=name, given=given, family=family))

        # Parse year from date
        year = None
        date_field = safe_str(row.get("date"))
        if date_field:
            try:
                year = int(date_field.split("-")[0])
            except (ValueError, IndexError):
                pass

        # Extract other fields
        doi = safe_str(row.get("DOI"))
        pmid = safe_str(row.get("pmid"))
        url = safe_str(row.get("url"))

        # Generate ID
        if doi:
            paper_id = f"doi:{doi}"
        elif pmid:
            paper_id = f"pmid:{pmid}"
        elif url:
            paper_id = url
        else:
            import hashlib
            paper_id = f"generated:{hashlib.md5(title.encode()).hexdigest()[:12]}"

        # Get citation count
        citation_count = None
        try:
            cit = row.get("citation_count", row.get("nb_citation"))
            if cit and not pd.isna(cit):
                citation_count = int(cit)
        except (ValueError, TypeError):
            pass

        return Paper(
            id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract or None,
            year=year,
            journal=safe_str(row.get("publicationTitle")) or safe_str(row.get("journal")),
            doi=doi or None,
            pmid=pmid or None,
            url=url or None,
            pdf_url=safe_str(row.get("pdf_url")) or None,
            citation_count=citation_count,
            source=PaperSource.SCILEX,
            keywords=[t.strip() for t in safe_str(row.get("tags")).split(", ")] if row.get("tags") else [],
            metadata={
                "archive": safe_str(row.get("archive"), "unknown"),
                "type": safe_str(row.get("itemType")) or safe_str(row.get("type")),
            },
        )


# For compatibility
SciLExSearchProvider = SciLExAdapter
