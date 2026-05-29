"""SciLEx adapter for Perspicacité v2.

This module provides an adapter to use SciLEx as a literature search provider.
Uses SciLEx's collection, then manually aggregates and converts to Papers.
"""

import asyncio
import contextlib
import json
import logging as _stdlib_logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper, PaperSource

logger = get_logger("perspicacite.search.scilex")


# Authoritative list of database/provider names callers can request. The
# names mirror the SciLEx dispatch table inside ``_scilex_search_sync`` plus
# the additional non-SciLEx providers wired into the domain aggregator
# (europepmc, ads, pubchem, inspire, google_scholar). The MCP layer filters
# user-supplied ``databases`` against this set before forwarding, so an
# unrecognised name is silently dropped rather than reaching providers.
KNOWN_DATABASES: frozenset[str] = frozenset({
    "arxiv",
    "crossref",
    "pubmed",
    "semantic_scholar",
    "openalex",
    "europepmc",
    "ads",
    "pubchem",
    "inspire",
    "dblp",
    "dblp_sparql",
    "google_scholar",
})


# F-33: cache the per-key validation result so we only hit SS once per
# adapter init (and not once per query).
_SS_KEY_CACHE: dict[str, bool] = {}


def _ss_key_is_valid(api_key: str) -> bool:
    """Return True if the given Semantic Scholar API key still works.

    Hits the cheapest SS endpoint with a tiny query. Treats network errors
    as "assume valid" so a transient outage doesn't drop a good key.
    On 401/403 returns False so the caller can fall through to unauth mode.
    Result is cached per-key for the process lifetime.
    """
    if not api_key:
        return False
    cached = _SS_KEY_CACHE.get(api_key)
    if cached is not None:
        return cached
    try:
        import httpx
        r = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": "ping", "limit": 1, "fields": "title"},
            headers={"x-api-key": api_key},
            timeout=5.0,
        )
        valid = r.status_code not in (401, 403)
        _SS_KEY_CACHE[api_key] = valid
        return valid
    except Exception as exc:
        logger.debug("ss_key_validation_error_assume_valid", error=str(exc))
        return True


class _QuotaLogCapture(_stdlib_logging.Handler):
    """Captures SciLEx's stdlib-logger PubMed quota warnings.

    SciLEx logs ``"PubMed API: Only N requests remaining in current period!"``
    via the root logger; we attach this handler for the duration of a
    SciLEx call, scan emitted messages for the quota pattern, and
    surface the remaining-count as a structured warning to the caller.
    """

    _QUOTA_RE = re.compile(r"Only (\d+) requests remaining")

    def __init__(self) -> None:
        super().__init__(level=_stdlib_logging.WARNING)
        self.last_remaining: int | None = None
        self.provider = "pubmed"

    def emit(self, record: _stdlib_logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            m = self._QUOTA_RE.search(msg)
            if m:
                self.last_remaining = int(m.group(1))
        except Exception:
            pass


@dataclass
class SciLExSearchResult:
    """Structured search result for the SciLEx adapter.

    ``dropped_apis`` lists any APIs the caller asked for that SciLEx
    doesn't know about. Surfaced upstream as a ``warnings`` field on
    MCP responses so external agents can tell their pick was partially
    honoured.
    """

    papers: list = field(default_factory=list)
    dropped_apis: list[str] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)


class SciLExAdapter:
    """Adapter to use SciLEx as a search provider."""

    name = "scilex"
    description = (
        "SciLEx multi-database academic literature search "
        "(Semantic Scholar, OpenAlex, PubMed, arXiv, HAL, DBLP)"
    )
    domains: ClassVar[list[str]] = ["general", "biomedical", "cs"]
    # SciLEx is itself a multi-API fan-out (4+ databases × every year in the
    # span, with per-API rate-limit retries), so it legitimately needs far
    # more than the 20s "reliable" budget — the aggregator was killing it
    # mid-collection (~24s) and returning zero results. "flaky" gives it
    # 2.25× (≈45s).
    tier: str = "flaky"
    retry: int = 0

    def __init__(self, api_config: dict[str, Any] | None = None):
        self._scilex_available = self._check_scilex()
        self.api_config = api_config or {}
        # F-19 (audit 2026-05-16): per-database error visibility. Populated
        # by the most-recent search() call. ``{}`` means "no errors";
        # callers can check this to distinguish "the upstream API was
        # down" from "the query had no hits".
        self.last_errors_by_database: dict[str, str] = {}
        self._last_dropped_apis: list[str] = []
        self._last_quota_warning: dict | None = None

    @classmethod
    def from_config(cls, config: Any) -> "SciLExAdapter":
        """Build adapter wired up to API keys declared in ``config.yml``.

        Reads ``pdf_download.semantic_scholar_api_key`` (and the publisher
        keys) and shapes them into the per-API ``api_config`` dict that
        SciLEx expects. Env-var fallbacks still apply inside
        :meth:`_build_api_config`.
        """
        api_config: dict[str, dict[str, Any]] = {}
        pdf = getattr(config, "pdf_download", None) if config else None
        if pdf is not None:
            mapping = {
                "semantic_scholar": getattr(pdf, "semantic_scholar_api_key", None),
                "springer": getattr(pdf, "springer_api_key", None),
            }
            for api, key in mapping.items():
                if key:
                    api_config[api] = {"api_key": key}
        return cls(api_config=api_config)

    def _check_scilex(self) -> bool:
        """Check if SciLEx is installed."""
        import importlib.util

        if importlib.util.find_spec("scilex") is not None:
            return True
        logger.warning("scilex_not_available")
        return False

    @property
    def available(self) -> bool:
        """True if the SciLEx package is importable."""
        return self._scilex_available

    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        apis: list[str] | None = None,
        article_type: str | None = None,
    ) -> list[Paper]:
        """Public search entry point. Retries once with normalised title
        when a title-like query returns zero results.

        Returns an empty list when the optional SciLEx package isn't
        installed. Callers should check ``self.available`` first to
        distinguish "SciLEx missing" from "search returned zero hits".
        """
        out = await self._search_once(
            query=query,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            apis=apis,
            article_type=article_type,
        )
        if out:
            return out

        from perspicacite.search.title_normalize import is_titlelike, normalize_title
        if not is_titlelike(query):
            return out
        normalised = normalize_title(query)
        if normalised == query or len(normalised) < 4:
            return out

        logger.info(
            "scilex_normalize_retry",
            original=query,
            normalized=normalised,
        )
        retried = await self._search_once(
            query=normalised,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            apis=apis,
            article_type=article_type,
        )
        for p in retried:
            if p.metadata is None:
                p.metadata = {}
            p.metadata["search_normalized_from"] = query
        return retried

    async def _search_once(
        self,
        *,
        query: str,
        max_results: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        apis: list[str] | None = None,
        article_type: str | None = None,
    ) -> list[Paper]:
        """Internal: do one actual search pass without normalize-retry.

        Returns an empty list when the optional SciLEx package isn't
        installed.
        """
        if not self._scilex_available:
            logger.warning("scilex_not_available_fallback")
            return []

        # Reset per-call error trail (F-19).
        self.last_errors_by_database = {}
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
        from scilex.crawlers.aggregate import (
            ArxivtoZoteroFormat,
            DBLPtoZoteroFormat,
            IEEEtoZoteroFormat,
            OpenAlextoZoteroFormat,
            PubMedtoZoteroFormat,
            SemanticScholartoZoteroFormat,
            SpringertoZoteroFormat,
            deduplicate,
        )
        from scilex.crawlers.collector_collection import CollectCollection

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

        # Default APIs — PubMed first (no key needed, reliable). Semantic
        # Scholar is intentionally last because its public tier rate-limits
        # aggressively and frequently errors out; we still query it for
        # CS coverage, but it shouldn't lead the fan-out.
        if apis is None:
            apis = ["pubmed", "openalex", "semantic_scholar"]

        # Defense in depth: SciLEx only knows the APIs in ``api_name_map``.
        # Callers sometimes pass the full user-selected provider list (e.g.
        # ``google_scholar``, ``europepmc``, ``core``) which would surface as
        # KeyError('google_scholar') inside SciLEx. Filter them out — those
        # providers are handled by the domain_aggregator directly.
        _unknown = [a for a in apis if a not in api_name_map]
        if _unknown:
            logger.info(
                "scilex_filtered_non_scilex_apis",
                filtered=_unknown,
                kept=[a for a in apis if a in api_name_map],
            )
        # Stash on self so search_with_warnings can include this list
        # in its structured result. Cleared at the start of each call.
        self._last_dropped_apis = list(_unknown)
        apis = [a for a in apis if a in api_name_map]
        if not apis:
            # User explicitly selected only non-SciLEx databases — SciLEx has
            # nothing to do; return empty cleanly so callers can handle it.
            logger.info("scilex_no_supported_apis_selected", filtered=_unknown)
            return []

        # Build the list of years to query. SciLEx's queryCompositor treats
        # this as DISCRETE years (one query per element, via product()), NOT a
        # [min, max] range — so we must expand to EVERY year in the span.
        # Otherwise [2023, 2026] queries only 2023 and 2026, silently skipping
        # 2024-2025 (and collapsing to 2023 since the current year is sparse).
        #
        # Default span is a 25-year window ending at the current year. The
        # previous default (last ~3 years) silently excluded foundational
        # biomedical literature — e.g. asking for "GLP-1 RA mechanisms" never
        # surfaced Drucker 2018, LEADER 2016, or STEP 2021 because all three
        # fell outside the 3-year window. (Bug B-3 in the 2026-05-25 audit.)
        current_year = datetime.now().year
        if year_min and year_max:
            _lo, _hi = year_min, year_max
        elif year_max:
            # Bug B-4: previously collapsed to [year_max, year_max] — a single
            # year. Use a sensible window ending at year_max so callers don't
            # have to specify both bounds.
            _lo, _hi = year_max - 25, year_max
        elif year_min:
            # Bug B-4: previously collapsed to [year_min, year_min] — a single
            # year. Pair year_min with the current year so e.g. year_min=2010
            # actually searches 2010 through today.
            _lo, _hi = year_min, current_year
        else:
            # Bug B-3: default is now a 25-year window. Covers most published
            # biomedical literature; callers wanting a narrower scan should
            # pass year_min/year_max explicitly.
            _lo, _hi = current_year - 25, current_year
        if _lo > _hi:
            _lo, _hi = _hi, _lo
        # Cap the span (keep the most recent years) so a very wide range
        # doesn't explode into one query per year × per API.
        _MAX_YEARS = 30
        if _hi - _lo + 1 > _MAX_YEARS:
            _lo = _hi - _MAX_YEARS + 1
        years = list(range(_lo, _hi + 1))
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

            _quota = _QuotaLogCapture()
            _root_logger = _stdlib_logging.getLogger()
            _root_logger.addHandler(_quota)
            try:
                # Phase 1: Collect — backends run concurrently; a single
                # backend failure does not poison the aggregate.
                logger.info("scilex_collection_start", query=query, apis=apis)
                collector = CollectCollection(main_config, api_config)

                queries_by_api = collector.queryCompositor()

                successful_backends, failed_backends = self._collect_all_backends(
                    collector=collector,
                    queries_by_api=queries_by_api,
                    max_results=max_results,
                )

                if failed_backends:
                    logger.warning(
                        "scilex_collection_partial",
                        failed=failed_backends,
                        succeeded=successful_backends,
                        note="Results from successful backends will still be returned.",
                    )

                # Phase 2: Manual aggregation
                logger.info("scilex_aggregation_start")
                repo_path = Path(tmpdir) / "perspicacite_search"

                if not repo_path.exists():
                    logger.warning(
                        "scilex_no_results_dir",
                        note="No backend produced output; all backends failed.",
                    )
                    return []

                all_records: list[Any] = []

                # Walk through API directories
                for api_dir in repo_path.iterdir():
                    if not api_dir.is_dir():
                        continue
                    if api_dir.name in ["config_used.yml", "citation_cache.db"]:
                        continue

                    api_name = api_dir.name
                    converter = api_converters.get(api_name)

                    if not converter:
                        logger.warning("scilex_no_converter_for_api", api=api_name)
                        continue

                    backend_records_before = len(all_records)

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
                                        logger.debug(
                                            "scilex_conversion_error",
                                            api=api_name,
                                            error=str(conv_error),
                                        )
                                        continue

                            except Exception as e:
                                logger.debug(
                                    "scilex_result_file_read_error",
                                    file=str(result_file),
                                    error=str(e),
                                )
                                continue

                    backend_count = len(all_records) - backend_records_before
                    logger.info(
                        "scilex_backend_ok",
                        backend=api_name,
                        results=backend_count,
                    )


                logger.info("scilex_collected_records", count=len(all_records))

                if not all_records:
                    logger.warning("scilex_no_results", query=query)
                    return []

                # === Build a pre-dedupe archive map ===
                # We need to remember EVERY archive (e.g. "pubmed",
                # "openalex") that returned each paper, so the UI's
                # Retrieval panel can show "pubmed 30, openalex 80" instead
                # of attributing every paper to the single archive that
                # happened to survive SciLEx's DOI dedupe. Key by DOI when
                # present, else by normalized title.
                pre_archives_by_doi: dict[str, set[str]] = {}
                pre_archives_by_title: dict[str, set[str]] = {}
                for rec in all_records:
                    _arch = (rec.get("archive") or "unknown")
                    _arch_norm = str(_arch).lower().replace("-", "_")
                    if _arch_norm == "unknown":
                        continue
                    _doi = (rec.get("DOI") or "").strip().lower()
                    if _doi:
                        pre_archives_by_doi.setdefault(_doi, set()).add(_arch_norm)
                    else:
                        _title = (rec.get("title") or "").strip().lower()[:120]
                        if _title:
                            pre_archives_by_title.setdefault(_title, set()).add(_arch_norm)

                # Create DataFrame
                df = pd.DataFrame(all_records)

                # Deduplicate
                try:
                    df_deduped = deduplicate(df)
                    logger.info("scilex_deduplicated", before=len(df), after=len(df_deduped))
                except Exception as e:
                    logger.debug(f"Deduplication error: {e}")
                    df_deduped = df

                # Rank the deduped pool by SciLEx's composite relevance score
                # (keywords 45% + quality 25% + itemtype 20% + citations 10%),
                # globally across databases AND years. SciLEx collects
                # year-by-year and our manual aggregation kept collection
                # order, so a plain head(max_results) returned only the
                # earliest year's block. Sorting by relevance here lets the
                # most on-topic papers (any year) survive the later truncation.
                try:
                    from scilex.aggregate_collect import _apply_relevance_ranking
                    df_deduped = _apply_relevance_ranking(
                        df_deduped,
                        keyword_groups=main_config["keywords"],
                        top_n=None,  # truncation happens after post-filters below
                        has_citations=True,
                        config=main_config,
                    )
                except Exception as e:
                    logger.debug(f"Relevance ranking skipped: {e}")

                # Convert to Paper models
                papers = self._map_dataframe_to_papers(df_deduped)

                # F-4: second-pass dedup by normalized title to catch
                # records that survived SciLEx's DOI-only dedup because the
                # two sources returned different DOIs (or one had None) for
                # the same paper. Casing + punctuation + whitespace are
                # collapsed before comparison.
                papers = self._dedupe_by_normalized_title(papers)

                # === Re-attach multi-archive provenance ===
                # After SciLEx's internal dedupe we only know the surviving
                # record's archive. Cross-reference with our pre-dedupe map
                # to recover the full set of archives that returned each
                # paper.
                for _p in papers:
                    _archives: set[str] = set()
                    if _p.doi:
                        _archives = pre_archives_by_doi.get(_p.doi.lower().strip(), set())
                    if not _archives and _p.title:
                        _archives = pre_archives_by_title.get(
                            _p.title.strip().lower()[:120], set()
                        )
                    if _archives:
                        _p.discovery_sources = sorted(_archives)

                # Post-filter by article_type
                if article_type:
                    papers = self._filter_by_article_type(papers, article_type)

                logger.info("scilex_collection_complete", query=query, found=len(papers))
                return papers[:max_results]

            except Exception as e:
                logger.error("scilex_collection_error", error=str(e))
                raise

            finally:
                _root_logger.removeHandler(_quota)
                # Surface the quota warning so search_with_warnings can include it.
                if _quota.last_remaining is not None and _quota.last_remaining < 10:
                    self._last_quota_warning = {
                        "kind": "rate_limit_low",
                        "provider": "pubmed",
                        "remaining": _quota.last_remaining,
                        "advice": (
                            "Add NCBI_API_KEY to lift quota from 3 r/s to 10 r/s. "
                            "Without a key, SciLEx is throttled aggressively."
                        ),
                    }

    def _collect_all_backends(
        self,
        collector: Any,
        queries_by_api: dict[str, list[dict]],
        max_results: int,
    ) -> tuple[list[str], list[str]]:
        """Fan out per-backend collection concurrently via ThreadPoolExecutor.

        One backend's failure (network error, rate-limit, missing API key) is
        isolated by _collect_from_backend; this helper makes the fan-out
        parallel so wall time equals the slowest backend, not the sum of all
        backends.

        Args:
            collector: SciLEx CollectCollection instance.
            queries_by_api: Mapping from capitalized backend name to the list
                of raw query dicts as returned by collector.queryCompositor().
            max_results: User-requested result cap; used to compute
                max_articles_per_query (= max_results * 2) per existing pattern.

        Returns:
            (successful_backends, failed_backends) — mutable lists populated
            during concurrent dispatch. CPython's GIL makes list.append
            thread-safe.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        successful: list[str] = []
        failed: list[str] = []

        def build_collect_list(api_name: str, queries: list[dict]) -> list[dict]:
            api_collect_list = []
            for idx, query_dict in enumerate(queries):
                # Copy to avoid mutating the caller's dicts across threads.
                query_dict = dict(query_dict)
                query_dict["id_collect"] = idx
                query_dict["total_art"] = 0
                query_dict["last_page"] = 0
                query_dict["coll_art"] = 0
                query_dict["state"] = 0
                query_dict["max_articles_per_query"] = max_results * 2
                api_collect_list.append({"query": query_dict, "api": api_name})
            return api_collect_list

        api_lists = {
            api_name: build_collect_list(api_name, queries)
            for api_name, queries in queries_by_api.items()
        }

        if not api_lists:
            return successful, failed

        # Concurrent fan-out — each backend runs in its own thread.
        # _collect_from_backend swallows collector exceptions and appends to
        # the shared lists. CPython list.append is GIL-safe.
        with ThreadPoolExecutor(max_workers=len(api_lists)) as executor:
            futures = [
                executor.submit(
                    self._collect_from_backend,
                    collector=collector,
                    api_name=api_name,
                    api_collect_list=api_collect_list,
                    successful_backends=successful,
                    failed_backends=failed,
                )
                for api_name, api_collect_list in api_lists.items()
            ]
            for f in as_completed(futures):
                # _collect_from_backend catches collector errors; surface any
                # unexpected helper-level exceptions here.
                f.result()

        return successful, failed

    def _collect_from_backend(
        self,
        collector: Any,
        api_name: str,
        api_collect_list: list[dict],
        successful_backends: list[str],
        failed_backends: list[str],
    ) -> None:
        """Collect results from a single backend, logging loudly on failure.

        Failures are caught and logged at WARNING level so that a single
        broken backend (network error, missing API key, rate-limit) does not
        poison the aggregate — the remaining backends' results are still
        returned to the caller.

        Args:
            collector: SciLEx CollectCollection instance.
            api_name: Capitalized backend name ("SemanticScholar", "OpenAlex", …).
            api_collect_list: Query dicts prepared for this backend.
            successful_backends: Mutable list; this backend's name is appended
                on success so the caller can log a summary.
            failed_backends: Mutable list; this backend's name is appended on
                failure so the caller can log a summary and warn the user.
        """
        try:
            logger.info("scilex_backend_collect_start", backend=api_name)
            collector.run_job_collects(api_collect_list)
            logger.info("scilex_backend_collect_ok", backend=api_name)
            successful_backends.append(api_name)
        except Exception as api_error:
            logger.warning(
                "scilex_backend_failed",
                backend=api_name,
                error=str(api_error),
                note=(
                    "This backend's results will be absent from the aggregate. "
                    "Other backends are unaffected."
                ),
            )
            failed_backends.append(api_name)

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

    @staticmethod
    def _normalize_title_for_dedupe(title: str | None) -> str:
        """Collapse a title to its dedupe fingerprint.

        Lowercased, ASCII-folded, leading articles stripped, all
        non-alphanumeric characters dropped, then truncated to 120 chars.
        Two records that differ only in casing, punctuation, or a leading
        "The"/"A"/"An" produce the same fingerprint.
        """
        if not title:
            return ""
        import re as _re
        import unicodedata as _ud
        s = _ud.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii")
        s = s.lower().strip()
        # Strip a single leading article — surprisingly common variant.
        for art in ("the ", "a ", "an "):
            if s.startswith(art):
                s = s[len(art):]
                break
        s = _re.sub(r"[^a-z0-9]+", "", s)
        return s[:120]

    def _dedupe_by_normalized_title(self, papers: list[Paper]) -> list[Paper]:
        """Keep the first paper for each normalized-title key. Records with
        empty titles are passed through unchanged."""
        seen: set[str] = set()
        out: list[Paper] = []
        for p in papers:
            key = self._normalize_title_for_dedupe(p.title)
            if not key:
                out.append(p)
                continue
            if key in seen:
                logger.debug("scilex_title_dedup_drop", title=p.title[:80])
                continue
            seen.add(key)
            out.append(p)
        return out

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
                # F-33: Validate the key before handing it to SciLEx. If the
                # configured Semantic Scholar key is rejected (stale / revoked),
                # drop it so the adapter falls through to the unauthenticated
                # public tier (still rate-limited but usable). Same pattern as
                # the F-29 snowball fix.
                if api == "semantic_scholar" and not _ss_key_is_valid(api_key):
                    logger.warning(
                        "scilex_ss_key_rejected",
                        api=scilex_api_name,
                        action="dropping_key_falling_through_to_unauth",
                    )
                    config[scilex_api_name] = {"api_key": ""}
                    continue
                config[scilex_api_name] = {"api_key": api_key}
                # Log only first/last 4 chars of key for security
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
                logger.info(f"API key configured for {scilex_api_name}", key_mask=masked_key, key_length=len(api_key))
            else:
                # SciLEx ≥ 0.1 reads config[api]["api_key"] unconditionally
                # (it's wrapped in dict.get(...) on the inside but the outer
                # code does ``api_config[api]["api_key"]`` directly for some
                # collectors). Pass an empty string so it doesn't KeyError.
                # APIs that genuinely don't need a key (OpenAlex, Arxiv,
                # PubMed) still work with this.
                config[scilex_api_name] = {"api_key": ""}
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
            # SciLEx writes the literal string "NA" (see scilex/constants.py
            # MISSING_VALUE = "NA") whenever an upstream API returns no value.
            # Treat it as missing so downstream truthiness checks behave.
            if value is None:
                return default
            if isinstance(value, float) and pd.isna(value):
                return default
            if not value:
                return default
            s = str(value)
            if s.strip().lower() == "na":
                return default
            return s

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
            with contextlib.suppress(ValueError, IndexError):
                year = int(date_field.split("-")[0])

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
            metadata=(lambda _archive: {
                "archive": _archive,
                "type": safe_str(row.get("itemType")) or safe_str(row.get("type")),
                # Seed ``sources`` with the underlying SciLEx-internal API
                # (normalised: lowercase, "Semantic-Scholar" → "semantic_scholar").
                # The domain_aggregator will later append "scilex" to this list;
                # the UI strips "scilex" before rendering, so the user sees
                # the REAL upstream DB (pubmed / openalex / etc.) instead of
                # a useless "scilex" chip. When the same DOI is also returned
                # by a direct provider (europepmc, ads, ...), the merge in
                # domain_aggregator produces a true multi-DB ``sources_all``.
                "sources": (
                    [_archive.lower().replace("-", "_")]
                    if _archive and _archive != "unknown"
                    else []
                ),
            })(safe_str(row.get("archive"), "unknown")),
        )


    async def search_with_warnings(
        self,
        query: str,
        max_results: int = 50,
        year_min: int | None = None,
        year_max: int | None = None,
        article_type: str | None = None,
        apis: list[str] | None = None,
        databases: list[str] | None = None,
    ) -> SciLExSearchResult:
        """Same as ``search`` but returns a structured result with warnings.

        Use this from MCP / API entry points where the caller benefits
        from knowing their api list was partially dropped. The plain
        ``search()`` method is unchanged for legacy callers that only
        want ``list[Paper]``.

        ``databases`` is the user-facing alias for ``apis`` used by the
        MCP surface; when both are passed, ``databases`` wins. Either is
        passed through to ``search`` which then filters to providers
        SciLEx actually dispatches to.
        """
        if databases is not None:
            apis = databases
        self._last_dropped_apis = []
        self._last_quota_warning = None
        papers = await self.search(
            query=query,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            article_type=article_type,
            apis=apis,
        )
        extra_warnings: list[dict] = []
        if self._last_quota_warning is not None:
            extra_warnings.append(self._last_quota_warning)
        return SciLExSearchResult(
            papers=papers,
            dropped_apis=list(self._last_dropped_apis),
            warnings=extra_warnings,
        )


# For compatibility
SciLExSearchProvider = SciLExAdapter
