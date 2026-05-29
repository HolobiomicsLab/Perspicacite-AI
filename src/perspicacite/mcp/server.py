"""MCP server implementation for Perspicacité v2.

Exposes scientific literature tools via the Model Context Protocol
so external agent systems (e.g., Mimosa-AI) can discover and use them.

Tools exposed:
- search_literature: Search academic databases
- general_web_search: General-web search (DuckDuckGo + Playwright); docs sites, GitHub, vendor blogs
- get_paper_content: Fetch full text + structured sections
- get_paper_references: Extract cited references from a paper
- list_knowledge_bases: List all KBs
- search_knowledge_base: Semantic search within a KB
- create_knowledge_base: Create a new KB
- add_papers_to_kb: Add papers to a KB
- generate_report: Synthesize a research report from a KB
- screen_papers: Score candidate papers by relevance to a query
- add_dois_to_kb: Bulk-add papers to a KB from a list of DOIs
- push_to_zotero: Push DOIs to the configured Zotero library
- build_kbs_from_zotero: Build one KB per Zotero top-level collection
- build_kb_from_search: Search SciLEx, filter, fetch PDFs, ingest into a KB
- zotero_list_collections: List all Zotero collections with sub-collection tree
- zotero_get_collection_items: Get papers in a collection with license classification
- zotero_get_paper_resources: Get ordered file access options for a paper
- zotero_ingest_collection_to_kb: Ingest a Zotero collection into a KB
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.pipeline.asb.response import build_asb_response_metadata
from perspicacite.pipeline.asb.run_ingest import ingest_asb_run as ingest_asb_run_pipeline
from perspicacite.pipeline.github_kb import (
    IngestSummary,
)
from perspicacite.pipeline.github_kb import (
    ingest_github_repo as ingest_github_repo_pipeline,
)
from perspicacite.pipeline.github_kb import (
    ingest_skill_bundle as ingest_skill_bundle_pipeline,
)
from perspicacite.pipeline.asb.collection_ingest import (
    ingest_asb_skill_collection,
)
from perspicacite.pipeline.asb.edam_filter import edam_pre_filter
from perspicacite.rag.paper_metadata_codec import decode_paper_metadata_json

logger = get_logger("perspicacite.mcp.server")

try:
    from fastmcp import Context, FastMCP

    mcp = FastMCP("perspicacite")
except ImportError:
    mcp = None
    Context = Any  # type: ignore[misc, assignment]


# =============================================================================
# Shared State
# =============================================================================


class MCPState:
    """Shared state object set during server startup.

    Provides MCP tools with access to Perspicacité internals
    (session store, vector store, embedding provider, config, etc.)
    without coupling to the web app module.
    """

    def __init__(self) -> None:
        self.session_store: Any = None
        self.vector_store: Any = None
        self.embedding_provider: Any = None
        self.config: Any = None
        self.llm_client: Any = None
        self.pdf_parser: Any = None
        self.tool_registry: Any = None
        self.provenance_store: Any = None
        # MCP doesn't run with the FastAPI JobRegistry; tools that need
        # one (fetch_paper_resources) fall back to a synchronous inline
        # registry when this is None.
        self.job_registry: Any = None
        self.initialized: bool = False

    async def initialize(self, config: Any) -> None:
        """Initialize all components from config."""
        if self.initialized:
            return

        from pathlib import Path

        from perspicacite.llm import AsyncLLMClient
        from perspicacite.llm.embeddings import create_embedding_provider
        from perspicacite.memory.session_store import SessionStore
        from perspicacite.pipeline.parsers.pdf import PDFParser
        from perspicacite.retrieval import ChromaVectorStore

        self.config = config

        # LLM client
        self.llm_client = AsyncLLMClient(config.llm)

        # Embedding provider — same factory the web app uses, so when the
        # primary OpenAI embedding fails (no OPENAI_API_KEY in env),
        # vectors transparently fall back to a local sentence-transformers
        # model instead of crashing the whole MCP tool call. Without
        # this, MCP `generate_report` and other retrieval tools die at
        # the embed step on dev boxes that haven't exported the key.
        self.embedding_provider = create_embedding_provider(
            model=config.knowledge_base.embedding_model,
        )

        # Vector store
        self.vector_store = ChromaVectorStore(
            persist_dir=str(config.database.chroma_path),
            embedding_provider=self.embedding_provider,
        )

        # Session store
        db_path = Path("./data/perspicacite.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_store = SessionStore(db_path)
        await self.session_store.init_db()

        # Provenance store (shares the same DB as the session store)
        from perspicacite.provenance.store import ProvenanceStore

        sidecar_dir = Path("./data/provenance")
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        self.provenance_store = ProvenanceStore(db_path=db_path, sidecar_dir=sidecar_dir)

        # PDF parser
        self.pdf_parser = PDFParser()

        # Tool registry for RAG engine
        from perspicacite.rag.tools import ToolRegistry

        self.tool_registry = ToolRegistry()

        self.initialized = True
        logger.info("mcp_state_initialized")


mcp_state = MCPState()


# =============================================================================
# Helper: JSON response builder
# =============================================================================


def _json_ok(data: dict[str, Any]) -> str:
    """Emit a successful MCP envelope.

    Carries both ``success: true`` (canonical, will remain) and
    ``ok: true`` (deprecated alias for backwards compat with
    pre-v3.x downstream clients). Plan to drop ``ok`` after the
    Scriptorium-v0.13 migration completes — see docs/MCP.md.
    """
    return json.dumps(
        {"success": True, "ok": True, **data},
        ensure_ascii=False,
        default=str,
    )


def _normalize_paper_id(paper_id: str) -> str:
    """Strip the ``doi:`` prefix some surfaces use, so lookups match the
    bare-DOI form stored in the KB. ``doi:10.1038/x`` → ``10.1038/x``.
    """
    return paper_id[4:] if paper_id.startswith("doi:") else paper_id


def _json_error(message: str, **extra: Any) -> str:
    """Emit an error MCP envelope. See _json_ok for the dual-key rationale."""
    return json.dumps(
        {"success": False, "ok": False, "error": message, **extra},
        default=str,
    )


async def _resolve_push_input(inp: dict, *, http_client: Any) -> tuple[dict, str, str]:
    """Normalize a push_to_zotero input dict into a ``paper`` dict ready for
    :meth:`ZoteroClient.create_item`.

    Returns ``(paper_dict, normalized_doi, normalized_url)``.

    Three routes:
    - ``doi``: fetches metadata via the unified pipeline.
    - ``url``: uses caller-supplied fields; mines OpenGraph / citation_*
      meta tags from the page if title/authors are missing.
    - ``bibtex``: parses a BibTeX string. Promotes ``doi`` (if any) into
      the DOI route, else falls back to URL/title-only.
    """
    # BibTeX route: parse and recurse with the parsed dict.
    if inp.get("bibtex"):
        try:
            import bibtexparser
        except ImportError as exc:
            raise RuntimeError("bibtexparser not installed; pip install bibtexparser") from exc
        import re as _re

        bib = bibtexparser.loads(inp["bibtex"])
        if not bib.entries:
            raise RuntimeError("bibtex string contained no entries")
        e = bib.entries[0]

        def _unbrace(s: str) -> str:
            """Recursively strip BibTeX ``{...}`` case-preservation braces.
            ``The {Evolving} {Role} of {LLM}`` → ``The Evolving Role of LLM``."""
            if not s:
                return s
            prev = None
            while s != prev:
                prev = s
                s = _re.sub(r"\{([^{}]*)\}", r"\1", s)
            # Collapse runs of whitespace introduced by removed braces
            return _re.sub(r"\s+", " ", s).strip()

        # arXiv promotion: prefer eprint+archivePrefix=arXiv (canonical),
        # fall back to a url like ``arxiv.org/abs/<id>``. The synthesized
        # ``10.48550/arXiv.<id>`` routes the entry through the DOI path
        # (preprint item_type, real PDF fetch) instead of being demoted
        # to a webpage when only ``url`` is present in the bib.
        eprint = (e.get("eprint") or "").strip()
        archive_prefix = (e.get("archiveprefix") or e.get("archivePrefix") or "").lower()
        arxiv_id: str | None = None
        if (
            archive_prefix == "arxiv" and _re.match(r"^[0-9]{4}\.[0-9]{4,6}$", eprint)
        ) or _re.match(r"^[0-9]{4}\.[0-9]{4,6}$", eprint):
            arxiv_id = eprint
        elif e.get("url"):
            m = _re.search(
                r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,6})",
                e.get("url", ""),
            )
            if m:
                arxiv_id = m.group(1)

        explicit_doi = (e.get("doi") or "").strip().rstrip(".") or None
        synthetic_doi = f"10.48550/arXiv.{arxiv_id}" if arxiv_id else None

        promoted_title = _unbrace(e.get("title", ""))
        promoted_authors = [
            _unbrace(a.strip()) for a in (e.get("author") or "").split(" and ") if a.strip()
        ]

        # Last-resort: if the bib entry has no DOI and no usable URL
        # (or the URL is to a non-academic host like github.com), try
        # title-based DOI discovery via scholarly metadata APIs. This
        # rescues entries like LangGraph/smolagents docs cited from
        # paper bibs, and bare ``@misc`` blocks that omit the arxiv
        # eprint field.
        resolved_doi: str | None = None
        bib_url = (e.get("url") or "").strip()
        if (
            not explicit_doi
            and not synthetic_doi
            and promoted_title
            and (not bib_url or _re.search(r"(github\.com|docs?\.)", bib_url))
        ):
            import os as _os

            from perspicacite.pipeline.download.title_resolver import (
                resolve_doi_from_title,
            )

            # Headless Chromium tier 5 is opt-in via env var. Avoids
            # ImportError + 150MB Chromium download for the common
            # case where the four HTTP tiers are enough. Agents with
            # a browser MCP available (e.g. ``claude-in-chrome``)
            # can pre-resolve the title themselves and pass the DOI
            # to ``push_to_zotero`` directly — see tool docstring.
            enable_browser = _os.getenv("PERSPICACITE_HEADLESS_BROWSER", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            try:
                resolved_doi = await resolve_doi_from_title(
                    promoted_title,
                    promoted_authors,
                    e.get("year"),
                    http_client=http_client,
                    enable_browser=enable_browser,
                )
            except Exception:
                resolved_doi = None

        promoted = {
            "doi": explicit_doi or synthetic_doi or resolved_doi,
            "url": e.get("url") or "",
            "title": promoted_title,
            "year": e.get("year"),
            "authors": promoted_authors,
            "journal": _unbrace(e.get("journal") or ""),
            "abstract": _unbrace(e.get("abstract") or ""),
        }
        promoted = {k: v for k, v in promoted.items() if v}
        return await _resolve_push_input(promoted, http_client=http_client)

    # DOI route: full metadata + abstract fetch via the unified pipeline.
    if inp.get("doi"):
        from perspicacite.pipeline.download import retrieve_paper_content

        doi = inp["doi"].strip().replace("https://doi.org/", "")
        content = await retrieve_paper_content(
            doi,
            http_client=http_client,
            pdf_parser=None,  # metadata-only here
        )
        paper: dict[str, Any] = dict(content.metadata or {})
        paper["doi"] = doi
        paper["abstract"] = content.abstract or paper.get("abstract")
        # Caller-supplied fields take precedence over auto-discovered ones
        for k in (
            "title",
            "authors",
            "year",
            "journal",
            "item_type",
            "url",
            "tags",
            "abstract",
            "repository",
            "archive_id",
        ):
            if inp.get(k):
                paper[k] = inp[k]
        url = paper.get("url") or ""
        return paper, doi, url

    # URL route: trust caller-supplied metadata; supplement with OG/citation_*.
    if inp.get("url"):
        url = inp["url"].strip()
        # YouTube URLs default to videoRecording (Zotero's native type
        # for video citations). Caller can still override via
        # explicit ``item_type``.
        derived_item_type = inp.get("item_type")
        if not derived_item_type:
            from perspicacite.pipeline.download.youtube import is_youtube_url

            if is_youtube_url(url):
                derived_item_type = "videoRecording"
        paper = {
            "url": url,
            "title": inp.get("title") or "",
            "authors": inp.get("authors") or [],
            "year": inp.get("year"),
            "abstract": inp.get("abstract") or "",
            "item_type": derived_item_type,
            "tags": inp.get("tags") or [],
            "repository": inp.get("repository") or "",
            "website_title": inp.get("website_title") or "",
        }
        if not paper["title"]:
            # Last-ditch: derive a title from the URL path so the Zotero
            # item isn't blank.
            from urllib.parse import urlparse

            paper["title"] = urlparse(url).path.strip("/").split("/")[-1] or url
        return paper, "", url

    raise RuntimeError(
        "push_to_zotero input requires one of: doi, url, bibtex; got keys="
        + ",".join(sorted(inp.keys()))
    )


def _require_state() -> MCPState | str:
    """Check that MCP state is initialized. Returns state or error string."""
    if not mcp_state.initialized:
        return _json_error("MCP server not initialized")
    return mcp_state


# =============================================================================
# Tool 1: search_literature
# =============================================================================


@mcp.tool()
async def search_literature(
    query: str,
    max_results: int = 20,
    year_min: int | None = None,
    year_max: int | None = None,
    article_type: str | None = None,
    databases: list[str] | None = None,
    min_relevance: float = 0.0,
    relevance_method: str = "bm25",
    exclude_kb: str | None = None,
    context: str | None = None,
    optimize_query: bool | None = None,
    enrich: bool = True,
) -> str:
    """
    Search academic databases for scientific papers matching a query.

    **Latency:** ~5-15s with parallel 3-backend fan-out (Phase B2); 15-50s on legacy serial path. Use >=60s HTTP timeout in clients.

    Args:
        query: Search query (keywords, phrases, or natural language)
        max_results: Maximum number of results to return (1-50)
        year_min: Earliest publication year (inclusive)
        year_max: Latest publication year (inclusive)
        article_type: Filter by type ("review", "article", "conference")
        databases: Databases to search. Options: semantic_scholar, openalex, pubmed, arxiv
        min_relevance: When > 0, post-filter results so only papers with
            ``relevance_score >= min_relevance`` are returned. Score is
            normalized to ``[0, 1]``. Default 0.0 keeps every hit
            (current behavior). Try 0.3 to drop clearly-off-topic hits,
            0.5+ for high precision.
        relevance_method: How to score relevance when filtering. Three
            tiers, in order of cost/accuracy:

            - ``"bm25"`` (default, nearly free) — BM25 token overlap on
              title+abstract vs query. Catches keyword-irrelevant hits.
            - ``"rerank"`` (~5ms/paper, CPU) — cross-encoder model. More
              accurate semantically; catches "wrong domain entirely"
              hits that share surface keywords.
            - ``"llm"`` (slowest, $ per paper) — LLM judge with reasons.
              Best for ambiguous topic overlap; returns ``reason`` field
              per paper.

            Each returned paper gets a ``relevance_score`` field; ``llm``
            also adds ``relevance_reason``.
        exclude_kb: Optional KB name. Papers whose DOI already exists in
            this knowledge base are removed from the results before
            returning, so callers only see literature not yet ingested.
        context: Optional. A short grounding excerpt from earlier in the
            conversation that disambiguates the query (e.g., a specific
            finding, entity, or scope the user has been focused on). Keep
            it short — one sentence or a short bullet is ideal. Skip
            entirely when the user has shifted topic. Max ~300 chars;
            truncated otherwise.
        optimize_query: Whether to run the LLM-assisted query rewrite
            before searching. ``True`` forces on, ``False`` forces off,
            ``None`` falls back to
            ``config.search.query_optimization.enabled`` (default True).
            The rewrite uses one cheap Haiku call to produce a clean
            scientific phrasing; on any failure (timeout, LLM error,
            unparseable output) we silently fall back to the verbatim
            query and surface ``fallback_reason`` in the response.
        enrich: When True (default), enrich returned papers via Crossref
            (fills missing abstracts, canonicalises author lists). Set
            False for raw provider data.

    Returns:
        JSON with list of papers including title, authors, year, doi, abstract.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    from perspicacite.rag.telemetry import ResponseMetadataCollector
    from perspicacite.search.domain_aggregator import build_aggregator
    from perspicacite.search.scilex_adapter import KNOWN_DATABASES

    # Response-level metadata collector. Mirrors the wiring in generate_report:
    # any telemetry events emitted during this call (tokens / cost from the
    # optimizer LLM, provider_progress / query_rephrased from the aggregator)
    # are aggregated and embedded in the final JSON response — useful for
    # callers that drop MCP progress notifications.
    _response_collector = ResponseMetadataCollector()

    # Filter caller-supplied database list against the authoritative set.
    # Unknown names are dropped silently (the frontend can pass anything);
    # if the filter empties the list, fall back to provider defaults.
    filtered_databases: list[str] | None = None
    if databases is not None:
        filtered_databases = [d for d in databases if d in KNOWN_DATABASES]
        dropped = sorted(set(databases) - KNOWN_DATABASES)
        if dropped:
            logger.warning("mcp_search_literature_unknown_db", dropped=dropped)
        if not filtered_databases:
            filtered_databases = None  # fall back to defaults

    try:
        aggregator = build_aggregator(state.config)
        if not aggregator.available:
            return _json_error(
                "No search providers are available. Install SciLEx with: "
                '`uv pip install -e ".[scilex]"` from the Perspicacité repo, '
                "or configure at least one search provider in config.yml.",
                scilex_available=False,
            )
        import perspicacite.search.query_optimizer as _qo_mod

        try:
            opt = await _qo_mod.optimize_query(
                query=query,
                context=context,
                app_state=state,
                optimize_enabled=optimize_query,
                sink=_response_collector,
            )
        except Exception as _qo_exc:
            # The optimizer fails closed: any unexpected error (bad config,
            # missing LLM client, etc.) degrades to "use the verbatim query".
            logger.warning("mcp_search_literature_optimizer_error", error=str(_qo_exc))
            opt = _qo_mod.OptimizationResult(
                searched_query=query,
                enabled=False,
                applied=False,
                context_used=False,
                fallback_reason="optimizer_error",
            )

        # When filtering by relevance, overfetch ~3x so the post-filter
        # has enough candidates to actually return ``max_results``
        # quality hits. Capped at SciLEx's per-DB ceiling.
        fetch_n = min(max_results * 3, 100) if min_relevance > 0 else max_results

        # Dispatch to the multi-provider aggregator. The ``databases``
        # filter is plumbed through so the aggregator can restrict its
        # fan-out and SciLEx can restrict its sub-providers — without
        # bypassing non-SciLEx providers (europepmc, ads, pubchem,
        # inspire, google_scholar, ...).
        papers = await aggregator.search(
            query=opt.searched_query,
            max_results=fetch_n,
            year_min=year_min,
            year_max=year_max,
            apis=filtered_databases,
            databases=filtered_databases,
            article_type=article_type,
        )

        # Collect structured warnings from SciLEx (e.g. unknown APIs dropped).
        mcp_warnings: list[dict] = []
        try:
            from perspicacite.search.scilex_adapter import SciLExAdapter

            for _prov in getattr(aggregator, "_providers", []):
                if isinstance(_prov, SciLExAdapter):
                    if _prov._last_dropped_apis:
                        mcp_warnings.append(
                            {
                                "kind": "unknown_apis_dropped",
                                "apis": list(_prov._last_dropped_apis),
                                "advice": (
                                    "Use the web_search MCP tool for non-SciLEx providers "
                                    "(google_scholar, europepmc, etc.)."
                                ),
                            }
                        )
                    if _prov._last_quota_warning is not None:
                        mcp_warnings.append(_prov._last_quota_warning)
                    break
        except Exception:
            pass

        # Crossref-enrich the returned papers (fills missing abstracts etc.).
        if enrich and papers:
            from perspicacite.pipeline.enrichment.crossref_enrich import enrich_papers

            try:
                papers = await asyncio.wait_for(enrich_papers(papers), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "mcp_search_literature_enrich_timeout", n_papers=len(papers)
                )
                # papers unchanged — proceed without enrichment
            except Exception as _ee:
                logger.warning("mcp_search_literature_enrich_failed", error=str(_ee))

        # ── Dedup against existing KB (optional) ───────────────────────
        if exclude_kb:
            from perspicacite.models.kb import chroma_collection_name_for_kb

            collection = chroma_collection_name_for_kb(exclude_kb)
            filtered_papers = []
            for paper in papers:
                if paper.doi:
                    try:
                        already = await state.vector_store.paper_exists(
                            collection,
                            paper.doi,
                        )
                        if already:
                            continue
                    except Exception:
                        pass  # dedup is best-effort; don't drop on error
                filtered_papers.append(paper)
            papers = filtered_papers

        # Convert Paper models to dicts
        results = []
        for p in papers:
            pd = {
                "id": p.id,
                "title": p.title,
                "year": p.year,
                "doi": p.doi,
                "abstract": p.abstract,
                "journal": p.journal,
                "citation_count": p.citation_count,
                "source": p.source.value if p.source else None,
                "url": p.url,
            }
            if p.authors:
                pd["authors"] = [
                    a.family if hasattr(a, "family") and a.family else str(a) for a in p.authors
                ]
            # Per-provider attribution: when the aggregator merged this Paper
            # from multiple providers, expose every contributing provider name.
            # Prefer ``discovery_sources`` (maintained by DomainAwareAggregator
            # through dedup merges) then fall back to ``metadata["sources"]``
            # (set by individual providers).  Union both so no attribution is lost.
            discovery = list(p.discovery_sources or [])
            meta_srcs = (p.metadata or {}).get("sources") or []
            all_srcs = list(dict.fromkeys(discovery + [s for s in meta_srcs if s not in discovery]))
            if all_srcs:
                pd["metadata"] = {"sources": all_srcs}
            results.append(pd)

        # Optional relevance filtering (tiers A/B/C)
        if min_relevance > 0.0 and results:
            from perspicacite.search.screening import (
                screen_papers,
                screen_papers_llm,
                screen_papers_rerank,
            )

            method = (relevance_method or "bm25").lower()
            if method == "bm25":
                scored = screen_papers(
                    results,
                    reference=query,
                    threshold=min_relevance,
                )
            elif method == "rerank":
                scored = await screen_papers_rerank(
                    results,
                    query=query,
                    threshold=min_relevance,
                )
            elif method == "llm":
                scored = await screen_papers_llm(
                    results,
                    query=query,
                    llm=state.llm_client,
                    threshold=min_relevance,
                )
            else:
                return _json_error(
                    f"unknown relevance_method '{relevance_method}'; "
                    "use 'bm25', 'rerank', or 'llm'",
                )
            filtered = []
            for r in scored:
                if not r.kept:
                    continue
                item = dict(r.item)
                item["relevance_score"] = round(r.score, 4)
                if r.reason:
                    item["relevance_reason"] = r.reason
                filtered.append(item)
                if len(filtered) >= max_results:
                    break
            results = filtered

        logger.info(
            "mcp_search_literature",
            query=query,
            searched_query=opt.searched_query,
            results=len(results),
            min_relevance=min_relevance,
            method=relevance_method if min_relevance > 0 else "none",
        )
        # F-19: surface per-database failures so external agents can tell
        # "no matches" from "the upstream DB was down".
        errors_by_db = dict(getattr(aggregator, "last_errors_by_database", {}))
        # Use the names of the actual providers that ran (accurate attribution).
        # Fall back to filtered_databases or the hardcoded legacy default only
        # when the aggregator has no _providers attribute.
        _actual_providers: list[str] = [
            getattr(p, "name", "unknown") for p in getattr(aggregator, "_providers", [])
        ]
        databases_queried = filtered_databases or _actual_providers or ["pubmed", "openalex", "semantic_scholar"]
        all_dbs_failed = (
            bool(errors_by_db) and len(errors_by_db) >= len(databases_queried) and not results
        )
        # F-34: always include errors_by_database (even when empty) so callers
        # can distinguish "this DB returned 0 results" from "this DB silently
        # failed". An empty value for a queried DB means "ran cleanly".
        errors_full: dict[str, str | None] = {db: None for db in databases_queried}
        errors_full.update(errors_by_db)

        payload: dict[str, Any] = {
            "query": query,
            "total_results": len(results),
            "papers": results,
            "warnings": mcp_warnings,
            "errors_by_database": errors_full,
            "original_query": query,
            "searched_query": opt.searched_query,
            "query_optimization": {
                "enabled": opt.enabled,
                "applied": opt.applied,
                "context_used": opt.context_used,
                "fallback_reason": opt.fallback_reason,
            },
        }
        # Merge response-level metadata extras (attempts / query_rephrasings /
        # usage). When no telemetry events flowed (current default for
        # search_literature, since the aggregator doesn't emit), this is a no-op.
        payload.update(_response_collector.as_response_extras())
        if all_dbs_failed:
            payload["success"] = False
            payload["error"] = "All queried databases failed; see errors_by_database for details."
            return _json_error(
                payload["error"],
                **{k: v for k, v in payload.items() if k != "error"},
            )
        return _json_ok(payload)

    except Exception as e:
        logger.error("mcp_search_literature_error", error=str(e))
        return _json_error(f"Search failed: {e}")


# =============================================================================
# Tool 1b: general_web_search
# =============================================================================


@mcp.tool()
async def general_web_search(
    query: str,
    max_results: int = 10,
    site_filter: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    relevance_method: str = "bm25",
    min_relevance: float = 0.0,
) -> str:
    """General-web search via DuckDuckGo HTML interface + headless Chromium.

    Distinct from ``search_literature`` / ``web_search`` which both target
    academic databases (semantic_scholar, openalex, pubmed, arxiv,
    google_scholar). Use this when you need software docs, GitHub repos,
    vendor blogs, README pages, or any other non-academic web content
    — e.g., to find a tool's documentation site after the paper mentions
    the tool name but not the URL.

    Returns a list of ``{title, url, snippet, relevance_score, relevance_reason}``
    dicts. Optional LLM-judged relevance filtering via ``relevance_method``
    + ``min_relevance`` mirrors ``search_literature``.

    **Requires** the ``[browser]`` extra (``uv pip install -e '.[browser]'``
    + ``playwright install chromium``). When the extra is missing, returns
    an empty result list with a clear warning in the logs.

    Args:
        query: Free-text web search query.
        max_results: Cap on returned results (1-50).
        site_filter: Optional list of domains to restrict to, e.g.
            ``["github.com", "*.github.io", "readthedocs.io"]``.
            DuckDuckGo's wildcard ``*.subdomain`` syntax is supported
            natively. Translates to OR'd ``site:`` operators.
        exclude_domains: Optional list of domains to exclude. Translates
            to ``-site:`` operators (e.g.
            ``["wikipedia.org", "w3.org"]``).
        relevance_method: How to score relevance when ``min_relevance > 0``.
            - ``"bm25"`` (default, free) — BM25 token overlap on
              title+snippet vs query.
            - ``"rerank"`` (~5ms/result, CPU) — cross-encoder model.
            - ``"llm"`` (slow, $) — LLM judge per result with reasons.
            Each returned result gets a ``relevance_score`` field; ``llm``
            also adds ``relevance_reason``.
        min_relevance: When > 0, drop results scoring below this
            threshold. 0.0 = keep every hit (default). Try 0.3 for
            mild filtering, 0.5+ for high precision.

    Returns:
        JSON with ``results`` list and ``count`` total. Each result
        has ``title``, ``url``, ``snippet`` and (when filtering)
        ``relevance_score`` + ``relevance_reason``.

    See also: ``ingest_urls_to_kb`` to feed the top-K result URLs into
    a knowledge base — natural pipeline for proactive enrichment.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    from perspicacite.search.duckduckgo_playwright import (
        DuckDuckGoPlaywrightProvider,
    )

    try:
        provider = DuckDuckGoPlaywrightProvider()
        raw_results = await provider.search(
            query=query,
            max_results=max(max_results * 3, 30) if min_relevance > 0.0 else max_results,
            site_filter=site_filter,
            exclude_domains=exclude_domains,
        )
    except Exception as e:
        logger.error("mcp_general_web_search_error", error=str(e), query=query[:80])
        return _json_error(f"General web search failed: {e}")

    if not raw_results:
        return _json_ok({"results": [], "count": 0, "method": "none"})

    # Map raw results to the screen_papers dict shape (title + abstract)
    # so we can reuse the same relevance-scoring tiers as search_literature.
    if min_relevance > 0.0:
        from perspicacite.search.screening import (
            screen_papers,
            screen_papers_llm,
            screen_papers_rerank,
        )

        method = (relevance_method or "bm25").lower()
        # screen_papers reads 'title' and 'abstract'; map snippet -> abstract.
        candidates = [
            {"title": r["title"], "abstract": r.get("snippet", ""), "_url": r["url"]}
            for r in raw_results
        ]
        if method == "bm25":
            scored = screen_papers(candidates, reference=query, threshold=min_relevance)
        elif method == "rerank":
            scored = await screen_papers_rerank(
                candidates, query=query, threshold=min_relevance
            )
        elif method == "llm":
            scored = await screen_papers_llm(
                candidates, query=query, llm=state.llm_client, threshold=min_relevance,
            )
        else:
            return _json_error(
                f"unknown relevance_method '{relevance_method}'; "
                "use 'bm25', 'rerank', or 'llm'"
            )

        filtered: list[dict[str, Any]] = []
        for r in scored:
            if not r.kept:
                continue
            item = {
                "title": r.item["title"],
                "url": r.item["_url"],
                "snippet": r.item.get("abstract", ""),
                "relevance_score": round(r.score, 4),
            }
            if r.reason:
                item["relevance_reason"] = r.reason
            filtered.append(item)
            if len(filtered) >= max_results:
                break
        return _json_ok({
            "results": filtered,
            "count": len(filtered),
            "method": method,
            "n_pre_filter": len(raw_results),
        })

    return _json_ok({
        "results": raw_results,
        "count": len(raw_results),
        "method": "none",
    })


# =============================================================================
# Tool 2: get_paper_content
# =============================================================================


@mcp.tool()
async def get_paper_content(
    doi: str,
    include_sections: bool = True,
) -> str:
    """
    Fetch full text and structured sections for a paper by DOI.

    Uses a unified pipeline: discovers sources via OpenAlex/Unpaywall, then
    tries PMC JATS XML, arXiv HTML, publisher PDF, and abstract in priority order.

    Args:
        doi: Paper DOI (e.g., "10.1234/example")
        include_sections: Whether to include section breakdowns

    Returns:
        JSON with content_type, full_text_length, sections, and references.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    import httpx

    from perspicacite.pipeline.download import retrieve_paper_content

    try:
        pdf_config = state.config.pdf_download
        pdf_kwargs: dict[str, Any] = {}
        if pdf_config:
            pdf_kwargs = {
                "unpaywall_email": pdf_config.unpaywall_email,
                "alternative_endpoint": pdf_config.alternative_endpoint,
                "wiley_tdm_token": pdf_config.wiley_tdm_token,
                "aaas_api_key": pdf_config.aaas_api_key,
                "rsc_api_key": pdf_config.rsc_api_key,
                "springer_api_key": pdf_config.springer_api_key,
            }

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            result = await retrieve_paper_content(
                doi,
                http_client=client,
                pdf_parser=state.pdf_parser,
                **pdf_kwargs,
            )

        if result.success and result.content_type in ("structured", "full_text"):
            # F-26 (audit 2026-05-17): include the full text body so callers
            # can actually consume the content. The previous shape only
            # exposed full_text_length, which let consumers see "the paper
            # exists" but not read it. We do still expose the length for
            # quick budgeting.
            resp: dict[str, Any] = {
                "doi": doi,
                "content_type": result.content_type,
                "content_source": result.content_source,
                "full_text": result.full_text or "",
                "full_text_length": len(result.full_text or ""),
                "attempts": list(result.attempts),
            }
            if include_sections and result.sections:
                resp["sections"] = result.sections
            if result.references:
                resp["references_count"] = len(result.references)
                resp["references"] = result.references
            return _json_ok(resp)

        if result.content_type == "abstract":
            return _json_ok(
                {
                    "doi": doi,
                    "content_type": "abstract",
                    "content_source": result.content_source,
                    "abstract": result.abstract,
                    "attempts": list(result.attempts),
                    "note": "Full text not available; returning abstract only",
                }
            )

        return _json_error(
            f"Could not retrieve content for DOI: {doi}",
            attempts=list(result.attempts),
        )

    except Exception as e:
        logger.error("mcp_get_paper_content_error", doi=doi, error=str(e))
        return _json_error(f"Content retrieval failed: {e}")


# =============================================================================
# Tool 3: get_paper_references
# =============================================================================


@mcp.tool()
async def get_paper_references(
    doi: str,
) -> str:
    """
    Get the list of cited references from a paper.

    Extracts references from JATS XML when available via PMC Open Access.
    Falls back to discovery metadata for non-PMC papers.

    Args:
        doi: Paper DOI

    Returns:
        JSON with list of referenced papers (doi, title, authors, year).
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    import httpx

    try:
        from perspicacite.pipeline.download.unified import _load_cached_references

        # Try loading cached refs first (from a previous content fetch)
        refs = _load_cached_references(doi)
        if refs:
            return _json_ok({"doi": doi, "references": refs, "total": len(refs)})

        # No cached refs — fetch content through unified pipeline to populate cache
        from perspicacite.pipeline.download import retrieve_paper_content

        pdf_config = getattr(state.config, "pdf_download", None)
        pdf_kwargs = {}
        if pdf_config:
            for key in (
                "unpaywall_email",
                "wiley_tdm_token",
                "elsevier_api_key",
                "aaas_api_key",
                "rsc_api_key",
                "springer_api_key",
            ):
                val = getattr(pdf_config, key, None)
                if val:
                    pdf_kwargs[key] = val

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            await retrieve_paper_content(
                doi,
                http_client=client,
                pdf_parser=state.pdf_parser,
                **pdf_kwargs,
            )

        # Try cache again after content fetch
        refs = _load_cached_references(doi)
        if refs:
            return _json_ok({"doi": doi, "references": refs, "total": len(refs)})

        return _json_ok(
            {
                "doi": doi,
                "references": [],
                "total": 0,
                "note": "References not available — JATS XML extraction only works for PMC Open Access papers",
            }
        )

    except Exception as e:
        logger.error("mcp_get_paper_references_error", doi=doi, error=str(e))
        return _json_error(f"Reference retrieval failed: {e}")


# =============================================================================
# Tool 4: list_knowledge_bases (DEFERRED: get_citation_network)
# =============================================================================


# Embedding families strong enough that a general cross-encoder reranker and a
# BM25 hybrid blend tend to HURT rather than help (the embedder already produces
# a near-perfect top-k). See docs/embedding_reranker_policy.md. Heuristic
# substring match on the embedding model id — conservative: anything not matched
# is treated as a "standard" embedder where rerank + hybrid help.
_STRONG_EMBEDDING_MARKERS = (
    "qwen3-embedding", "codestral-embed", "text-embedding-3-large",
    "gemini-embedding", "e5-mistral", "gte-qwen", "embed-v",  # cohere embed-v*
)


def _retrieval_hint(embedding_model: str) -> dict:
    """Derive a reranker/hybrid recommendation from the KB's embedding model.

    Heuristic, backed by docs/embedding_reranker_policy.md: strong
    instruction-tuned / large API embedders should NOT be reranked or
    BM25-blended; weaker local embedders (MiniLM, base BGE, SPECTER2, PubMedBERT)
    benefit from both.
    """
    m = (embedding_model or "").lower()
    strong = any(marker in m for marker in _STRONG_EMBEDDING_MARKERS)
    return {
        "embedding_strength": "strong" if strong else "standard",
        "recommended_reranker": not strong,
        "recommended_hybrid": not strong,
        "note": (
            "strong embedder — prefer pure vector (rerank + BM25 hybrid tend to hurt)"
            if strong else
            "standard embedder — cross-encoder rerank + BM25 hybrid usually help"
        ),
    }


@mcp.tool()
async def list_knowledge_bases() -> str:
    """
    List all available knowledge bases.

    Returns:
        JSON with list of KBs including name, description, paper/chunk counts,
        the embedding_model each KB was ingested with, and a retrieval_hint
        (reranker/hybrid recommendation derived from the embedding's strength).

    IMPORTANT for agents: a KB can only be queried with its OWN embedding_model.
    When fanning a query across multiple KBs (kb_names=[...]), only group KBs that
    share the same embedding_model — mixed-embedding groups are rejected at query
    time. Use the embedding_model field to pick compatible KBs and the
    retrieval_hint to decide whether to enable reranking/hybrid for each.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        kbs = await state.session_store.list_kbs()
        result = []
        for kb in kbs:
            embedding_model = getattr(kb, "embedding_model", None)
            result.append(
                {
                    "name": kb.name,
                    "description": kb.description,
                    "paper_count": kb.paper_count,
                    "chunk_count": kb.chunk_count,
                    "embedding_model": embedding_model,
                    "retrieval_hint": _retrieval_hint(embedding_model or ""),
                    "created_at": str(kb.created_at) if hasattr(kb, "created_at") else None,
                }
            )
        return _json_ok({"knowledge_bases": result})
    except Exception as e:
        logger.error("mcp_list_kbs_error", error=str(e))
        return _json_error(f"Failed to list KBs: {e}")


# =============================================================================
# Tool 5: search_knowledge_base
# =============================================================================


@mcp.tool()
async def search_knowledge_base(
    query: str,
    kb_name: str = "default",
    top_k: int = 5,
    kb_names: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> str:
    """
    Search within a specific knowledge base (or multiple KBs) using semantic similarity.

    Args:
        query: Search query
        kb_name: Knowledge base name (single-KB path)
        top_k: Number of top results to return
        kb_names: Optional list of KBs to query together. All KBs must share the same
            embedding model. When provided and len > 1, supersedes kb_name and returns
            chunks tagged with their source KB. When exactly 1 entry, treated as
            single KB via kb_name.
        year_min: Restrict to papers published in or after this year (inclusive).
        year_max: Restrict to papers published in or before this year (inclusive).

    Returns:
        JSON with matching chunks including paper_id, chunk_text, relevance_score,
        and (in multi-KB mode) kb_name per chunk.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        # Multi-KB path
        if kb_names and len(kb_names) > 1:
            if year_min is not None or year_max is not None:
                logger.warning(
                    "search_kb_multi_year_filter_ignored",
                    year_min=year_min,
                    year_max=year_max,
                    note="multi-KB filter passthrough is a Wave 4.2 followup",
                )
            from perspicacite.retrieval.multi_kb import MultiKBRetriever, check_embedding_compat

            metas = [await state.session_store.get_kb_metadata(n) for n in kb_names]
            for i, meta in enumerate(metas):
                if meta is None:
                    return _json_error(f"Knowledge base not found: {kb_names[i]}")
            compat_msg = check_embedding_compat(metas)
            if compat_msg:
                return _json_error(compat_msg)

            retr = MultiKBRetriever(
                vector_store=state.vector_store,
                embedding_service=state.embedding_provider,
                kb_metas=metas,
            )
            results = await retr.search(query, top_k=top_k)

            chunks = []
            for r in results:
                meta_obj = r.get("metadata")
                meta_dict = meta_obj.__dict__ if hasattr(meta_obj, "__dict__") else (meta_obj or {})
                # Decode the ASB-style paper_metadata_json (if present)
                # so the response helper can read it directly.
                pm_dict = decode_paper_metadata_json(meta_obj)
                chunks.append(
                    {
                        "paper_id": r.get("paper_id"),
                        "title": meta_dict.get("title") if isinstance(meta_dict, dict) else None,
                        "section": meta_dict.get("section")
                        if isinstance(meta_dict, dict)
                        else None,
                        "chunk_text": r.get("text", ""),
                        "relevance_score": r.get("score"),
                        "doi": meta_dict.get("doi") if isinstance(meta_dict, dict) else None,
                        "kb_name": r.get("kb_name"),
                        "metadata": pm_dict,
                    }
                )

            asb_md = build_asb_response_metadata(chunks)
            return _json_ok(
                {
                    "query": query,
                    "kb_names": kb_names,
                    "results": chunks,
                    "asb_metadata": asb_md,
                }
            )

        # Single-KB path (original behaviour, unchanged)
        effective_kb_name = kb_names[0] if (kb_names and len(kb_names) == 1) else kb_name

        from perspicacite.models.kb import chroma_collection_name_for_kb
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

        collection_name = chroma_collection_name_for_kb(effective_kb_name)

        # Verify KB exists
        kb_meta = await state.session_store.get_kb_metadata(effective_kb_name)
        if not kb_meta:
            return _json_error(f"Knowledge base '{effective_kb_name}' not found")

        # Use the embedding model the KB was indexed with.  If the server's
        # current default has a different dimension (e.g. after switching from
        # text-embedding-3-small to all-MiniLM) ChromaDB rejects the query
        # with a dimension-mismatch error.  Matching on kb_meta.embedding_model
        # means old 1536-dim KBs automatically use OpenAI for the query vector.
        # Split on "|" in case the stored name is a legacy composite like
        # "text-embedding-3-small|all-MiniLM-L6-v2"; use the first entry.
        #
        # IMPORTANT: reuse the server's pre-loaded embedding_provider when the
        # KB model matches — avoids reloading large sentence-transformer models
        # (e.g. stella 1.5B, BGE-M3) from disk on every search_knowledge_base
        # call, which caused OOM crashes with concurrent eval requests.
        from perspicacite.llm.embeddings import create_embedding_provider

        _kb_model_name = kb_meta.embedding_model.split("|")[0].strip()
        # Normalise: strip the "st:" routing prefix before comparing names.
        _norm = lambda s: s.removeprefix("st:").strip()
        if _norm(_kb_model_name) == _norm(state.embedding_provider.model_name):
            _kb_embedding = state.embedding_provider
        else:
            _kb_embedding = create_embedding_provider(_kb_model_name, use_local_fallback=False)
        dkb = DynamicKnowledgeBase(
            state.vector_store,
            _kb_embedding,
            config=KnowledgeBaseConfig(
                vector_size=_kb_embedding.dimension,
            ),
        )
        dkb.collection_name = collection_name
        dkb._initialized = True

        # Build year-bounded filters (Wave 4.2).
        from perspicacite.models.search import SearchFilters

        filters = None
        if year_min is not None or year_max is not None:
            filters = SearchFilters(year_min=year_min, year_max=year_max)

        results = await dkb.search(query, top_k=top_k, filters=filters)

        # ``DynamicKnowledgeBase.search`` returns plain dicts shaped as
        # {"text", "score", "paper_id", "metadata", "kb_name"} — not the
        # RetrievedChunk objects this serializer previously assumed.
        # Treating them as objects produced empty paper_id/title/doi and
        # stuffed the entire dict-repr into chunk_text (audit R-18).
        chunks = []
        for r in results:
            if isinstance(r, dict):
                meta_obj = r.get("metadata") or {}
                meta_dict = (
                    meta_obj.__dict__
                    if hasattr(meta_obj, "__dict__")
                    else dict(meta_obj)
                    if isinstance(meta_obj, dict)
                    else {}
                )
                pm_dict = decode_paper_metadata_json(meta_dict)
                chunks.append(
                    {
                        "paper_id": r.get("paper_id") or meta_dict.get("paper_id"),
                        "title": meta_dict.get("title"),
                        "section": meta_dict.get("section"),
                        "chunk_text": r.get("text", ""),
                        "relevance_score": r.get("score"),
                        "doi": meta_dict.get("doi"),
                        "kb_name": r.get("kb_name"),
                        "year": meta_dict.get("year"),
                        "content_type": meta_dict.get("content_type"),
                        "metadata": pm_dict,
                    }
                )
            else:
                meta = getattr(r, "metadata", None) or {}
                if hasattr(meta, "__dict__"):
                    meta = meta.__dict__
                # Decode the ASB-style paper_metadata_json (if present)
                # so the response helper can read it directly.
                pm_dict = decode_paper_metadata_json(meta)
                chunks.append(
                    {
                        "paper_id": meta.get("paper_id") if isinstance(meta, dict) else None,
                        "title": meta.get("title") if isinstance(meta, dict) else None,
                        "section": meta.get("section") if isinstance(meta, dict) else None,
                        "chunk_text": getattr(r, "text", str(r)),
                        "relevance_score": getattr(r, "score", None),
                        "doi": meta.get("doi") if isinstance(meta, dict) else None,
                        "metadata": pm_dict,
                    }
                )

        asb_md = build_asb_response_metadata(chunks)
        return _json_ok(
            {
                "query": query,
                "kb_name": effective_kb_name,
                "results": chunks,
                "asb_metadata": asb_md,
            }
        )

    except Exception as e:
        logger.error("mcp_search_kb_error", kb_name=kb_name, error=str(e))
        return _json_error(f"KB search failed: {e}")


# =============================================================================
# Tool 5b: search_skill_kb  (EDAM-prefiltered skill search)
# =============================================================================


@mcp.tool()
async def search_skill_kb(
    query: str,
    kb_name: str = "default",
    top_k: int = 10,
    edam_operation: str | None = None,
    edam_topics: list[str] | None = None,
) -> str:
    """Search a skill KB with an optional EDAM IRI pre-filter.

    Wraps ``search_knowledge_base`` with an EDAM-IRI metadata pre-filter
    that cuts the candidate set before embedding ranking — ~10× precision
    multiplier at scale for skill libraries with EDAM annotations.

    Used by the L2 layer of the layered skill-retrieval architecture (§6 of
    the 2026-05-24 ASB-Skills release design):

      L0: Collection README description
      L1: ``_router/SKILL.md`` (auto-generated per collection), loaded by default
      L2: **this tool** — EDAM pre-filter + embed rank
      L3: Load specific SKILL.md body via ``Read`` tool

    Returns top-k skill IRIs (from ``metadata.skill_iri``) plus the
    filtered chunk list for the caller to use or forward to L3.

    **Fail-open:** chunks with no EDAM metadata always pass through to
    preserve backward compat with legacy skill bundles that predate EDAM
    annotation.

    Args:
        query: Free-text question or topic.
        kb_name: Target KB containing the skill collection.
        top_k: Number of chunks to request from the underlying KB search.
            EDAM filtering happens post-retrieval; the final result may
            have fewer items than top_k when filtering is active.
        edam_operation: Optional EDAM operation IRI
            (e.g. ``"http://edamontology.org/operation_3215"``).
            When provided, restricts to skills annotated with this operation.
        edam_topics: Optional list of EDAM topic IRIs
            (e.g. ``["http://edamontology.org/topic_3172"]``).
            When provided, restricts to skills sharing at least one topic.

    Returns:
        JSON with ``success``, ``query``, ``kb_name``, ``results``
        (filtered chunks), ``skill_iris`` (deduplicated, score-ordered),
        and ``edam_filter_applied`` (bool).
    """
    import json as _json

    # Step 1: run the standard search
    raw_str = await search_knowledge_base(
        query=query,
        kb_name=kb_name,
        top_k=top_k,
    )
    raw = _json.loads(raw_str)

    # Propagate failures immediately
    if not raw.get("success"):
        return raw_str

    chunks = raw.get("results") or []

    # Step 2: EDAM pre-filter
    edam_filter_applied = bool(edam_operation or edam_topics)
    if edam_filter_applied:
        chunks = edam_pre_filter(
            chunks,
            edam_operation=edam_operation,
            edam_topics=edam_topics,
        )

    # Step 3: Deduplicate skill IRIs (preserve score order)
    seen_iris: set[str] = set()
    skill_iris: list[str] = []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        iri = meta.get("skill_iri") or ""
        if iri and iri not in seen_iris:
            seen_iris.add(iri)
            skill_iris.append(iri)

    return _json_ok({
        "query": query,
        "kb_name": kb_name,
        "results": chunks,
        "skill_iris": skill_iris,
        "edam_filter_applied": edam_filter_applied,
        "edam_operation": edam_operation,
        "edam_topics": edam_topics,
    })


# =============================================================================
# Tool 6: create_knowledge_base
# =============================================================================


@mcp.tool()
async def create_knowledge_base(
    name: str,
    description: str = "",
) -> str:
    """
    Create a new empty knowledge base.

    Args:
        name: KB name (alphanumeric, hyphens, underscores)
        description: Optional description

    Returns:
        JSON with created KB details.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        from perspicacite.models.kb import (
            ChunkConfig,
            KnowledgeBase,
            chroma_collection_name_for_kb,
        )

        collection_name = chroma_collection_name_for_kb(name)

        # Check for duplicate
        existing = await state.session_store.get_kb_metadata(name)
        if existing:
            return _json_error(f"Knowledge base '{name}' already exists")

        # Create ChromaDB collection
        await state.vector_store.create_collection(collection_name)

        # Save metadata
        kb = KnowledgeBase(
            name=name,
            description=description or "Created via MCP",
            collection_name=collection_name,
            embedding_model=state.embedding_provider.model_name,
            chunk_config=ChunkConfig(
                chunk_size=state.config.knowledge_base.chunk_size,
                chunk_overlap=state.config.knowledge_base.chunk_overlap,
            ),
        )
        await state.session_store.save_kb_metadata(kb)

        logger.info("mcp_create_kb", name=name)
        return _json_ok(
            {
                "name": name,
                "description": kb.description,
                "collection_name": collection_name,
                "paper_count": 0,
                "chunk_count": 0,
            }
        )

    except Exception as e:
        logger.error("mcp_create_kb_error", name=name, error=str(e))
        return _json_error(f"KB creation failed: {e}")


# =============================================================================
# Tool 7: add_papers_to_kb
# =============================================================================


@mcp.tool()
async def add_papers_to_kb(
    kb_name: str,
    papers: list[dict],
) -> str:
    """
    Add papers to a knowledge base with automatic PDF download and indexing.

    Each paper dict should have at least 'title' and optionally 'doi', 'year',
    'authors', 'abstract', 'citations'.

    Args:
        kb_name: Target knowledge base name
        papers: List of paper dicts to add

    Returns:
        JSON with counts of added/skipped papers and PDF download stats.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        import hashlib

        import httpx

        from perspicacite.models.kb import chroma_collection_name_for_kb
        from perspicacite.models.papers import Author, Paper, PaperSource
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

        collection_name = chroma_collection_name_for_kb(kb_name)
        kb_meta = await state.session_store.get_kb_metadata(kb_name)
        if not kb_meta:
            return _json_error(f"Knowledge base '{kb_name}' not found")

        # Convert paper dicts to Paper models
        paper_models: list[Paper] = []
        for pd in papers:
            paper_id = pd.get("doi") or hashlib.md5(pd.get("title", "").encode()).hexdigest()[:12]

            authors = []
            for a in pd.get("authors", []):
                if isinstance(a, str):
                    authors.append(Author(family=a, given="", name=a))
                elif isinstance(a, dict):
                    authors.append(
                        Author(
                            family=a.get("family", ""),
                            given=a.get("given", ""),
                            name=a.get("name", ""),
                        )
                    )

            paper = Paper(
                id=paper_id,
                title=pd.get("title", ""),
                authors=authors,
                year=pd.get("year"),
                doi=pd.get("doi"),
                abstract=pd.get("abstract"),
                citation_count=pd.get("citations"),
                journal=pd.get("journal"),
                url=pd.get("url"),
                pdf_url=pd.get("pdf_url"),
                source=PaperSource.USER_UPLOAD,
                keywords=pd.get("keywords", []),
                metadata=pd.get("metadata", {}),
            )
            paper_models.append(paper)

        # Download PDFs
        pdf_stats = {"attempted": 0, "success": 0, "failed": 0}
        pdf_config = state.config.pdf_download
        pdf_kwargs: dict[str, Any] = {}
        if pdf_config:
            pdf_kwargs = {
                "unpaywall_email": pdf_config.unpaywall_email,
                "alternative_endpoint": pdf_config.alternative_endpoint,
                "wiley_tdm_token": pdf_config.wiley_tdm_token,
                "aaas_api_key": pdf_config.aaas_api_key,
                "rsc_api_key": pdf_config.rsc_api_key,
                "springer_api_key": pdf_config.springer_api_key,
            }

        from perspicacite.pipeline.download import retrieve_paper_content
        import asyncio as _asyncio_local

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            # First pass: handle papers that need no network (pre-supplied
            # full_text, skip_content_fetch, missing/non-canonical DOI). These
            # are O(1) per paper and don't benefit from parallelism.
            fetch_idxs: list[int] = []
            for i, (paper, pd) in enumerate(zip(paper_models, papers)):
                # Caller can supply pre-fetched text directly via `full_text`
                # (or pass `skip_content_fetch=True`) to bypass the slow
                # Crossref/PMC/Unpaywall lookup loop. Useful for benchmark
                # corpora where the abstract is the only signal we want and
                # the supplied "doi" is a non-resolvable benchmark id
                # (e.g. ``scifact:4983`` or ``beir:nfcorpus:MED-10``).
                supplied_full_text = pd.get("full_text")
                if supplied_full_text:
                    paper.full_text = supplied_full_text
                    continue
                if pd.get("skip_content_fetch"):
                    continue
                if not paper.doi:
                    continue
                # Heuristic: real DOIs start with "10." (Crossref prefix). Anything
                # else is a synthetic benchmark id; skipping saves 1-5 s of
                # failed lookups per paper at no information loss.
                if not paper.doi.startswith("10."):
                    continue
                fetch_idxs.append(i)

            # Second pass: fetch the real-DOI papers in parallel. The previous
            # implementation iterated serially, so a batch of 5 took 5 × per-paper
            # time. With asyncio.gather we issue all fetches concurrently against
            # the shared httpx client; the upstream APIs (PMC, Unpaywall, Crossref,
            # OpenAlex) all support concurrent connections fine.
            pdf_stats["attempted"] += len(fetch_idxs)

            async def _fetch(idx: int) -> tuple[int, str | None, bool]:
                paper = paper_models[idx]
                try:
                    result = await retrieve_paper_content(
                        paper.doi,
                        url=paper.url,
                        http_client=client,
                        pdf_parser=state.pdf_parser,
                        **pdf_kwargs,
                    )
                    if result.success and result.full_text:
                        return idx, result.full_text, True
                    return idx, None, False
                except Exception:
                    return idx, None, False

            if fetch_idxs:
                outcomes = await _asyncio_local.gather(
                    *[_fetch(i) for i in fetch_idxs],
                    return_exceptions=False,
                )
                for idx, full_text, ok in outcomes:
                    if ok and full_text:
                        paper_models[idx].full_text = full_text
                        pdf_stats["success"] += 1
                    else:
                        pdf_stats["failed"] += 1

        # Add to vector store
        dkb_config = KnowledgeBaseConfig(
            vector_size=state.embedding_provider.dimension,
            chunk_size=state.config.knowledge_base.chunk_size,
            chunk_overlap=state.config.knowledge_base.chunk_overlap,
            chunking_method=state.config.knowledge_base.chunking_method,
        )
        dkb = DynamicKnowledgeBase(
            state.vector_store,
            state.embedding_provider,
            config=dkb_config,
        )
        dkb.collection_name = collection_name
        dkb._initialized = True

        chunks_added = await dkb.add_papers(paper_models, include_full_text=True)

        # Update metadata
        kb_meta.paper_count = (kb_meta.paper_count or 0) + len(paper_models)
        kb_meta.chunk_count = (kb_meta.chunk_count or 0) + chunks_added
        await state.session_store.save_kb_metadata(kb_meta)

        logger.info(
            "mcp_add_papers",
            kb_name=kb_name,
            papers=len(paper_models),
            chunks=chunks_added,
        )

        return _json_ok(
            {
                "kb_name": kb_name,
                "added_papers": len(paper_models),
                "added_chunks": chunks_added,
                "pdf_download": pdf_stats,
            }
        )

    except Exception as e:
        logger.error("mcp_add_papers_error", kb_name=kb_name, error=str(e))
        return _json_error(f"Failed to add papers: {e}")


# =============================================================================
# Tool 8: generate_report
# =============================================================================


@mcp.tool()
async def generate_report(
    query: str,
    kb_name: str = "default",
    mode: str = "advanced",
    max_papers: int = 10,
    recency_weight: float = 0.0,
    kb_names: list[str] | None = None,
    task_id: str | None = None,
    max_total_seconds: float | None = None,
    batch_size: int | None = None,
    crossref_concurrency: int | None = None,
    screen_method: str | None = None,
    screen_threshold: float | None = None,
    max_papers_to_download: int | None = None,
    databases: list[str] | None = None,
    extract_claims: bool = False,
    domains: list[str] | None = None,
    bm25_weight: float | None = None,
    vector_weight: float | None = None,
    use_hybrid: bool | None = None,
    use_reranker: bool | None = None,
    ctx: Context | None = None,
) -> str:
    """
    Generate a synthesized research report from a knowledge base.

    Uses Perspicacité's RAG pipeline (retrieval + LLM synthesis) to answer
    a research question using papers in the specified KB.

    **Latency:** 30-120s depending on KB size + LLM. Use >=180s HTTP timeout.

    Args:
        query: Research question to answer
        kb_name: Knowledge base to query (single-KB path)
        kb_names: Optional list of KBs to query together. All KBs must share the same
            embedding model. When provided and len > 1, supersedes kb_name.
            When exactly 1 entry, treated as single KB via kb_name.
        mode: RAG mode (default "advanced"; an unknown value falls back to
            "advanced"). One of:
              - "basic": quick single-pass retrieval + synthesis, no rerank.
              - "advanced": screening + rerank with query expansion (default).
              - "deep_research": deep multi-cycle research with planning + reflection (recommended; "profound" is a deprecated alias that still works).
              - "contradiction": surfaces conflicting evidence across papers
                (agreement / disagreement / open questions).
              - "agentic": multi-step, intent-driven orchestration — delegates
                to the AgenticOrchestrator (tool use, iterative replanning).
              - "literature_survey": broad survey with theme clustering and
                paper recommendations.
        max_papers: Maximum papers to reference in the report
        recency_weight: Optional recency bias (0.0 = disabled, 1.0 = full recency). When > 0,
            retrieved chunks are re-scored toward more recent papers using exponential decay.
        max_total_seconds: Override the per-mode wall-clock budget (30–1800 s). Applies to
            the "deep_research" mode's cycle loop. None uses the config-file default.
        batch_size: Override the abstract-analysis batch size for "literature_survey" mode
            (1–100 papers per batch). None uses the config-file default (20).
        crossref_concurrency: Override Crossref enrichment concurrency (1–10). None uses
            the default (2 without mailto env var, 6 with).
        screen_method: Optional relevance screening method used by modes that
            perform a screening step. One of "bm25", "rerank", "llm". Unknown
            values are dropped (mode falls back to its config default).
        screen_threshold: Optional screening threshold in [0, 1]. Values outside
            the range are clamped. None means the mode's config default applies.
        max_papers_to_download: Optional hard cap on the number of full-text
            papers downloaded during the report (1–50). Out-of-range values are
            clamped. None means the mode's default applies.
        databases: Optional list of search databases (e.g. ["arxiv", "pubmed"]).
            Unknown names are dropped with a warning. None means the mode's
            default fan-out (semantic_scholar, openalex, pubmed) applies.
        extract_claims: When True, run the indicium claim-extraction pipeline
            over the cited sources after the report is generated and attach the
            result as an ``indicia`` list in the response payload. Defaults to
            False so existing callers are unaffected.
        domains: Optional list of domain adapter IDs (e.g. ["metabolomics"]).
            Multiple IDs are composed into a CompositeAdapter; applies when
            ``extract_claims=True``. Requires indicium-adapters; silently ignored
            if not available.
        bm25_weight: Override BM25 retrieval weight in [0.0, 1.0] for hybrid
            retrieval modes (advanced, deep_research). None uses the config
            default (0.5). Set to 1.0 for pure keyword retrieval, 0.0 for
            pure vector retrieval. Complementary to vector_weight.
        vector_weight: Override vector (semantic) retrieval weight in [0.0, 1.0]
            for hybrid retrieval modes. None uses the config default (0.5).
            Complementary to bm25_weight. Both can be set independently for
            asymmetric hybrid weighting.
        use_hybrid: Override hybrid (vector+BM25) retrieval for this call.
            None = mode default. False = pure vector. Set False for KBs built
            with a strong instruction-tuned embedder (Qwen3-Embedding,
            codestral-embed, text-embedding-3-large) — BM25 blending hurts them.
            Check list_knowledge_bases' retrieval_hint.recommended_hybrid.
        use_reranker: Override cross-encoder reranking of web-search results.
            None = server default. False = no rerank. Set False for the same
            strong-embedder KBs (see retrieval_hint.recommended_reranker). Both
            help weak embedders (MiniLM) and hurt strong ones — see
            docs/embedding_reranker_policy.md.

    Returns:
        JSON object with the following keys:

        Always present:
          - ``report`` (str): synthesized answer with inline citations
          - ``sources`` (list): cited papers (doi, title, authors, year, section)
          - ``papers_used`` (int): number of sources cited
          - ``query`` (str): echoed query
          - ``kb_name`` / ``kb_names``: echoed KB selection
          - ``mode`` (str): echoed RAG mode used
          - ``message_id`` (str): unique report ID for referencing in conversations
          - ``task_id`` (str | null): async task ID if one was provided
          - ``asb_metadata`` (dict | null): ASB provenance block when run via ASB
          - ``iteration_count`` (int): number of RAG cycles completed
            (always 1 for basic/advanced; 1–N for deep_research; 0 on cancellation)
          - ``completion_reason`` (str): why the run ended — ``"complete"``,
            ``"time_budget_exceeded"``, ``"early_exit_confidence"``, or
            ``"cancelled"``
          - ``diagnostic`` (dict | null): mode-specific internals, e.g.
            ``{"cycles_completed": 2, "papers_retrieved": 14}`` for deep_research
          - ``_provenance`` (dict): authorship/trust envelope so host agents do
            not mistake LLM synthesis for source-verified text. Keys:
            ``provider`` / ``model`` (the LLM that wrote ``report``),
            ``rag_cycles_executed`` (int), ``ai_generated_fields``
            (``["report"]`` — these fields are model-authored, not grounded),
            and ``untrusted_sources`` (DOIs/titles of retrieved sources whose
            content is attacker-influenceable). The ``report`` text may
            misrepresent or propagate injected steering from these sources.

        Present only when ``extract_claims=True``:
          - ``indicia`` (list): typed claim dicts (5-slot SuperPattern + ECO/CiTO)
          - ``claims_valid`` (bool): SHACL validation result
          - ``validation_report`` (str): SHACL error detail (only when invalid)
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    # Backward compat: "profound" is the deprecated alias for "deep_research"
    if mode == "profound":
        logger.warning(
            "generate_report_deprecated_mode",
            mode="profound",
            suggested="deep_research",
        )
        mode = "deep_research"

    import uuid as _uuid

    if not task_id:
        task_id = f"mcp-{_uuid.uuid4().hex[:12]}"

    # Emit the task_id immediately via ctx so the client can cancel.
    if ctx is not None:
        try:
            await ctx.report_progress(
                progress=0,
                total=100,
                message=f"Task started — task_id={task_id}",
            )
        except Exception:
            pass

    # Bind ctx for any nested LLM call via sampling. We use the
    # contextvar token directly here (rather than the `with` form) to
    # avoid re-indenting this tool's large body.
    from perspicacite.llm.mcp_sampling import _mcp_ctx as _sampling_ctx

    _sampling_token = _sampling_ctx.set(ctx) if ctx is not None else None
    try:
        message_id = str(uuid.uuid4())

        from perspicacite.models.rag import RAGMode, RAGRequest
        from perspicacite.rag.engine import RAGEngine

        # Resolve effective kb_name / kb_names
        effective_kb_name = kb_name
        effective_kb_names: list[str] | None = None

        if kb_names and len(kb_names) > 1:
            from perspicacite.retrieval.multi_kb import check_embedding_compat

            metas = [await state.session_store.get_kb_metadata(n) for n in kb_names]
            for i, meta in enumerate(metas):
                if meta is None:
                    return _json_error(f"Knowledge base not found: {kb_names[i]}")
            compat_msg = check_embedding_compat(metas)
            if compat_msg:
                return _json_error(compat_msg)
            effective_kb_names = kb_names
        elif kb_names and len(kb_names) == 1:
            effective_kb_name = kb_names[0]

        if effective_kb_names is None:
            # Single-KB: verify it exists
            kb_meta = await state.session_store.get_kb_metadata(effective_kb_name)
            if not kb_meta:
                return _json_error(f"Knowledge base '{effective_kb_name}' not found")

        engine = RAGEngine(
            llm_client=state.llm_client,
            vector_store=state.vector_store,
            embedding_provider=state.embedding_provider,
            tool_registry=state.tool_registry,
            config=state.config,
            session_store=getattr(state, "session_store", None),
            # MCPState duck-types as the AppState protocol (carries .config
            # and .llm_client); RAGEngine.auto-attach will set
            # request.app_state = state for every mode dispatched here.
            # Closes the Tier 3.5 loop so query optimization runs on
            # MCP-originated requests instead of silently no-op'ing.
            app_state=state,
        )
        engine.provenance_store = getattr(state, "provenance_store", None)

        # Collect full response from streaming generator
        report_text = ""
        sources: list[dict] = []

        mode_map = {
            "basic": RAGMode.BASIC,
            "advanced": RAGMode.ADVANCED,
            "deep_research": RAGMode.DEEP_RESEARCH,
            "profound": RAGMode.PROFOUND,
            "agentic": RAGMode.AGENTIC,
            "literature_survey": RAGMode.LITERATURE_SURVEY,
            "contradiction": RAGMode.CONTRADICTION,
        }
        rag_mode = mode_map.get(mode, RAGMode.ADVANCED)

        # Resolve provider/model from server-side config so MCP respects
        # llm.default_provider / llm.default_model rather than the
        # hard-coded RAGRequest defaults (deepseek). Matches the fix in
        # web/routers/chat.py::_stream_rag_mode.
        default_provider = "deepseek"
        default_model = "deepseek-chat"
        cfg_llm = getattr(state, "config", None)
        if cfg_llm is not None and getattr(cfg_llm, "llm", None) is not None:
            default_provider = cfg_llm.llm.default_provider or default_provider
            default_model = cfg_llm.llm.default_model or default_model

        # Clamp / validate the new knobs at the MCP boundary so that
        # internal callers (and RAGRequest itself) can trust the values.
        if screen_threshold is not None:
            screen_threshold = max(0.0, min(1.0, float(screen_threshold)))
        if max_papers_to_download is not None:
            max_papers_to_download = max(1, min(50, int(max_papers_to_download)))
        if screen_method is not None and screen_method not in ("bm25", "rerank", "llm"):
            logger.warning(
                "mcp_generate_report_unknown_screen_method",
                method=screen_method,
            )
            screen_method = None

        filtered_databases: list[str] | None = None
        if databases is not None:
            from perspicacite.search.scilex_adapter import KNOWN_DATABASES

            filtered_databases = [d for d in databases if d in KNOWN_DATABASES]
            dropped = sorted(set(databases) - set(KNOWN_DATABASES))
            if dropped:
                logger.warning(
                    "mcp_generate_report_unknown_db",
                    dropped=dropped,
                )
            if not filtered_databases:
                filtered_databases = None

        rag_request = RAGRequest(
            query=query,
            kb_name=effective_kb_name,
            kb_names=effective_kb_names,
            mode=rag_mode,
            recency_weight=recency_weight if recency_weight > 0 else None,
            provider=default_provider,
            model=default_model,
            task_id=task_id,
            max_total_seconds=max_total_seconds,
            batch_size=batch_size,
            crossref_concurrency=crossref_concurrency,
            screen_method=screen_method,
            screen_threshold=screen_threshold,
            max_papers_to_download=max_papers_to_download,
            databases=filtered_databases,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
        )

        # Build telemetry sink and attach to the request so each RAG mode
        # can read it via getattr(request, "telemetry_sink", None).
        # The SSE chat path never sets this field, so legacy code hits
        # the `or []` fallback and behaves identically to before.
        #
        # The ResponseMetadataCollector always runs (regardless of ctx) so
        # the final JSON response carries attempts/query_rephrasings/usage
        # even when MCP progress notifications get dropped.
        from perspicacite.rag.telemetry import ResponseMetadataCollector

        _response_collector = ResponseMetadataCollector()

        if ctx is not None:
            from perspicacite.mcp.progress_adapter import MCPProgressAdapter
            from perspicacite.rag.telemetry import CallbackTelemetrySink

            _progress_adapter = MCPProgressAdapter(ctx)
            _progress_sink = CallbackTelemetrySink(_progress_adapter.on_event)

            class _FanOutSink:
                """Fan-out wrapper: forwards each event to multiple sinks."""

                def __init__(self, *sinks: Any) -> None:
                    self._sinks = sinks
                    # Mirror events into a buffer so legacy code that reads
                    # ``sink.events`` keeps working.
                    self.events: list[dict] = []

                def append(self, event: dict) -> None:
                    self.events.append(event)
                    for s in self._sinks:
                        try:
                            s.append(event)
                        except Exception:
                            pass

                async def on_event_async(self, event: dict) -> None:
                    self.events.append(event)
                    for s in self._sinks:
                        try:
                            fn = getattr(s, "on_event_async", None)
                            if fn is not None:
                                await fn(event)
                            else:
                                s.append(event)
                        except Exception:
                            pass

            rag_request.telemetry_sink = _FanOutSink(  # type: ignore[attr-defined]
                _progress_sink, _response_collector
            )
        else:
            rag_request.telemetry_sink = _response_collector  # type: ignore[attr-defined]

        # Hard wall-clock cap for long-running modes (profound, literature_survey).
        # Prevents the MCP tool from blocking indefinitely when a single RAG cycle
        # takes longer than expected. Default: 600 s (10 min).
        _generate_report_timeout_s: float = 600.0
        if mode in ("profound", "deep_research"):
            _generate_report_timeout_s = float(
                getattr(rag_request, "max_total_seconds", None) or 540.0
            )

        report_iterations: int | None = None
        report_completion_reason: str | None = None
        report_diagnostic: dict | None = None

        cancelled_reason: str | None = None
        try:
            async with asyncio.timeout(_generate_report_timeout_s):
                async for event in engine.query_stream(rag_request, message_id=message_id):
                    if event.event == "content":
                        import json as _json

                        payload = _json.loads(event.data)
                        report_text += payload.get("delta", "")
                    elif event.event == "source":
                        import json as _json

                        src = _json.loads(event.data)
                        sources.append(
                            {
                                "title": src.get("title"),
                                "authors": src.get("authors"),
                                "year": src.get("year"),
                                "doi": src.get("doi"),
                                "relevance_score": src.get("relevance_score"),
                                "section": src.get("section"),
                                "kb_name": src.get("kb_name"),
                                "metadata": src.get("metadata"),
                            }
                        )
                    elif event.event == "metadata":
                        import json as _json

                        _meta = _json.loads(event.data) if isinstance(event.data, str) else {}
                        report_iterations = _meta.get("iteration_count")
                        report_completion_reason = _meta.get("completion_reason")
                    elif event.event == "diagnostic":
                        import json as _json

                        report_diagnostic = _json.loads(event.data) if isinstance(event.data, str) else {}
                    elif event.event == "error":
                        # Modes signal cancellation by yielding an error event with
                        # ``reason="cancelled"``. Surface this as a structured response
                        # field so MCP clients can distinguish a cancelled partial
                        # result from a normally-completed report. Other error events
                        # (e.g. embedding-mismatch) flow through unchanged.
                        import json as _json

                        _err = _json.loads(event.data) if isinstance(event.data, str) else {}
                        if _err.get("reason") == "cancelled":
                            cancelled_reason = "cancelled"
                            break
        except asyncio.TimeoutError:
            logger.warning(
                "mcp_generate_report_timeout",
                query=query,
                mode=mode,
                timeout_s=_generate_report_timeout_s,
                partial_chars=len(report_text),
            )
            cancelled_reason = "time_budget_exceeded"

        asb_md = build_asb_response_metadata(
            [{"metadata": s.get("metadata") if isinstance(s, dict) else None} for s in sources]
        )

        def _build_provenance(cycles: int) -> dict[str, Any]:
            """Provenance envelope flagging which response fields are LLM-authored.

            ``report`` is model synthesis, not source-verified text. Host agents
            otherwise see it next to factual ``sources``/``papers_used`` and
            assume the whole report is citation-grounded. Every retrieved source
            is attacker-influenceable (a poisoned preprint can steer synthesis),
            so all of them are listed as untrusted.
            """
            return {
                "provider": default_provider,
                "model": default_model,
                "rag_cycles_executed": cycles,
                "ai_generated_fields": ["report"],
                "untrusted_sources": [
                    (s.get("doi") or s.get("title"))
                    for s in sources
                    if isinstance(s, dict) and (s.get("doi") or s.get("title"))
                ],
            }

        if cancelled_reason == "cancelled":
            logger.info(
                "mcp_generate_report_cancelled",
                query=query,
                task_id=task_id,
                partial_chars=len(report_text),
            )
            _cancelled_payload = {
                "query": query,
                "kb_name": effective_kb_name,
                "kb_names": effective_kb_names,
                "mode": mode,
                "report": report_text,  # partial, may be empty
                "sources": sources,
                "papers_used": len(sources),
                "message_id": message_id,
                "cancelled": True,
                "task_id": task_id,
                "asb_metadata": asb_md,
                "iteration_count": report_iterations if report_iterations is not None else 0,
                "completion_reason": report_completion_reason or "cancelled",
                "diagnostic": report_diagnostic,
                "_provenance": _build_provenance(
                    report_iterations if report_iterations is not None else 0
                ),
            }
            _cancelled_payload.update(_response_collector.as_response_extras())
            return _json_ok(_cancelled_payload)

        logger.info("mcp_generate_report", query=query, kb_name=effective_kb_name, mode=mode)

        indicia = None
        claims_valid: bool | None = None
        validation_report: str | None = None
        if extract_claims:
            try:
                from perspicacite.pipeline.claims import extract_claims as _extract_claims, validate_claims
                _passages = [
                    {
                        "chunk_text": s.get("section") or s.get("title", ""),
                        "source": {"doi": s.get("doi"), "title": s.get("title")},
                    }
                    for s in sources
                ]

                # Resolve domain adapter(s) — compose if multiple (best-effort)
                domain_adapter = None
                if domains:
                    try:
                        from indicium_adapters import discover_adapters, compose_adapters
                        discovered = discover_adapters()
                        valid = [discovered[d] for d in domains if d in discovered]
                        domain_adapter = compose_adapters(valid) if valid else None
                    except ImportError:
                        pass  # indicium-adapters not installed — proceed without adapter

                indicia = await _extract_claims(
                    llm_client=state.llm_client, passages=_passages, context=query,
                    domain_adapter=domain_adapter)

                conforms, val_report = validate_claims(indicia, domain_adapter=domain_adapter) if indicia else (True, "")
                claims_valid = conforms
                if not conforms:
                    validation_report = val_report
            except ImportError:
                indicia = None

        _final_payload = {
            "query": query,
            "kb_name": effective_kb_name,
            "kb_names": effective_kb_names,
            "mode": mode,
            "report": report_text,
            "sources": sources,
            "papers_used": len(sources),
            "message_id": message_id,
            "asb_metadata": asb_md,
            "iteration_count": report_iterations if report_iterations is not None else 1,
            "completion_reason": report_completion_reason or "complete",
            "diagnostic": report_diagnostic,
            "_provenance": _build_provenance(
                report_iterations if report_iterations is not None else 1
            ),
        }
        _final_payload.update(_response_collector.as_response_extras())
        if indicia is not None:
            _final_payload["indicia"] = indicia
            _final_payload["claims_valid"] = claims_valid
            if validation_report is not None:
                _final_payload["validation_report"] = validation_report
        return _json_ok(_final_payload)

    except Exception as e:
        logger.error("mcp_generate_report_error", query=query, error=str(e))
        return _json_error(f"Report generation failed: {e}")
    finally:
        if _sampling_token is not None:
            _sampling_ctx.reset(_sampling_token)


# =============================================================================
# Tool 9: screen_papers
# =============================================================================


@mcp.tool()
async def screen_papers(
    candidates: list[str] | list[dict],
    query: str,
    method: str = "bm25",
    threshold: float = 0.3,
    max_results: int = 50,
    ctx: Context | None = None,
) -> str:
    """Score candidate papers by relevance to a research query.

    Each item in ``candidates`` may be either:
      - A plain **string** — a DOI (e.g. "10.1038/nature12373") or a paper
        title. DOIs trigger an OA-content lookup so the abstract can be
        used in scoring.
      - A **dict** — {"doi": "...", "title": "...", "abstract": "..."}.
        Use this when you already have the abstract and want to skip the
        DOI lookup.

    Args:
        candidates: Strings (DOIs or titles) OR dicts with doi/title/abstract.
        query: The research query / topic to screen against.
        method: "bm25" (fast, no LLM) or "llm" (LLM-rated 0-1 with one-line reasons).
        threshold: Keep papers scoring >= this value (0..1).
        max_results: Cap on the number of returned items.

    Returns:
        JSON with keys: query, method, screened (list of
        doi/title/score/kept/reason).
    """
    state = _require_state()
    if isinstance(state, str):
        return state
    try:
        import httpx

        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.search.screening import screen_papers as _bm25
        from perspicacite.search.screening import screen_papers_llm as _llm

        items: list[dict] = []
        # Split dicts (already have metadata) from strings (need lookup).
        dicts_already = [c for c in candidates if isinstance(c, dict)]
        strings_only = [c for c in candidates if isinstance(c, str)]

        # Accept user-supplied metadata dicts as-is.
        for d in dicts_already:
            items.append(
                {
                    "doi": (d.get("doi") or "").strip().replace("https://doi.org/", "") or None,
                    "title": d.get("title") or d.get("doi") or "(untitled)",
                    "abstract": d.get("abstract") or "",
                }
            )

        # Only spin up an HTTP client if at least one string looks like a DOI.
        doi_like = [
            c for c in strings_only if c.strip().lower().startswith("10.") or "doi.org/" in c
        ]
        if doi_like:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                for c in strings_only:
                    if c.strip().lower().startswith("10.") or "doi.org/" in c:
                        doi = c.strip().replace("https://doi.org/", "")
                        try:
                            r = await retrieve_paper_content(
                                doi, http_client=client, pdf_parser=None
                            )
                            md = r.metadata or {}
                            items.append(
                                {
                                    "doi": doi,
                                    "title": md.get("title") or doi,
                                    "abstract": r.abstract or md.get("abstract") or "",
                                }
                            )
                        except Exception:
                            items.append({"doi": doi, "title": doi, "abstract": ""})
                    else:
                        items.append({"title": c, "abstract": ""})
        else:
            items.extend([{"title": c, "abstract": ""} for c in strings_only])

        if method == "llm":
            from perspicacite.llm.client import resolve_stage_model
            from perspicacite.llm.mcp_sampling import use_mcp_context

            sp, sm = resolve_stage_model(state.config, "screening")
            with use_mcp_context(ctx):
                results = await _llm(
                    items,
                    query=query,
                    llm=state.llm_client,
                    threshold=threshold,
                    model=sm,
                    provider=sp,
                )
        else:
            results = _bm25(items, reference=query, method="bm25", threshold=threshold)

        screened = []
        for r in results[:max_results]:
            entry: dict = {"score": r.score, "kept": r.kept, "reason": r.reason}
            if r.item.get("doi"):
                entry["doi"] = r.item["doi"]
            entry["title"] = r.item.get("title")
            screened.append(entry)
        logger.info(
            "mcp_screen_papers",
            n=len(candidates),
            method=method,
            kept=sum(e["kept"] for e in screened),
        )
        return _json_ok({"query": query, "method": method, "screened": screened})
    except Exception as e:
        logger.error("mcp_screen_papers_error", error=str(e))
        return _json_error(f"Screening failed: {e}")


# =============================================================================
# Tool 10: add_dois_to_kb
# =============================================================================


def _add_dois_warnings(added: int, failed: list, skipped: list) -> list[str]:
    """Caller-facing warnings for ``add_dois_to_kb`` so a 0-paper outcome is not
    mistaken for success: surfaces the all-failed / all-duplicate / partial-failure
    cases (the KB may be empty or unchanged despite ``success: true``)."""
    warnings: list[str] = []
    n_failed, n_skipped = len(failed), len(skipped)
    if added == 0:
        if n_failed:
            warnings.append(
                f"No papers could be fetched: {n_failed} DOI(s) failed, so the KB gained no "
                "papers. Check OA availability or try a preprint DOI (see 'failed')."
            )
        if n_skipped:
            warnings.append(
                f"{n_skipped} DOI(s) were already in the KB (skipped); nothing new added."
            )
    elif n_failed:
        warnings.append(
            f"{n_failed} of the requested DOI(s) failed to fetch (see 'failed')."
        )
    return warnings


@mcp.tool()
async def add_dois_to_kb(
    kb_name: str,
    dois: list[str],
) -> str:
    """
    Bulk-add papers to a knowledge base from a list of DOIs.

    For each DOI the tool fetches full text via the unified download pipeline,
    deduplicates against existing KB content, and indexes the result.

    **Latency:** 5-30s per DOI (resolve + fetch + embed). Scale with batch size; use >=120s for small batches.

    Args:
        kb_name: Target knowledge base name
        dois: List of DOIs to add (max 200 per call)

    Returns:
        JSON with added_papers, added_chunks, skipped_duplicates, failed, pdf_download stats.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    if len(dois) > 200:
        return _json_error("At most 200 DOIs per request")

    try:
        from perspicacite.models.kb import chroma_collection_name_for_kb
        from perspicacite.models.papers import Author, Paper, PaperSource
        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

        kb_meta = await state.session_store.get_kb_metadata(kb_name)
        if not kb_meta:
            return _json_error(f"Knowledge base '{kb_name}' not found")

        collection_name = chroma_collection_name_for_kb(kb_name)

        pdf_config = state.config.pdf_download
        pdf_kwargs: dict[str, Any] = {}
        cookies_path: str | None = None
        if pdf_config:
            pdf_kwargs = {
                "unpaywall_email": pdf_config.unpaywall_email,
                "alternative_endpoint": pdf_config.alternative_endpoint,
                "wiley_tdm_token": pdf_config.wiley_tdm_token,
                "aaas_api_key": pdf_config.aaas_api_key,
                "rsc_api_key": pdf_config.rsc_api_key,
                "springer_api_key": pdf_config.springer_api_key,
            }
            if pdf_config.cache_pdfs:
                pdf_kwargs["pdf_cache_dir"] = pdf_config.cache_dir
            cookies_path = pdf_config.cookies_path

        papers_to_add: list[Paper] = []
        skipped: list[dict] = []
        failed: list[dict] = []
        metadata_only: list[dict] = []  # F-28/F-30
        dl: dict[str, int] = {"attempted": 0, "success": 0, "failed": 0, "metadata_only": 0}

        from perspicacite.pipeline.download.cookies import (
            build_authenticated_client,
        )

        async with build_authenticated_client(cookies_path=cookies_path) as client:
            for raw_doi in dois:
                doi = (raw_doi or "").strip().replace("https://doi.org/", "")
                if not doi:
                    continue

                if await state.vector_store.paper_exists(collection_name, doi):
                    skipped.append({"doi": doi})
                    continue

                dl["attempted"] += 1
                try:
                    result = await retrieve_paper_content(
                        doi,
                        http_client=client,
                        pdf_parser=state.pdf_parser,
                        # When the user has cookies configured, let the DOI
                        # path fall back to a cookie-authenticated landing
                        # capture (parity with ingest_url for paywalled papers).
                        enable_landing_capture=bool(cookies_path),
                        **pdf_kwargs,
                    )
                except Exception as e:
                    failed.append({"doi": doi, "reason": str(e)})
                    dl["failed"] += 1
                    continue

                if not result or not result.success:
                    attempts = list(getattr(result, "attempts", []) or [])
                    failed.append(
                        {
                            "doi": doi,
                            "reason": "; ".join(f"{a['source']}:{a['status']}" for a in attempts)
                            or "no content",
                            "attempts": attempts,
                        }
                    )
                    dl["failed"] += 1
                    continue

                md = result.metadata or {}
                paper = Paper(
                    id=doi,
                    title=md.get("title") or f"Reference {doi}",
                    authors=[Author(name=a) for a in (md.get("authors") or [])],
                    year=md.get("year"),
                    doi=doi,
                    abstract=result.abstract or md.get("abstract"),
                    journal=md.get("journal"),
                    source=PaperSource.OPENALEX,
                    content_type=getattr(result, "content_type", None),
                    license=md.get("license"),
                )
                if result.full_text:
                    paper.full_text = result.full_text
                    dl["success"] += 1
                else:
                    dl["metadata_only"] += 1
                    metadata_only.append(
                        {
                            "doi": doi,
                            "content_type": paper.content_type,
                            "attempts": list(getattr(result, "attempts", []) or []),
                        }
                    )
                papers_to_add.append(paper)

        added_with_full_text = sum(1 for p in papers_to_add if getattr(p, "full_text", None))
        added_metadata_only = len(papers_to_add) - added_with_full_text
        if not papers_to_add:
            return _json_ok(
                {
                    "kb_name": kb_name,
                    "added_papers": 0,
                    "added_with_full_text": 0,
                    "added_metadata_only": 0,
                    "added_chunks": 0,
                    "skipped_duplicates": len(skipped),
                    "failed": failed,
                    "metadata_only": metadata_only,
                    "pdf_download": dl,
                    "warnings": _add_dois_warnings(len(papers_to_add), failed, skipped),
                }
            )

        dkb = DynamicKnowledgeBase(
            state.vector_store,
            state.embedding_provider,
        )
        dkb.collection_name = collection_name
        dkb._initialized = True

        chunks_added = await dkb.add_papers(papers_to_add, include_full_text=True)

        kb_meta.paper_count = (kb_meta.paper_count or 0) + len(papers_to_add)
        kb_meta.chunk_count = (kb_meta.chunk_count or 0) + chunks_added
        await state.session_store.save_kb_metadata(kb_meta)

        logger.info(
            "mcp_add_dois_to_kb",
            kb_name=kb_name,
            papers=len(papers_to_add),
            chunks=chunks_added,
        )

        return _json_ok(
            {
                "kb_name": kb_name,
                "added_papers": len(papers_to_add),
                "added_with_full_text": added_with_full_text,
                "added_metadata_only": added_metadata_only,
                "added_chunks": chunks_added,
                "skipped_duplicates": len(skipped),
                "failed": failed,
                "metadata_only": metadata_only,
                "pdf_download": dl,
                "warnings": _add_dois_warnings(len(papers_to_add), failed, skipped),
            }
        )

    except Exception as e:
        logger.error("mcp_add_dois_to_kb_error", kb_name=kb_name, error=str(e))
        return _json_error(f"Failed to add DOIs: {e}")


# =============================================================================
# Tool 11: push_to_zotero
# =============================================================================


@mcp.tool()
async def push_to_zotero(
    dois: list[str] | str | None = None,
    items: list[dict] | None = None,
    library_id: str | None = None,
    collection_key: str | None = None,
    attach_pdf: bool = False,
    attach_supplementary: bool = False,
    youtube_correct: bool = False,
) -> str:
    """Push one or more papers to a Zotero library — by DOI, URL, or BibTeX.

    Fetches metadata via the unified pipeline and calls
    :meth:`ZoteroClient.create_item` for each item. Skips duplicates
    automatically (by DOI when present, else by URL); the dedup is
    immune to Zotero's eventually-consistent search index thanks to a
    recent-items fallback scan.

    Three input routes (mix freely in ``items`` or use the DOI shortcut
    ``dois`` for the legacy single-route call):

    1. **DOI route** — ``{"doi": "10.xxxx/yyy"}`` (or just a bare string
       in ``dois``). Fetches metadata + optionally PDF via the unified
       pipeline; creates a ``journalArticle`` (or ``preprint`` for
       ``10.48550/arXiv.*`` DOIs).

    2. **URL route** — ``{"url": "https://example.com/...",
       "title": "...", "authors": [...]}``. For pages without metadata
       this is degraded — the caller is expected to supply at least a
       title (the URL ingest pipeline in ``ingest_url`` will mine the
       page for full metadata if needed). Creates a ``webpage`` item by
       default; pass ``"item_type": "computerProgram"`` for GitHub
       repos.

    3. **BibTeX route** — ``{"bibtex": "@misc{key, title={...}, ...}"}``.
       Parsed locally via bibtexparser; treated like the DOI route
       when a ``doi`` field is present, else URL route. For entries
       with no DOI and no usable URL, a title-based resolver walks
       OpenAlex -> Crossref -> Semantic Scholar -> arXiv to recover a
       DOI. Setting ``PERSPICACITE_HEADLESS_BROWSER=1`` adds a fifth
       Chromium -> Google Scholar tier (requires the ``[browser]``
       extra + ``playwright install chromium``).

    **Agent-side hint:** if you have a browser MCP available (e.g.,
    ``claude-in-chrome``), you can pre-resolve a title to a DOI yourself
    by searching Google Scholar in the browser, then pass the DOI
    directly to this tool. That works without the optional Chromium
    extra and lets you confirm ambiguous matches visually.

    Optionally attaches the cached PDF and/or supplementary files (only
    works for DOI-route items today; URL-route items get HTML capture
    via the upcoming ``ingest_url`` tool / Priority 3b HTML fallback).

    Cloud-only — the local desktop API rejects writes and attachment
    upload via the documented 3-step protocol. Group libraries also
    require the cloud API.

    Args:
        dois: Legacy: a single DOI string or list of DOIs (max 100).
            Convenience for the DOI-only call shape; equivalent to
            passing ``items=[{"doi": ...}, ...]``.
        items: Mixed list of input dicts. Each dict carries at least
            one of: ``doi``, ``url``, ``bibtex``. Optional fields:
            ``title``, ``authors``, ``year``, ``abstract``, ``journal``,
            ``item_type``, ``repository``, ``archive_id``, ``url``,
            ``tags``. Max 100 items per call.
        library_id: Override ``config.yml`` ``zotero.library_id`` for
            this call. Useful for multi-group pushes without restart.
        collection_key: Override ``config.yml`` ``zotero.collection_key``.
        attach_pdf: If True, upload the cached PDF as a child attachment
            (DOI route only).
        attach_supplementary: If True, upload any
            ``data/capsules/<paper_id>/supplementary/files/*`` as
            additional child attachments.
        youtube_correct: For YouTube URLs, run LLM cleanup on the
            auto-captions before attaching (default ``False``). When
            False the rendered transcript Markdown carries a warning
            header so downstream KB chunks are tagged as
            auto-captions.

    Returns:
        JSON ``{"created": [...], "skipped": [], "failed": [...]}`` where
        each entry carries the original input plus ``{"key": "...",
        "attached_pdf"?, "attached_supplementary"?, "attached_html"?}``.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    cfg = getattr(state.config, "zotero", None)
    if cfg is None or not cfg.enabled or not cfg.api_key:
        return _json_error("zotero_not_configured")
    # library_id / collection_key may come from the call or from config.
    effective_library_id = library_id or cfg.library_id
    effective_collection_key = (
        collection_key if collection_key is not None else (cfg.collection_key or "")
    )
    if not effective_library_id:
        return _json_error("library_id required (pass via argument or set zotero.library_id)")

    # Normalize the two input shapes (dois shortcut, items list) into a
    # uniform list of input dicts.
    inputs: list[dict] = []
    if items:
        inputs.extend(items)
    if dois is not None:
        doi_list = [dois] if isinstance(dois, str) else dois
        inputs.extend({"doi": d} for d in doi_list)
    if not inputs:
        return _json_error("either dois or items must be provided")
    if len(inputs) > 100:
        return _json_error("at most 100 items per call")

    pdf_config = state.config.pdf_download
    cache_dir = pdf_config.cache_dir if (pdf_config and pdf_config.cache_pdfs) else None
    cookies_path = pdf_config.cookies_path if pdf_config else None

    try:
        from perspicacite.integrations.zotero import ZoteroClient
        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.pipeline.download.cookies import (
            build_authenticated_client,
        )
        from perspicacite.pipeline.download.pdf_cache import (
            cached_pdf_path,
        )

        created: list[dict] = []
        failed: list[dict] = []

        async with build_authenticated_client(cookies_path=cookies_path) as http_client:
            zotero = ZoteroClient(
                api_key=cfg.api_key,
                library_id=effective_library_id,
                library_type=cfg.library_type,
                collection_key=effective_collection_key,
                base_url=getattr(cfg, "base_url", "") or None,
                http_client=http_client,
            )
            for inp in inputs:
                # Resolve the input to a `paper` dict via the appropriate
                # route (DOI / URL / BibTeX). Errors are recorded against
                # the original input so the caller can correlate.
                route_err: str | None = None
                doi: str = ""
                url: str = ""
                paper: dict[str, Any] = {}
                try:
                    paper, doi, url = await _resolve_push_input(inp, http_client=http_client)
                except Exception as exc:
                    route_err = str(exc)
                if route_err is not None:
                    failed.append({"input": inp, "reason": route_err})
                    continue
                identifier = doi or url or (paper.get("title") or "<no-id>")
                try:
                    key = await zotero.create_item(paper)
                    if not key:
                        failed.append({"input": inp, "reason": "no key returned"})
                        continue
                    entry: dict[str, Any] = {"key": key, "identifier": identifier}
                    if doi:
                        entry["doi"] = doi
                    if url:
                        entry["url"] = url

                    # Step 2: attachments. Two independent paths:
                    # - PDF attach (opt-in via attach_pdf, DOI route only).
                    # - HTML attach: automatic for URL-route items (where
                    #   the HTML is the only content available), or as a
                    #   fallback for DOI-route items when the PDF is
                    #   missing/too-large.
                    pdf_path = None
                    pdf_too_large = False
                    if attach_pdf and doi:
                        pdf_path = cached_pdf_path(doi, cache_dir) if cache_dir else None
                        if pdf_path is None and state.pdf_parser is not None:
                            # Trigger a full fetch (which also populates the
                            # cache for next time) before uploading.
                            await retrieve_paper_content(
                                doi,
                                http_client=http_client,
                                pdf_parser=state.pdf_parser,
                                unpaywall_email=pdf_config.unpaywall_email,
                                wiley_tdm_token=pdf_config.wiley_tdm_token,
                                aaas_api_key=pdf_config.aaas_api_key,
                                rsc_api_key=pdf_config.rsc_api_key,
                                springer_api_key=pdf_config.springer_api_key,
                                pdf_cache_dir=cache_dir,
                            )
                            pdf_path = cached_pdf_path(doi, cache_dir) if cache_dir else None
                        if pdf_path is None and cache_dir:
                            # retrieve_paper_content returned at the
                            # structured-text tier (e.g. arXiv HTML) before
                            # reaching the PDF tier — never caching a PDF.
                            # When the caller specifically asked for a PDF
                            # attachment, force a PDF-only fetch.
                            from perspicacite.pipeline.download.unified import (
                                download_paper_pdf,
                            )

                            await download_paper_pdf(
                                doi,
                                http_client=http_client,
                                unpaywall_email=pdf_config.unpaywall_email,
                                wiley_tdm_token=pdf_config.wiley_tdm_token,
                                aaas_api_key=pdf_config.aaas_api_key,
                                rsc_api_key=pdf_config.rsc_api_key,
                                springer_api_key=pdf_config.springer_api_key,
                                pdf_cache_dir=cache_dir,
                            )
                            pdf_path = cached_pdf_path(doi, cache_dir)
                        # Size-cap check (Priority 3b extension): if the
                        # PDF exceeds the configured max_pdf_attach_bytes,
                        # skip the upload and fall through to HTML capture.
                        # Saves Zotero quota on huge review articles
                        # (Chem Rev surveys can be 50+ MB) where the user
                        # usually has the file locally and just needs the
                        # bibliographic record + landing-page snapshot.
                        pdf_too_large = False
                        max_pdf_bytes = getattr(pdf_config, "max_pdf_attach_bytes", 0) or 0
                        if pdf_path is not None and max_pdf_bytes > 0:
                            try:
                                size_b = pdf_path.stat().st_size
                                if size_b > max_pdf_bytes:
                                    pdf_too_large = True
                                    entry["attached_pdf"] = False
                                    entry["pdf_attach_skipped"] = (
                                        f"pdf_too_large ({size_b} bytes "
                                        f"> max_pdf_attach_bytes {max_pdf_bytes})"
                                    )
                                    entry["pdf_size_bytes"] = size_b
                            except OSError:
                                pass

                        if pdf_path is not None and not pdf_too_large:
                            try:
                                att_key = await zotero.upload_attachment(
                                    parent_item_key=key,
                                    file_path=str(pdf_path),
                                    filename=pdf_path.name,
                                    content_type="application/pdf",
                                )
                                entry["attached_pdf"] = bool(att_key)
                            except Exception as exc:
                                entry["pdf_attach_error"] = str(exc)
                        elif pdf_path is None:
                            entry["attached_pdf"] = False
                            entry["pdf_attach_error"] = "no PDF available"

                    # YouTube special-case: when the item URL is a
                    # YouTube video, attach the LLM-corrected transcript
                    # (as Markdown) instead of the generic HTML capture.
                    # This gives ASB / KB consumers searchable spoken
                    # content with [mm:ss] timestamps. Falls through to
                    # the HTML path on any error.
                    attached_transcript = False
                    if paper.get("url") or url:
                        from perspicacite.pipeline.download.youtube import (
                            fetch_youtube_transcript,
                            is_youtube_url,
                        )

                        target_url = paper.get("url") or url
                        if is_youtube_url(target_url):
                            try:
                                md, _yt_title = await fetch_youtube_transcript(
                                    target_url,
                                    http_client=http_client,
                                    llm_client=state.llm_client,
                                    correct_with_llm=youtube_correct,
                                )
                                import re as _re_yt
                                from pathlib import Path as _Path

                                if cache_dir:
                                    yt_dir = _Path(cache_dir).expanduser() / "youtube"
                                else:
                                    yt_dir = _Path.home() / ".cache" / "perspicacite" / "youtube"
                                yt_dir.mkdir(parents=True, exist_ok=True)
                                slug = _re_yt.sub(
                                    r"[^a-zA-Z0-9.-]+",
                                    "_",
                                    target_url.lower(),
                                )[:120]
                                yt_path = yt_dir / f"{slug}.md"
                                yt_path.write_text(md, encoding="utf-8")
                                att_key = await zotero.upload_attachment(
                                    parent_item_key=key,
                                    file_path=str(yt_path),
                                    filename=yt_path.name,
                                    content_type="text/markdown",
                                )
                                entry["attached_transcript"] = bool(att_key)
                                entry["transcript_chars"] = len(md)
                                attached_transcript = True
                            except Exception as exc:
                                entry["transcript_attach_error"] = str(exc)

                    # HTML attach: always for URL-route items, or as a
                    # fallback when the requested PDF couldn't be
                    # attached. The same capture path handles both —
                    # ``capture_landing_html`` falls back to a
                    # bibliographic stub when the live page is blocked.
                    # Skipped for YouTube items that already got a
                    # transcript attachment above.
                    need_html = not attached_transcript and (
                        (not doi) or (attach_pdf and (pdf_path is None or pdf_too_large))
                    )
                    if need_html:
                        try:
                            from perspicacite.pipeline.download.html_capture import (
                                capture_landing_html,
                            )

                            html_attach = await capture_landing_html(
                                doi=doi,
                                landing_url=paper.get("url") or url,
                                abstract=paper.get("abstract") or "",
                                title=paper.get("title") or "",
                                http_client=http_client,
                                cache_dir=cache_dir,
                            )
                            if html_attach is not None:
                                att_key = await zotero.upload_attachment(
                                    parent_item_key=key,
                                    file_path=str(html_attach.path),
                                    filename=html_attach.path.name,
                                    content_type="text/html",
                                )
                                entry["attached_html"] = bool(att_key)
                                entry["html_source"] = html_attach.tier
                                entry["html_chars"] = html_attach.char_count
                        except Exception as exc:
                            entry["html_attach_error"] = str(exc)

                    # Step 3 (optional): supplementary attachments from capsule.
                    # DOI route only — URL-route items don't have a capsule path.
                    if attach_supplementary and doi:
                        from pathlib import Path

                        si_dir = (
                            Path(state.config.capsule.root)
                            / doi.replace("/", "_")
                            / "supplementary"
                            / "files"
                        )
                        attached_si: list[str] = []
                        si_errors: list[dict] = []
                        if si_dir.exists():
                            for f in sorted(si_dir.glob("*")):
                                if not f.is_file():
                                    continue
                                try:
                                    att_key = await zotero.upload_attachment(
                                        parent_item_key=key,
                                        file_path=str(f),
                                        filename=f.name,
                                    )
                                    if att_key:
                                        attached_si.append(f.name)
                                except Exception as exc:
                                    si_errors.append({"file": f.name, "error": str(exc)})
                        entry["attached_supplementary"] = attached_si
                        if si_errors:
                            entry["si_attach_errors"] = si_errors

                    created.append(entry)
                except Exception as exc:
                    failed.append({"input": inp, "reason": str(exc)})

        logger.info(
            "mcp_push_to_zotero",
            created=len(created),
            failed=len(failed),
            attach_pdf=attach_pdf,
            attach_supplementary=attach_supplementary,
        )
        return _json_ok({"created": created, "skipped": [], "failed": failed})

    except Exception as e:
        logger.error("mcp_push_to_zotero_error", error=str(e))
        return _json_error(f"Failed to push to Zotero: {e}")


# =============================================================================
# Tool 11b: push_notes_to_zotero
# =============================================================================


@mcp.tool()
async def push_notes_to_zotero(
    notes: list[dict],
    library_id: str | None = None,
) -> str:
    """Attach text notes to existing Zotero items.

    Each entry in ``notes`` must supply **content** and a way to identify
    the parent item — either ``item_key`` (direct Zotero key) or ``doi``
    (looked up in the library first):

    .. code-block:: json

        [
          {"item_key": "ABCD1234", "content": "Key finding: …", "tags": ["ai-forge"]},
          {"doi": "10.1234/xyz",   "content": "Reviewer note: …"}
        ]

    Fields per entry:

    - **content** *(str, required)* — plain text or Markdown; converted to
      Zotero note HTML automatically.
    - **item_key** *(str)* — direct Zotero item key.  Use when you already
      have the key from ``push_to_zotero`` or ``zotero_get_collection_items``.
    - **doi** *(str)* — DOI of the parent item; resolved to a Zotero key via
      a library search.  Ignored when ``item_key`` is also provided.
    - **tags** *(list[str], optional)* — Zotero tags to attach to the note.

    Returns a JSON object with ``created`` (list of ``{note_key, parent_key}``)
    and ``failed`` (list of ``{input, reason}``) arrays.

    **Typical use-cases**

    - Attach a Perspicacité RAG summary to the Zotero record of a paper so
      the annotation survives outside the KB.
    - Record reviewer feedback, decision rationale, or usage notes alongside
      the reference.
    - Export proposal-drafting annotations back to Zotero for archival.
    """
    try:
        cfg = mcp_state.config
        zotero_cfg = cfg.zotero if cfg else None
        if not zotero_cfg or not zotero_cfg.api_key:
            return _json_error(
                "Zotero is not configured. Set zotero.api_key and zotero.library_id in config.yml."
            )

        from perspicacite.integrations.zotero import ZoteroClient

        lib_id = library_id or str(zotero_cfg.library_id or "")
        lib_type = getattr(zotero_cfg, "library_type", "user") or "user"

        client = ZoteroClient(
            api_key=zotero_cfg.api_key,
            library_id=lib_id,
            library_type=lib_type,
            base_url=getattr(zotero_cfg, "base_url", "") or None,
        )

        created: list[dict] = []
        failed: list[dict] = []

        for entry in notes:
            content = entry.get("content")
            if not content:
                failed.append({"input": entry, "reason": "missing required field: content"})
                continue

            item_key = entry.get("item_key")
            doi = entry.get("doi")
            tags = entry.get("tags") or []

            # Resolve parent key via DOI when item_key not provided
            if not item_key:
                if not doi:
                    failed.append(
                        {
                            "input": entry,
                            "reason": "must supply item_key or doi to identify the parent item",
                        }
                    )
                    continue
                item_key = await client._find_existing_by_doi(doi)
                if not item_key:
                    failed.append(
                        {
                            "input": entry,
                            "reason": f"DOI {doi!r} not found in library {lib_id}",
                        }
                    )
                    continue

            try:
                note_key = await client.create_note(
                    parent_item_key=item_key,
                    content=content,
                    tags=tags,
                )
                created.append({"note_key": note_key, "parent_key": item_key})
            except Exception as exc:
                failed.append({"input": entry, "reason": str(exc)})

        logger.info(
            "mcp_push_notes_to_zotero",
            created=len(created),
            failed=len(failed),
        )
        return _json_ok({"created": created, "failed": failed})

    except Exception as e:
        logger.error("mcp_push_notes_to_zotero_error", error=str(e))
        return _json_error(f"Failed to push notes to Zotero: {e}")


# =============================================================================
# Tool 11c: ingest_url
# =============================================================================


@mcp.tool()
async def ingest_url(
    url: str,
    push_to_zotero_collection: str | None = None,
    library_id: str | None = None,
    attach_html: bool = True,
) -> str:
    """Ingest a URL into Zotero (and optionally a KB) — HTML-first.

    Closes the no-DOI gap for vendor docs (Anthropic blog posts, GitHub
    READMEs, OpenReview entries, preprints.org pages, generic publisher
    landing pages). Extracts metadata from the page's ``citation_*`` /
    OpenGraph tags, plus URL-pattern-specific routes for GitHub and
    OpenReview that use the publisher API directly.

    When ``push_to_zotero_collection`` is set, a Zotero item is created
    in that collection (under ``library_id`` or the configured library).
    When ``attach_html`` is True (default), an HTML snapshot of the
    landing page is captured and attached using the same three-tier
    classifier as the PDF-fallback path (``full_text_html`` /
    ``extended_abstract`` / ``bibliographic_stub``).

    Args:
        url: The page URL. Routes:
            - ``github.com/<owner>/<repo>`` → repo + README via GitHub API.
            - ``openreview.net/forum?id=*`` → note via OpenReview API.
            - ``preprints.org/manuscript/*`` → preprint metadata via meta tags.
            - everything else → generic ``citation_*`` / OG mining.
        push_to_zotero_collection: Zotero collection key to drop the item
            into. None = skip Zotero (extract metadata only).
        library_id: Override ``zotero.library_id`` for this call.
        attach_html: If True, also capture and attach an HTML snapshot.

    Returns:
        JSON ``{"url": ..., "item_type": ..., "title": ..., "doi": ...,
        "zotero_key"?: ..., "attached_html"?: bool, "html_tier"?: ...}``
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    from perspicacite.pipeline.download.cookies import build_authenticated_client
    from perspicacite.pipeline.download.html_capture import capture_landing_html
    from perspicacite.pipeline.download.url_extractors import extract_url

    pdf_config = state.config.pdf_download
    cookies_path = pdf_config.cookies_path if pdf_config else None
    cache_dir = pdf_config.cache_dir if (pdf_config and pdf_config.cache_pdfs) else None

    async with build_authenticated_client(cookies_path=cookies_path) as http_client:
        try:
            paper = await extract_url(url, http_client=http_client)
        except Exception as exc:
            return _json_error(f"url_extraction_failed: {exc}")

        result: dict[str, Any] = {
            "url": url,
            "item_type": paper.get("item_type"),
            "title": paper.get("title"),
            "doi": paper.get("doi") or "",
            "ingest_format": paper.get("ingest_format"),
        }

        if push_to_zotero_collection:
            cfg = getattr(state.config, "zotero", None)
            if cfg is None or not cfg.enabled or not cfg.api_key:
                return _json_error("zotero_not_configured")
            effective_library_id = library_id or cfg.library_id
            if not effective_library_id:
                return _json_error("library_id required for Zotero push")
            from perspicacite.integrations.zotero import ZoteroClient

            zotero = ZoteroClient(
                api_key=cfg.api_key,
                library_id=effective_library_id,
                library_type=cfg.library_type,
                collection_key=push_to_zotero_collection,
                base_url=getattr(cfg, "base_url", "") or None,
                http_client=http_client,
            )
            try:
                key = await zotero.create_item(paper)
                result["zotero_key"] = key
            except Exception as exc:
                result["zotero_error"] = str(exc)

            if attach_html and result.get("zotero_key"):
                try:
                    cap = await capture_landing_html(
                        doi=paper.get("doi") or "",
                        landing_url=url,
                        abstract=paper.get("abstract") or "",
                        title=paper.get("title") or "",
                        http_client=http_client,
                        cache_dir=cache_dir,
                    )
                    if cap is not None:
                        att_key = await zotero.upload_attachment(
                            parent_item_key=result["zotero_key"],
                            file_path=str(cap.path),
                            filename=cap.path.name,
                            content_type="text/html",
                        )
                        result["attached_html"] = bool(att_key)
                        result["html_tier"] = cap.tier
                        result["html_chars"] = cap.char_count
                except Exception as exc:
                    result["html_attach_error"] = str(exc)

        return _json_ok(result)


# =============================================================================
# Tool 12: build_kbs_from_zotero
# =============================================================================


@mcp.tool()
async def build_kbs_from_zotero(
    top_level_collection_keys: list[str] | None = None,
    include_unfiled: bool = True,
    plan_only: bool = False,
    library_id: str | None = None,
    library_type: str | None = None,
) -> dict[str, Any]:
    """Build one KB per Zotero top-level collection.

    Args:
        top_level_collection_keys: Optional filter. None = all top-level collections.
        include_unfiled: Also include items not in any collection (default True).
        plan_only: If True, return the preview only without executing.
        library_id: Optional override for zotero.library_id from config.
            Pass this to switch libraries per-call without restarting.
            e.g. for BioMedOmicsAI → "5691738", MetaboLinkAI → "5453037".
        library_type: Optional override for zotero.library_type ("user" or
            "group"). Defaults to the configured value.

    Returns either {"plan": [...]} (plan-only) or {"per_kb": [...]} (executed).
    Requires zotero.enabled=true and credentials in config.yml.
    """
    from perspicacite.integrations import zotero_ingest
    from perspicacite.integrations.zotero import ZoteroClient

    cfg = getattr(getattr(mcp_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled):
        return {"error": "Zotero not configured (zotero.enabled)"}

    # Allow per-call overrides for library_id / library_type so an agent
    # can drive multiple libraries (e.g. BioMedOmicsAI + MetaboLinkAI)
    # without restarting the server.
    eff_library_id = library_id or cfg.library_id
    eff_library_type = library_type or cfg.library_type
    if not eff_library_id:
        return {"error": "library_id required (pass as argument or set zotero.library_id)"}

    base_url = getattr(cfg, "base_url", "") or None
    is_local = base_url and ("localhost" in base_url or "127.0.0.1" in base_url)
    if not cfg.api_key and not is_local:
        return {"error": "Zotero api_key required for non-local base_url"}

    client = ZoteroClient(
        api_key=cfg.api_key,
        library_id=eff_library_id,
        library_type=eff_library_type,
        collection_key=cfg.collection_key,
        base_url=base_url,
    )
    # Resolve real group name so KB names are scoped per-library.
    library_name = await client.get_library_name() or "Library"
    plan = await zotero_ingest.plan_kbs_from_zotero(
        client,
        top_level_collection_keys=top_level_collection_keys,
        include_unfiled=include_unfiled,
        library_label=library_name,
    )
    if plan_only:
        return {"library_name": library_name, "plan": [p.model_dump() for p in plan]}

    # Inline execution with a no-op registry — MCP returns the final summary.
    class _InlineRegistry:
        def __init__(self) -> None:
            self.result: dict[str, Any] | None = None
            self.err: str | None = None

        async def publish(self, jid: str, ev: dict[str, Any]) -> None:
            return None

        async def finish(self, jid: str, res: dict[str, Any]) -> None:
            self.result = res

        async def fail(self, jid: str, err: str) -> None:
            self.err = err

    reg = _InlineRegistry()
    try:
        await zotero_ingest.build_kbs_from_zotero(
            client,
            plan=plan,
            app_state=mcp_state,
            registry=reg,
            job_id="mcp-inline",
        )
    except Exception as exc:
        return {"error": str(exc)}
    if reg.err is not None:
        return {"error": reg.err}
    return reg.result or {"per_kb": []}


@mcp.tool()
async def ingest_local_documents(
    kb_name: str,
    paths: list[str],
    recursive: bool = True,
) -> dict:
    """Ingest local files or directories into a KB.

    Files must be absolute paths under one of `local_docs.allowed_roots`.
    If allowed_roots is empty, this tool refuses all calls.
    """

    from perspicacite.integrations.local_docs import (
        LocalDocsDisabledError,
        LocalDocsValidationError,
        validate_local_path,
    )
    from perspicacite.integrations.local_docs import (
        ingest_local_documents as _ingest,
    )

    allowed = list(getattr(mcp_state.config.local_docs, "allowed_roots", []) or [])
    validated: list[Path] = []
    try:
        for raw in paths:
            validated.append(validate_local_path(raw, allowed_roots=allowed))
    except LocalDocsDisabledError as exc:
        return {"error": str(exc)}
    except LocalDocsValidationError as exc:
        return {"error": str(exc)}

    class _Reg:
        async def publish(self, jid, ev):
            pass

        async def finish(self, jid, res):
            self._res = res

        async def fail(self, jid, err):
            self._err = err

    reg = _Reg()
    return await _ingest(
        kb_name=kb_name,
        paths=validated,
        app_state=mcp_state,
        registry=reg,
        job_id="mcp-inline",
        recursive=recursive,
    )


@mcp.tool()
async def add_local_papers_to_kb(
    kb_name: str,
    papers: list[dict],
) -> str:
    """Add locally-stored documents to a KB with user-provided metadata.

    Bridges the gap between ``ingest_local_documents`` (full text, filename
    as title) and ``add_papers_to_kb`` (rich metadata, DOI-only download).
    This tool accepts both a local file path AND explicit metadata so the KB
    entry has a proper title, authors, year, and searchable full text.

    Each paper dict requires:
      - ``file``  (str) — absolute path to a local PDF, Markdown, or text file.
                          Must be under ``local_docs.allowed_roots`` in config.
      - ``title`` (str) — human-readable title shown in search results.

    Optional fields (all improve retrieval quality):
      - ``authors``  list[str]  e.g. ["Alice Smith", "Bob Jones"]
      - ``year``     int
      - ``abstract`` str        ingested as extra context alongside the full text
      - ``keywords`` list[str]
      - ``doi``      str        used as the stable paper_id when present

    When to use vs. alternatives:
      - Use ``ingest_local_documents`` when you don't have metadata and
        the filename is a sufficient identifier.
      - Use ``add_papers_to_kb`` when you have DOIs and want automatic
        PDF download from the web.
      - Use this tool when you have local files AND want proper metadata
        (proposal PDFs, lab reports, preprints without DOIs, etc.).

    Returns JSON with ``added_chunks``, ``papers_added``, and per-file status.
    """
    import hashlib

    from perspicacite.integrations.local_docs import (
        LocalDocsDisabledError,
        LocalDocsValidationError,
        validate_local_path,
    )
    from perspicacite.models.papers import Author, Paper, PaperSource
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

    state = _require_state()
    if isinstance(state, str):
        return state

    allowed = list(getattr(state.config.local_docs, "allowed_roots", []) or [])
    kb_meta = await state.session_store.get_kb_metadata(kb_name)
    if not kb_meta:
        return _json_error(f"Knowledge base '{kb_name}' not found")

    from perspicacite.models.kb import chroma_collection_name_for_kb

    collection_name = chroma_collection_name_for_kb(kb_name)
    dkb_config = KnowledgeBaseConfig(
        vector_size=state.embedding_provider.dimension,
        chunk_size=state.config.knowledge_base.chunk_size,
        chunk_overlap=state.config.knowledge_base.chunk_overlap,
        chunking_method=state.config.knowledge_base.chunking_method,
    )
    dkb = DynamicKnowledgeBase(state.vector_store, state.embedding_provider, config=dkb_config)
    dkb.collection_name = collection_name
    dkb._initialized = True

    results: list[dict] = []
    total_chunks = 0

    for pd in papers:
        raw_file = pd.get("file", "")
        title = pd.get("title", "")
        if not raw_file:
            results.append({"file": raw_file, "status": "error", "reason": "missing 'file' field"})
            continue
        if not title:
            results.append({"file": raw_file, "status": "error", "reason": "missing 'title' field"})
            continue

        try:
            fp = validate_local_path(raw_file, allowed_roots=allowed)
        except LocalDocsDisabledError as exc:
            return _json_error(str(exc))
        except LocalDocsValidationError as exc:
            results.append({"file": raw_file, "status": "error", "reason": str(exc)})
            continue

        # Parse full text from the local file.
        from perspicacite.integrations.local_docs import infer_content_type

        content_type, _ = infer_content_type(fp)
        full_text: str | None = None
        if content_type == "pdf":
            if state.pdf_parser is not None:
                try:
                    parsed = await state.pdf_parser.parse(fp)
                    full_text = parsed.text or None
                except Exception as exc:
                    results.append(
                        {"file": raw_file, "status": "error", "reason": f"PDF parse failed: {exc}"}
                    )
                    continue
        else:
            try:
                full_text = fp.read_text(encoding="utf-8", errors="replace") or None
            except Exception as exc:
                results.append(
                    {"file": raw_file, "status": "error", "reason": f"Read failed: {exc}"}
                )
                continue

        doi = pd.get("doi")
        paper_id = doi if doi else f"generated:{hashlib.md5(title.encode()).hexdigest()[:12]}"

        authors = [
            Author(name=a) if isinstance(a, str) else Author(**a) for a in pd.get("authors", [])
        ]
        abstract = pd.get("abstract")
        # Prepend abstract to full text so it's always retrievable even for
        # chunked documents where the first page may be split across chunks.
        if abstract and full_text:
            full_text = f"{abstract}\n\n{full_text}"
        elif abstract:
            full_text = abstract

        paper = Paper(
            id=paper_id,
            title=title,
            authors=authors,
            year=pd.get("year"),
            doi=doi,
            abstract=abstract,
            keywords=pd.get("keywords", []),
            source=PaperSource.LOCAL,
            full_text=full_text,
        )

        try:
            n = await dkb.add_papers([paper], include_full_text=True)
            total_chunks += n
            results.append({"file": raw_file, "title": title, "status": "ok", "chunks": n})
        except Exception as exc:
            results.append(
                {"file": raw_file, "title": title, "status": "error", "reason": str(exc)}
            )

    papers_ok = sum(1 for r in results if r["status"] == "ok")
    kb_meta.chunk_count = (kb_meta.chunk_count or 0) + total_chunks
    kb_meta.paper_count = (kb_meta.paper_count or 0) + papers_ok
    await state.session_store.save_kb_metadata(kb_meta)

    return _json_ok({"papers_added": papers_ok, "added_chunks": total_chunks, "results": results})


@mcp.tool()
async def ingest_urls_to_kb(
    kb_name: str,
    urls: list[str],
    youtube_correct: bool = False,
) -> dict:
    """Ingest arbitrary URLs into ``kb_name`` as searchable KB content.

    For each URL:

    - **GitHub repository URLs** (``github.com/owner/repo``) fetch the
      raw README via the GitHub API. The README's existing Markdown
      structure (headings, code blocks) flows directly into the
      heading-aware Markdown chunker.
    - **YouTube videos** fetch the public transcript via
      ``youtube-transcript-api`` and prepend a warning header noting
      the auto-caption origin + one-sentence context (title + channel)
      so downstream chunks carry the "may be garbled" signal. Pass
      ``youtube_correct=True`` to enable an LLM cleanup pass first
      (opt-in for cost; a 1-hour talk runs ~$0.10-0.50 to clean).
    - **Other URLs** fetch the HTML and convert to Markdown. With the
      optional ``[html-ingest]`` extra installed (trafilatura), the
      conversion strips boilerplate (nav, footers, ads) and preserves
      heading hierarchy. Without it, falls back to a basic
      BeautifulSoup text extraction.

    The converted Markdown is written to ``data/url_cache/<slug>.md``
    and then ingested via the regular local-docs path — so it picks
    up the same heading-aware chunking, embeddings, and KB-log
    bookkeeping as a hand-fed ``.md`` file would. Cache files are
    persisted across runs; re-ingesting the same URL re-fetches and
    overwrites.

    Returns ``{"added_chunks": N, "files": M, "results": [...]}``
    with per-URL status (``ok`` / ``fetch_failed`` / ``empty``).

    Args:
        kb_name: Target KB name (must exist; create with
            ``create_knowledge_base`` first).
        urls: Up to 50 URLs in one call.
        youtube_correct: Run LLM cleanup on YouTube auto-captions
            (default ``False`` to keep ingest free). When ``False``,
            the rendered Markdown carries a warning header so chunks
            are tagged as auto-captions for downstream consumers.
    """
    state = _require_state()
    if isinstance(state, str):
        return {"error": state}
    if not urls:
        return {"added_chunks": 0, "files": 0, "results": []}
    if len(urls) > 50:
        return {"error": "url batch too large (max 50)"}

    import re as _re
    from pathlib import Path

    import httpx

    from perspicacite.integrations.local_docs import (
        ingest_local_documents as _ingest,
    )
    from perspicacite.pipeline.download.arxiv import is_arxiv_url
    from perspicacite.pipeline.download.url_to_markdown import (
        fetch_url_as_markdown,
    )

    def _arxiv_id_from_url(u: str) -> str | None:
        """Extract bare arxiv id from /abs/ or /pdf/ URL forms."""
        m = _re.search(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})", u)
        if m:
            return m.group(1)
        m = _re.search(r"arxiv\.org/(?:abs|pdf)/([a-z\-]+/[0-9]+)", u)
        return m.group(1) if m else None

    cache_dir = Path("data/url_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    written_paths: list[Path] = []
    arxiv_dois: list[tuple[str, str]] = []  # (original_url, arxiv_doi)
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
    ) as http_client:
        for url in urls:
            # Route arxiv URLs through the structured-fulltext DOI pipeline
            # rather than the generic HTML fetcher — the abstract page is
            # the only thing url-to-markdown would extract from an /abs/
            # URL, whereas the DOI pipeline pulls the full paper.
            if is_arxiv_url(url):
                aid = _arxiv_id_from_url(url)
                if aid:
                    arxiv_dois.append((url, f"10.48550/arxiv.{aid}"))
                    results.append(
                        {
                            "url": url,
                            "status": "arxiv_routed",
                            "doi": f"10.48550/arxiv.{aid}",
                        }
                    )
                    continue

            slug = _re.sub(r"[^a-zA-Z0-9.]+", "_", url.lower())[:120] or "url"
            dest = cache_dir / f"{slug}.md"
            try:
                md, title = await fetch_url_as_markdown(
                    url,
                    http_client=http_client,
                    llm_client=state.llm_client,
                    youtube_correct=youtube_correct,
                )
                # Prepend the title as an H1 so the markdown chunker
                # uses it as the top-level section anchor.
                if title and not md.lstrip().startswith("#"):
                    md = f"# {title}\n\n{md}"
                dest.write_text(md, encoding="utf-8")
                written_paths.append(dest)
                results.append(
                    {
                        "url": url,
                        "status": "ok",
                        "chars": len(md),
                        "path": str(dest),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "url": url,
                        "status": "fetch_failed",
                        "error": str(exc)[:200],
                    }
                )

    # Process arxiv URLs via the structured-fulltext DOI ingest path.
    arxiv_chunks = 0
    if arxiv_dois:
        from perspicacite.models.papers import Author, Paper, PaperSource
        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

        kb_meta = await state.session_store.get_kb_metadata(kb_name)
        if kb_meta is not None:
            dkb = DynamicKnowledgeBase(
                vector_store=state.vector_store,
                embedding_service=state.embedding_provider,
            )
            dkb.collection_name = kb_meta.collection_name
            dkb._initialized = True
            arxiv_papers: list[Paper] = []
            for original_url, doi in arxiv_dois:
                try:
                    pc = await retrieve_paper_content(
                        doi,
                        pdf_parser=state.pdf_parser,
                        url=original_url,
                    )
                except Exception as exc:
                    for r in results:
                        if r.get("doi") == doi:
                            r["status"] = "arxiv_fetch_failed"
                            r["error"] = str(exc)[:200]
                    continue
                if not pc.success:
                    for r in results:
                        if r.get("doi") == doi:
                            r["status"] = "arxiv_no_content"
                    continue
                md = pc.metadata or {}
                p = Paper(
                    id=doi,
                    title=md.get("title") or doi,
                    authors=[Author(name=a) for a in (md.get("authors") or [])],
                    year=md.get("year"),
                    doi=doi,
                    abstract=pc.abstract or md.get("abstract"),
                    full_text=pc.full_text,
                    source=PaperSource.OPENALEX,
                    content_type=pc.content_type,
                    url=original_url,
                )
                arxiv_papers.append(p)
                for r in results:
                    if r.get("doi") == doi:
                        r["status"] = "ok"
                        r["content_type"] = pc.content_type
                        r["chars"] = len(pc.full_text or pc.abstract or "")
            if arxiv_papers:
                added = await dkb.add_papers(arxiv_papers, include_full_text=True)
                arxiv_chunks = added
                kb_meta.paper_count += len(arxiv_papers)
                kb_meta.chunk_count += added
                await state.session_store.save_kb_metadata(kb_meta)

    if not written_paths and not arxiv_dois:
        return {"added_chunks": 0, "files": 0, "results": results}
    if not written_paths:
        return {
            "added_chunks": int(arxiv_chunks),
            "files": len(arxiv_dois),
            "results": results,
        }

    class _Reg:
        async def publish(self, jid, ev):
            pass

        async def finish(self, jid, res):
            self._res = res

        async def fail(self, jid, err):
            self._err = err

    reg = _Reg()
    ingest_result = await _ingest(
        kb_name=kb_name,
        paths=written_paths,
        app_state=mcp_state,
        registry=reg,
        job_id="mcp-url-ingest",
        recursive=False,
    )
    return {
        "added_chunks": int(ingest_result.get("added_chunks", 0)) + int(arxiv_chunks),
        "files": int(ingest_result.get("files", 0)) + len(arxiv_dois),
        "results": results,
    }


@mcp.tool()
async def build_capsule(
    paper_id: str,
    kb_name: str,
    force: bool = False,
) -> dict:
    """Build (or rebuild) a per-paper capsule.

    Enumerates papers in ``kb_name``'s vector-store collection, finds the row
    matching ``paper_id``, reconstructs a Paper, locates a cached PDF (if any),
    and calls ``capsule_builder.build_capsule``.

    **Latency:** 5-60s per paper (figures + SI + code fetch). Scale with paper artifact count.
    """
    from perspicacite.pipeline.capsule_builder import (
        build_capsule as _build,
    )
    from perspicacite.pipeline.capsule_builder import (
        locate_cached_pdf,
        resolve_paper_from_metadata,
    )

    kb = await mcp_state.session_store.get_kb_metadata(kb_name)
    if kb is None:
        return {"error": f"KB '{kb_name}' not found"}
    rows = await mcp_state.vector_store.list_paper_metadata(kb.collection_name)
    row = next((r for r in rows if r.get("paper_id") == _normalize_paper_id(paper_id)), None)
    if row is None:
        return {"error": f"paper '{paper_id}' not found in KB '{kb_name}'"}
    paper = resolve_paper_from_metadata(row)
    pdf_path = locate_cached_pdf(row)
    return await _build(
        paper=paper,
        pdf_path=pdf_path,
        kb_name=kb_name,
        app_state=mcp_state,
        force=force,
    )


@mcp.tool()
async def build_capsules_for_kb(
    kb_name: str,
    force: bool = False,
) -> dict:
    """Build capsules for every paper in ``kb_name``.

    Returns ``{total, built, skipped, errored, per_paper: [...]}``.

    **Latency:** minutes for a full KB. Prefer async job dispatch.
    """
    from perspicacite.pipeline.capsule_builder import (
        build_capsule as _build,
    )
    from perspicacite.pipeline.capsule_builder import (
        locate_cached_pdf,
        resolve_paper_from_metadata,
    )

    kb = await mcp_state.session_store.get_kb_metadata(kb_name)
    if kb is None:
        return {
            "error": f"KB '{kb_name}' not found",
            "total": 0,
            "built": 0,
            "skipped": 0,
            "errored": 0,
            "per_paper": [],
        }
    rows = await mcp_state.vector_store.list_paper_metadata(kb.collection_name)
    per_paper = []
    counts = {"built": 0, "skipped": 0, "errored": 0}
    for row in rows:
        paper = resolve_paper_from_metadata(row)
        pdf_path = locate_cached_pdf(row)
        try:
            res = await _build(
                paper=paper,
                pdf_path=pdf_path,
                kb_name=kb_name,
                app_state=mcp_state,
                force=force,
            )
            status = res.get("status", "errored")
            counts[status] = counts.get(status, 0) + 1
            per_paper.append({"paper_id": paper.id, **res})
        except Exception as exc:
            counts["errored"] += 1
            per_paper.append({"paper_id": paper.id, "status": "errored", "error": str(exc)})
    return {"total": len(rows), **counts, "per_paper": per_paper}


@mcp.tool()
async def fetch_paper_resources(
    kb_name: str,
    paper_id: str,
    kinds: list[str] | None = None,
    ingest: bool = True,
    force: bool = False,
) -> dict:
    """Fetch external resources mined into the paper's capsule resources.json.

    Resources fetched per ``kinds`` (default = all supported: github/zenodo/doi).
    With ``ingest=True``, fetched text-like files are routed into the KB as
    ``is_external=True`` chunks tagged with ``parent_paper_id=<paper_id>``.

    **Latency:** 5-30s (network-bound; PDF + figures + SI).
    """
    from perspicacite.pipeline.capsule_builder import (
        capsule_dir_for,
        resolve_paper_from_metadata,
    )
    from perspicacite.pipeline.external.fetch_orchestrator import (
        fetch_paper_resources as _fetch,
    )

    kb = await mcp_state.session_store.get_kb_metadata(kb_name)
    if kb is None:
        return {"error": f"KB '{kb_name}' not found"}
    rows = await mcp_state.vector_store.list_paper_metadata(kb.collection_name)
    row = next((r for r in rows if r.get("paper_id") == _normalize_paper_id(paper_id)), None)
    if row is None:
        return {"error": f"paper '{paper_id}' not found in KB '{kb_name}'"}
    paper = resolve_paper_from_metadata(row)
    cap_dir = capsule_dir_for(paper, root=mcp_state.config.capsule.root)
    paper._kb_name = kb_name

    if mcp_state.job_registry is None:
        # Synchronous fallback registry — collects events into a list.
        class _LocalReg:
            def __init__(self):
                self.events: list[dict] = []

            async def publish(self, _job_id, payload):
                self.events.append(payload)

            async def finish(self, _job_id, payload):
                self.events.append({"type": "done", **payload})

            async def fail(self, _job_id, msg):
                self.events.append({"type": "error", "error": msg})

        reg = _LocalReg()
        result = await _fetch(
            paper=paper,
            capsule_dir=cap_dir,
            kinds=kinds,
            app_state=mcp_state,
            registry=reg,
            job_id="local",
            ingest=ingest,
            force=force,
        )
        return result
    job_id = await mcp_state.job_registry.create("external_fetch", total=0)
    result = await _fetch(
        paper=paper,
        capsule_dir=cap_dir,
        kinds=kinds,
        app_state=mcp_state,
        registry=mcp_state.job_registry,
        job_id=job_id,
        ingest=ingest,
        force=force,
    )
    return {"job_id": job_id, **result}


# =============================================================================
# Tool 17: fetch_supplementary
# =============================================================================


@mcp.tool()
async def fetch_supplementary(
    kb_name: str,
    paper_id: str,
    max_bytes_per_file: int = 50_000_000,
    max_bytes_per_record: int = 200_000_000,
    text_only: bool = False,
    force: bool = False,
) -> dict:
    """Download the Supplementary Information files listed in a paper's capsule.

    Reads ``<capsule>/supplementary/index.json`` (built during capsule
    creation via discover_supplementary — PMC JATS → Springer ESM → ACS),
    fetches each file, writes them to
    ``<capsule>/supplementary/files/<filename>``, and records a summary
    at ``<capsule>/supplementary/fetched.json``.

    **Latency:** 5-30s (publisher SI fetch).

    Args:
        kb_name: Knowledge base containing the paper.
        paper_id: Paper DOI (with or without ``doi:`` prefix).
        max_bytes_per_file: Skip individual files larger than this.
            Default 50 MB. Raise this if you need to pull big SI archives.
        max_bytes_per_record: Stop the loop once cumulative bytes for
            this paper's SI exceed this. Default 200 MB.
        text_only: When True, skip mime types we can't easily chunk
            (zip, tar, mp4, octet-stream). PDFs/XLSX/CSV/TXT are kept.
        force: Re-download even if fetched.json already exists.

    Returns:
        {"fetched": [...], "skipped": [...], "bytes": int} or {"error": ...}.
    """
    import json as _json

    from perspicacite.pipeline.capsule_builder import (
        capsule_dir_for,
        resolve_paper_from_metadata,
    )
    from perspicacite.pipeline.download.supplementary import (
        download_supplementary_to_capsule,
    )

    kb = await mcp_state.session_store.get_kb_metadata(kb_name)
    if kb is None:
        return {"error": f"KB '{kb_name}' not found"}
    rows = await mcp_state.vector_store.list_paper_metadata(kb.collection_name)
    norm_id = _normalize_paper_id(paper_id)
    row = next((r for r in rows if r.get("paper_id") == norm_id), None)
    if row is None:
        return {"error": f"paper '{paper_id}' not found in KB '{kb_name}'"}
    paper = resolve_paper_from_metadata(row)
    cap = capsule_dir_for(paper, root=mcp_state.config.capsule.root)
    index_path = cap / "supplementary" / "index.json"
    if not index_path.exists():
        return {
            "error": "no supplementary/index.json — build the capsule first "
            "(build_capsule or build_capsules_for_kb)",
            "capsule_dir": str(cap),
        }
    fetched_path = cap / "supplementary" / "fetched.json"
    if fetched_path.exists() and not force:
        return {
            "skipped": "already_fetched",
            "summary": _json.loads(fetched_path.read_text(encoding="utf-8")),
            "capsule_dir": str(cap),
        }
    manifest = _json.loads(index_path.read_text(encoding="utf-8"))
    result = await download_supplementary_to_capsule(
        cap,
        manifest.get("items") or [],
        max_bytes_per_file=max_bytes_per_file,
        max_bytes_per_record=max_bytes_per_record,
        text_only=text_only,
    )
    return {"capsule_dir": str(cap), **result}


# =============================================================================
# Tool 18: route_kbs
# =============================================================================


@mcp.tool()
async def route_kbs(
    query: str,
    candidate_kbs: list[str] | None = None,
    method: str = "bm25",
    top_k: int = 3,
    score_threshold: float = 0.1,
    ctx: Context | None = None,
) -> dict:
    """Pick the most-relevant KBs for a query without actually running it.

    Useful for agents that want to decide where to look before
    committing to retrieval — pass the returned ``hits`` list as
    ``kb_names`` to ``generate_report`` / ``search_knowledge_base`` /
    ``/api/chat``.

    Args:
        query: The research question.
        candidate_kbs: Optional restricted list (KB names). ``None`` =
            consider every KB in the session store.
        method: ``"bm25"`` (default, no LLM call) or ``"llm"`` (one
            cheap LLM call scores every KB; better on semantic
            mismatches).
        top_k: Max KBs to return.
        score_threshold: Drop KBs whose normalized score is below this.

    Returns:
        ``{"hits": [{"kb_name", "score", "reason", "sampled_titles"}, …]}``
    """
    from perspicacite.rag.kb_router import auto_route_kbs

    state = _require_state()
    if isinstance(state, str):
        return {"error": state}

    all_kbs = await state.session_store.list_kbs()
    if candidate_kbs:
        wanted = set(candidate_kbs)
        all_kbs = [k for k in all_kbs if k.name in wanted]
    if not all_kbs:
        return {"hits": [], "note": "no candidate KBs"}

    from perspicacite.llm.client import resolve_stage_model
    from perspicacite.llm.mcp_sampling import use_mcp_context

    route_provider, route_model = resolve_stage_model(state.config, "routing")
    with use_mcp_context(ctx):
        hits = await auto_route_kbs(
            query=query,
            kb_metas=all_kbs,
            vector_store=state.vector_store,
            method=method,
            top_k=top_k,
            score_threshold=score_threshold,
            llm_client=state.llm_client,
            llm_model=route_model,
            llm_provider=route_provider,
        )
    return {"hits": [h.to_dict() for h in hits]}


# =============================================================================
# Tool 19: build_kb_from_search
# =============================================================================


@mcp.tool()
async def build_kb_from_search(
    query: str,
    kb_name: str,
    max_results: int = 20,
    databases: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_citations: int | None = None,
    require_abstract: bool = False,
    article_type: str | None = None,
    create_if_missing: bool = True,
    description: str | None = None,
    dry_run: bool = False,
    screen_method: str | None = None,
    screen_threshold: float = 0.5,
    kb_aware: bool = False,
    kb_aware_terms: int = 8,
    rephrase: int = 0,
    ctx: Context | None = None,
) -> str:
    """Build or enrich a KB from a SciLEx multi-database search.

    Runs ``query`` against Semantic Scholar / OpenAlex / PubMed / arXiv
    (configurable), applies year / citation / abstract filters, then
    fetches PDFs and ingests them into ``kb_name``. Creates the KB
    when it doesn't exist (unless ``create_if_missing=False``).

    Use this when an agent wants to spin up a focused KB for a topic
    before doing real RAG over it — one tool call gets you from
    "query string" to "queryable KB" without manual DOI shuffling.

    **Latency:** minutes to tens of minutes (fan-out + per-paper enrichment). Run async via /api/jobs/* or use long client timeouts.

    Args:
        query: Free-text research question (used verbatim by SciLEx).
        kb_name: Target KB; created if missing and ``create_if_missing``.
        max_results: Max hits to pull from SciLEx (1–100).
        databases: SciLEx APIs to query (default: semantic_scholar,
            openalex, pubmed). Other options: arxiv, ieee, springer, dblp.
        min_year / max_year: Drop hits outside this range.
        min_citations: Drop hits below this citation count (uses
            citation_count when SciLEx provides it; treats missing as 0).
        require_abstract: Drop hits without an abstract.
        article_type: Optional "review" / "article" / "conference".
        create_if_missing: When False, error if KB doesn't already exist.
        description: KB description (used only when creating).
        dry_run: Return the filtered DOI list without fetching PDFs.

    Returns:
        JSON :class:`IngestReport` — search counts, filter reasons,
        added papers/chunks, PDF stats, list of selected DOIs.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    if max_results < 1 or max_results > 100:
        return _json_error("max_results must be 1..100")

    try:
        from perspicacite.llm.mcp_sampling import use_mcp_context
        from perspicacite.pipeline.search_to_kb import (
            SearchFilter,
            search_filter_and_ingest,
        )

        flt = SearchFilter(
            min_year=min_year,
            max_year=max_year,
            min_citations=min_citations,
            require_doi=True,
            require_abstract=require_abstract,
        )
        with use_mcp_context(ctx):
            report = await search_filter_and_ingest(
                app_state=state,
                query=query,
                kb_name=kb_name,
                max_results=max_results,
                databases=databases,
                flt=flt,
                article_type=article_type,
                create_if_missing=create_if_missing,
                description=description,
                dry_run=dry_run,
                screen_method=screen_method,
                screen_threshold=screen_threshold,
                kb_aware=kb_aware,
                kb_aware_terms=kb_aware_terms,
                rephrase=rephrase,
            )
        logger.info(
            "mcp_build_kb_from_search",
            query=query,
            kb=kb_name,
            searched=report.searched,
            candidates=report.candidates,
            added=report.added_papers,
        )
        return _json_ok(report.to_dict())
    except Exception as e:
        logger.error(
            "mcp_build_kb_from_search_error",
            query=query,
            kb=kb_name,
            error=str(e),
        )
        return _json_error(f"build_kb_from_search failed: {e}")


# =============================================================================
# Tool 20: export_kb
# =============================================================================


@mcp.tool()
async def export_kb(
    kb_name: str,
    out_dir: str,
    with_pdfs: bool = True,
    with_supplementary: bool = False,
    overwrite: bool = False,
) -> str:
    """Export a KB as BibTeX + cached-PDF folder for Zotero / ZotFile import.

    Produces ``<out_dir>/<kb_name>.bib`` plus optional
    ``<out_dir>/papers/<sanitized-doi>.pdf`` for every paper with a
    cached PDF. The BibTeX file references each PDF via the BetterBibTeX
    ``file`` field so Zotero attaches them on import. With
    ``with_supplementary=True``, also copies any
    ``data/capsules/<paper_id>/supplementary/files/*`` into
    ``<out_dir>/supplementary/<paper_id>/``.

    Use this for the citation-manager-agnostic preservation path —
    Zotero-free, filesystem-only, portable, git-friendly.

    Args:
        kb_name: KB to export.
        out_dir: Destination directory (created if missing).
        with_pdfs: Copy cached PDFs and reference them in the BibTeX file.
        with_supplementary: Copy supplementary files from capsules.
        overwrite: Replace an existing ``<kb_name>.bib`` if present.

    Returns:
        JSON :class:`ExportReport` with counts and paths.
    """
    state = _require_state()
    if isinstance(state, str):
        return state
    try:
        from perspicacite.pipeline.export_kb import export_kb as _export

        report = await _export(
            app_state=state,
            kb_name=kb_name,
            out_dir=out_dir,
            with_pdfs=with_pdfs,
            with_supplementary=with_supplementary,
            overwrite=overwrite,
        )
        logger.info(
            "mcp_export_kb",
            kb=kb_name,
            out=out_dir,
            papers=report.papers,
            pdfs=report.pdfs_copied,
        )
        return _json_ok(report.to_dict())
    except FileExistsError as exc:
        return _json_error(str(exc))
    except Exception as e:
        logger.error("mcp_export_kb_error", kb=kb_name, error=str(e))
        return _json_error(f"export_kb failed: {e}")


# =============================================================================
# Tool 21: expand_kb_via_citations
# =============================================================================


@mcp.tool()
async def expand_kb_via_citations(
    kb_name: str,
    direction: str = "both",
    max_per_seed: int = 10,
    seed_dois: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_citations: int | None = None,
    require_abstract: bool = False,
    screen_method: str | None = None,
    screen_threshold: float = 0.5,
    query: str | None = None,
    dry_run: bool = False,
    ctx: Context | None = None,
) -> str:
    """Grow a KB by following citation edges from its existing papers.

    Forward snowball: papers that cite the seeds. Backward snowball:
    papers the seeds cite. Uses OpenAlex (no SciLEx dependency).
    Optionally screens candidates by BM25 / LLM relevance against
    ``query`` (or the KB description) before ingest.

    **Latency:** minutes (cite-graph traversal + per-paper fetching). Prefer async job dispatch.

    Args:
        kb_name: Target KB. Must already exist.
        direction: ``"forward"`` / ``"backward"`` / ``"both"``.
        max_per_seed: Cap on hits per seed per direction (max 25). Note
            the parameter name is ``max_per_seed`` — not ``max_papers``
            — and applies per-seed-per-direction, so total papers added
            is at most ``max_per_seed * len(seeds) * directions``.
        seed_dois: Restrict to these seeds. ``None`` = every DOI in
            the KB.
        min_year / max_year / min_citations / require_abstract:
            Pre-screen filters.
        screen_method / screen_threshold: Optional LLM/BM25 relevance
            gate; ``query`` falls back to the KB description.
        dry_run: Skip PDF fetch + ingest; return the candidate DOIs only.

    Returns:
        JSON :class:`SnowballReport`.
    """
    state = _require_state()
    if isinstance(state, str):
        return state
    try:
        from perspicacite.llm.mcp_sampling import use_mcp_context
        from perspicacite.pipeline.search_to_kb import SearchFilter
        from perspicacite.pipeline.snowball import expand_kb_via_citations as _expand

        flt = SearchFilter(
            min_year=min_year,
            max_year=max_year,
            min_citations=min_citations,
            require_doi=True,
            require_abstract=require_abstract,
        )
        with use_mcp_context(ctx):
            report = await _expand(
                app_state=state,
                kb_name=kb_name,
                direction=direction,
                max_per_seed=max_per_seed,
                seed_dois=seed_dois,
                flt=flt,
                screen_method=screen_method,
                screen_threshold=screen_threshold,
                query=query,
                dry_run=dry_run,
            )
        logger.info(
            "mcp_expand_kb_via_citations",
            kb=kb_name,
            direction=direction,
            raw_hits=report.raw_hits,
            added=report.papers_added,
        )
        return _json_ok(report.to_dict())
    except Exception as e:
        logger.error("mcp_expand_kb_via_citations_error", kb=kb_name, error=str(e))
        return _json_error(f"expand_kb_via_citations failed: {e}")


# =============================================================================
# Tool 22: delete_knowledge_base
# =============================================================================


@mcp.tool()
async def delete_knowledge_base(
    name: str,
    keep_collection: bool = False,
) -> str:
    """Permanently delete a KB (metadata row + Chroma collection).

    Cached PDFs in ``pdf_download.cache_dir`` are NOT removed — they
    are per-DOI artefacts useful for future ingests. To purge them,
    delete the matching ``<doi>.pdf`` / ``<doi>.meta.json`` files
    by hand.

    Args:
        name: KB name.
        keep_collection: When True, drop only the metadata row and
            leave the underlying Chroma collection in place (orphans
            the embeddings; useful when re-attaching).

    Returns:
        JSON ``{"deleted": bool, "collection_dropped": bool, "name": ...}``.
    """
    state = _require_state()
    if isinstance(state, str):
        return state
    try:
        kb = await state.session_store.get_kb_metadata(name)
        if not kb:
            return _json_error(f"KB '{name}' not found")
        collection_dropped = False
        collection_error: str | None = None
        if not keep_collection and kb.collection_name:
            try:
                await state.vector_store.delete_collection(kb.collection_name)
                collection_dropped = True
            except Exception as exc:
                collection_error = str(exc)
        deleted = await state.session_store.delete_kb_metadata(name)
        logger.info(
            "mcp_delete_kb",
            name=name,
            deleted=deleted,
            collection_dropped=collection_dropped,
        )
        return _json_ok(
            {
                "name": name,
                "deleted": deleted,
                "collection_dropped": collection_dropped,
                "collection_error": collection_error,
            }
        )
    except Exception as e:
        logger.error("mcp_delete_kb_error", name=name, error=str(e))
        return _json_error(f"delete_knowledge_base failed: {e}")


# =============================================================================
# Tool 23: enrich_kb_from_cite_graph_tool
# =============================================================================


@mcp.tool()
async def enrich_kb_from_cite_graph_tool(
    kb_name: str,
    tool: str | None = None,
    doi: str | None = None,
    openalex_id: str | None = None,
    max_papers: int | None = None,
    dry_run: bool = True,
) -> dict:
    """MCP tool: cite-graph enrichment preview.

    Resolves a library/tool name (or explicit DOI/OpenAlex id) to a
    canonical paper, fetches OpenAlex citing works, filters and scores
    them, and returns a ranked list of CiteHit records.

    v1: dry-run only. Returns ranked CiteHit records as dicts.

    **Latency:** minutes (cite-graph + per-paper fetch). Prefer async job dispatch.

    Args:
        kb_name: Target KB name (used for context; no ingest in v1).
        tool: Library/tool name to resolve to its canonical DOI.
        doi: Skip the resolver and use this DOI directly as the seed.
        openalex_id: Skip the resolver and DOI lookup; use this OpenAlex
            Work id (e.g. ``W3177828909``) directly as the seed.
        max_papers: Override the max_papers cap from config.
        dry_run: Preview only — no ingest (default True; v1 always behaves
            as dry-run regardless of this flag).

    Returns:
        Dict ``{"hits": [...]}`` where each hit contains doi, title,
        year, citation_count, is_oa, venue, score, score_breakdown.
    """
    from perspicacite.pipeline.cite_graph import enrich_kb_from_cite_graph

    cfg = mcp_state.config
    kb_cfg = cfg.knowledge_base
    if max_papers is not None:
        kb_cfg.cite_graph.max_papers = max_papers
    hits = await enrich_kb_from_cite_graph(
        tool=tool,
        doi=doi,
        openalex_id=openalex_id,
        kb_config=kb_cfg,
        existing_dois=set(),
        dry_run=dry_run,
    )
    return {
        "hits": [
            {
                "doi": h.doi,
                "title": h.title,
                "year": h.year,
                "citation_count": h.citation_count,
                "is_oa": h.is_oa,
                "venue": h.venue,
                "score": h.score,
                "score_breakdown": h.score_breakdown,
            }
            for h in hits
        ]
    }


@mcp.tool()
async def ingest_asb_run(
    asb_run_dir: str,
    kb_name: str | None = None,
    include: list[str] | None = None,
    mode: str = "composite",
) -> str:
    """
    Ingest an Agent Skill Bundle (ASB) run directory into Perspicacité KBs.

    The ASB run dir is expected to contain skills/_index.json and/or
    cards/task_*.json+md pairs and optional workflow_dag.json. Both the
    2026-05-15 (pair edges, executable=bool) and 2026-05-16+ (dict edges
    with port labels, executable=dict) schemas are supported.

    **Latency:** 30-300s depending on bundle size + backing-paper DOI
    count. Each backing paper goes through the full DOI ingest path
    (resolve + fetch + embed). Use >=300s HTTP timeout in clients;
    prefer async dispatch for large bundles.

    Args:
        asb_run_dir: Absolute path to the ASB run directory.
        kb_name: Target KB name. Defaults to the run-dir basename.
        include: Subset of ["skills", "workflows"]. Defaults to both.
        mode: "composite" (single KB) or "per-skill" (one KB per skill;
            workflows still composite).

    Returns:
        JSON envelope with kb_names, skills_ingested, workflows_ingested,
        papers_ingested, failed, workflow_dag.
    """
    state = _require_state()
    if isinstance(state, str):
        return state  # already _json_error

    if not include:
        include = ["skills", "workflows"]

    try:
        result = await ingest_asb_run_pipeline(
            asb_run_dir=asb_run_dir,
            kb_name=kb_name,
            include=tuple(include),
            mode=mode,
            app_state=state,
        )
    except Exception as e:
        logger.error("mcp_ingest_asb_run_error", error=str(e))
        return _json_error(f"ASB ingest failed: {e}")

    logger.info(
        "mcp_ingest_asb_run",
        kb_names=result.get("kb_names"),
        skills=result.get("skills_ingested"),
        workflows=result.get("workflows_ingested"),
    )
    return _json_ok(result)


# =============================================================================
# GitHub repo + skill-bundle ingest (Task 7 of 2026-05-15 plan)
# =============================================================================


def _summary_to_dict(summary: IngestSummary) -> dict[str, Any]:
    """Flatten an :class:`IngestSummary` dataclass for the MCP envelope.

    Using ``dataclasses.asdict`` keeps the JSON shape in sync with the
    dataclass: when fields are added upstream they surface here without
    a separate edit. (The ``Co-Authored-By`` linked-papers list of
    ``(kind, value)`` tuples serializes as JSON arrays of two strings.)
    """
    import dataclasses

    return dataclasses.asdict(summary)


@mcp.tool()
async def ingest_github_repo(
    url: str,
    kb_name: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> str:
    """Ingest a GitHub repository into a Perspicacité KB.

    Walks the repo (filtered by include/exclude globs), chunks markdown,
    Python (docstrings only), and Jupyter notebooks, and stores them in
    ``kb_name``. Use ``ingest_skill_bundle`` instead when the repo
    contains a ``bundle.yml`` you want parsed for paper refs.

    **Latency:** 30-180s depending on repo size. Tarball is cached by
    commit SHA so re-ingest is fast. Use >=240s HTTP timeout.

    Args:
        url: GitHub URL (https://github.com/<org>/<repo>[@ref]).
        kb_name: Target KB name (created if missing).
        include: Optional list of include globs (gitwildmatch).
        exclude: Optional list of exclude globs.

    Returns:
        JSON envelope with kb_name, repo_org, repo_name, commit_sha,
        files_added, chunks_added, mode.
    """
    state = _require_state()
    if isinstance(state, str):
        return state  # already _json_error

    # Build a ContentSpec only when the caller supplied at least one
    # override. ``None`` lets the orchestrator apply its own defaults.
    content = None
    if include or exclude:
        from perspicacite.pipeline.github.bundle import ContentSpec

        defaults = ContentSpec()
        content = ContentSpec(
            include=list(include) if include else list(defaults.include),
            exclude=list(exclude) if exclude else list(defaults.exclude),
        )

    try:
        summary = await ingest_github_repo_pipeline(
            url=url,
            kb_name=kb_name,
            config=state.config,
            vector_store=state.vector_store,
            embedding_service=state.embedding_provider,
            session_store=state.session_store,
            content=content,
        )
    except Exception as e:
        logger.error("mcp_ingest_github_repo_error", url=url, error=str(e))
        return _json_error(f"GitHub repo ingest failed: {e}")

    logger.info(
        "mcp_ingest_github_repo",
        url=url,
        kb_name=summary.kb_name,
        files_added=summary.files_added,
        chunks_added=summary.chunks_added,
    )
    return _json_ok(_summary_to_dict(summary))


@mcp.tool()
async def ingest_skill_bundle(
    source: str,
    kb_name: str | None = None,
    ingest_linked_papers: bool = True,
    source_format: str = "legacy",
) -> str:
    """Ingest an agentic-science-builder skill bundle into Perspicacité.

    Supports two source formats:

    * **``legacy``** (default) — original ASB run-dir format with
      ``bundle.yml`` / ``bundle.yaml``. The bundle source can be a local
      directory path OR a GitHub URL. Linked papers (DOIs in the manifest
      + README) are routed through the existing DOI ingest path when
      ``ingest_linked_papers=True``.

    * **``asb-skill-collection-v1``** — reads an ASB-Skills release
      collection directory containing ``collection.yaml``,
      ``tools.lock.yaml``, ``tools/*.yaml``, ``catalogue.jsonld``, and
      ``skills/<slug>/SKILL.md`` per §7.1 of the 2026-05-24 ASB-Skills
      release design. Produces three chunk-types per skill:
      ``summary`` (Overview section), ``procedure`` (Procedure section),
      and ``tools`` (tool registry entries). Writes
      ``kb_metadata/ontology_refs.json`` and
      ``kb_metadata/skill_index.json`` as side-files.

    **Latency:**
    - ``legacy``: 60-600s per bundle + linked-paper count. Use >=600s timeout.
    - ``asb-skill-collection-v1``: 30-300s depending on skill count + DOI ingest.

    Args:
        source: Local path OR (legacy only) GitHub URL.
        kb_name: Target KB name. None → derived from bundle name.
        ingest_linked_papers: If True (default), DOIs from the bundle are
            added to the same KB.
        source_format: ``"legacy"`` (default) or ``"asb-skill-collection-v1"``.

    Returns:
        JSON envelope. For ``legacy``: kb_name, bundle_name, files_added,
        chunks_added, linked_papers_added, linked_papers_skipped_non_doi.
        For ``asb-skill-collection-v1``: kb_name, collection_name,
        skills_ingested, papers_added, failed, kb_metadata_written.
    """
    _VALID_FORMATS = {"legacy", "asb-skill-collection-v1"}
    if source_format not in _VALID_FORMATS:
        return _json_error(
            f"Unknown source_format '{source_format}'. "
            f"Valid values: {sorted(_VALID_FORMATS)}"
        )

    state = _require_state()
    if isinstance(state, str):
        return state  # already _json_error

    # ---- asb-skill-collection-v1 branch ----
    if source_format == "asb-skill-collection-v1":
        collection_dir = Path(source)
        if not collection_dir.is_dir():
            return _json_error(
                f"source_format='asb-skill-collection-v1' requires a local "
                f"directory path; got: {source!r}"
            )
        target_kb = kb_name or collection_dir.name

        try:
            result = await ingest_asb_skill_collection(
                collection_dir=collection_dir,
                kb_name=target_kb,
                app_state=state,
                ingest_linked_papers=ingest_linked_papers,
            )
        except Exception as e:
            logger.error(
                "mcp_ingest_skill_bundle_v1_error",
                source=source,
                error=str(e),
            )
            return _json_error(f"ASB-Skill collection v1 ingest failed: {e}")

        logger.info(
            "mcp_ingest_skill_bundle_v1",
            source=source,
            kb_name=result["kb_name"],
            skills_ingested=result["skills_ingested"],
        )
        return _json_ok(result)

    # ---- legacy branch (original logic, unchanged) ----
    # Match the CLI's path-vs-URL detection: existing local path → Path,
    # anything else (URL or non-existent path) → raw string. The
    # orchestrator parses URL strings itself.
    source_arg: Path | str
    candidate = Path(source)
    if candidate.exists():
        source_arg = candidate
    else:
        source_arg = source

    try:
        summary = await ingest_skill_bundle_pipeline(
            source=source_arg,
            kb_name=kb_name,
            config=state.config,
            vector_store=state.vector_store,
            embedding_service=state.embedding_provider,
            session_store=state.session_store,
            ingest_linked_papers=ingest_linked_papers,
            app_state_for_doi_ingest=state if ingest_linked_papers else None,
        )
    except Exception as e:
        logger.error("mcp_ingest_skill_bundle_error", source=source, error=str(e))
        return _json_error(f"Skill bundle ingest failed: {e}")

    logger.info(
        "mcp_ingest_skill_bundle",
        source=source,
        kb_name=summary.kb_name,
        bundle_name=summary.bundle_name,
        files_added=summary.files_added,
        chunks_added=summary.chunks_added,
        linked_papers_added=summary.linked_papers_added,
    )
    return _json_ok(_summary_to_dict(summary))


# =============================================================================
# Tool 13: zotero_list_collections
# =============================================================================

_zotero_collections_cache: dict[str, tuple[list, float]] = {}
_COLLECTION_CACHE_TTL = 3600.0  # 1 hour


def _build_collection_tree(flat: list[dict], parent_key: str | None = None) -> list[dict]:
    result = []
    for coll in flat:
        data = coll.get("data") or {}
        pc = data.get("parentCollection")
        coll_parent = None if (pc is False or not pc) else pc
        if coll_parent == parent_key:
            result.append(
                {
                    "id": coll["key"],
                    "name": data.get("name") or "",
                    "parent_id": parent_key,
                    "item_count": None,
                    "subcollections": _build_collection_tree(flat, parent_key=coll["key"]),
                }
            )
    return result


@mcp.tool()
async def zotero_list_collections(
    library_id: str | None = None,
    include_subcollections: bool = True,
) -> dict:
    """List all Zotero collections (with sub-collection tree).

    Args:
        library_id: Override the configured library_id for this call.
        include_subcollections: If True (default), return a nested tree.
            If False, return only top-level collections.

    Returns:
        {"collections": [...], "library_id": str, "library_type": str}
        Each collection: {"id", "name", "parent_id", "item_count", "subcollections"}
    """
    import time

    import httpx

    from perspicacite.integrations.zotero import ZoteroClient

    cfg = getattr(getattr(mcp_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled and cfg.api_key):
        return {
            "error": "ZOTERO_NOT_CONFIGURED",
            "message": "Zotero not enabled or api_key missing",
        }

    eff_library_id = library_id or cfg.library_id
    if not eff_library_id:
        return {"error": "ZOTERO_NOT_CONFIGURED", "message": "library_id required"}

    base_url = getattr(cfg, "base_url", "") or None

    cache_key = f"{eff_library_id}:{cfg.library_type}"
    cached = _zotero_collections_cache.get(cache_key)
    if cached and time.time() < cached[1]:
        flat = cached[0]
    else:
        client = ZoteroClient(
            api_key=cfg.api_key,
            library_id=eff_library_id,
            library_type=cfg.library_type,
            base_url=base_url,
        )
        try:
            flat = await client.list_collections()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 403:
                return {"error": "ZOTERO_AUTH_FAILED", "message": "Zotero API returned 403"}
            if status == 429:
                ra = exc.response.headers.get("retry-after") or "60"
                return {"error": "ZOTERO_RATE_LIMITED", "retry_after_s": float(ra)}
            if status == 404:
                return {
                    "error": "LIBRARY_NOT_FOUND",
                    "message": f"Library {eff_library_id} not found",
                }
            return {"error": "ZOTERO_ERROR", "message": str(exc)}
        _zotero_collections_cache[cache_key] = (flat, time.time() + _COLLECTION_CACHE_TTL)

    if include_subcollections:
        collections = _build_collection_tree(flat, parent_key=None)
    else:
        collections = [
            {
                "id": c["key"],
                "name": (c.get("data") or {}).get("name") or "",
                "parent_id": None,
                "item_count": None,
                "subcollections": [],
            }
            for c in flat
            if not (c.get("data") or {}).get("parentCollection")
        ]

    return {
        "collections": collections,
        "library_id": eff_library_id,
        "library_type": cfg.library_type,
    }


# =============================================================================
# Tool 14: zotero_get_collection_items
# =============================================================================

import base64 as _base64


def _encode_cursor(start: int) -> str:
    return _base64.b64encode(str(start).encode()).decode()


def _decode_cursor(cursor: str) -> int:
    try:
        return int(_base64.b64decode(cursor.encode()).decode())
    except Exception:
        return 0


@mcp.tool()
async def zotero_get_collection_items(
    collection_id: str,
    library_id: str | None = None,
    include_abstract: bool = True,
    limit: int = 200,
    cursor: str | None = None,
) -> dict:
    """Return papers in a Zotero collection with metadata and license classification.

    Args:
        collection_id: Zotero collection key (e.g. "ABC123").
        library_id: Override the configured library_id.
        include_abstract: Include abstractNote in each item (default True).
        limit: Page size, max 500 (default 200).
        cursor: Opaque pagination token from a previous call's ``next_cursor``.

    Returns:
        {"collection_id", "items": [...], "total": int, "next_cursor": str | None}
        Each item: {"zotero_key", "doi", "title", "authors", "year", "abstract",
                    "item_type", "tags", "license": {...}, "has_attachments"}
    """
    import asyncio

    import httpx

    from perspicacite.integrations.zotero import ZoteroClient
    from perspicacite.integrations.zotero_license import LicenseClassifier

    cfg = getattr(getattr(mcp_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled and cfg.api_key):
        return {
            "error": "ZOTERO_NOT_CONFIGURED",
            "message": "Zotero not enabled or api_key missing",
        }

    eff_library_id = library_id or cfg.library_id
    if not eff_library_id:
        return {"error": "ZOTERO_NOT_CONFIGURED", "message": "library_id required"}

    base_url = getattr(cfg, "base_url", "") or None
    client = ZoteroClient(
        api_key=cfg.api_key,
        library_id=eff_library_id,
        library_type=cfg.library_type,
        base_url=base_url,
    )

    try:
        all_items = await client.list_items_in_collection(
            collection_id, include_subcollections=True
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 403:
            return {"error": "ZOTERO_AUTH_FAILED"}
        if status == 429:
            ra = exc.response.headers.get("retry-after") or "60"
            return {"error": "ZOTERO_RATE_LIMITED", "retry_after_s": float(ra)}
        if status == 404:
            return {
                "error": "COLLECTION_NOT_FOUND",
                "message": f"Collection {collection_id} not found",
            }
        return {"error": "ZOTERO_ERROR", "message": str(exc)}

    total = len(all_items)
    limit = max(1, min(limit, 500))
    start = _decode_cursor(cursor) if cursor else 0
    if start < 0 or start > total:
        return {"error": "INVALID_CURSOR", "message": "Cursor is stale or invalid"}
    page = all_items[start : start + limit]
    next_start = start + len(page)
    next_cursor = _encode_cursor(next_start) if next_start < total else None

    clf = LicenseClassifier()
    async with httpx.AsyncClient() as http:

        async def _classify_item(it: dict) -> dict:
            data = it.get("data") or {}
            doi = data.get("DOI") or None
            creators = data.get("creators") or []
            authors = [
                (
                    (cr.get("firstName") or "") + " " + (cr.get("lastName") or cr.get("name") or "")
                ).strip()
                for cr in creators
            ]
            year_str = str(data.get("date") or "")[:4]
            year = int(year_str) if year_str.isdigit() else None
            tags = [(t.get("tag") or "") for t in (data.get("tags") or [])]

            if doi:
                lic = await clf.classify(doi, zotero_item=it, http_client=http)
            else:
                lic = clf.classify_zotero_tags(it) or clf.heuristic(is_oa=False)

            # Resolve child-attachment keys so downstream tools (and ASB's
            # MCP bridge) can request the bytes via zotero_get_attachment_bytes
            # without a second round-trip per item to discover them.
            try:
                attachments = await client.get_item_attachments(it.get("key") or "")
                attachment_keys = [a.get("key") for a in attachments if a.get("key")]
            except httpx.HTTPError:
                attachment_keys = []

            return {
                "zotero_key": it.get("key"),
                "doi": doi,
                "title": data.get("title") or "",
                "authors": [a for a in authors if a],
                "year": year,
                "abstract": data.get("abstractNote") if include_abstract else None,
                "item_type": data.get("itemType") or "journalArticle",
                "tags": tags,
                "license": {
                    "spdx": lic.spdx,
                    "classification": lic.classification,
                    "policy": lic.policy,
                    "source": lic.source,
                },
                "attachment_keys": attachment_keys,
                "has_attachments": bool(attachment_keys),
            }

        items = await asyncio.gather(*(_classify_item(it) for it in page))

    return {
        "collection_id": collection_id,
        "items": list(items),
        "total": total,
        "next_cursor": next_cursor,
    }


# =============================================================================
# Tool 15: zotero_get_paper_resources
# =============================================================================


@mcp.tool()
async def zotero_get_paper_resources(
    doi: str | None = None,
    zotero_key: str | None = None,
    library_id: str | None = None,
) -> dict:
    """Return file access options for a paper (local path first, then remote URLs).

    Exactly one of ``doi`` or ``zotero_key`` must be provided.

    Args:
        doi: The paper's DOI.
        zotero_key: Zotero item key (use when DOI is ambiguous).
        library_id: Override the configured library_id.

    Returns:
        {"doi", "zotero_key", "license": {...}, "resources": [...], "notes": []}
        Each resource: {"role", "filename", "access": [{"type", "path"|"url", "via"?}]}
    """
    import asyncio

    import httpx

    from perspicacite.integrations.zotero import ZoteroClient
    from perspicacite.integrations.zotero_license import LicenseClassifier
    from perspicacite.integrations.zotero_resources import ResourceLocator

    if not doi and not zotero_key:
        return {"error": "INVALID_ARGUMENTS", "message": "Provide doi or zotero_key"}

    cfg = getattr(getattr(mcp_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled and cfg.api_key):
        return {
            "error": "ZOTERO_NOT_CONFIGURED",
            "message": "Zotero not enabled or api_key missing",
        }

    eff_library_id = library_id or cfg.library_id
    if not eff_library_id:
        return {"error": "ZOTERO_NOT_CONFIGURED", "message": "library_id required"}

    base_url = getattr(cfg, "base_url", "") or None
    client = ZoteroClient(
        api_key=cfg.api_key,
        library_id=eff_library_id,
        library_type=cfg.library_type,
        base_url=base_url,
    )

    try:
        if zotero_key:
            items = await client._paginated(f"/items/{zotero_key}")
            zotero_item = items[0] if items else None
            if zotero_item is None:
                return {"error": "PAPER_NOT_FOUND", "message": f"Key {zotero_key} not found"}
        else:
            items = await client._paginated("/items", params={"q": doi, "qmode": "everything"})
            matched = [
                it
                for it in items
                if (it.get("data") or {}).get("DOI", "").lower().strip() == doi.lower().strip()
            ]
            if not matched:
                return {"error": "PAPER_NOT_FOUND", "message": f"DOI {doi} not in library"}
            if len(matched) > 1:
                return {
                    "error": "AMBIGUOUS_DOI",
                    "message": f"DOI {doi} matches {len(matched)} items; pass zotero_key",
                    "keys": [it.get("key") for it in matched],
                }
            zotero_item = matched[0]

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 403:
            return {"error": "ZOTERO_AUTH_FAILED"}
        if status == 429:
            ra = exc.response.headers.get("retry-after") or "60"
            return {"error": "ZOTERO_RATE_LIMITED", "retry_after_s": float(ra)}
        return {"error": "ZOTERO_ERROR", "message": str(exc)}

    item_doi = (zotero_item.get("data") or {}).get("DOI") or doi or ""
    item_key = zotero_item.get("key") or zotero_key

    clf = LicenseClassifier()
    async with httpx.AsyncClient() as http:
        attachments, lic = await asyncio.gather(
            client.get_item_attachments(item_key),
            clf.classify(item_doi, zotero_item=zotero_item, http_client=http),
        )

    rl = ResourceLocator(mcp_state.config)
    resources = rl.build(doi=item_doi, zotero_item=zotero_item, attachments=attachments)

    return {
        "doi": item_doi,
        "zotero_key": item_key,
        "license": {
            "spdx": lic.spdx,
            "classification": lic.classification,
            "policy": lic.policy,
            "source": lic.source,
        },
        "resources": resources,
        "notes": [],
    }


# =============================================================================
# Tool 16: zotero_ingest_collection_to_kb
# =============================================================================
# Tool 16b: zotero_get_attachment_bytes
# =============================================================================


@mcp.tool()
async def zotero_get_attachment_bytes(
    attachment_key: str,
    library_id: str | None = None,
) -> dict[str, Any]:
    """Return the raw bytes of a Zotero attachment as base64.

    Companion to ``zotero_get_collection_items``' new ``attachment_keys``
    field: callers iterate the returned keys and fetch each one via this
    tool to get the actual PDF/HTML/etc. content for downstream
    processing (ASB capsule synthesis, KB ingest, archival).

    Args:
        attachment_key: Zotero key of an attachment child item.
        library_id: Override ``config.yml`` ``zotero.library_id`` for
            this call. Useful when the attachment lives in a different
            group than the server's default.

    Returns:
        ``{filename, content_b64, content_type, size_bytes, role_hint?,
        license_spdx?}``. Errors surface as ``{"error": ..., "message": ...}``.
    """
    import base64

    cfg = getattr(getattr(mcp_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled and cfg.api_key):
        return {
            "error": "ZOTERO_NOT_CONFIGURED",
            "message": "Zotero not enabled or api_key missing",
        }
    eff_library_id = library_id or cfg.library_id
    if not eff_library_id:
        return {"error": "ZOTERO_NOT_CONFIGURED", "message": "library_id required"}

    from perspicacite.integrations.zotero import ZoteroClient

    client = ZoteroClient(
        api_key=cfg.api_key,
        library_id=eff_library_id,
        library_type=cfg.library_type,
        base_url=getattr(cfg, "base_url", "") or None,
    )

    # Fetch the attachment metadata first (filename, contentType, tags
    # that may encode role_hint or license).
    c = await client._client()
    try:
        meta_r = await c.get(
            f"{client._base()}/items/{attachment_key}",
            headers=client._headers(),
        )
    except httpx.HTTPError as exc:
        return {"error": "ZOTERO_FETCH_FAILED", "message": str(exc)}
    if meta_r.status_code == 404:
        return {
            "error": "ATTACHMENT_NOT_FOUND",
            "message": f"No attachment with key {attachment_key!r}",
        }
    if meta_r.status_code != 200:
        return {
            "error": "ZOTERO_ERROR",
            "message": f"HTTP {meta_r.status_code} fetching attachment metadata",
        }
    meta = (meta_r.json() or {}).get("data") or {}
    filename = meta.get("filename") or meta.get("title") or attachment_key
    content_type = meta.get("contentType") or "application/octet-stream"

    # Then the binary content.
    data = await client.download_attachment_bytes(attachment_key)
    if data is None:
        return {
            "error": "ATTACHMENT_BYTES_UNAVAILABLE",
            "message": (
                "Attachment exists but bytes couldn't be fetched. "
                "Common causes: linked file (not uploaded to Zotero), "
                "snapshot-only, or storage quota exceeded."
            ),
        }

    # Surface optional metadata if Zotero tags encode it
    # (convention: role:main_article, license:CC-BY-4.0, etc.)
    role_hint = None
    license_spdx = None
    for t in meta.get("tags") or []:
        tag = (t.get("tag") or "").strip()
        low = tag.lower()
        if low.startswith("role:"):
            role_hint = tag.split(":", 1)[1]
        elif low.startswith("license:"):
            license_spdx = tag.split(":", 1)[1]

    return {
        "filename": filename,
        "content_b64": base64.b64encode(data).decode("ascii"),
        "content_type": content_type,
        "size_bytes": len(data),
        "role_hint": role_hint,
        "license_spdx": license_spdx,
    }


# =============================================================================
# Tool 17: zotero_ingest_collection_to_kb
# =============================================================================

# Strong references to background tasks (prevent GC before completion)
_zotero_ingest_tasks: set = set()


@mcp.tool()
async def zotero_ingest_collection_to_kb(
    collection_id: str,
    kb_name: str | None = None,
    library_id: str | None = None,
    force_reingest: bool = False,
) -> dict:
    """Ingest a Zotero collection into a Perspicacité KB.

    If the server has a job registry (running under the full web server),
    the ingest runs as a background task and the call returns immediately
    with a ``job_id`` and ``poll_url``. Otherwise the ingest runs inline
    and the finished result is returned directly.

    Args:
        collection_id: Zotero collection key (e.g. "ABC123").
        kb_name: KB name override; defaults to a sanitized version of the
            collection name.
        library_id: Override the configured library_id.
        force_reingest: Re-embed papers already in the KB (default False).

    Returns (async mode):
        {"job_id", "kb_name", "collection_id", "item_count", "status": "running", "poll_url"}
    Returns (inline mode):
        {"per_kb": [...]} from build_kbs_from_zotero
    """
    import asyncio

    import httpx

    from perspicacite.integrations import zotero_ingest
    from perspicacite.integrations.zotero import ZoteroClient

    cfg = getattr(getattr(mcp_state, "config", None), "zotero", None)
    if not (cfg and cfg.enabled and cfg.api_key):
        return {
            "error": "ZOTERO_NOT_CONFIGURED",
            "message": "Zotero not enabled or api_key missing",
        }

    eff_library_id = library_id or cfg.library_id
    if not eff_library_id:
        return {"error": "ZOTERO_NOT_CONFIGURED", "message": "library_id required"}

    base_url = getattr(cfg, "base_url", "") or None
    client = ZoteroClient(
        api_key=cfg.api_key,
        library_id=eff_library_id,
        library_type=cfg.library_type,
        base_url=base_url,
    )

    try:
        plan = await zotero_ingest.plan_kbs_from_zotero(
            client,
            top_level_collection_keys=[collection_id],
            include_unfiled=False,
            library_label=eff_library_id,
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 403:
            return {"error": "ZOTERO_AUTH_FAILED"}
        if status == 429:
            ra = exc.response.headers.get("retry-after") or "60"
            return {"error": "ZOTERO_RATE_LIMITED", "retry_after_s": float(ra)}
        if status == 404:
            return {
                "error": "COLLECTION_NOT_FOUND",
                "message": f"Collection {collection_id} not found",
            }
        return {"error": "ZOTERO_ERROR", "message": str(exc)}

    if not plan:
        return {
            "error": "COLLECTION_NOT_FOUND",
            "message": f"Collection {collection_id} produced no plan entries",
        }

    entry = plan[0]
    effective_kb = kb_name or entry.kb_name

    if kb_name:
        entry = zotero_ingest.ZoteroKBPlanEntry(
            kb_name=kb_name,
            source_collection_key=entry.source_collection_key,
            source_collection_name=entry.source_collection_name,
            item_count=entry.item_count,
            with_doi_count=entry.with_doi_count,
            with_pdf_count=entry.with_pdf_count,
            with_notes_count=entry.with_notes_count,
        )

    registry = getattr(mcp_state, "job_registry", None)

    if registry is not None:
        job_id = await registry.create("zotero_collection_ingest", total=entry.item_count)
        task = asyncio.create_task(
            zotero_ingest.build_kbs_from_zotero(
                client,
                plan=[entry],
                app_state=mcp_state,
                registry=registry,
                job_id=job_id,
            )
        )
        _zotero_ingest_tasks.add(task)
        task.add_done_callback(_zotero_ingest_tasks.discard)
        base_cfg = getattr(mcp_state.config, "server", None)
        port = getattr(base_cfg, "port", 5468) if base_cfg else 5468
        return {
            "job_id": job_id,
            "kb_name": effective_kb,
            "collection_id": collection_id,
            "item_count": entry.item_count,
            "status": "running",
            "poll_url": f"http://localhost:{port}/api/jobs/{job_id}/events",
        }

    # Inline mode (MCP-only context, no registry)
    class _InlineReg:
        def __init__(self) -> None:
            self.result: dict[str, Any] | None = None
            self.err: str | None = None

        async def publish(self, jid: str, ev: dict[str, Any]) -> None:
            return None

        async def finish(self, jid: str, res: dict[str, Any]) -> None:
            self.result = res

        async def fail(self, jid: str, err: str) -> None:
            self.err = err

    reg = _InlineReg()
    try:
        await zotero_ingest.build_kbs_from_zotero(
            client,
            plan=[entry],
            app_state=mcp_state,
            registry=reg,
            job_id="mcp-inline",
        )
    except Exception as exc:
        return {"error": str(exc)}
    if reg.err is not None:
        return {"error": reg.err}
    return reg.result or {"per_kb": []}


# =============================================================================
# Resource
# =============================================================================


@mcp.tool()
async def web_search(
    query: str,
    databases: list[str] | None = None,
    max_results: int = 10,
    enrich: bool = True,
    optimize_query: bool = True,
    ctx: Context | None = None,
) -> str:
    """Live academic web search across user-selected databases.

    Wraps the shared aggregator pipeline (semantic_scholar, openalex,
    pubmed, arxiv via SciLEx + standalone google_scholar, europepmc,
    core, etc.) with Crossref enrichment + MiniLM rerank. Returns a
    JSON-encoded payload with ``papers``, ``warnings``, and
    ``telemetry_summary`` (per-provider hit counts).

    Distinct from ``search_literature`` (SciLEx-only) and
    ``generate_report`` (heavy mode-bound RAG). Use this when you
    just want a focused literature lookup with cleaned-up metadata.

    Args:
        query: free-text scientific query
        databases: list of provider names (default: semantic_scholar,
            openalex, pubmed). Pass google_scholar / europepmc / core
            for the standalone aggregator providers.
        max_results: cap on returned papers (1-50)
        enrich: when True (default) runs Crossref enrichment on the
            returned papers — fills missing abstracts and canonicalises
            author lists. Set False for raw provider data.
        optimize_query: when True, runs the LLM-assisted keyword rewrite
            before searching.
        ctx: MCP context for live progress notifications (injected
            automatically by fastmcp; do not pass manually).

    Returns:
        JSON string: {"papers": [...], "warnings": [...],
                      "telemetry_summary": {"by_provider": {...}}}
    """
    import json as _json

    from perspicacite.rag.resolve_papers import resolve_papers_pipeline
    from perspicacite.rag.telemetry import (
        CallbackTelemetrySink,
        ListTelemetrySink,
    )

    # Choose a sink that buffers events for the telemetry_summary.
    # When ctx is present, also forward events as live MCP progress
    # notifications via MCPProgressAdapter.
    if ctx is not None:
        from perspicacite.mcp.progress_adapter import MCPProgressAdapter

        _adapter = MCPProgressAdapter(ctx)
        sink: Any = CallbackTelemetrySink(_adapter.on_event)
    else:
        sink = ListTelemetrySink()

    # Use the MCP server's own state (carries .config + .llm_client) so
    # the query optimizer runs on MCP-originated web_search calls. Without
    # this, optimize_query=True is silently a no-op.
    _state = _require_state()
    try:
        papers = await resolve_papers_pipeline(
            query=query,
            databases=databases,
            max_docs=max(1, min(50, int(max_results))),
            app_state=_state,
            telemetry=sink,
            enrich=enrich,
            rerank=True,
            optimize_query=bool(optimize_query),
        )
    except Exception as exc:
        return _json.dumps(
            {
                "papers": [],
                "warnings": [],
                "error": f"web_search_failed: {exc}",
            }
        )

    # Build response payload — scan buffered events for per-provider hit counts
    by_provider: dict[str, int] = {}
    for ev in getattr(sink, "events", []) or []:
        if ev.get("kind") == "provider_progress" and ev.get("phase") == "done":
            by_provider.update(ev.get("by_provider", {}) or {})

    serialised: list[dict] = []
    for p in papers:
        serialised.append(
            {
                "title": p.title,
                "authors": [a.name for a in (p.authors or [])],
                "year": p.year,
                "journal": p.journal,
                "doi": p.doi,
                "url": p.url,
                "abstract": p.abstract,
                "discovery_sources": list(p.discovery_sources or []),
                "enrichment_sources": list(p.enrichment_sources or []),
            }
        )

    return _json.dumps(
        {
            "papers": serialised,
            "warnings": [],  # provider-level warnings flow via search_with_warnings;
            # for direct web_search the aggregator surfaces them
            # in logs, not in the payload (yet).
            "telemetry_summary": {"by_provider": by_provider},
        }
    )


# =============================================================================
# cancel_task — abort an in-flight generate_report / search_to_kb / web_search
# =============================================================================


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """Mark a running MCP task as cancelled.

    Long-running tools (``generate_report``, ``search_to_kb``,
    ``web_search``) check the cancellation registry at safe points
    (between RAG cycles / batches / iterations) and return early
    when the registry says their task_id has been cancelled.

    The task_id is the same one returned in the first progress
    notification of the cancellable tool's response.

    Returns:
        JSON: {"ok": true, "task_id": str, "was_running": bool}
        ``was_running`` is best-effort — we cannot perfectly distinguish
        a task that already finished from one that never existed.
    """
    import json as _json

    from perspicacite.rag.cancellation import mark_cancelled

    if not task_id:
        return _json.dumps({"ok": False, "error": "missing task_id"})
    await mark_cancelled(task_id)
    return _json.dumps(
        {
            "ok": True,
            "task_id": task_id,
            "was_running": True,  # see docstring — best-effort
        }
    )


# =============================================================================
# Tool: search_by_passage
# =============================================================================


@mcp.tool()
async def search_by_passage(
    text: str,
    kb_name: str = "default",
    kb_names: list[str] | None = None,
    k: int = 5,
    min_score: float | None = None,
) -> str:
    """
    Retrieve KB passages semantically similar to an arbitrary input text
    (a sentence, paragraph, or claim you already have in hand).

    When to use (vs ``search_knowledge_base``): use this when your input IS a
    chunk of source text — e.g. you want to find supporting/related passages
    for a sentence you are about to write, check whether a claim is backed by
    the KB, or de-duplicate against existing material. Reach for
    ``search_knowledge_base`` instead when you have a *question* and want a
    synthesized answer rather than raw matching chunks. Unlike
    ``search_knowledge_base``, every match here carries ``license_id`` and a
    structured ``source`` record so the caller can make citation decisions on
    the returned chunks. See also ``get_relevant_passages`` for keyword/query
    style retrieval with an adaptive empty-result retry.

    Args:
        text: Free-form input text (sentence, paragraph, claim). 1–4000 chars.
        kb_name: Single-KB scope (default "default"); used when ``kb_names``
            is None or has 1 entry.
        kb_names: Optional list of KBs to query together (default None). All
            must share the same embedding model; when len > 1 it supersedes
            ``kb_name``.
        k: Top-k matches to return (default 5, max 50).
        min_score: Optional similarity floor in [0, 1] (default None); matches
            scoring below it are dropped.

    Returns:
        A JSON string. On success: {"success": True, "results": [{chunk_id,
        chunk_text, score, source: {doi, title, authors, year, bibkey,
        source_url, license_id}, kb_name}, ...]}. On failure:
        {"success": False, "error": "..."}.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        from perspicacite.retrieval.passage_search import search_passages

        # Multi-KB path
        if kb_names and len(kb_names) > 1:
            from perspicacite.retrieval.multi_kb import (
                MultiKBRetriever,
                check_embedding_compat,
            )

            metas = [await state.session_store.get_kb_metadata(n) for n in kb_names]
            for i, meta in enumerate(metas):
                if meta is None:
                    return _json_error(f"Knowledge base not found: {kb_names[i]}")
            compat_msg = check_embedding_compat(metas)
            if compat_msg:
                return _json_error(compat_msg)

            retriever = MultiKBRetriever(
                vector_store=state.vector_store,
                embedding_service=state.embedding_provider,
                kb_metas=metas,
            )
        else:
            from perspicacite.models.kb import chroma_collection_name_for_kb
            from perspicacite.rag.dynamic_kb import (
                DynamicKnowledgeBase,
                KnowledgeBaseConfig,
            )

            effective_kb = kb_names[0] if (kb_names and len(kb_names) == 1) else kb_name
            kb_meta = await state.session_store.get_kb_metadata(effective_kb)
            if not kb_meta:
                return _json_error(f"Knowledge base '{effective_kb}' not found")
            retriever = DynamicKnowledgeBase(
                state.vector_store,
                state.embedding_provider,
                config=KnowledgeBaseConfig(
                    vector_size=state.embedding_provider.dimension,
                ),
            )
            retriever.collection_name = chroma_collection_name_for_kb(effective_kb)
            retriever._initialized = True

        matches = await search_passages(retriever, text=text, k=k, min_score=min_score)

        return _json_ok(
            {
                "results": [
                    {
                        "chunk_id": m.chunk_id,
                        "chunk_text": m.chunk_text,
                        "score": m.score,
                        "source": {
                            "doi": m.source.doi,
                            "title": m.source.title,
                            "authors": m.source.authors,
                            "year": m.source.year,
                            "bibkey": m.source.bibkey,
                            "source_url": m.source.source_url,
                            "license_id": m.source.license_id,
                        },
                        "kb_name": m.kb_name,
                    }
                    for m in matches
                ]
            }
        )

    except ValueError as e:
        return _json_error(str(e))
    except Exception as e:
        logger.error("mcp_search_by_passage_error", error=str(e))
        return _json_error(f"search_by_passage failed: {e}")


# =============================================================================
# Tool: get_relevant_passages (with adaptive retry)
# =============================================================================


async def _rephrase_query(query: str, *, context: str | None = None) -> str | None:
    """Wrap the search.query_optimizer for one-shot rephrasing.

    Returns None when the optimizer can't suggest a rewrite (we then bail
    on adaptive retry rather than loop). Internal helper; patched in tests.

    Note: the underlying ``optimize_query`` takes an ``app_state`` kwarg and
    returns an ``OptimizationResult``. We pass ``mcp_state`` (which exposes
    the same ``config`` / ``llm_client`` attributes the optimizer needs) and
    map the result back to a plain string (or ``None`` when no rewrite was
    applied).
    """
    try:
        from perspicacite.search.query_optimizer import optimize_query

        if not mcp_state.initialized or mcp_state.config is None:
            return None

        result = await optimize_query(
            query=query,
            context=context,
            app_state=mcp_state,
            optimize_enabled=True,
        )
        if not result or not result.applied:
            return None
        refined = (result.searched_query or "").strip()
        if not refined or refined == query.strip():
            return None
        return refined
    except Exception as e:
        logger.warning("query_rephrase_failed", error=str(e), query=query)
        return None


@mcp.tool()
async def get_relevant_passages(
    query: str,
    kb_name: str = "default",
    kb_names: list[str] | None = None,
    k: int = 10,
    paper_doi: str | None = None,
    adaptive: bool = False,
) -> str:
    """
    Keyword/query-style passage retrieval with an optional adaptive re-query
    when the first attempt returns nothing.

    When to use (vs ``search_by_passage``): use this when your input is a
    *search query* — keywords or a short prompt — rather than a verbatim chunk
    of source text. It is the better default when you are exploring ("what
    does the KB say about X?") and want raw passages back. Use
    ``search_by_passage`` instead when you already hold a sentence/paragraph
    and want passages similar to that exact text. Setting ``adaptive=True``
    makes the server run the query optimizer once and retry if the first pass
    finds zero passages — handy for terse or jargon-heavy queries. The
    response always reports ``attempts`` (1 or 2 entries) so the caller can
    see whether the retry fired and what the rephrased query was.

    Args:
        query: Search query (keywords / short prompt).
        kb_name: Single-KB scope (default "default").
        kb_names: Optional multi-KB list (default None); all KBs must share the
            same embedding model. When len > 1 it supersedes ``kb_name``.
        k: Top-k passages per attempt (default 10, max 50).
        paper_doi: Optional DOI scope-filter (default None; reserved, not yet
            enforced).
        adaptive: When True (default False), retry once with a rephrased query
            if the first attempt returns no passages.

    Returns:
        A JSON string. On success: {"success": True, "passages": [{text,
        source_doi, source_url, license_id, score, kb_name}, ...], "attempts":
        [{query, hit_count}, ...], "refined_query": "..." | None}. On failure:
        {"success": False, "error": "..."}.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        from perspicacite.retrieval.passage_search import search_passages

        # Build retriever (same pattern as search_by_passage).
        if kb_names and len(kb_names) > 1:
            from perspicacite.retrieval.multi_kb import (
                MultiKBRetriever,
                check_embedding_compat,
            )

            metas = [await state.session_store.get_kb_metadata(n) for n in kb_names]
            for i, meta in enumerate(metas):
                if meta is None:
                    return _json_error(f"Knowledge base not found: {kb_names[i]}")
            compat_msg = check_embedding_compat(metas)
            if compat_msg:
                return _json_error(compat_msg)
            retriever = MultiKBRetriever(
                vector_store=state.vector_store,
                embedding_service=state.embedding_provider,
                kb_metas=metas,
            )
        else:
            from perspicacite.models.kb import chroma_collection_name_for_kb
            from perspicacite.rag.dynamic_kb import (
                DynamicKnowledgeBase,
                KnowledgeBaseConfig,
            )

            effective_kb = kb_names[0] if (kb_names and len(kb_names) == 1) else kb_name
            kb_meta = await state.session_store.get_kb_metadata(effective_kb)
            if not kb_meta:
                return _json_error(f"Knowledge base '{effective_kb}' not found")
            retriever = DynamicKnowledgeBase(
                state.vector_store,
                state.embedding_provider,
                config=KnowledgeBaseConfig(
                    vector_size=state.embedding_provider.dimension,
                ),
            )
            retriever.collection_name = chroma_collection_name_for_kb(effective_kb)
            retriever._initialized = True

        attempts: list[dict] = []
        matches = await search_passages(retriever, text=query, k=k)
        attempts.append({"query": query, "hit_count": len(matches)})
        refined: str | None = None

        if adaptive and not matches:
            refined = await _rephrase_query(query)
            if refined:
                matches = await search_passages(retriever, text=refined, k=k)
                attempts.append({"query": refined, "hit_count": len(matches)})

        return _json_ok(
            {
                "passage_count": len(matches),
                "passages": [
                    {
                        "text": m.chunk_text,
                        "source_doi": m.source.doi,
                        "source_url": m.source.source_url,
                        "license_id": m.source.license_id,
                        "score": m.score,
                        "kb_name": m.kb_name,
                    }
                    for m in matches
                ],
                "attempts": attempts,
                "refined_query": refined,
            }
        )

    except ValueError as e:
        return _json_error(str(e))
    except Exception as e:
        logger.error("mcp_get_relevant_passages_error", error=str(e))
        return _json_error(f"get_relevant_passages failed: {e}")


# =============================================================================
# Tool: extract_parameters_from_passages
# =============================================================================

_PARAM_EXTRACTION_PROMPT = """\
You are extracting numeric experimental or methodological parameters from
scientific passages. Return a JSON array of objects with keys:
  name, type ("numeric"|"categorical"), typical, units, min, max,
  source_doi, source_quote, confidence (0..1)

Only include parameters explicitly stated in the passages. If a value is
absent, omit that key. Skip parameters not relevant to {context}.
"""


def _passages_type_message(passages: object) -> str | None:
    """Actionable message when ``passages`` is the wrong type — typically a
    JSON-encoded string instead of the list of passage dicts returned by
    get_relevant_passages / search_by_passage. Returns ``None`` when acceptable."""
    if isinstance(passages, str):
        return (
            "passages must be a list of passage dicts from get_relevant_passages "
            "(or search_by_passage), not a string — did you JSON-encode it?"
        )
    return None


@mcp.tool()
async def extract_parameters_from_passages(
    passages: list[dict],
    context: str | None = None,
    parameter_families: list[str] | None = None,
    model: str | None = None,
) -> str:
    """
    Extract structured *numeric* parameters (thresholds, concentrations,
    ranges, temperatures, pH, durations) from a list of passages using an LLM
    with JSON-schema-style output.

    When to use (vs ``extract_failure_modes_from_passages``): use this when you
    want the quantitative settings a method depends on — the knobs and their
    values/units/ranges. Use ``extract_failure_modes_from_passages`` instead
    when you want the qualitative things that go wrong (symptoms, causes,
    mitigations). Typical pipeline: call ``search_by_passage`` /
    ``get_relevant_passages`` to gather candidate chunks, then feed the
    returned passage dicts straight into this tool. Only parameters explicitly
    stated in the passages are returned; ``license_id`` on each passage
    controls how quotes are handled.

    Args:
        passages: List of {text, source_doi, license_id?, source_url?} dicts —
            the shape returned by the passage-search tools. Passages with no
            ``text`` are skipped.
        context: Optional domain/skill hint to focus extraction (default None).
        parameter_families: Optional list of family names to bias the LLM
            toward (default None), e.g. ["threshold","concentration","pH",
            "temperature"].
        model: Optional LiteLLM-style "provider/model" override (default None
            uses the server's configured default model).

    Returns:
        A JSON string. On success: {"success": True, "parameters": [{name,
        type, typical, units, min, max, source_doi, source_quote, confidence},
        ...]}. On failure: {"success": False, "error": "..."}.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    if (msg := _passages_type_message(passages)) is not None:
        return _json_error(msg)

    try:
        from perspicacite.pipeline.extraction import (
            Passage,
            annotate_anchor_status,
            extract_structured,
            handle_quote_for_license,
        )

        passage_objs = [
            Passage(
                text=str(p.get("text", "")),
                source_doi=str(p.get("source_doi", "")),
                license_id=p.get("license_id"),
                source_url=p.get("source_url"),
            )
            for p in passages
            if p.get("text")
        ]

        families = parameter_families or [
            "threshold",
            "concentration",
            "pH",
            "temperature",
            "time",
            "rate",
        ]
        prompt = _PARAM_EXTRACTION_PROMPT.format(context=context or "general")
        prompt += f"\nFocus on these families when relevant: {', '.join(families)}."

        # Hard cap: entire extraction (all batches) must finish within 80s.
        try:
            async with asyncio.timeout(80.0):
                records = await extract_structured(
                    llm_client=state.llm_client,
                    passages=passage_objs,
                    prompt_template=prompt,
                    schema={},
                    what="parameters",
                    context=context,
                    dedup_key=lambda r: (r.get("name"), r.get("units")),
                    model=model,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "mcp_extract_parameters_timeout",
                n_passages=len(passage_objs),
                timeout_s=80.0,
            )
            records = []  # return empty list on timeout rather than error

        # R3 — verify each record's quote against the source passages BEFORE the
        # license-tier rewrite, so verification sees the model's original quote.
        records = annotate_anchor_status(records, passage_objs)

        # Apply license-tier policy to source_quote on each record.
        doi_to_license = {p.source_doi: p.license_id for p in passage_objs}
        cleaned: list[dict] = []
        for r in records:
            quote = r.get("source_quote")
            if quote:
                tier_quote = handle_quote_for_license(
                    str(quote),
                    license_id=doi_to_license.get(r.get("source_doi", "")),
                    paraphraser=None,  # MVP: drop when paraphraser is unavailable
                )
                if tier_quote is None:
                    r = {k: v for k, v in r.items() if k != "source_quote"}
                else:
                    r = {**r, "source_quote": tier_quote}
            cleaned.append(r)

        return _json_ok({"parameters": cleaned})

    except Exception as e:
        logger.error("mcp_extract_parameters_error", error=str(e))
        return _json_error(f"extract_parameters_from_passages failed: {e}")


# =============================================================================
# Tool: extract_failure_modes_from_passages
# =============================================================================

_FAILURE_EXTRACTION_PROMPT = """\
You are extracting failure modes, limitations, caveats, and pitfalls from
scientific passages. Return a JSON array of objects with keys:
  symptom (one sentence), root_cause, mitigation, source_doi,
  source_quote, confidence (0..1)

Only include failure modes explicitly stated. Skip generic disclaimers.
Domain context: {context}.
"""


@mcp.tool()
async def extract_failure_modes_from_passages(
    passages: list[dict],
    context: str | None = None,
    model: str | None = None,
) -> str:
    """
    Extract structured *failure modes* (symptoms, likely causes, and
    mitigations) from a list of passages using an LLM.

    When to use (vs ``extract_parameters_from_passages``): use this when you
    want the qualitative ways a method or system breaks — what goes wrong, why,
    and how to avoid it. Use ``extract_parameters_from_passages`` instead when
    you want quantitative settings (thresholds, concentrations, ranges).
    Typical pipeline: gather chunks with ``search_by_passage`` /
    ``get_relevant_passages``, then pass those passage dicts directly here.
    Records are de-duplicated by symptom, and each passage's ``license_id``
    governs how its source quote is handled.

    Args:
        passages: List of {text, source_doi, license_id?, source_url?} dicts —
            the shape returned by the passage-search tools. Passages with no
            ``text`` are skipped.
        context: Optional domain/skill hint to focus extraction (default None).
        model: Optional LiteLLM-style "provider/model" override (default None
            uses the server's configured default model).

    Returns:
        A JSON string. On success: {"success": True, "failure_modes":
        [{symptom, root_cause, mitigation, source_doi, source_quote,
        confidence}, ...]}. On failure: {"success": False, "error": "..."}.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    if (msg := _passages_type_message(passages)) is not None:
        return _json_error(msg)

    try:
        from perspicacite.pipeline.extraction import (
            Passage,
            annotate_anchor_status,
            extract_structured,
            handle_quote_for_license,
        )

        passage_objs = [
            Passage(
                text=str(p.get("text", "")),
                source_doi=str(p.get("source_doi", "")),
                license_id=p.get("license_id"),
                source_url=p.get("source_url"),
            )
            for p in passages
            if p.get("text")
        ]

        prompt = _FAILURE_EXTRACTION_PROMPT.format(context=context or "general")

        # Hard cap: entire extraction (all batches) must finish within 80s.
        # Per-batch asyncio.wait_for(50s) doesn't help when LiteLLM/tenacity
        # retries eat CancelledError — the outer timeout is the real guard.
        try:
            async with asyncio.timeout(80.0):
                records = await extract_structured(
                    llm_client=state.llm_client,
                    passages=passage_objs,
                    prompt_template=prompt,
                    schema={},
                    what="failure_modes",
                    context=context,
                    dedup_key=lambda r: (str(r.get("symptom", "")).strip().lower(),),
                    model=model,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "mcp_extract_failure_modes_timeout",
                n_passages=len(passage_objs),
                timeout_s=80.0,
            )
            records = []  # return empty list on timeout rather than error

        # R3 — verify quotes before the license-tier rewrite.
        records = annotate_anchor_status(records, passage_objs)

        doi_to_license = {p.source_doi: p.license_id for p in passage_objs}
        cleaned: list[dict] = []
        for r in records:
            quote = r.get("source_quote")
            if quote:
                tier_quote = handle_quote_for_license(
                    str(quote),
                    license_id=doi_to_license.get(r.get("source_doi", "")),
                    paraphraser=None,
                )
                if tier_quote is None:
                    r = {k: v for k, v in r.items() if k != "source_quote"}
                else:
                    r = {**r, "source_quote": tier_quote}
            cleaned.append(r)

        return _json_ok({"failure_modes": cleaned})

    except Exception as e:
        logger.error("mcp_extract_failure_modes_error", error=str(e))
        return _json_error(f"extract_failure_modes_from_passages failed: {e}")


@mcp.tool()
async def extract_claims_from_passages(
    passages: list[dict],
    context: str | None = None,
    model: str | None = None,
    domains: list[str] | None = None,
) -> str:
    """Extract typed scientific claims (Bucur 5-slot SuperPattern + ECO-typed
    evidence) from retrieved passages, validated against the indicium standard.

    Each claim has context/subject/qualifier/relation/object plus evidence
    (DOI + exact quote + ECO evidence_type), and an ``_anchor`` record
    {status, matched_index, quote_exact, score} from verifying its quote against
    the source passages (R3): status is verified / repaired / unverified, and
    quote_exact is present only when verified/repaired (never fabricated for an
    unverified quote). Returns
    {success, claims:[...], claims_valid: bool, validation_report?: str}.
    Use after retrieval (e.g. get_relevant_passages) to produce a machine-
    readable, standards-compliant claim set instead of prose.

    Args:
        passages: List of passage dicts (as returned by get_relevant_passages).
        context: Optional extraction context appended to the LLM prompt.
        model: Optional LLM model override.
        domains: Optional list of domain adapter IDs (e.g. ["metabolomics"]).
            Multiple IDs are composed into a single CompositeAdapter so all
            adapters' enrichment and SHACL shapes are applied together.
            Requires indicium-adapters to be installed; silently ignored if not.
    """
    state = _require_state()
    if isinstance(state, str):
        return state
    if (msg := _passages_type_message(passages)) is not None:
        return _json_error(msg)

    try:
        from perspicacite.pipeline.claims import extract_claims, validate_claims
    except ImportError:
        return _json_error("indicium not installed; reinstall with the 'indicia' extra")

    # Resolve domain adapter(s) — compose if multiple (best-effort; skip if not installed)
    adapter = None
    if domains:
        try:
            from indicium_adapters import discover_adapters, compose_adapters
            discovered = discover_adapters()
            valid = [discovered[d] for d in domains if d in discovered]
            adapter = compose_adapters(valid) if valid else None
        except ImportError:
            pass  # indicium-adapters not installed — proceed without adapter

    # get_relevant_passages emits flat {text, source_doi}; extract_claims and
    # anchor_claims both read {chunk_text, source:{doi}}. Normalize so the model
    # actually sees passage text AND the anchor verifier has real candidates —
    # without this, anchoring would silently see empty text and tag everything
    # unverified.
    norm_passages = [
        {
            "chunk_text": p.get("chunk_text") or p.get("text", ""),
            "source": {"doi": p.get("source_doi") or (p.get("source") or {}).get("doi")},
        }
        for p in passages
        if isinstance(p, dict)
    ]

    claims = await extract_claims(
        llm_client=state.llm_client, passages=norm_passages, context=context, model=model,
        domain_adapter=adapter)

    # R3 — verify each claim's quote against the passages it was drawn from and
    # tag it. Fail-open: every claim is kept, each carrying claim["_anchor"]
    # {status, matched_index, quote_exact, score}. norm_passages is index-aligned
    # with the list the model saw, so matched_index is meaningful.
    from perspicacite.indicium_layer.anchor import anchor_claims

    claims = anchor_claims(claims, norm_passages)

    conforms, report = validate_claims(claims, domain_adapter=adapter) if claims else (True, "")
    out: dict = {"claims": claims, "claims_valid": conforms}
    if not conforms:
        out["validation_report"] = report
    return _json_ok(out)


@mcp.tool()
async def export_astra(claims: list[dict]) -> str:
    """Project indicium claims onto ASTRA Insights (astra-spec.org).

    Returns {success, insights:[{id, claim, evidence:[{doi|artifact, quote:{exact}}]}}]}.
    Each claim is flattened (5-slot SuperPattern -> a claim string); full typing
    stays in the claim set. Claims without an 'id' get a positional one.
    Use to hand a Perspicacité claim set to an ASTRA-consuming analysis tool.
    """
    state = _require_state()
    if isinstance(state, str):
        return state
    try:
        from indicium import claim_to_insight
    except ImportError:
        return _json_error("indicium not installed; reinstall with the 'indicia' extra")
    insights = []
    for i, c in enumerate(claims):
        c = {**c, "id": c.get("id", f"c{i}")}
        insights.append(claim_to_insight(c))
    return _json_ok({"insights": insights})


@mcp.tool()
async def build_claim_graph(
    kb_name: str,
    *,
    refresh: bool = False,
    max_pairs_per_claim: int = 20,
    model: str | None = None,
) -> str:
    """Build (or incrementally refresh) the claim graph for a KB.

    Extracts indicium claims from every passage, SHACL-validates, upserts
    Claim/Evidence/Passage/Paper nodes, prunes candidate pairs, then runs a
    CiTO classification pass. Outputs go to ``data/claim_graphs/<kb_name>/``.

    Args:
        kb_name: KB to build.
        refresh: When True, ignore manifest and rebuild every paper.
        max_pairs_per_claim: Cap on per-claim CiTO fan-out (default 20).
        model: Optional LLM override (default: server's configured model).

    Returns:
        JSON: ``{success, kb_name, claims_added, edges_added,
                 pairs_classified, papers_processed, duration_seconds}``.
    """
    state = _require_state()
    if isinstance(state, str):
        return state
    try:
        import indicium  # noqa: F401
    except ImportError:
        return _json_error("indicia extra not installed; uv sync --extra indicia")

    import pathlib

    from perspicacite.indicium_layer.builder import build_claim_graph as _build
    from perspicacite.indicium_layer.store import ClaimGraphStore
    from perspicacite.models.kb import chroma_collection_name_for_kb

    collection = chroma_collection_name_for_kb(kb_name)

    # Fetch all papers and chunks async, then expose via sync providers to builder.
    papers_list = await state.vector_store.list_paper_metadata(collection)
    papers_dict = {p["paper_id"]: p for p in papers_list}
    paper_ids = list(papers_dict.keys())
    chunks_list = await state.vector_store.get_chunks_by_paper_ids(collection, paper_ids)

    passages_by_pid: dict[str, list[dict]] = {pid: [] for pid in paper_ids}
    for chunk in chunks_list:
        pid = chunk.metadata.paper_id
        cs = chunk.metadata.char_span
        passages_by_pid.setdefault(pid, []).append(
            {
                "chunk_idx": chunk.metadata.chunk_index,
                "text": chunk.text,
                "char_start": cs[0] if cs else None,
                "char_end": cs[1] if cs else None,
            }
        )

    data_dir = pathlib.Path("data/claim_graphs") / kb_name
    store = ClaimGraphStore(kb_name, data_dir=data_dir, backend="oxigraph")

    # Resolve the model: explicit override takes priority; otherwise use the
    # "claim_graph" stage config (falls back to default_model).  Passing an
    # explicit model to _build bypasses the free-auto chain so structured-output
    # capable models are used for claim extraction.
    if model is None:
        from perspicacite.llm.client import resolve_stage_model

        _, model = resolve_stage_model(state.config, "claim_graph")

    try:
        result = await _build(
            kb_name=kb_name,
            store=store,
            llm_client=state.llm_client,
            papers_provider=lambda: papers_dict,
            passages_provider=lambda pid: passages_by_pid.get(pid, []),
            refresh=refresh,
            max_pairs_per_claim=max_pairs_per_claim,
            model=model,
            anchor_strict=state.config.anchor.strict,
            anchor_near_threshold=state.config.anchor.near_threshold,
            anchor_audit_dir=state.config.anchor.audit_dir,
        )
    finally:
        store.close()

    return _json_ok(
        {
            "kb_name": result.kb_name,
            "claims_added": result.claims_added,
            "edges_added": result.edges_added,
            "pairs_classified": result.pairs_classified,
            "papers_processed": result.papers_processed,
            "duration_seconds": result.duration_seconds,
        }
    )


@mcp.tool()
async def claim_graph_status(kb_name: str) -> str:
    """Return the claim-graph manifest + last build summary for a KB.

    Returns:
        JSON: ``{success, kb_name, paper_count, indicium_schema_version,
                 builder_version, last_build_iso, schema_drift}``.
    """
    try:
        import indicium  # noqa: F401
    except ImportError:
        return _json_error("indicia extra not installed; uv sync --extra indicia")
    from perspicacite.indicium_layer.invalidation import schema_version_changed
    from perspicacite.indicium_layer.manifest import read_manifest

    m = read_manifest(kb_name)
    return _json_ok(
        {
            "kb_name": m.kb_name,
            "paper_count": len(m.paper_hashes),
            "indicium_schema_version": m.indicium_schema_version,
            "builder_version": m.builder_version,
            "last_build_iso": m.last_build_iso,
            "schema_drift": schema_version_changed(m),
        }
    )


def _open_claim_graph_store_for_kb(kb_name: str):
    """Test seam: lets tests substitute an in-memory store."""
    import pathlib

    from perspicacite.indicium_layer.store import ClaimGraphStore

    data_dir = pathlib.Path("data/claim_graphs") / kb_name
    return ClaimGraphStore(kb_name, data_dir=data_dir, backend="oxigraph")


@mcp.tool()
async def query_claim_graph(
    kb_name: str,
    query_name: str,
    kwargs: dict | None = None,
) -> str:
    """Run a typed query against a KB's claim graph.

    Available queries (see indicium_layer.queries):
      - claims_supporting(subject_or_iri, min_eco_grade=None)
      - claims_disputing(target_iri)
      - evidence_trace(claim_iri, max_depth=3)
      - papers_with_claim_pattern(subject=None, relation=None, object=None)
      - neighbors(claim_iri, edge_types=None)

    Args:
        kb_name: KB whose claim graph to query.
        query_name: One of the five typed queries above.
        kwargs: Query-specific keyword arguments (see signatures).

    Returns:
        JSON: ``{success, query, kb_name, rows: [{...}, ...]}``.
    """
    try:
        import indicium  # noqa: F401
    except ImportError:
        return _json_error("indicia extra not installed; uv sync --extra indicia")

    from perspicacite.indicium_layer import queries as _q

    _query_table = {
        "claims_supporting": _q.claims_supporting,
        "claims_disputing": _q.claims_disputing,
        "evidence_trace": _q.evidence_trace,
        "papers_with_claim_pattern": _q.papers_with_claim_pattern,
        "neighbors": _q.neighbors,
    }
    fn = _query_table.get(query_name)
    if fn is None:
        return _json_error(f"unknown query_name: {query_name}; valid: {sorted(_query_table)}")
    store = _open_claim_graph_store_for_kb(kb_name)
    try:
        rows = fn(store, kb_name, **(kwargs or {}))
    finally:
        store.close()
    return _json_ok(
        {
            "query": query_name,
            "kb_name": kb_name,
            "rows": rows,
        }
    )


@mcp.tool()
async def get_claim_figures(
    kb_name: str,
    claim_iri: str,
) -> str:
    """Return Figure nodes associated with a claim in the KB's claim graph.

    Traverses the claim graph for Figure nodes linked to the given claim
    IRI via prov:wasDerivedFrom.  Useful for surfacing the visual evidence
    behind a claim (microscopy plates, data plots, etc.) when the claim was
    grounded in a figure rather than a text passage.

    Args:
        kb_name:   KB whose claim graph to query.
        claim_iri: IRI of the claim (e.g. ``kb://my_kb/claim/abc123``).

    Returns:
        JSON ``{success, kb_name, claim_iri, figures: [{figure, figure_id,
        caption, figure_type, source_doi}]}``
    """
    try:
        import indicium  # noqa: F401
    except ImportError:
        return _json_error("indicia extra not installed; uv sync --extra indicia")

    from perspicacite.indicium_layer.queries import figures_for_claim

    store = _open_claim_graph_store_for_kb(kb_name)
    try:
        rows = figures_for_claim(store, kb_name, claim_iri)
    finally:
        store.close()

    return _json_ok({
        "kb_name": kb_name,
        "claim_iri": claim_iri,
        "figures": rows,
    })


@mcp.tool()
async def get_claim_links(
    kb_name: str,
    claim_iri: str,
) -> str:
    """Return ClaimLink nodes where the given claim is from_claim or to_claim.

    Queries the KB's claim graph for all ClaimLink nodes (typed cito:Citation
    per Indicium v1.3) where from_claim or to_claim equals claim_iri. Returns
    both outgoing edges (where this claim makes an assertion about another) and
    incoming edges (where another claim references this one).

    Requires build_claim_graph to have been run on the KB first. ClaimLinks are
    written using the full 13-predicate CiTO vocabulary: supports, disputes,
    qualifies, citesForInformation, contradicts, refines, extends, agreesWith,
    disagreesWith, updates, corrects, usesDataFrom, usesMethodIn.

    Args:
        kb_name:   KB whose claim graph to query.
        claim_iri: IRI of the claim (e.g. ``kb://my_kb/claim/abc123``).

    Returns:
        JSON ``{success, kb_name, claim_iri, links: [{link_iri, from_claim,
        to_claim, link_type, direction}]}`` where direction is "outgoing" or
        "incoming".
    """
    try:
        import indicium  # noqa: F401
    except ImportError:
        return _json_error("indicia extra not installed; uv sync --extra indicia")

    from perspicacite.indicium_layer.queries import claim_links_for_claim

    store = _open_claim_graph_store_for_kb(kb_name)
    try:
        rows = claim_links_for_claim(store, kb_name, claim_iri)
    finally:
        store.close()

    return _json_ok({
        "kb_name": kb_name,
        "claim_iri": claim_iri,
        "links": rows,
    })


@mcp.tool()
async def claim_graph_export(
    kb_name: str,
    format: str = "nquads",
) -> str:
    """Export a KB's claim graph as N-Quads, Turtle, JSON-LD, or RO-Crate.

    Serialises every triple in the KB's indicium claim graph to the requested
    format and returns the result inline. Useful for archiving a KB's claim
    graph, feeding it to an external triple-store, or packaging it as an
    RO-Crate research object.

    Args:
        kb_name: KB whose claim graph to export.
        format: Serialisation format — one of ``"nquads"``, ``"turtle"``,
            ``"jsonld"``, ``"rocrate"`` (default: ``"nquads"``).

    Returns:
        JSON ``{success, kb_name, format, data}`` where ``data`` is a string
        (nquads/turtle) or a list/dict (jsonld/rocrate).
    """
    try:
        import indicium  # noqa: F401
    except ImportError:
        return _json_error("indicia extra not installed; uv sync --extra indicia")

    _valid_formats = ("nquads", "turtle", "jsonld", "rocrate")
    if format not in _valid_formats:
        return _json_error(
            f"unsupported format '{format}'; choose one of: {list(_valid_formats)}"
        )

    store = _open_claim_graph_store_for_kb(kb_name)
    try:
        if store._backend == "memory":
            # rdflib ConjunctiveGraph — full serialisation support
            assert store._g is not None
            if format == "nquads":
                result: Any = store._g.serialize(format="nquads")
            elif format == "turtle":
                serialised = store._g.serialize(format="turtle")
                result = serialised
            else:
                raw_jld = store._g.serialize(format="json-ld")
                try:
                    jld_list = json.loads(raw_jld) if isinstance(raw_jld, str) else raw_jld
                except Exception:
                    jld_list = []
                if format == "jsonld":
                    result = jld_list
                else:  # rocrate
                    result = {
                        "@context": "https://w3id.org/ro/crate/1.1/context",
                        "@graph": [
                            {
                                "@type": "CreativeWork",
                                "@id": "ro-crate-metadata.json",
                                "about": {"@id": "./"},
                                "conformsTo": {
                                    "@id": "https://w3id.org/ro/crate/1.1"
                                },
                            },
                            {
                                "@type": "Dataset",
                                "@id": "./",
                                "name": f"Perspicacite claim graph: {kb_name}",
                                "hasPart": [{"@id": "claim-graph.jsonld"}],
                            },
                        ]
                        + (jld_list if isinstance(jld_list, list) else []),
                    }
        else:
            # oxigraph backend — Store only supports quad formats (N-Quads/TriG); round-trip
            # via N-Quads→rdflib for all three output formats since pyoxigraph has no JSON-LD
            # serialiser and text/turtle / n-triples are triples-only formats that raise
            # "A RDF format supporting datasets was expected" on a named-graph Store.
            import io

            from rdflib import ConjunctiveGraph as _CG

            buf = io.BytesIO()
            store._oxistore.dump(buf, "application/n-quads")
            g = _CG()
            g.parse(data=buf.getvalue().decode("utf-8"), format="nquads")

            if format == "nquads":
                result = buf.getvalue().decode("utf-8")
            elif format == "turtle":
                result = g.serialize(format="turtle")
            else:
                raw_jld = g.serialize(format="json-ld")
                try:
                    jld_list = json.loads(raw_jld) if isinstance(raw_jld, str) else raw_jld
                except Exception:
                    jld_list = []
                if format == "jsonld":
                    result = jld_list
                else:  # rocrate
                    result = {
                        "@context": "https://w3id.org/ro/crate/1.1/context",
                        "@graph": [
                            {
                                "@type": "CreativeWork",
                                "@id": "ro-crate-metadata.json",
                                "about": {"@id": "./"},
                                "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                            },
                            {
                                "@type": "Dataset",
                                "@id": "./",
                                "name": f"Perspicacite claim graph: {kb_name}",
                                "hasPart": [{"@id": "claim-graph.jsonld"}],
                            },
                        ]
                        + (jld_list if isinstance(jld_list, list) else []),
                    }
    finally:
        store.close()

    return _json_ok({"kb_name": kb_name, "format": format, "data": result})


@mcp.tool()
async def suggest_databases(query: str, hints: list[str] | None = None) -> str:
    """
    Recommend which literature databases to search for a given query.

    Use this BEFORE ``search_literature`` or ``generate_report`` when you are
    unsure which databases to pass: it returns a topic-relevant shortlist so a
    search hits the right sources instead of a blind broad sweep.

    Deterministic: the recommendation comes from a keyword topic heuristic over
    the query (plus optional ``hints``); no LLM is involved, so the same input
    always yields the same output. Examples: biomedical → pubmed/europepmc;
    machine learning / physics → arxiv; high-energy physics → inspire;
    chemistry → pubchem. A broad default (semantic_scholar, openalex, crossref)
    is always included so a recommendation is never empty.

    Args:
        query: The research question or topic to search for.
        hints: Optional extra terms (e.g. ["chemistry"]) to steer the topic match.

    Returns:
        JSON {"success": True, "recommended": [...], "reasoning": "...",
        "all_known": [...]} where ``all_known`` is the sorted set of every
        database name accepted by ``search_literature``.
    """
    from perspicacite.search.database_advisor import suggest_databases_for_query
    from perspicacite.search.scilex_adapter import KNOWN_DATABASES

    suggestion = suggest_databases_for_query(query, hints=hints)
    return _json_ok(
        {
            "recommended": suggestion.databases,
            "reasoning": suggestion.reasoning,
            "all_known": sorted(KNOWN_DATABASES),
        }
    )


@mcp.tool()
async def get_usage_guide() -> str:
    """
    Return the authoritative guide to using Perspicacité over MCP.

    Call this FIRST when planning multi-step research: it returns the server's
    capabilities, decision rules (translate non-English queries, set
    ``optimize_query``, call ``suggest_databases``, pick the right tool and
    mode/screening, read the ``{success}`` envelope), a documented entry for
    every registered tool (``name``, ``purpose``, ``when_to_use``, ``key_knobs``),
    and recommended knob defaults.

    Returns:
        JSON {"success": True, "capabilities": [...], "decision_rules": [...],
        "tools": [...], "knob_defaults": {...}}.
    """
    from perspicacite.mcp.usage_guide import build_usage_guide

    return _json_ok(build_usage_guide())


_TOOL_NAMES: list[str] = [
    "search_literature",
    "get_paper_content",
    "get_paper_references",
    "list_knowledge_bases",
    "search_knowledge_base",
    "search_skill_kb",
    "search_by_passage",
    "get_relevant_passages",
    "extract_parameters_from_passages",
    "extract_failure_modes_from_passages",
    "create_knowledge_base",
    "add_papers_to_kb",
    "generate_report",
    "screen_papers",
    "add_dois_to_kb",
    "push_to_zotero",
    "push_notes_to_zotero",
    "build_kbs_from_zotero",
    "ingest_local_documents",
    "add_local_papers_to_kb",
    "build_capsule",
    "fetch_supplementary",
    "build_capsules_for_kb",
    "fetch_paper_resources",
    "route_kbs",
    "build_kb_from_search",
    "export_kb",
    "expand_kb_via_citations",
    "delete_knowledge_base",
    "enrich_kb_from_cite_graph_tool",
    "zotero_list_collections",
    "zotero_get_collection_items",
    "zotero_get_paper_resources",
    "zotero_ingest_collection_to_kb",
    "ingest_asb_run",
    "ingest_github_repo",
    "ingest_skill_bundle",
    "web_search",
    "cancel_task",
    "suggest_databases",
    "build_claim_graph",
    "claim_graph_status",
    "query_claim_graph",
    "get_claim_figures",
    "get_claim_links",
    "claim_graph_export",
    "get_usage_guide",
    "extract_claims_from_passages",
    "export_astra",
]


@mcp.resource("perspicacite://info")
async def get_info() -> str:
    """Perspicacité capabilities and status."""
    return json.dumps(
        {
            "name": "Perspicacité v2",
            "description": "AI-powered scientific literature research assistant",
            "tools": _TOOL_NAMES,
            "tool_count": len(_TOOL_NAMES),
            "initialized": mcp_state.initialized,
        }
    )


# =============================================================================
# KB resources (Wave 5.1)
# =============================================================================

from perspicacite.mcp import resources as _resources  # noqa: E402


@mcp.resource("perspicacite://kbs")
async def _kbs_resource() -> str:
    """List all knowledge bases."""
    return await _resources.kbs_resource()


@mcp.resource("perspicacite://kb/{name}")
async def _kb_resource(name: str) -> str:
    """Metadata + sub-resource URIs for a single KB."""
    return await _resources.kb_resource(name)


@mcp.resource("perspicacite://kb/{name}/papers")
async def _kb_papers_resource(name: str) -> str:
    """Papers in a KB, sourced from the kb_log (fallback to Chroma)."""
    return await _resources.kb_papers_resource(name)


@mcp.resource("perspicacite://kb/{name}/log")
async def _kb_log_resource(name: str) -> str:
    """Most-recent KB-log events (bounded by kb.mcp_resource_max_events)."""
    return await _resources.kb_log_resource(name)


# =============================================================================
# Canned prompts (Wave 5.2)
# =============================================================================

from perspicacite.mcp import prompts as _prompts  # noqa: E402


@mcp.prompt()
def literature_review(
    topic: str, kb_name: str | None = None, max_papers: int = 30
) -> list[dict[str, Any]]:
    """Run a literature review on a topic, optionally against a KB."""
    return _prompts.literature_review(topic, kb_name, max_papers)


@mcp.prompt()
def compare_papers(paper_a: str, paper_b: str, kb_name: str | None = None) -> list[dict[str, Any]]:
    """Compare two papers side-by-side."""
    return _prompts.compare_papers(paper_a, paper_b, kb_name)


@mcp.prompt()
def summarize_kb(kb_name: str, max_papers: int = 50) -> list[dict[str, Any]]:
    """Summarize an entire knowledge base in five paragraphs."""
    return _prompts.summarize_kb(kb_name, max_papers)


@mcp.prompt()
def ingest_dois(kb_name: str, dois: list[str]) -> list[dict[str, Any]]:
    """Ingest a list of DOIs into a KB."""
    return _prompts.ingest_dois(kb_name, dois)


@mcp.prompt()
def screen_topic(topic: str, kb_name: str, threshold: float = 0.6) -> list[dict[str, Any]]:
    """Screen a KB for papers relevant to a topic."""
    return _prompts.screen_topic(topic, kb_name, threshold)
