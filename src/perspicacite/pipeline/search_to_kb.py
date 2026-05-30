"""SciLEx-driven KB build / enrich pipeline.

Glues three existing pieces together so the user can grow a KB from a
plain-text query:

1. :class:`perspicacite.search.scilex_adapter.SciLExAdapter` — multi-DB
   academic search (Semantic Scholar, OpenAlex, PubMed, arXiv, …).
2. Light client-side filters (year, citations, abstract presence, DOI
   presence) so we don't waste a PDF fetch on garbage hits.
3. The same DOI → PDF → chunk → embed pipeline that ``add_dois_to_kb``
   already uses.

The big win over "search then manually paste DOIs into add_dois_to_kb"
is that this is one tool, callable from MCP or CLI, with the de-dup
and KB-auto-create UX baked in. Claude Code (or any agent) can call
``build_kb_from_search`` after running its own exploratory queries to
spin up a focused KB before doing real RAG over it.

Public surface:

- :func:`run_search` — query SciLEx, return raw :class:`Paper` list.
- :func:`apply_filters` — pure function over :class:`Paper` list.
- :func:`ingest_dois_into_kb` — shared "add these DOIs to this KB"
  implementation reused by MCP and CLI.
- :func:`search_filter_and_ingest` — the one-shot orchestrator that
  ties the three together.

Nothing here owns any state; the caller passes ``app_state`` (the
already-initialized :class:`perspicacite.web.state.AppState`) so we
share its vector store / embedding provider / session store / PDF
parser. That keeps test wiring trivial and avoids re-initializing
expensive resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.search_to_kb")


@dataclass
class SearchFilter:
    """Client-side filters applied to a SciLEx hit list.

    All filters are AND-combined; ``None`` means "no constraint". The
    intent is to drop obvious non-fits before paying for a PDF fetch,
    not to do real screening — call ``screen_papers`` separately for
    LLM-based relevance grading.
    """

    min_year: int | None = None
    max_year: int | None = None
    min_citations: int | None = None
    require_doi: bool = True
    require_abstract: bool = False


# Tiny stopword list — enough to skim generic filler off KB title text.
# Not trying to be a real NLP toolkit; KB-aware expansion just wants
# the topic-bearing nouns to surface so SciLEx finds adjacent papers.
_KB_AWARE_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "from", "by", "with", "is", "are", "was", "were", "be", "been",
    "being", "as", "this", "that", "these", "those", "we", "our",
    "their", "his", "her", "its", "it", "into", "via", "using", "use",
    "used", "uses", "show", "shown", "study", "studies", "based",
    "novel", "new", "method", "methods", "approach", "paper", "papers",
    "result", "results", "data", "datasets",
    "imported", "references", "smoke", "test", "audit", "demo", "kb",
}


def _kb_aware_term_candidates(text: str) -> list[str]:
    """Tokenize ``text`` to lower-cased alpha words (>=4 chars), with
    stopwords removed. Used both for description and per-title term
    extraction."""
    import re
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
    return [
        t.lower() for t in tokens
        if len(t) >= 4 and t.lower() not in _KB_AWARE_STOPWORDS
    ]


_REPHRASE_PROMPT = """You are helping search scientific literature. The user wants to
maximise coverage of papers on this topic by issuing multiple SciLEx
queries with different phrasings. Generate {n} alternate phrasings of
the query below. Each phrasing should:

- Preserve the scientific meaning exactly.
- Use different terminology — synonyms, controlled vocabulary
  (MeSH-style), spelled-out vs abbreviated forms, related sub-fields.
- Be one short phrase (3-10 words). No leading numbers or bullets.

Return JSON only, in this exact shape:
{{"variants": ["...", "...", "..."]}}

Original query:
{query}
"""


async def rephrase_query(
    *,
    query: str,
    num_variants: int,
    llm_client: Any,
    model: str = "claude-haiku-4-5",
    provider: str = "anthropic",
) -> list[str]:
    """Generate ``num_variants`` alternate phrasings via one LLM call.

    Returns ``[]`` on parse failure or missing client — callers should
    treat that as "no rephrase, fall back to single-query search".
    The original query is *not* included in the return value; callers
    add it explicitly to the search loop.
    """
    if num_variants < 1 or llm_client is None:
        return []
    import json as _json
    prompt = _REPHRASE_PROMPT.format(n=num_variants, query=query)
    try:
        text = await llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            provider=provider,
            max_tokens=400,
            temperature=0.4,
            stage="search_to_kb_rephrase",
        )
    except Exception as exc:
        logger.warning("rephrase_query_llm_failed", error=str(exc))
        return []
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.strip().rstrip("`")
    try:
        obj = _json.loads(cleaned)
    except Exception as exc:
        logger.warning(
            "rephrase_query_unparseable",
            error=str(exc), sample=cleaned[:200],
        )
        return []
    variants = obj.get("variants") or []
    out: list[str] = []
    seen: set[str] = {query.strip().lower()}
    for v in variants:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out[:num_variants]


async def kb_aware_query(
    *,
    app_state: Any,
    kb_name: str,
    query: str,
    max_terms: int = 8,
    sample_papers: int = 20,
) -> tuple[str, list[str]]:
    """Augment ``query`` with KB-derived topic terms.

    Returns ``(augmented_query, injected_terms)``. The augmented query
    is the original ``query`` followed by the top-N most-frequent
    content words across the KB's description + sampled paper titles
    (after stopword removal). This biases SciLEx toward the KB's
    existing topic surface without overriding the user's intent.

    When the KB doesn't exist or has no titles/description we return
    ``(query, [])`` unchanged — callers can treat the empty injection
    as "KB-aware was a no-op".

    Why simple frequency and not an LLM rephrasing call:
    - Free, deterministic, debuggable (the user sees the injected terms).
    - Works on KBs of any size, including 1-paper KBs.
    - LLM rephrasing is a separate feature (--rephrase) that handles
      the "find lexically different but semantically equivalent
      phrasings" problem.
    """
    kb_meta = await app_state.session_store.get_kb_metadata(kb_name)
    if not kb_meta:
        return query, []
    desc = (kb_meta.description or "").strip()
    # Skip generic placeholder descriptions — same set the kb_router uses.
    if desc.lower() in {
        "imported from references.bib", "smoke test", "audit test kb",
        "test", "demo",
    }:
        desc = ""
    from perspicacite.models.kb import chroma_collection_name_for_kb
    collection = kb_meta.collection_name or chroma_collection_name_for_kb(kb_name)
    try:
        rows = await app_state.vector_store.list_paper_metadata(collection)
    except Exception as exc:
        logger.warning("kb paper metadata listing failed", error=str(exc))
        rows = []
    titles = [r.get("title") or "" for r in rows[:sample_papers]]

    # Bag-of-words frequency across description + titles.
    counts: dict[str, int] = {}
    for t in _kb_aware_term_candidates(desc):
        counts[t] = counts.get(t, 0) + 2  # weight description higher
    for tt in titles:
        for t in _kb_aware_term_candidates(tt):
            counts[t] = counts.get(t, 0) + 1
    # Remove any term already present in the query (don't duplicate).
    q_lower = query.lower()
    for k in list(counts.keys()):
        if k in q_lower:
            counts.pop(k, None)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [k for k, _v in ranked[:max_terms]]
    if not top:
        return query, []
    augmented = f"{query} {' '.join(top)}"
    return augmented, top


@dataclass
class IngestReport:
    """Result of one end-to-end search→ingest run."""

    query: str
    kb_name: str
    kb_created: bool = False
    augmented_query: str | None = None
    injected_terms: list[str] = field(default_factory=list)
    rephrase_variants: list[str] = field(default_factory=list)
    rephrase_hits_per_variant: dict[str, int] = field(default_factory=dict)
    searched: int = 0
    filtered_out: int = 0
    candidates: int = 0
    screened_out: int = 0
    after_screen: int = 0
    added_papers: int = 0
    added_chunks: int = 0
    skipped_duplicates: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    pdf_download: dict[str, int] = field(default_factory=dict)
    selected_dois: list[str] = field(default_factory=list)
    filter_reasons: dict[str, int] = field(default_factory=dict)
    screen_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


async def screen_candidates(
    papers: list[Any],
    *,
    query: str,
    method: str,
    threshold: float,
    llm_client: Any = None,
    app_state: Any = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Score candidates by relevance to ``query`` and drop those below threshold.

    Thin wrapper around :func:`perspicacite.search.screening.screen_papers`
    / ``screen_papers_llm`` that takes SciLEx-style :class:`Paper`
    objects (rather than dicts). ``method`` is ``"bm25"`` (no LLM) or
    ``"llm"`` (cheap LLM call per batch). Returns the surviving papers
    and the full score table (kept + dropped, for the IngestReport).
    """
    if not papers:
        return [], []
    items: list[dict[str, Any]] = []
    for p in papers:
        items.append({
            "doi": (getattr(p, "doi", None) or "").strip() or None,
            "title": getattr(p, "title", None) or "",
            "abstract": getattr(p, "abstract", None) or "",
        })
    if method == "llm" and llm_client is None:
        logger.warning("screen_papers_llm_no_client_falling_back_to_bm25")
        method = "bm25"
    if method == "llm":
        from perspicacite.llm.client import resolve_stage_model
        from perspicacite.search.screening import screen_papers_llm as _llm
        if app_state is not None:
            screen_provider, screen_model = resolve_stage_model(
                app_state.config, "screening",
            )
        else:
            screen_provider, screen_model = (None, None)
        results = await _llm(
            items, query=query, llm=llm_client, threshold=threshold,
            model=screen_model, provider=screen_provider,
        )
    else:
        from perspicacite.search.screening import screen_papers as _bm25
        results = _bm25(items, reference=query, method="bm25", threshold=threshold)
    # Map results back to original Papers by DOI (preferred) or title.
    by_doi: dict[str, Any] = {}
    by_title: dict[str, Any] = {}
    for p in papers:
        if getattr(p, "doi", None):
            by_doi[p.doi.lower().strip()] = p
        if getattr(p, "title", None):
            by_title[p.title.strip().lower()] = p
    kept: list[Any] = []
    score_table: list[dict[str, Any]] = []
    for r in results:
        doi_key = (r.item.get("doi") or "").lower().strip()
        title_key = (r.item.get("title") or "").strip().lower()
        paper = by_doi.get(doi_key) or by_title.get(title_key)
        score_table.append({
            "doi": r.item.get("doi"),
            "title": r.item.get("title"),
            "score": float(r.score),
            "kept": bool(r.kept),
            "reason": r.reason,
        })
        if paper is not None and r.kept:
            kept.append(paper)
    return kept, score_table


async def run_search(
    *,
    query: str,
    max_results: int,
    databases: list[str] | None,
    year_min: int | None,
    year_max: int | None,
    article_type: str | None = None,
    config: Any | None = None,
) -> list[Any]:
    """Run a multi-DB search via DomainAwareAggregator.

    Falls back to SciLEx-only when config is not provided.
    Returns an empty list when no providers are available.
    """
    if config is not None:
        from perspicacite.search.domain_aggregator import build_aggregator
        aggregator = build_aggregator(config)
        if not aggregator.available:
            logger.warning("run_search_no_providers_available")
            return []
        papers = await aggregator.search(
            query=query,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            apis=databases or ["semantic_scholar", "openalex", "pubmed"],
        )
    else:
        from perspicacite.search.scilex_adapter import SciLExAdapter
        adapter = SciLExAdapter()
        if not adapter.available:
            logger.warning(
                "search_to_kb_scilex_missing",
                advice="install with: uv pip install -e \".[scilex]\"",
            )
            return []
        papers = await adapter.search(
            query=query,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            apis=databases or ["semantic_scholar", "openalex", "pubmed"],
            article_type=article_type,
        )
    logger.info("search_to_kb_search", query=query, hits=len(papers))
    return list(papers)


def apply_filters(
    papers: list[Any],
    flt: SearchFilter,
) -> tuple[list[Any], dict[str, int]]:
    """Drop papers that fail any active filter. Returns (kept, reasons).

    ``reasons`` is a histogram (``{reason: count}``) suitable for
    surfacing back to the user — they often want to know "why was my
    18-hit search reduced to 3 candidates?".
    """
    kept: list[Any] = []
    reasons: dict[str, int] = {}
    seen_dois: set[str] = set()
    for p in papers:
        doi = (getattr(p, "doi", None) or "").lower().strip()
        if flt.require_doi and not doi:
            reasons["no_doi"] = reasons.get("no_doi", 0) + 1
            continue
        if doi and doi in seen_dois:
            reasons["duplicate_doi"] = reasons.get("duplicate_doi", 0) + 1
            continue
        year = getattr(p, "year", None)
        if flt.min_year is not None and (year is None or year < flt.min_year):
            reasons["below_min_year"] = reasons.get("below_min_year", 0) + 1
            continue
        if flt.max_year is not None and (year is None or year > flt.max_year):
            reasons["above_max_year"] = reasons.get("above_max_year", 0) + 1
            continue
        if flt.min_citations is not None:
            cites = getattr(p, "citation_count", None) or 0
            if cites < flt.min_citations:
                reasons["below_min_citations"] = (
                    reasons.get("below_min_citations", 0) + 1
                )
                continue
        if flt.require_abstract and not (getattr(p, "abstract", None) or "").strip():
            reasons["no_abstract"] = reasons.get("no_abstract", 0) + 1
            continue
        if doi:
            seen_dois.add(doi)
        kept.append(p)
    return kept, reasons


async def _create_kb_if_missing(
    app_state: Any,
    kb_name: str,
    description: str | None,
) -> tuple[Any, bool]:
    """Return (kb_meta, created). Mirrors the create_knowledge_base
    MCP tool's logic so callers don't have to.

    Raises :class:`~perspicacite.rag.kb_compat.EmbeddingModelConflictError`
    if the KB exists but was built with a different embedding model
    than the current ``app_state.embedding_provider``.
    """
    from perspicacite.models.kb import (
        ChunkConfig,
        KnowledgeBase,
        chroma_collection_name_for_kb,
    )
    from perspicacite.rag.kb_compat import check_embedding_compat_for_ingest

    existing = await app_state.session_store.get_kb_metadata(kb_name)
    # Catch embedding-model mismatch at ingest time so multi-KB
    # queries don't fail surprisingly later. See 2026-05-16 backlog.
    check_embedding_compat_for_ingest(
        kb_meta=existing,
        embedding_service=app_state.embedding_provider,
    )
    if existing:
        return existing, False
    collection_name = chroma_collection_name_for_kb(kb_name)
    await app_state.vector_store.create_collection(collection_name)
    kb = KnowledgeBase(
        name=kb_name,
        description=description or "Built from SciLEx search via search_to_kb",
        collection_name=collection_name,
        embedding_model=app_state.embedding_provider.model_name,
        chunk_config=ChunkConfig(
            chunk_size=app_state.config.knowledge_base.chunk_size,
            chunk_overlap=app_state.config.knowledge_base.chunk_overlap,
        ),
    )
    await app_state.session_store.save_kb_metadata(kb)
    return kb, True


async def ingest_dois_into_kb(
    app_state: Any,
    kb_name: str,
    dois: list[str],
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Add each DOI's full-text paper to ``kb_name``.

    Mirrors the body of the ``add_dois_to_kb`` MCP tool — both call
    sites build a per-DOI :class:`Paper`, hand it to
    :class:`DynamicKnowledgeBase`, and update KB metadata counts.

    Wave 3.3: this function is crash-resilient via
    :class:`~perspicacite.pipeline.checkpoint.CheckpointStore`. On
    re-run with the same ``kb_name`` and DOIs, already-processed
    entries are skipped. Pass ``resume=False`` to ignore the checkpoint
    and start fresh; pass ``retry_failed=True`` to retry entries that
    previously failed.
    """
    from pathlib import Path as _Path

    from perspicacite.models.kb import chroma_collection_name_for_kb
    from perspicacite.models.papers import Author, Paper, PaperSource
    from perspicacite.pipeline.checkpoint import CheckpointStore
    from perspicacite.pipeline.download import retrieve_paper_content
    from perspicacite.pipeline.download.cookies import build_authenticated_client
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

    # ---- checkpoint setup (Wave 3.3) -----------------------------------
    ck_dir = _Path(getattr(app_state.config.knowledge_base, "checkpoint_dir", "data/checkpoints"))
    ckpt = CheckpointStore(
        path=ck_dir / f"{kb_name}__ingest_dois.json",
        kb_name=kb_name,
        operation="ingest_dois",
    )
    if not resume:
        ckpt.delete()
    ck_state = ckpt.load_or_create(planned_ids=list(dois))
    dois_to_process = list(ck_state.remaining_ids(retry_failed=retry_failed))
    # --------------------------------------------------------------------

    # ---- KB log setup (Wave 4.3) ----------------------------------------
    from perspicacite.pipeline.kb_log import KBEvent, KBLogWriter
    log_dir = _Path(getattr(app_state.config.knowledge_base, "log_dir", "data/kb_logs"))
    kb_log = KBLogWriter(path=log_dir / f"{kb_name}.jsonl")
    # ---------------------------------------------------------------------

    kb_meta = await app_state.session_store.get_kb_metadata(kb_name)
    if not kb_meta:
        raise ValueError(f"Knowledge base '{kb_name}' not found")
    # Catch embedding-model mismatch before embedding any new chunks.
    # 2026-05-16: previously this site silently re-embedded with the
    # currently-configured provider, so the KB ended up with chunks
    # from two different models.
    from perspicacite.rag.kb_compat import check_embedding_compat_for_ingest
    check_embedding_compat_for_ingest(
        kb_meta=kb_meta,
        embedding_service=app_state.embedding_provider,
    )
    collection_name = chroma_collection_name_for_kb(kb_name)

    pdf_config = app_state.config.pdf_download
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
    failed: list[dict[str, str]] = []
    dl: dict[str, int] = {"attempted": 0, "success": 0, "failed": 0}

    async with build_authenticated_client(cookies_path=cookies_path) as client:
        for raw_doi in dois_to_process:
            doi = (raw_doi or "").strip().replace("https://doi.org/", "")
            if not doi:
                continue
            if await app_state.vector_store.paper_exists(collection_name, doi):
                skipped.append({"doi": doi})
                ck_state.record(doi, "skipped")
                ckpt.save(ck_state)
                kb_log.append(KBEvent(
                    event="paper_skipped", kb_name=kb_name, paper_id=doi,
                    source_command="ingest_dois_into_kb",
                ))
                continue
            dl["attempted"] += 1
            try:
                result = await retrieve_paper_content(
                    doi,
                    http_client=client,
                    pdf_parser=app_state.pdf_parser,
                    abstract_only=(
                        getattr(
                            getattr(
                                getattr(app_state, "config", None),
                                "knowledge_base", None,
                            ),
                            "ingest_mode", "auto",
                        ) == "abstract_only"
                    ),
                    **pdf_kwargs,
                )
            except Exception as e:
                logger.warning("paper ingest failed", error=str(e))
                failed.append({"doi": doi, "reason": str(e)})
                dl["failed"] += 1
                ck_state.record(doi, "failed", reason=str(e))
                ckpt.save(ck_state)
                kb_log.append(KBEvent(
                    event="paper_failed", kb_name=kb_name, paper_id=doi,
                    reason=str(e)[:500], source_command="ingest_dois_into_kb",
                ))
                continue
            if not result or not result.success:
                failed.append({"doi": doi, "reason": "no content"})
                dl["failed"] += 1
                ck_state.record(doi, "failed", reason="no content")
                ckpt.save(ck_state)
                kb_log.append(KBEvent(
                    event="paper_failed", kb_name=kb_name, paper_id=doi,
                    reason="no content", source_command="ingest_dois_into_kb",
                ))
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
                content_type=getattr(result, "content_type", None),
                license=md.get("license"),
                source=PaperSource.OPENALEX,
            )
            if result.full_text:
                paper.full_text = result.full_text
                dl["success"] += 1
            elif result.abstract:
                # abstract_only mode — abstract without full text is a valid result
                dl["success"] += 1
            else:
                dl["failed"] += 1
            papers_to_add.append(paper)
            kb_log.append(KBEvent(
                event="paper_added", kb_name=kb_name, paper_id=doi,
                title=md.get("title"),
                source_command="ingest_dois_into_kb",
            ))
            # Mark as added immediately on successful retrieval (Wave 3.3).
            ck_state.record(doi, "added")
            ckpt.save(ck_state)

    added_chunks = 0
    if papers_to_add:
        dkb = DynamicKnowledgeBase(
            app_state.vector_store,
            app_state.embedding_provider,
        )
        dkb.collection_name = collection_name
        dkb._initialized = True
        added_chunks = await dkb.add_papers(papers_to_add, include_full_text=True)
        kb_meta.paper_count = (kb_meta.paper_count or 0) + len(papers_to_add)
        kb_meta.chunk_count = (kb_meta.chunk_count or 0) + added_chunks
        await app_state.session_store.save_kb_metadata(kb_meta)

    # Clean up checkpoint on clean completion (Wave 3.3).
    if ck_state.is_complete():
        ckpt.delete()

    return {
        "added_papers": len(papers_to_add),
        "added_chunks": added_chunks,
        "skipped_duplicates": len(skipped),
        "failed": failed,
        "pdf_download": dl,
    }


async def search_filter_and_ingest(
    *,
    app_state: Any,
    query: str,
    kb_name: str,
    max_results: int = 20,
    databases: list[str] | None = None,
    flt: SearchFilter | None = None,
    article_type: str | None = None,
    create_if_missing: bool = True,
    description: str | None = None,
    dry_run: bool = False,
    screen_method: str | None = None,
    screen_threshold: float = 0.5,
    kb_aware: bool = False,
    kb_aware_terms: int = 8,
    rephrase: int = 0,
    rephrase_model: str | None = None,
    rephrase_provider: str | None = None,
) -> IngestReport:
    """End-to-end: SciLEx search → filter → optionally create KB → ingest.

    ``dry_run=True`` runs everything up to but not including PDF fetch —
    useful from the CLI when you want to see which DOIs would be added
    before paying for the download.
    """
    flt = flt or SearchFilter()
    report = IngestReport(query=query, kb_name=kb_name)

    # KB-aware expansion (optional): if the KB already exists, mix
    # its top topic terms into the query so SciLEx surfaces papers
    # adjacent to the existing literature. No-op when the KB is new.
    effective_query = query
    if kb_aware:
        augmented, injected = await kb_aware_query(
            app_state=app_state, kb_name=kb_name,
            query=query, max_terms=kb_aware_terms,
        )
        if injected:
            effective_query = augmented
            report.augmented_query = augmented
            report.injected_terms = injected
            logger.info(
                "search_to_kb_kb_aware",
                original=query, injected=injected,
            )

    # Optional LLM-rephrased multi-query expansion: one extra LLM call
    # generates N variants, we run SciLEx once per variant (+ once for
    # the original) and merge. Trades cost for coverage on queries
    # where lexical choice matters (e.g. "metabolomics annotation" vs
    # "metabolite identification" vs "mass spec annotation").
    queries_to_run = [effective_query]
    if rephrase > 0:
        from perspicacite.llm.client import resolve_stage_model
        eff_provider, eff_model = (
            (rephrase_provider, rephrase_model)
            if rephrase_provider and rephrase_model
            else resolve_stage_model(app_state.config, "rephrase")
        )
        variants = await rephrase_query(
            query=effective_query,
            num_variants=rephrase,
            llm_client=app_state.llm_client,
            model=eff_model,
            provider=eff_provider,
        )
        if variants:
            queries_to_run.extend(variants)
            report.rephrase_variants = variants
            logger.info(
                "search_to_kb_rephrase",
                original=effective_query, variants=variants,
            )

    # Fan out across all queries, dedup hits by DOI before filtering.
    seen_dois: set[str] = set()
    merged_papers: list[Any] = []
    for q in queries_to_run:
        hits = await run_search(
            query=q,
            max_results=max_results,
            databases=databases,
            year_min=flt.min_year,
            year_max=flt.max_year,
            article_type=article_type,
            config=getattr(app_state, "config", None),
        )
        report.rephrase_hits_per_variant[q] = len(hits)
        for p in hits:
            doi = (getattr(p, "doi", None) or "").lower().strip()
            if doi:
                if doi in seen_dois:
                    continue
                seen_dois.add(doi)
            merged_papers.append(p)
    papers = merged_papers
    report.searched = len(papers)
    if not papers:
        return report

    kept, reasons = apply_filters(papers, flt)
    report.candidates = len(kept)
    report.filtered_out = report.searched - report.candidates
    report.filter_reasons = reasons

    # Optional LLM / BM25 relevance screen between filter and ingest.
    # Filters drop hits that can't be ingested (no DOI, wrong year);
    # screening drops hits that aren't *topically* on-question. Run
    # in this order so the LLM only sees viable candidates.
    if screen_method and kept:
        survivors, score_table = await screen_candidates(
            kept, query=query, method=screen_method,
            threshold=screen_threshold, llm_client=app_state.llm_client,
            app_state=app_state,
        )
        report.screened_out = len(kept) - len(survivors)
        report.after_screen = len(survivors)
        report.screen_scores = score_table
        kept = survivors

    report.selected_dois = [
        (getattr(p, "doi", None) or "").strip() for p in kept if getattr(p, "doi", None)
    ]
    if not kept or dry_run:
        return report

    # KB create-or-resolve
    if create_if_missing:
        _, created = await _create_kb_if_missing(
            app_state, kb_name, description=description,
        )
        report.kb_created = created
    else:
        existing = await app_state.session_store.get_kb_metadata(kb_name)
        if not existing:
            raise ValueError(
                f"KB '{kb_name}' not found and create_if_missing=False"
            )

    ingest = await ingest_dois_into_kb(
        app_state, kb_name, report.selected_dois,
    )
    report.added_papers = ingest["added_papers"]
    report.added_chunks = ingest["added_chunks"]
    report.skipped_duplicates = ingest["skipped_duplicates"]
    report.failed = ingest["failed"]
    report.pdf_download = ingest["pdf_download"]
    logger.info(
        "search_to_kb_done",
        query=query, kb_name=kb_name,
        searched=report.searched, candidates=report.candidates,
        added=report.added_papers, chunks=report.added_chunks,
    )
    return report
