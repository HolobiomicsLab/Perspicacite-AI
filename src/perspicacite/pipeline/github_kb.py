"""Top-level orchestrator for GitHub-repo / skill-bundle KB ingest.

This is the public surface of the 2026-05-15 GitHub / skill-bundle
ingest pipeline. Lower layers (``pipeline/github/fetcher.py``,
``pipeline/github/bundle.py``, ``pipeline/github/walk.py``,
``pipeline/github/chunk_producer.py``) do one job each; this module
wires them together against a KB target.

Three entry points:

* :func:`ingest_github_repo` — fetch a GitHub repo, walk + chunk, add
  Papers to a single KB. Never auto-routes linked papers (raw-repo
  mode; the user opts into chunked README only).
* :func:`ingest_skill_bundle` — parse the bundle.yml manifest (or fall
  back to README-only), walk + chunk, and OPTIONALLY route the
  manifest's deduplicated DOI list through
  :func:`perspicacite.pipeline.search_to_kb.ingest_dois_into_kb` so
  cited papers land in the same KB. ArXiv / PMC IDs surface in the
  summary as ``linked_papers_skipped_non_doi`` but aren't routed (v1
  routes DOIs only).
* :func:`ingest_skill_bundles_batch` — iterate a directory full of
  bundle subdirs. Per-skill mode produces one KB per bundle; composite
  mode (``composite_kb=<name>``) routes every bundle's chunks into the
  same KB, with ``source_skill`` chunk metadata differentiating them.

Design references:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md`` Task 5
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from perspicacite.config.schema import Config
from perspicacite.models.kb import (
    ChunkConfig,
    KnowledgeBase,
    chroma_collection_name_for_kb,
)
from perspicacite.pipeline.external_id_resolver import (
    resolve_arxiv_to_doi,
    resolve_pmc_to_doi,
)
from perspicacite.pipeline.github.bundle import (
    BundleManifest,
    ContentSpec,
)
from perspicacite.pipeline.github.chunk_producer import papers_from_directory
from perspicacite.pipeline.github.fetcher import (
    GitHubFetcher,
    parse_repo_url,
)
from perspicacite.pipeline.search_to_kb import ingest_dois_into_kb
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

__all__ = [
    "IngestSummary",
    "ingest_github_repo",
    "ingest_skill_bundle",
    "ingest_skill_bundles_batch",
]

logger = logging.getLogger(__name__)


IngestMode = Literal["repo", "per-skill", "composite"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class IngestSummary:
    """Per-target summary returned by the three orchestrators.

    Attributes
    ----------
    kb_name : str
        The KB that received the chunks. For composite-mode batch runs
        this is the same name across every summary.
    bundle_name : str | None
        ``manifest.name`` when ingesting a bundle; ``None`` for raw
        repos.
    repo_org, repo_name : str | None
        Owner + repo set only for the GitHub-repo path; ``None`` for
        local bundle sources.
    commit_sha : str | None
        Resolved commit SHA when the source is a GitHub URL; ``None``
        for local paths.
    files_added : int
        Count of files survived the include/exclude globs and produced
        a Paper. (Files that survive globs but read empty produce a
        Paper that the KB layer may drop — these still count here so
        operators see what the chunker tried.)
    chunks_added : int
        Count of vector-store rows actually written. Returned by
        :meth:`DynamicKnowledgeBase.add_papers`.
    linked_papers_added : int
        DOIs routed through :func:`ingest_dois_into_kb` that succeeded.
        Always ``0`` for the raw-repo path and when
        ``ingest_linked_papers=False``.
    linked_papers_skipped_non_doi : list[tuple[str, str]]
        Mined ``(kind, value)`` pairs that were NOT routed. ArXiv +
        PMC ids are first run through
        :mod:`perspicacite.pipeline.external_id_resolver`; only the
        unresolvable ones land here. Everything else (any future
        ``kind`` the manifest may grow) is reported as-is so the
        operator can route them manually.
    linked_papers_resolved_via_external_id : int
        Count of arXiv / PMC ids that the upstream resolver (arXiv → DOI
        and PMC → DOI) turned into DOIs before routing. Always ``0`` on
        the raw-repo path and when ``ingest_linked_papers=False``.
    external_links_logged : int
        Count of ``external_link`` Wave-4.3 KB-log events emitted for
        non-paper URLs (datasets / tools) mined from README + docs.
        Always ``0`` on the raw-repo path; non-zero for bundle ingests
        when the bundle's prose cites at least one non-paper URL.
    mode : "repo" | "per-skill" | "composite"
        Discriminates the call site so a single ``list[IngestSummary]``
        from the batch path stays self-describing.
    """

    kb_name: str
    bundle_name: str | None
    repo_org: str | None
    repo_name: str | None
    commit_sha: str | None
    files_added: int
    chunks_added: int
    linked_papers_added: int
    linked_papers_skipped_non_doi: list[tuple[str, str]]
    mode: IngestMode
    external_links_logged: int = 0
    linked_papers_resolved_via_external_id: int = 0


# ---------------------------------------------------------------------------
# ingest_github_repo
# ---------------------------------------------------------------------------


async def ingest_github_repo(
    *,
    url: str,
    kb_name: str,
    config: Config,
    vector_store: Any,
    embedding_service: Any,
    session_store: Any,
    fetcher: GitHubFetcher | None = None,
    content: ContentSpec | None = None,
    focus: str | None = None,
    max_files: int | None = None,
) -> IngestSummary:
    """Fetch a GitHub repo, walk + chunk it, write Papers into ``kb_name``.

    Args
    ----
    url : str
        GitHub URL. See :func:`parse_repo_url` for accepted shapes.
    kb_name : str
        Target KB. Created if it doesn't exist.
    config : Config
        Pipeline config. The ``github`` and ``knowledge_base`` blocks
        are consulted here; ``bundles`` is unused on the raw-repo path.
    vector_store, embedding_service, session_store : Any
        Storage seams. The orchestrator never reaches into AppState
        directly so unit/integration tests can pass deterministic
        fakes.
    fetcher : GitHubFetcher, optional
        Override for the fetcher. Tests pass a ``MagicMock(spec=...)``
        so no network is touched. In production this is constructed
        from ``config.github``.
    content : ContentSpec, optional
        Include/exclude globs for the walker. Defaults to the bundle
        defaults (markdown + python + notebooks + yaml).

    Notes
    -----
    *  Raw-repo mode never auto-routes linked papers — that's a
       bundle-only feature (the bundle author signed off on the
       citation list via ``papers:``; a random repo's mined DOIs are
       too noisy to trust).
    *  The synthetic manifest carries the org + repo on
       :attr:`BundleManifest.raw` so the chunk producer's ``Paper.id``
       prefix encodes the real repo coordinates rather than the
       placeholder ``bundle/<name>``.
    """
    ref = parse_repo_url(url)

    if fetcher is None:
        fetcher = _build_default_fetcher(config)

    root, sha = await fetcher.fetch(ref)
    walk_root = Path(root) / ref.subpath if ref.subpath else Path(root)

    # Synthetic manifest: the producer expects a BundleManifest for
    # author propagation + source_skill metadata. We bake org + repo
    # into ``raw`` so the chunk producer's id-builder picks them up.
    manifest = BundleManifest(
        name=f"{ref.org}__{ref.repo}",
        readme_only=False,
        raw={"org": ref.org, "repo": ref.repo},
        directory=walk_root,
    )

    papers = papers_from_directory(
        walk_root,
        manifest,
        commit_sha=sha,
        content=content,
    )

    # Relevance-gated "key code" selection: for large repos (mzmine has
    # thousands of source files), keep only the code files most relevant to the
    # tool's methodology — ranked against ``focus`` (e.g. the tool README + the
    # benchmark's research questions). Docs/markdown are always kept. Opt-in:
    # callers that pass no focus (MCP tool, skill bundles) ingest everything.
    if focus and max_files and len(papers) > max_files:
        kept = _select_relevant_papers(papers, focus=focus, max_files=max_files)
        logger.info(
            f"github_relevance_gate repo={ref.org}/{ref.repo} "
            f"files={len(papers)} kept={len(kept)} max_files={max_files}")
        papers = kept

    chunks_added = await _add_papers_to_kb(
        kb_name=kb_name,
        papers=papers,
        config=config,
        vector_store=vector_store,
        embedding_service=embedding_service,
        session_store=session_store,
        description=f"GitHub repo ingest: {ref.org}/{ref.repo}@{sha}",
    )

    return IngestSummary(
        kb_name=kb_name,
        bundle_name=None,
        repo_org=ref.org,
        repo_name=ref.repo,
        commit_sha=sha,
        files_added=len(papers),
        chunks_added=chunks_added,
        linked_papers_added=0,
        linked_papers_skipped_non_doi=[],
        mode="repo",
    )


# ---------------------------------------------------------------------------
# ingest_skill_bundle
# ---------------------------------------------------------------------------


async def ingest_skill_bundle(
    *,
    source: Path | str,
    kb_name: str | None,
    config: Config,
    vector_store: Any,
    embedding_service: Any,
    session_store: Any,
    fetcher: GitHubFetcher | None = None,
    ingest_linked_papers: bool = True,
    app_state_for_doi_ingest: Any = None,
) -> IngestSummary:
    """Ingest one skill bundle into a KB.

    Args
    ----
    source : Path | str
        Local directory (``Path``) OR GitHub URL (``str``). Strings
        that don't look like a URL are coerced to ``Path``.
    kb_name : str | None
        Target KB. When ``None`` the orchestrator derives one from
        ``config.bundles.default_kb_name_template`` using the bundle's
        ``name``.
    ingest_linked_papers : bool
        When ``True`` (default), the manifest's deduplicated DOI list
        is routed through :func:`ingest_dois_into_kb`. ArXiv / PMC IDs
        are reported in ``linked_papers_skipped_non_doi`` but not
        routed (v1 only resolves DOIs).
    app_state_for_doi_ingest : Any
        Required when ``ingest_linked_papers=True`` because
        :func:`ingest_dois_into_kb` reads PDF-download config, the
        checkpoint store, and the vector store off it. The orchestrator
        deliberately keeps it separate from the explicit
        ``vector_store`` / ``session_store`` kwargs so the bundle-walk
        path stays usable with mocks alone.

    Raises
    ------
    ValueError
        If ``ingest_linked_papers=True`` but no ``app_state_for_doi_ingest``
        was supplied.
    """
    bundle_dir = await _resolve_bundle_source(source, config=config, fetcher=fetcher)
    manifest = BundleManifest.from_directory(bundle_dir)

    if kb_name is None:
        kb_name = config.bundles.default_kb_name_template.format(
            name=manifest.name,
        )

    papers = papers_from_directory(bundle_dir, manifest, commit_sha=None)

    chunks_added = await _add_papers_to_kb(
        kb_name=kb_name,
        papers=papers,
        config=config,
        vector_store=vector_store,
        embedding_service=embedding_service,
        session_store=session_store,
        description=(manifest.description or f"Skill bundle: {manifest.name}"),
    )

    linked_added, skipped_non_doi, resolved_count = await _route_linked_papers(
        manifest=manifest,
        kb_name=kb_name,
        ingest_linked_papers=ingest_linked_papers,
        app_state_for_doi_ingest=app_state_for_doi_ingest,
    )

    external_links_logged = _emit_external_link_events(
        manifest=manifest,
        kb_name=kb_name,
        config=config,
    )

    return IngestSummary(
        kb_name=kb_name,
        bundle_name=manifest.name,
        repo_org=None,
        repo_name=None,
        commit_sha=None,
        files_added=len(papers),
        chunks_added=chunks_added,
        linked_papers_added=linked_added,
        linked_papers_skipped_non_doi=skipped_non_doi,
        mode="per-skill",
        external_links_logged=external_links_logged,
        linked_papers_resolved_via_external_id=resolved_count,
    )


# ---------------------------------------------------------------------------
# ingest_skill_bundles_batch
# ---------------------------------------------------------------------------


async def ingest_skill_bundles_batch(
    *,
    root: Path,
    config: Config,
    vector_store: Any,
    embedding_service: Any,
    session_store: Any,
    composite_kb: str | None = None,
    ingest_linked_papers: bool = True,
    app_state_for_doi_ingest: Any = None,
) -> list[IngestSummary]:
    """Walk ``root`` and ingest each immediate subdir as a bundle.

    Composite mode (``composite_kb=<name>``) routes every bundle's
    chunks into the same KB; per-skill mode (default) creates one KB
    per bundle. Returns one :class:`IngestSummary` per discovered
    bundle in deterministic (sorted) order.

    Notes
    -----
    *  Order is deterministic so the resulting summary list survives
       golden-file comparisons in downstream tests.
    *  Composite mode rewrites each summary's ``mode`` to ``"composite"``
       so the caller can distinguish a single-bundle ingest from one of
       N composite shards without re-checking ``kb_name`` equality.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"batch root {root!r} is not a directory")

    summaries: list[IngestSummary] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue

        target_kb = composite_kb if composite_kb else None
        summary = await ingest_skill_bundle(
            source=sub,
            kb_name=target_kb,
            config=config,
            vector_store=vector_store,
            embedding_service=embedding_service,
            session_store=session_store,
            ingest_linked_papers=ingest_linked_papers,
            app_state_for_doi_ingest=app_state_for_doi_ingest,
        )
        if composite_kb:
            # Re-stamp mode + kb_name so the summary reads as "composite"
            # for downstream consumers without depending on the inner
            # call having known the composite context.
            summary = replace(summary, kb_name=composite_kb, mode="composite")
        summaries.append(summary)

    return summaries


# ---------------------------------------------------------------------------
# Internals — KB seam
# ---------------------------------------------------------------------------


async def _add_papers_to_kb(
    *,
    kb_name: str,
    papers: list,
    config: Config,
    vector_store: Any,
    embedding_service: Any,
    session_store: Any,
    description: str,
) -> int:
    """Create KB if missing, hydrate a DKB, add papers, update counts.

    Mirrors the pattern from
    :func:`perspicacite.pipeline.search_to_kb._create_kb_if_missing`
    plus :func:`ingest_dois_into_kb`. Lives here (not a module helper)
    so tests can intercept the orchestrator's behaviour without
    monkey-patching the underlying KB primitives.

    Raises :class:`~perspicacite.rag.kb_compat.EmbeddingModelConflictError`
    if the KB already exists with a different ``embedding_model`` than
    the current ``embedding_service`` reports. Check runs BEFORE any
    ``create_collection`` / ``save_kb_metadata`` calls so a conflict
    leaves no partial state behind.
    """
    from perspicacite.rag.kb_compat import check_embedding_compat_for_ingest

    collection_name = chroma_collection_name_for_kb(kb_name)

    kb_meta = await session_store.get_kb_metadata(kb_name)
    check_embedding_compat_for_ingest(
        kb_meta=kb_meta,
        embedding_service=embedding_service,
    )
    if kb_meta is None:
        await vector_store.create_collection(collection_name)
        chunk_cfg = ChunkConfig(
            chunk_size=config.knowledge_base.chunk_size,
            chunk_overlap=config.knowledge_base.chunk_overlap,
        )
        kb_meta = KnowledgeBase(
            name=kb_name,
            description=description,
            collection_name=collection_name,
            embedding_model=getattr(
                embedding_service, "model_name", "unknown"
            ),
            chunk_config=chunk_cfg,
        )
        await session_store.save_kb_metadata(kb_meta)

    if not papers:
        return 0

    # Symbol-aware code chunking. The chunk producer reduces .py to docstrings
    # (a deliberate low-noise choice) but dumps every OTHER code file whole, so
    # Java/C++/R repos (e.g. mzmine, 833MB) flood the KB with license headers,
    # imports, getters and UI boilerplate. Route those through chunk_code so
    # each chunk is one function/class (tree-sitter for compiled langs, regex
    # for R), like the local-docs path already does. Python stays on the
    # producer's docstring path; markdown/text/notebooks stay on the token path.
    added_chunks = await _ingest_code_papers_symbolwise(
        papers=papers, collection_name=collection_name, config=config,
        vector_store=vector_store, embedding_service=embedding_service,
    )

    kb_meta.paper_count = (kb_meta.paper_count or 0) + len(papers)
    kb_meta.chunk_count = (kb_meta.chunk_count or 0) + added_chunks
    await session_store.save_kb_metadata(kb_meta)
    return added_chunks


# Extension -> chunk_code language for the symbol-aware path. Includes Python
# (AST symbol chunks with bodies + docstrings) for parity with the local-docs
# ingest path. Only languages chunk_code actually handles are listed.
_CODE_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".pyx": "python",
    ".java": "java", ".kt": "kotlin",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c": "cpp",
    ".h": "cpp", ".hpp": "cpp",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".swift": "swift", ".cs": "csharp",
    ".r": "r", ".rmd": "r",
}


def _select_relevant_papers(papers: list, *, focus: str, max_files: int) -> list:
    """Keep docs/README always + the top-``max_files`` code files most relevant
    to ``focus`` (lexical term-overlap over path/title/abstract/head-of-source).

    Lexical (not embedding) ranking keeps this cheap on mega-repos — no extra
    embedding pass over thousands of files just to decide what to ingest."""
    import re as _re

    terms = list(dict.fromkeys(_re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", focus.lower())))
    # Drop ubiquitous English/programming stopwords that match everything.
    _stop = {"the", "and", "for", "with", "this", "that", "from", "data", "using",
             "code", "file", "test", "import", "return", "self", "def", "class",
             "function", "method", "value", "result", "output", "input"}
    terms = [t for t in terms if t not in _stop][:60]
    if not terms:
        return papers

    def _is_doc(p) -> bool:
        md = p.metadata if isinstance(getattr(p, "metadata", None), dict) else {}
        rel = str(md.get("rel_path", "")).lower()
        return md.get("content_kind") == "github_markdown" or "readme" in rel

    def _score(p) -> int:
        md = p.metadata if isinstance(getattr(p, "metadata", None), dict) else {}
        hay = " ".join([
            str(md.get("rel_path", "")), p.title or "", p.abstract or "",
            (p.full_text or "")[:4000],
        ]).lower()
        return sum(hay.count(t) for t in terms)

    docs = [p for p in papers if _is_doc(p)]
    code = [p for p in papers if not _is_doc(p)]
    code_ranked = sorted(code, key=_score, reverse=True)
    matched = [p for p in code_ranked if _score(p) > 0][:max_files]
    # If nothing matched the focus terms, fall back to the top files by rank so
    # we never ingest docs-only (a thin KB is worse than a broad one).
    chosen_code = matched if matched else code_ranked[:max_files]
    return docs + chosen_code


async def _ingest_code_papers_symbolwise(
    *, papers: list, collection_name: str, config: Config,
    vector_store: Any, embedding_service: Any,
) -> int:
    """Chunk recognized code files per-symbol; everything else via the default
    token chunker. Returns total chunks added."""
    from pathlib import Path as _Path

    from perspicacite.pipeline.chunking_dispatch import chunk_document

    code_chunks: list = []
    rest: list = []
    for p in papers:
        md = p.metadata if isinstance(getattr(p, "metadata", None), dict) else {}
        rel = md.get("rel_path") or ""
        lang = _CODE_EXT_TO_LANG.get(_Path(rel).suffix.lower())
        text = getattr(p, "full_text", None)
        if not (lang and text):
            rest.append(p)
            continue
        try:
            # chunk_document → chunk_code (tree-sitter for compiled langs, regex
            # for R), falling back to LangChain language-aware splitting when no
            # symbol backend applies — never a whole-file token dump.
            chs = await chunk_document(text, p, content_type="code",
                                       language=lang, config=config.knowledge_base)
        except Exception as exc:  # never let one file break ingest
            logger.warning(f"github_code_chunk_failed rel={rel} lang={lang} error={exc}")
            chs = None
        if chs:
            code_chunks.extend(chs)
        else:
            rest.append(p)  # last-resort → token path

    added = 0
    if code_chunks:
        texts = [c.text for c in code_chunks]
        embeds = await embedding_service.embed(texts)
        # strict: providers drop blank inputs, so a short list would silently
        # shift every vector onto the wrong chunk. The store's own guard cannot
        # see this path — these chunks arrive already embedded.
        for c, e in zip(code_chunks, embeds, strict=True):
            c.embedding = e
        await vector_store.add_documents(collection=collection_name, chunks=code_chunks)
        added += len(code_chunks)

    if rest:
        dkb = DynamicKnowledgeBase(vector_store, embedding_service)
        dkb.collection_name = collection_name
        dkb._initialized = True
        added += await dkb.add_papers(rest, include_full_text=True)
    return added


# ---------------------------------------------------------------------------
# Internals — linked-paper routing
# ---------------------------------------------------------------------------


async def _route_linked_papers(
    *,
    manifest: BundleManifest,
    kb_name: str,
    ingest_linked_papers: bool,
    app_state_for_doi_ingest: Any,
) -> tuple[int, list[tuple[str, str]], int]:
    """Route the manifest's linked papers through the DOI ingest pipeline.

    arXiv + PMC refs are first translated to DOIs upstream via
    :mod:`perspicacite.pipeline.external_id_resolver`; only the
    unresolvable ones surface in the returned ``skipped`` list. Refs
    of any future ``kind`` are passed through to ``skipped`` as-is.

    Returns
    -------
    (added, skipped, resolved_via_external_id)
        - ``added`` — papers ``ingest_dois_into_kb`` reported as added.
        - ``skipped`` — ``(kind, value)`` tuples we could not route.
        - ``resolved_via_external_id`` — count of arXiv/PMC ids whose
          DOI resolution succeeded.
    """
    if not ingest_linked_papers:
        return 0, [], 0

    refs = manifest.collect_paper_refs()
    # The manifest may yield (doi, "10.x/y"), (arxiv, "..."), (pmc, "..."),
    # or anything else a future schema version adds. Bucket by kind so
    # we know what to resolve vs. what to pass through unchanged.
    dois: set[str] = {value for (kind, value) in refs if kind == "doi"}
    arxiv_ids = sorted({value for (kind, value) in refs if kind == "arxiv"})
    pmc_ids = sorted({value for (kind, value) in refs if kind == "pmc"})
    other_refs: list[tuple[str, str]] = sorted(
        (
            (kind, value)
            for (kind, value) in refs
            if kind not in ("doi", "arxiv", "pmc")
        ),
        key=lambda kv: (kv[0], kv[1]),
    )

    resolved_count = 0
    skipped: list[tuple[str, str]] = list(other_refs)

    for arxiv_id in arxiv_ids:
        try:
            doi = await resolve_arxiv_to_doi(arxiv_id)
        except Exception as exc:  # defensive — resolver is contracted not to raise
            logger.warning(
                "github_kb.arxiv_resolve_unexpected_error",
                extra={"arxiv_id": arxiv_id, "error": str(exc)},
            )
            doi = None
        if doi:
            if doi not in dois:  # dedup against existing DOI set
                dois.add(doi)
            resolved_count += 1
        else:
            skipped.append(("arxiv", arxiv_id))

    for pmc_id in pmc_ids:
        try:
            doi = await resolve_pmc_to_doi(pmc_id)
        except Exception as exc:
            logger.warning(
                "github_kb.pmc_resolve_unexpected_error",
                extra={"pmc_id": pmc_id, "error": str(exc)},
            )
            doi = None
        if doi:
            if doi not in dois:
                dois.add(doi)
            resolved_count += 1
        else:
            skipped.append(("pmc", pmc_id))

    # Keep the returned skipped list sorted for deterministic golden-file
    # comparisons in downstream tests.
    skipped.sort(key=lambda kv: (kv[0], kv[1]))

    if not dois:
        return 0, skipped, resolved_count

    if app_state_for_doi_ingest is None:
        raise ValueError(
            "app_state_for_doi_ingest is required when ingest_linked_papers=True"
        )

    try:
        result = await ingest_dois_into_kb(
            app_state_for_doi_ingest,
            kb_name=kb_name,
            dois=sorted(dois),
        )
    except Exception as exc:  # defensive: bundle ingest must still succeed
        logger.warning(
            "github_kb.linked_paper_ingest_failed",
            extra={"kb_name": kb_name, "error": str(exc)},
        )
        return 0, skipped, resolved_count

    if isinstance(result, dict):
        added = (
            result.get("added_papers")
            or result.get("added")
            or 0
        )
        try:
            added = int(added)
        except (TypeError, ValueError):
            added = 0
    else:
        added = 0
    return added, skipped, resolved_count


# ---------------------------------------------------------------------------
# Internals — external-link KB log emission (Task 9)
# ---------------------------------------------------------------------------


def _emit_external_link_events(
    *,
    manifest: BundleManifest,
    kb_name: str,
    config: Config,
) -> int:
    """Walk the manifest's README + docs prose, mine non-paper URLs,
    and emit one ``external_link`` Wave-4.3 KB-log event per URL.

    Returns the number of events emitted. Best-effort: a write failure
    on the KB log doesn't propagate (the underlying :class:`KBLogWriter`
    swallows write errors), and any unexpected error here is logged but
    not surfaced.
    """
    try:
        bag = manifest.collect_external_links()
    except Exception as exc:  # defensive — provenance is best-effort
        logger.warning(
            "github_kb.external_link_mining_failed",
            extra={"kb_name": kb_name, "error": str(exc)},
        )
        return 0

    if not bag.datasets and not bag.tools:
        return 0

    # Local imports keep the orchestrator import-cycle-free for callers
    # that mock these seams in unit tests.
    from pathlib import Path as _Path

    from perspicacite.pipeline.kb_log import KBEvent, KBLogWriter

    log_dir = _Path(
        getattr(config.knowledge_base, "log_dir", "data/kb_logs")
    )
    kb_log = KBLogWriter(path=log_dir / f"{kb_name}.jsonl")

    emitted = 0
    for url in bag.datasets:
        kb_log.append(KBEvent(
            event="external_link",
            kb_name=kb_name,
            paper_id="",
            source_command="ingest_skill_bundle",
            extra={"url": url, "category": "dataset"},
        ))
        emitted += 1
    for url in bag.tools:
        kb_log.append(KBEvent(
            event="external_link",
            kb_name=kb_name,
            paper_id="",
            source_command="ingest_skill_bundle",
            extra={"url": url, "category": "tool"},
        ))
        emitted += 1
    return emitted


# ---------------------------------------------------------------------------
# Internals — fetcher + source resolution
# ---------------------------------------------------------------------------


def _build_default_fetcher(config: Config) -> GitHubFetcher:
    """Production-path fetcher constructor.

    The orchestrator builds this lazily so tests can pass a mock
    fetcher without paying for ``mkdir(cache_dir)`` on every test run.
    """
    import os

    token = os.environ.get(config.github.token_env_var)
    return GitHubFetcher(
        token=token,
        cache_dir=Path(config.github.cache_dir),
        cache_max_mb=config.github.cache_max_mb,
        user_agent=config.github.user_agent,
        api_base=config.github.api_base,
    )


async def _resolve_bundle_source(
    source: Path | str,
    *,
    config: Config,
    fetcher: GitHubFetcher | None,
) -> Path:
    """Return a local directory to walk.

    * ``Path`` → returned as-is.
    * ``str`` that parses as a GitHub URL → fetched (with the optional
      ``ref.subpath`` applied so a ``/tree/.../sub/bundle`` URL points
      at the bundle root, not the repo root).
    * Any other ``str`` → coerced to ``Path``.
    """
    if isinstance(source, Path):
        return source

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        if fetcher is None:
            fetcher = _build_default_fetcher(config)
        ref = parse_repo_url(source)
        root, _sha = await fetcher.fetch(ref)
        return Path(root) / ref.subpath if ref.subpath else Path(root)

    return Path(source)
