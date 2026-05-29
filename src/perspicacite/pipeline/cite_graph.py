"""Cite-graph enrichment orchestrator (2026-05-15 spec).

Given a library/tool name (or explicit DOI), resolves to a canonical
paper, walks the OpenAlex forward-citation graph, filters + scores
citing works, and (optionally) ingests survivors via the existing
DOI-ingest path.

This module owns:
- ``CiteHit`` (a citing-paper record)
- ``apply_cite_graph_filters`` (cheap drops)
- ``score_cite_hit`` (final ranking)
- ``enrich_kb_from_cite_graph`` (orchestrator — Task 6)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from perspicacite.config.schema import CiteGraphConfig
from perspicacite.pipeline.external.fetch_github import fetch_github_repo

_GITHUB_REPO_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)", re.IGNORECASE)


def _github_repo_for_work(_client, oa_work: dict, *, headers: dict) -> str | None:
    """Best-effort extraction of ``owner/repo`` from an OpenAlex work record.

    Scans DOI, primary_location.landing_page_url, and alternate landing-page
    URLs for a github.com link. Returns ``owner/repo`` or None.
    """
    blobs: list[str] = []
    blobs.append(str(oa_work.get("doi") or ""))
    pl = oa_work.get("primary_location") or {}
    if isinstance(pl, dict):
        blobs.append(str(pl.get("landing_page_url") or ""))
    for url_field in ("alternate_landing_page_urls", "external_ids"):
        v = oa_work.get(url_field)
        if isinstance(v, list):
            for item in v:
                blobs.append(str(item))
    for blob in blobs:
        m = _GITHUB_REPO_RE.search(blob)
        if m:
            return m.group(1).rstrip("/").removesuffix(".git")
    return None


@dataclass
class CiteHit:
    """A citing paper record — built from an OpenAlex work."""
    doi: str
    title: str
    year: int
    venue: str | None
    citation_count: int
    is_oa: bool
    abstract: str | None = None
    github_url: str | None = None
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    scripts: list[dict] = field(default_factory=list)


def _normalize_citations(citations: int) -> float:
    if citations <= 0:
        return 0.0
    return min(math.log10(citations + 1) / 3.0, 1.0)


def _recency_score(year: int, *, now_year: int) -> float:
    age = max(now_year - year, 0)
    return 0.5 ** (age / 5.0)


_WORD_RE = re.compile(r"\w+")

# Common English stopwords — minimal, just enough to strip filler from
# paper titles. Not a full NLTK list; we want fast and dependency-free.
_TITLE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "via", "we", "were", "with", "without",
    "using", "use", "uses", "used", "their", "these", "those", "based",
    "new", "novel", "recent", "approach", "method", "methods",
    "paper", "study", "studies", "result", "results",
})


def tool_synonyms_from_seed(
    *,
    tool: str | None,
    seed_title: str | None,
) -> list[str]:
    """Build a synonyms list for cite-graph ranking.

    Includes the tool name plus content-word tokens from the seed
    paper's title (lowercased, deduped, stopword-filtered, short tokens
    dropped). Order: tool first, then unique title tokens in title order.
    """
    syns: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        t = token.lower().strip()
        if not t or len(t) < 3:
            return
        if t in _TITLE_STOPWORDS:
            return
        if t in seen:
            return
        seen.add(t)
        syns.append(t)

    if tool:
        _add(tool)
    if seed_title:
        for token in _WORD_RE.findall(seed_title):
            _add(token)
    return syns


def _keyword_match(text: str | None, synonyms: list[str]) -> float:
    """Score how well the abstract text matches the tool synonym list.

    Each synonym is matched either as a whole token or — for hyphenated
    names like ``openff-evaluator`` — by checking that all of its word
    parts appear in the text.
    """
    if not text or not synonyms:
        return 0.0
    text_lower = text.lower()
    tokens = {w.lower() for w in _WORD_RE.findall(text)}
    if not tokens:
        return 0.0
    hits = 0
    for syn in synonyms:
        if not syn:
            continue
        sl = syn.lower()
        # Exact word-token match
        if sl in tokens or sl in text_lower:
            hits += 1
        else:
            # All word-parts of the synonym appear as tokens
            parts = _WORD_RE.findall(sl)
            if parts and all(p in tokens for p in parts):
                hits += 1
    return min(hits / max(len(synonyms), 1), 1.0)


def score_cite_hit(
    hit: CiteHit,
    tool_synonyms: list[str],
    config: CiteGraphConfig,
    *,
    now_year: int,
    source_count: int = 1,
) -> float:
    """Compute hit.score from the four signal components plus optional multi-source bonus."""
    cit = _normalize_citations(hit.citation_count)
    rec = _recency_score(hit.year, now_year=now_year)
    oa = 1.0 if hit.is_oa else 0.5
    match = _keyword_match(hit.abstract, tool_synonyms)
    s = (
        config.w_citations * cit
        + config.w_recency   * rec
        + config.w_oa        * oa
        + config.w_match     * match
    )
    if source_count >= 2:
        s = min(s + config.multi_source_bonus, 1.0)
    hit.score = round(s, 4)
    hit.score_breakdown = {
        "citations": round(cit, 4),
        "recency": round(rec, 4),
        "oa": round(oa, 4),
        "match": round(match, 4),
        "multi_source_bonus": config.multi_source_bonus if source_count >= 2 else 0.0,
    }
    return hit.score


def apply_cite_graph_filters(
    hits: list[CiteHit],
    *,
    config: CiteGraphConfig,
    existing_dois: set[str],
    now_year: int,
) -> list[CiteHit]:
    """Drop hits that fail cheap rejects (year, citations, denylist, dedup)."""
    min_year = now_year - config.min_year_offset
    deny = {v.lower() for v in config.venue_denylist}
    out: list[CiteHit] = []
    for h in hits:
        if h.year < min_year:
            continue
        if h.citation_count < config.min_citations:
            continue
        if h.doi in existing_dois:
            continue
        if h.venue and h.venue.lower() in deny:
            continue
        out.append(h)
    return out


# --- Orchestrator (Task 6) -----------------------------------------


import httpx

from perspicacite.pipeline.library_doi import resolve_library_paper
from perspicacite.pipeline.snowball import (
    OPENALEX_BASE,
    _fetch_seed_work,
    fetch_cited_by_works,
    # Re-exported into this module's namespace so the DOI-resolution path is
    # patchable/observable from tests (test_cite_graph_openalex_id asserts the
    # openalex_id path skips it). Not called directly in this module's body.
    openalex_id_for_doi,  # noqa: F401
)


async def _resolve_and_fetch(
    *,
    tool: str | None,
    doi: str | None,
    openalex_id: str | None,
    headers: dict[str, str],
    client: httpx.AsyncClient,
    max_results: int,
) -> tuple[list[dict], str | None]:
    """Resolve the library to a seed work, then fetch OpenAlex citing works.

    Returns a (citing_works, seed_title) tuple. The title — when known —
    is used by the orchestrator to expand tool_synonyms for topic-aware
    scoring.

    When ``openalex_id`` is supplied, the resolver and DOI lookup are
    skipped entirely; the OpenAlex Work id is used as the seed directly.
    """
    if openalex_id:
        seed_url = f"{OPENALEX_BASE}/works/{openalex_id}"
        try:
            resp = await client.get(seed_url, headers=headers, timeout=20.0)
            if resp.status_code == 200:
                seed_work = resp.json() or {"id": f"https://openalex.org/{openalex_id}"}
            else:
                seed_work = {"id": f"https://openalex.org/{openalex_id}"}
        except httpx.HTTPError:
            seed_work = {"id": f"https://openalex.org/{openalex_id}"}
        seed_title = (
            (seed_work.get("title") or seed_work.get("display_name"))
            if isinstance(seed_work, dict) else None
        )
        works = await fetch_cited_by_works(
            client, seed_work=seed_work, max_results=max_results, headers=headers,
        )
        return works, seed_title

    seed_doi: str | None = doi
    if seed_doi is None:
        if not tool:
            return ([], None)
        return ([], None)

    seed_work = await _fetch_seed_work(client, seed_doi, headers)
    if seed_work is None:
        return ([], None)
    seed_title = seed_work.get("title")
    works = await fetch_cited_by_works(
        client, seed_work=seed_work,
        max_results=max_results, headers=headers,
    )
    return (works, seed_title)


def _hit_from_oa_work(work: dict) -> CiteHit | None:
    """Project an OpenAlex work dict into a CiteHit. Returns None when
    the work has no DOI."""
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    if not doi:
        doi = (work.get("ids") or {}).get("doi", "").replace("https://doi.org/", "")
    if not doi:
        return None
    title = work.get("title") or ""
    year = int(work.get("publication_year") or 0)
    cit = int(work.get("cited_by_count") or 0)
    oa = bool((work.get("open_access") or {}).get("is_oa"))
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    # Reconstruct abstract from inverted index (best-effort).
    inv = work.get("abstract_inverted_index") or None
    abstract = None
    if isinstance(inv, dict) and inv:
        try:
            positions: list[tuple[int, str]] = []
            for word, idxs in inv.items():
                for i in idxs:
                    positions.append((i, word))
            positions.sort()
            abstract = " ".join(w for _, w in positions)
        except Exception:
            abstract = None
    return CiteHit(
        doi=doi, title=title, year=year, venue=venue,
        citation_count=cit, is_oa=oa, abstract=abstract,
    )


async def enrich_kb_from_cite_graph(
    *,
    tool: str | None = None,
    doi: str | None = None,
    openalex_id: str | None = None,
    kb_config,
    existing_dois: set[str],
    dry_run: bool = False,
    now_year: int | None = None,
) -> list[CiteHit]:
    """Resolve library/DOI/OpenAlex-id → fetch citing works → filter+score → top-N.

    Returns the ranked CiteHit list (top max_papers). Synonyms for the
    abstract-match signal are expanded with content-word tokens from
    the seed paper's title via :func:`tool_synonyms_from_seed`.

    Pass exactly one of ``tool`` / ``doi`` / ``openalex_id`` (or both
    ``tool`` and ``doi`` — DOI wins). When ``openalex_id`` is supplied,
    DOI resolution is bypassed entirely.

    The ``dry_run`` parameter is reserved for future ingest plumbing —
    currently it has no effect on behaviour.
    """
    import datetime as _dt
    if now_year is None:
        now_year = _dt.datetime.now(_dt.UTC).year

    if not tool and not doi and not openalex_id:
        raise ValueError("must supply tool, doi, or openalex_id")

    cfg = kb_config.cite_graph

    # When only a tool name is given, resolve it to a DOI here so
    # ``_resolve_and_fetch`` can stay narrowly focused on the OpenAlex
    # round-trip (and easy to mock in tests).
    seed_doi: str | None = doi
    if openalex_id is None and seed_doi is None and tool:
        paper = await resolve_library_paper(
            tool,
            bundle=None, github_repo=None,
            config_map=dict(kb_config.library_paper_map),
            readme_text=None,
        )
        if paper is None:
            return []
        seed_doi = paper.doi

    async with httpx.AsyncClient() as client:
        works, seed_title = await _resolve_and_fetch(
            tool=tool,
            doi=seed_doi,
            openalex_id=openalex_id,
            headers={},
            client=client,
            max_results=cfg.max_papers * 4,
        )

        # Fetch COCI citations concurrently when we have a DOI
        coci_works: list[dict] = []
        if seed_doi:
            from perspicacite.pipeline.download.opencitations import (
                fetch_opencitations_citations,
            )
            coci_works = await fetch_opencitations_citations(seed_doi, http_client=client)

    # Build DOI→source_count map for multi-source bonus
    doi_source_counts: dict[str, int] = {}
    for w in works:
        raw_doi = (w.get("doi") or "").replace("https://doi.org/", "")
        if raw_doi:
            doi_source_counts[raw_doi] = doi_source_counts.get(raw_doi, 0) + 1
    for w in coci_works:
        raw_doi = (w.get("doi") or "").replace("https://doi.org/", "")
        if raw_doi:
            doi_source_counts[raw_doi] = doi_source_counts.get(raw_doi, 0) + 1

    all_works = works + coci_works

    raw_hits: list[CiteHit] = []
    work_by_doi: dict[str, dict] = {}
    for w in all_works:
        h = _hit_from_oa_work(w)
        if h is not None:
            raw_hits.append(h)
            work_by_doi[h.doi] = w

    filtered = apply_cite_graph_filters(
        raw_hits, config=cfg, existing_dois=existing_dois, now_year=now_year,
    )

    synonyms = tool_synonyms_from_seed(tool=tool, seed_title=seed_title)
    for h in filtered:
        sc = doi_source_counts.get(h.doi, 1)
        score_cite_hit(h, synonyms, cfg, now_year=now_year, source_count=sc)

    filtered.sort(key=lambda h: h.score, reverse=True)
    top = filtered[: cfg.max_papers]

    if cfg.include_scripts and not dry_run:
        import tempfile
        from pathlib import Path as _Path
        cache_dir = _Path(tempfile.gettempdir()) / "perspicacite_cite_graph_repos"
        for h in top:
            oa_work = work_by_doi.get(h.doi)
            if oa_work is None:
                continue
            try:
                repo = _github_repo_for_work(None, oa_work, headers={})
                if not repo:
                    continue
                blob = await fetch_github_repo(
                    repo, cache_dir=cache_dir, ttl_seconds=30 * 86400,
                )
                if isinstance(blob, dict):
                    scripts = (blob.get("scripts") or [])[:3]
                    h.scripts = list(scripts)
            except Exception:
                continue

    return top
