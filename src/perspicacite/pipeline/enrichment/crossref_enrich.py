"""Crossref-based metadata enrichment helpers.

Provides three public entry points:

- ``canonicalize_candidates``  : patches a list of *dict candidates* in
  place using Crossref (title/authors/year/journal/abstract). Used by
  the basic mode's web fallback pipeline.
- ``backfill_dois``            : for candidates with no DOI but a title,
  resolves the DOI via Crossref title search with a word-overlap safety
  check. Same dict-shape contract as ``canonicalize_candidates``.
- ``enrich_papers``            : the *Paper*-object façade. Converts
  each ``Paper`` to a dict, runs the two passes above, and writes
  results back to the Paper. Used by agentic / literature_survey /
  the new ``web_search`` MCP tool.

All three honour the ``CROSSREF_MAILTO`` (or ``UNPAYWALL_EMAIL``) env
var for the Crossref polite pool — when set, concurrency rises from 2
to 6 and the per-request 250 ms spacing is dropped.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import urllib.parse
from typing import Any

import httpx

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.enrichment.crossref")


async def backfill_dois(
    candidates: list[dict[str, Any]],
    http: httpx.AsyncClient,
    *,
    sem: asyncio.Semaphore,
    mailto: str | None,
    throttle: Any = None,
) -> int:
    """Resolve missing DOIs by title-searching Crossref.

    For each candidate with a title but no DOI, issues one Crossref
    ``GET /works?query.title=...&rows=1`` and copies the best-match
    DOI into the candidate when title token overlap passes >= 0.5.
    Mutates ``candidates`` in place. Returns the number of resolved
    DOIs for logging.
    """
    targets = [
        c for c in candidates
        if not c.get("doi") and (c.get("title") or "").strip()
    ]
    if not targets:
        return 0

    def _title_tokens(s: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}

    headers = {"User-Agent": f"perspicacite/2 (mailto:{mailto})"} if mailto else {}
    resolved = 0

    async def _one(c: dict[str, Any]) -> None:
        nonlocal resolved
        title = c["title"]
        async with sem:
            if throttle is not None:
                with contextlib.suppress(Exception):
                    await throttle()
            try:
                q = urllib.parse.quote(title[:200])
                url = f"https://api.crossref.org/works?query.title={q}&rows=1"
                resp = await http.get(url, headers=headers, timeout=15.0)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.debug(
                    "crossref_title_lookup_failed",
                    title=title[:60], error=str(e),
                )
                return
        items = (data.get("message") or {}).get("items") or []
        if not items:
            return
        match = items[0]
        match_title = (match.get("title") or [""])[0]
        a, b = _title_tokens(title), _title_tokens(match_title)
        if not a or not b:
            return
        overlap = len(a & b) / max(len(a), len(b))
        if overlap < 0.5:
            logger.debug(
                "crossref_title_lookup_low_overlap",
                title=title[:60], match=match_title[:60],
                overlap=round(overlap, 2),
            )
            return
        doi = match.get("DOI")
        if doi:
            c["doi"] = doi
            resolved += 1

    await asyncio.gather(*(_one(c) for c in targets), return_exceptions=True)
    logger.info(
        "crossref_doi_backfill",
        attempted=len(targets), resolved=resolved,
    )
    return resolved


async def canonicalize_candidates(
    candidates: list[dict[str, Any]],
    *,
    concurrency: int | None = None,
) -> None:
    """Enrich each candidate dict via Crossref in place.

    Rate-limit-safe: Semaphore(2) without mailto, Semaphore(6) with.
    Spacing throttle (250 ms) is dropped when ``CROSSREF_MAILTO`` is set.
    First runs a DOI-backfill pass for candidates that have a title but
    no DOI, then a canonicalization pass that fills title/authors/year/
    journal/abstract whenever Crossref has a value and the candidate's
    field is empty.

    Sets ``c["enrichment_sources"] = ["crossref"]`` on every patched
    candidate so the UI can render a "+Crossref" chip.

    Args:
        concurrency: Optional override for the Semaphore concurrency limit.
            When None, defaults to 2 (no mailto) or 6 (with mailto).
            Bounded to [1, 10] by the ``RAGRequest.crossref_concurrency``
            field validator when coming from a per-call override.
    """
    from perspicacite.pipeline.download.crossref import enrich_from_crossref

    targets = [c for c in candidates if c.get("doi")]
    mailto = (
        os.getenv("CROSSREF_MAILTO")
        or os.getenv("UNPAYWALL_EMAIL")
        or None
    )

    _default_concurrency = 2 if not mailto else 6
    sem = asyncio.Semaphore(concurrency if concurrency is not None else _default_concurrency)
    _spacing_lock = asyncio.Lock()
    _last_call_t = {"t": 0.0}
    _min_spacing = 0.25 if not mailto else 0.0

    async def _throttle() -> None:
        if _min_spacing <= 0:
            return
        async with _spacing_lock:
            import time as _t
            now = _t.monotonic()
            gap = now - _last_call_t["t"]
            if gap < _min_spacing:
                await asyncio.sleep(_min_spacing - gap)
            _last_call_t["t"] = _t.monotonic()

    async with httpx.AsyncClient(timeout=15.0) as http:
        try:
            await backfill_dois(
                candidates, http, sem=sem, mailto=mailto, throttle=_throttle,
            )
        except Exception as e:
            logger.debug("crossref_doi_backfill_skipped", error=str(e))

        targets[:] = [c for c in candidates if c.get("doi")]

        async def _one(c: dict[str, Any]) -> None:
            async with sem:
                await _throttle()
                patch: dict[str, Any] = {}
                for attempt in range(2):
                    try:
                        patch = await enrich_from_crossref(
                            c["doi"], http_client=http,
                            base_metadata={}, mailto=mailto,
                        )
                        break
                    except Exception as e:
                        msg = str(e)
                        if "429" in msg and attempt == 0:
                            await asyncio.sleep(1.5)
                            await _throttle()
                            continue
                        logger.debug(
                            "crossref_one_failed",
                            doi=c.get("doi"), error=msg,
                        )
                        return
            if not patch:
                return

            def _empty(v: Any) -> bool:
                return v is None or v == "" or v == []

            for k in ("title", "authors", "year", "journal", "abstract"):
                if patch.get(k) and _empty(c.get(k)):
                    c[k] = patch[k]
            enrichers = c.setdefault("enrichment_sources", [])
            if "crossref" not in enrichers:
                enrichers.append("crossref")

        await asyncio.gather(
            *[_one(c) for c in targets], return_exceptions=True,
        )

    logger.info(
        "crossref_canonicalized",
        attempted=len(targets), candidates=len(candidates),
        mailto_polite_pool=bool(mailto),
    )


async def enrich_papers(papers: list, *, concurrency: int | None = None) -> list:
    """Crossref-enrich a list of Paper objects in place.

    Converts each Paper to a dict candidate (carrying only the fields
    Crossref cares about), runs ``canonicalize_candidates``, then
    writes the patched values back to the Paper. Records enrichment
    provenance under ``paper.enrichment_sources`` (a list).

    The original Paper objects are returned (same list, mutated). This
    is the public entry point for agentic, literature_survey, the
    standalone MCP ``web_search`` tool, and the unified
    ``resolve_papers_pipeline`` introduced in Tier 3.

    Args:
        concurrency: Optional override forwarded to ``canonicalize_candidates``.
            Typically sourced from ``RAGRequest.crossref_concurrency``.
    """
    if not papers:
        return papers

    from perspicacite.models.papers import Author

    candidates: list[dict[str, Any]] = []
    for p in papers:
        authors_list: list[str] = []
        for a in (p.authors or []):
            n = getattr(a, "name", None) or ""
            if n:
                authors_list.append(n)
        candidates.append({
            "_paper_ref": p,
            "title": p.title,
            "authors": authors_list,
            "year": p.year,
            "journal": p.journal,
            "doi": p.doi,
            "abstract": p.abstract or "",
        })

    await canonicalize_candidates(candidates, concurrency=concurrency)

    for c in candidates:
        p = c["_paper_ref"]
        if not p.title and c.get("title"):
            p.title = c["title"]
        if not p.year and c.get("year"):
            p.year = c["year"]
        if not p.journal and c.get("journal"):
            p.journal = c["journal"]
        if not p.doi and c.get("doi"):
            p.doi = c["doi"]
        if not p.abstract and c.get("abstract"):
            p.abstract = c["abstract"]
        if not p.authors and c.get("authors"):
            p.authors = [Author(name=str(n)) for n in c["authors"] if n]

        if c.get("enrichment_sources"):
            for src in c["enrichment_sources"]:
                if src not in p.enrichment_sources:
                    p.enrichment_sources.append(src)

    return papers
