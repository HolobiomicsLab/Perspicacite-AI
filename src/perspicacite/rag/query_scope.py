"""Resolve which papers the user is asking about (quoted titles, DOIs) and constrain retrieval."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.rag.query_scope")

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\])>'\",]+", re.IGNORECASE)
# Double- or single-quoted segments (length bounds reduce false positives)
QUOTED_PATTERN = re.compile(r'"([^"]{12,600})"|\'([^\']{12,600})\'')

_STOPWORDS = frozenset(
    "the a an of in for to and or on at by with from as is are was were be been being "
    "this that these those it its into than then not no we our their they he she his her "
    "which what when where who how all any each both than".split()
)


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    for ch in ("\u2013", "\u2014", "\u2212", "\u2010"):
        s = s.replace(ch, "-")
    return s.lower()


def _token_set(s: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", _normalize_text(s))
    return {t for t in raw if len(t) > 1 and t not in _STOPWORDS}


def _normalize_doi(doi: str) -> str:
    d = (doi or "").strip().lower()
    for p in ("https://doi.org/", "http://dx.doi.org/", "doi:"):
        if d.startswith(p):
            d = d[len(p) :].strip()
    return d.rstrip(".,);")


def extract_paper_references(query: str) -> tuple[list[str], list[str]]:
    """Return (quoted_strings, dois) extracted from the query."""
    quoted: list[str] = []
    for m in QUOTED_PATTERN.finditer(query):
        q = (m.group(1) or m.group(2) or "").strip()
        if len(q) >= 12:
            quoted.append(q)
    dois = []
    seen: set[str] = set()
    for m in DOI_PATTERN.finditer(query):
        d = _normalize_doi(m.group(0))
        if d and d not in seen:
            seen.add(d)
            dois.append(d)
    return quoted, dois


def _title_match_score(needle: str, title: str | None) -> float:
    if not title:
        return 0.0
    nt, tt = _normalize_text(needle), _normalize_text(title)
    if not nt or not tt:
        return 0.0
    if nt in tt or tt in nt:
        return 1.0
    a, b = _token_set(needle), _token_set(title)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = max(len(a), len(b))
    return inter / denom if denom else 0.0


@dataclass
class PaperScopeResult:
    """Papers the user explicitly referenced and optional note for the prompt."""

    forced_paper_ids: list[str] = field(default_factory=list)
    max_papers: int = 5
    scope_note: str | None = None


async def resolve_paper_scope_for_query(
    query: str,
    collection: str,
    vector_store: Any,
    *,
    max_papers_override: int | None = None,
    hard_cap: int = 5,
) -> PaperScopeResult:
    """Match quoted titles / DOIs in ``query`` to KB papers via chunk metadata."""
    quoted, dois = extract_paper_references(query)
    if not quoted and not dois:
        cap = min(hard_cap, max_papers_override or hard_cap)
        return PaperScopeResult(max_papers=cap)

    if not hasattr(vector_store, "list_paper_metadata"):
        logger.warning("query_scope_no_list_paper_metadata", collection=collection)
        cap = min(hard_cap, max_papers_override or hard_cap)
        return PaperScopeResult(max_papers=cap)

    papers = await vector_store.list_paper_metadata(collection)
    if not papers:
        cap = min(hard_cap, max_papers_override or hard_cap)
        return PaperScopeResult(max_papers=cap)

    forced: list[str] = []
    notes: list[str] = []

    for d in dois:
        for p in papers:
            pd = _normalize_doi(str(p.get("doi") or ""))
            if pd and pd == d:
                pid = p.get("paper_id")
                if pid and pid not in forced:
                    forced.append(str(pid))
                break

    for q in quoted:
        best_pid: str | None = None
        best_score = 0.0
        for p in papers:
            title = p.get("title")
            sc = _title_match_score(q, str(title) if title else None)
            if sc > best_score:
                best_score = sc
                pid = p.get("paper_id")
                best_pid = str(pid) if pid else None
        if best_pid and best_score >= 0.32 and best_pid not in forced:
            forced.append(best_pid)
            if best_score < 0.85:
                notes.append(
                    f"A paper was matched to your quoted title with moderate confidence "
                    f"({best_score:.0%}); prefer confirming with DOI if the answer seems off."
                )

    if forced:
        if len(forced) == 1:
            max_p = 1
        else:
            max_p = min(hard_cap, len(forced), max_papers_override or hard_cap)
        note = "\n".join(notes) if notes else None
        if len(forced) == 1:
            titles = [p.get("title") for p in papers if str(p.get("paper_id")) == forced[0]]
            t0 = titles[0] if titles else forced[0]
            scope_note = (
                f"The user is asking specifically about this paper from the knowledge base: {t0}. "
                f"Prioritize it; do not dilute the answer with other papers unless they are clearly "
                f"necessary for context."
            )
            if note:
                scope_note = scope_note + " " + note
        else:
            scope_note = (
                "The user referenced multiple specific papers; address each as asked. "
                + (" ".join(notes) if notes else "")
            ).strip()
        return PaperScopeResult(
            forced_paper_ids=forced,
            max_papers=max_p,
            scope_note=scope_note,
        )

    cap = min(hard_cap, max_papers_override or hard_cap)
    return PaperScopeResult(max_papers=cap)


def merge_scope_with_candidates(
    candidate_order: list[str],
    paper_scores: dict[str, float],
    scope: PaperScopeResult | None,
    cap: int,
) -> list[str]:
    """Merge forced paper IDs with vector-ranked candidates and apply ``cap``."""
    if not scope or not scope.forced_paper_ids:
        return candidate_order[:cap]

    forced = list(scope.forced_paper_ids)
    out: list[str] = []
    seen: set[str] = set()
    for pid in forced:
        if pid and pid not in seen:
            out.append(pid)
            seen.add(pid)
    for pid in candidate_order:
        if pid not in seen:
            out.append(pid)
            seen.add(pid)
    if len(forced) == 1:
        return out[: max(1, min(cap, scope.max_papers))]
    return out[: min(cap, scope.max_papers, len(out))]
