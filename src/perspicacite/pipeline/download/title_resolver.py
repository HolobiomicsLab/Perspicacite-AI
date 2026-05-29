"""Title-based DOI discovery for entries with no DOI or usable URL.

When the bibtex/url route can't produce a routable identifier, this
module tries to recover a DOI by querying scholarly metadata APIs
with the paper title. Used as a last-resort fallback inside
``_resolve_push_input`` (mcp/server.py) so URL-only / title-only
entries can still route through the DOI path (real PDF fetch +
preprint item_type) instead of being demoted to a webpage shell.

Tiers (cheapest -> broadest):

1. OpenAlex   ``works?search=<title>``
2. Crossref   ``works?query.bibliographic=<title>&query.author=<lastname>``
3. Semantic Scholar ``paper/search?query=<title>``
4. arXiv      ``query?search_query=ti:"<title>"`` (returns arXiv DOI)
5. Headless Chromium -> Google Scholar (opt-in; needs the ``browser``
   extra and ``playwright install chromium``). Used when HTTP tiers
   miss — e.g., bib entries with title typos that defeat fuzzy match,
   or very-new papers not yet indexed by the JSON APIs. Each DOI
   scraped from the Scholar SERP is verified via Crossref before
   being accepted.

Each tier validates the top hits against the bib entry:

- first-author last-name overlap (case-insensitive substring),
- year +/-1 (when both sides have one),
- candidate title length within 60-150% of the bib title.

Returns the first DOI that passes validation, or ``None``.

Agent-side note: clients with a browser MCP available (e.g. the
``claude-in-chrome`` server) can also pre-resolve a title to a DOI
themselves and pass it to ``push_to_zotero`` directly. The Chromium
tier here exists for non-agent server-side callers (batch ingest,
``build_kbs_from_zotero``, CLI flows) that can't reach an agent.
"""
from __future__ import annotations

import os
import re
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger(__name__)


def _last_name(author: str) -> str:
    """Extract last name. Accepts ``"Smith, John"`` or ``"John Smith"``."""
    author = (author or "").strip()
    if not author:
        return ""
    if "," in author:
        return author.split(",", 1)[0].strip().lower()
    parts = author.split()
    return parts[-1].lower() if parts else ""


# Common BibTeX/Zotero placeholder author names that should be treated
# as "no author known", not as a real surname. Without this, validation
# treats ``["Unknown"]`` as target_lastname="unknown" and erroneously
# accepts any candidate whose surname is a substring of "unknown".
_JUNK_AUTHOR_TOKENS = {"unknown", "anonymous", "anon", "n/a", "na"}


def _is_junk_author(s: str) -> bool:
    cleaned = re.sub(r"[{}]", "", s or "").strip().lower()
    return (not cleaned) or cleaned in _JUNK_AUTHOR_TOKENS


def _author_tokens(authors: list[str] | None) -> set[str]:
    """Pull every meaningful name-token (>=3 letters, alpha) out of an
    author list. We pool first+last because Zotero/BibTeX entries are
    routinely inconsistent — Chinese names get stored with given +
    family swapped, single-name authors land in ``lastName``, etc.
    Validation then needs any candidate-author overlap, not just a
    rigid first-author surname match."""
    out: set[str] = set()
    for a in authors or []:
        if _is_junk_author(a):
            continue
        cleaned = re.sub(r"[{}]", "", a)
        for part in re.split(r"[,\s]+", cleaned):
            p = part.strip().lower()
            if len(p) >= 3 and p.isalpha():
                out.add(p)
    return out


_TITLE_STOPWORDS = {
    "the", "an", "of", "for", "and", "or", "in", "on", "to", "with",
    "is", "are", "be", "by", "from", "as", "at", "via", "a",
}


def _title_tokens(title: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9]+", (title or "").lower())
        if len(t) >= 3 and t not in _TITLE_STOPWORDS
    }


def _title_similarity(a: str, b: str) -> float:
    """Jaccard similarity on title content-word tokens (3+ chars, no
    stopwords). Returns 0.0 when either side has no usable tokens."""
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _validate_match(
    candidate_title: str,
    candidate_authors: list[str],
    candidate_year: int | None,
    target_title: str,
    target_authors: list[str] | None,
    target_year: int | None,
) -> bool:
    """Validate a candidate scholar API result against the bib entry.

    Rules:

    - Author overlap: target token set (any-position name parts, junk
      placeholders stripped) must share at least one 4+ char token with
      the candidate's token set. Skipped only when the target has no
      real authors — in that case the title similarity floor is raised.
    - Year proximity: ±1 when both sides have one.
    - Title similarity (Jaccard on content-word tokens): >=0.4
      normally, >=0.6 when no usable target author exists. This is the
      backstop against the "title-length-only" failure mode that lets
      arbitrary unrelated DOIs through for ``["Unknown"]`` entries.
    """
    if not candidate_title:
        return False

    target_tokens = _author_tokens(target_authors)
    cand_tokens = _author_tokens(candidate_authors)
    if target_tokens:
        overlap = {t for t in target_tokens & cand_tokens if len(t) >= 4}
        if not overlap:
            return False

    if target_year is not None and candidate_year is not None:
        try:
            if abs(int(candidate_year) - int(target_year)) > 1:
                return False
        except (TypeError, ValueError):
            pass

    if target_tokens:
        return _title_similarity(candidate_title, target_title) >= 0.4

    # No real author signal — short generic titles like "Model Context
    # Protocol Specification" or "LangGraph: Build resilient language
    # agents" share enough common terms with unrelated journal papers
    # ("Model-based protocol specification" etc.) to clear a 0.6
    # Jaccard floor. Tighten by requiring both (a) >= 0.6 Jaccard AND
    # (b) at least 4 shared content tokens — both are easy to satisfy
    # for genuine matches but hard to satisfy by accident.
    target_t = _title_tokens(target_title)
    cand_t = _title_tokens(candidate_title)
    if not target_t or not cand_t:
        return False
    shared = target_t & cand_t
    if len(shared) < 4:
        return False
    return len(shared) / len(target_t | cand_t) >= 0.6


async def _try_openalex(
    title: str,
    target_authors: list[str] | None,
    year: int | None,
    *,
    http_client: Any,
) -> str | None:
    mailto = os.getenv("OPENALEX_MAILTO") or os.getenv("UNPAYWALL_EMAIL") or ""
    params: dict[str, Any] = {"search": title, "per-page": 5}
    if mailto:
        params["mailto"] = mailto
    r = await http_client.get(
        "https://api.openalex.org/works", params=params, timeout=20.0,
    )
    if r.status_code != 200:
        return None
    for w in (r.json() or {}).get("results", []) or []:
        c_title = (w.get("title") or "").strip()
        c_authors = [
            ((a.get("author") or {}).get("display_name") or "")
            for a in (w.get("authorships") or [])
        ]
        c_year = w.get("publication_year")
        c_doi = (w.get("doi") or "").replace("https://doi.org/", "").strip()
        if c_doi and _validate_match(
            c_title, c_authors, c_year, title, target_authors, year,
        ):
            return c_doi
    return None


async def _try_crossref(
    title: str,
    target_authors: list[str] | None,
    year: int | None,
    *,
    http_client: Any,
) -> str | None:
    params: dict[str, Any] = {"query.bibliographic": title, "rows": 5}
    # Hint Crossref's author-aware ranking with whatever author signal
    # we have. Doesn't need to be the canonical surname — Crossref
    # tokenizes the query field generously.
    if target_authors:
        first_real = next(
            (a for a in target_authors if not _is_junk_author(a)),
            "",
        )
        if first_real:
            params["query.author"] = _last_name(first_real) or first_real
    r = await http_client.get(
        "https://api.crossref.org/works", params=params, timeout=20.0,
    )
    if r.status_code != 200:
        return None
    items = ((r.json() or {}).get("message") or {}).get("items") or []
    for w in items:
        titles = w.get("title") or []
        c_title = (titles[0] if titles else "").strip()
        c_authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in (w.get("author") or [])
            if a.get("family")
        ]
        c_year: int | None = None
        date_parts = ((w.get("issued") or {}).get("date-parts") or [[None]])[0]
        if date_parts and date_parts[0]:
            try:
                c_year = int(date_parts[0])
            except (TypeError, ValueError):
                c_year = None
        c_doi = (w.get("DOI") or "").strip()
        if c_doi and _validate_match(
            c_title, c_authors, c_year, title, target_authors, year,
        ):
            return c_doi
    return None


async def _try_semantic_scholar(
    title: str,
    target_authors: list[str] | None,
    year: int | None,
    *,
    http_client: Any,
) -> str | None:
    params: dict[str, Any] = {
        "query": title,
        "limit": 5,
        "fields": "title,authors,year,externalIds",
    }
    r = await http_client.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        timeout=20.0,
    )
    if r.status_code != 200:
        return None
    for w in (r.json() or {}).get("data") or []:
        c_title = (w.get("title") or "").strip()
        c_authors = [(a.get("name") or "") for a in (w.get("authors") or [])]
        c_year = w.get("year")
        ext = w.get("externalIds") or {}
        c_doi = (ext.get("DOI") or "").strip()
        if not c_doi and ext.get("ArXiv"):
            c_doi = f"10.48550/arXiv.{ext['ArXiv'].strip()}"
        if c_doi and _validate_match(
            c_title, c_authors, c_year, title, target_authors, year,
        ):
            return c_doi
    return None


async def _try_arxiv(
    title: str,
    target_authors: list[str] | None,
    year: int | None,
    *,
    http_client: Any,
) -> str | None:
    # arXiv API doesn't return JSON; entries come back as Atom XML.
    # We parse the minimum we need with a regex rather than pulling
    # in an XML parser dependency.
    #
    # Use AND-of-tokens, not phrase-exact: arXiv's ``ti:"<exact>"``
    # requires the exact title phrase, which fails for bib entries
    # whose stored title is the journal/OpenReview form but the
    # arXiv title is a shorter version (e.g. ``EvoPrompt: Connecting
    # LLMs ...`` vs ``Connecting Large Language Models ...``).
    content_toks = sorted(
        _title_tokens(title), key=len, reverse=True,
    )[:5]
    if not content_toks:
        return None
    search = " AND ".join(f"ti:{t}" for t in content_toks)
    params: dict[str, Any] = {
        "search_query": search,
        "start": 0,
        "max_results": 5,
    }
    r = await http_client.get(
        "https://export.arxiv.org/api/query",
        params=params,
        timeout=20.0,
    )
    if r.status_code != 200:
        return None
    text = r.text or ""
    entries = re.split(r"<entry>", text)[1:]
    for entry in entries:
        title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
        cand_title = (title_m.group(1) if title_m else "").strip()
        cand_title = re.sub(r"\s+", " ", cand_title)
        authors = re.findall(r"<name>(.*?)</name>", entry)
        published = re.search(r"<published>(\d{4})", entry)
        cand_year = int(published.group(1)) if published else None
        id_m = re.search(
            r"<id>https?://arxiv\.org/abs/([0-9]{4}\.[0-9]{4,6})",
            entry,
        )
        if not id_m:
            continue
        arxiv_id = id_m.group(1)
        if _validate_match(
            cand_title, authors, cand_year, title, target_authors, year,
        ):
            return f"10.48550/arXiv.{arxiv_id}"
    return None


async def _try_chromium_scholar(
    title: str,
    target_authors: list[str] | None,
    year: int | None,
    *,
    http_client: Any,
) -> str | None:
    """Headless Chromium -> Google Scholar -> DOI extraction + Crossref verify.

    Opt-in tier. Returns ``None`` immediately when ``playwright`` is
    not importable. When Chromium isn't installed either, the
    ``async_playwright`` launch raises and the tier logs + returns
    ``None``.

    Strategy: render the Scholar SERP for ``"<title> <year> <author>"``,
    extract DOI patterns from the rendered HTML (max 5 unique), then
    confirm each via Crossref ``/works/<doi>`` and validate the
    returned metadata against the bib entry. The Crossref verify step
    is essential — Scholar SERP DOIs can come from neighboring
    "cited by" / "related works" links, not the actual hit.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except ImportError:
        logger.info(
            "title_resolver_chromium_skipped",
            reason="playwright_not_installed",
        )
        return None

    from urllib.parse import quote

    query_parts = [title]
    if year:
        query_parts.append(str(year))
    if target_authors:
        first_real = next(
            (a for a in target_authors if not _is_junk_author(a)),
            "",
        )
        if first_real:
            query_parts.append(_last_name(first_real) or first_real)
    scholar_url = (
        "https://scholar.google.com/scholar?q=" + quote(" ".join(query_parts))
    )

    html = ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                page = await ctx.new_page()
                await page.goto(
                    scholar_url, wait_until="domcontentloaded", timeout=30000,
                )
                html = await page.content()
            finally:
                await browser.close()
    except Exception as exc:
        logger.info("title_resolver_chromium_failed", error=str(exc))
        return None

    # Pull DOI candidates from the rendered HTML. The pattern is
    # deliberately conservative — Crossref's syntactically valid set
    # is wider, but this catches everything we'd ever match against.
    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
    seen: set[str] = set()
    candidates: list[str] = []
    for raw in doi_pattern.findall(html):
        cleaned = raw.rstrip('.,;)"\'')
        if cleaned not in seen:
            seen.add(cleaned)
            candidates.append(cleaned)
        if len(candidates) >= 5:
            break

    for doi in candidates:
        try:
            r = await http_client.get(
                f"https://api.crossref.org/works/{doi}",
                timeout=15.0,
            )
            if r.status_code != 200:
                continue
            msg = ((r.json() or {}).get("message") or {})
            cand_title = ((msg.get("title") or [""])[0]).strip()
            cand_authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in (msg.get("author") or [])
                if a.get("family")
            ]
            cand_year: int | None = None
            date_parts = (
                (msg.get("issued") or {}).get("date-parts") or [[None]]
            )[0]
            if date_parts and date_parts[0]:
                try:
                    cand_year = int(date_parts[0])
                except (TypeError, ValueError):
                    cand_year = None
            if _validate_match(
                cand_title, cand_authors, cand_year, title, target_authors, year,
            ):
                return doi
        except Exception as exc:
            logger.debug("title_resolver_crossref_verify_failed", doi=doi, error=str(exc))
            continue
    return None


async def resolve_doi_from_title(
    title: str,
    authors: list[str] | None,
    year: int | str | None,
    *,
    http_client: Any,
    enable_browser: bool = False,
) -> str | None:
    """Try to discover a DOI from a paper title via scholarly metadata APIs.

    Walks OpenAlex -> Crossref -> Semantic Scholar -> arXiv; first
    validated match wins. With ``enable_browser=True``, appends a
    headless-Chromium Google-Scholar tier as the final fallback. A
    match requires either author-token overlap (when the bib entry has
    a real author) or strong title-word overlap, plus year +/-1.

    Returns the discovered DOI bare (``"10.1234/abc"``), or ``None``
    if no tier produced a validated hit. Network/HTTP errors at any
    tier are logged and treated as misses — the resolver never raises.
    """
    if not title or len(title.strip()) < 10:
        return None
    title = title.strip()

    year_int: int | None = None
    if year is not None:
        try:
            year_int = int(str(year)[:4])
        except (TypeError, ValueError):
            year_int = None

    tiers: list[tuple[str, Any]] = [
        ("openalex", _try_openalex),
        ("crossref", _try_crossref),
        ("semantic_scholar", _try_semantic_scholar),
        ("arxiv", _try_arxiv),
    ]
    if enable_browser:
        tiers.append(("chromium_scholar", _try_chromium_scholar))

    for tier_name, fn in tiers:
        try:
            doi = await fn(
                title, authors, year_int, http_client=http_client,
            )
            if doi:
                logger.info(
                    "title_resolver_match",
                    tier=tier_name,
                    doi=doi,
                    title=title[:80],
                )
                return doi
        except Exception as exc:
            logger.info(
                "title_resolver_tier_failed",
                tier=tier_name,
                error=str(exc),
            )

    logger.info(
        "title_resolver_no_match",
        title=title[:80],
        authors=(authors or [])[:3],
        year=year_int,
        browser_used=enable_browser,
    )
    return None
