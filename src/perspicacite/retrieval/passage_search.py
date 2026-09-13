"""Passage-level semantic retrieval.

This module is the shared core behind two MCP tools — ``search_by_passage``
(text input from the caller, e.g. a paragraph) and ``get_relevant_passages``
(keyword query with optional adaptive retry). Both end up calling
:func:`search_passages` with a fully-constructed retriever.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

MAX_TEXT_CHARS = 4000
MAX_K = 50


class _AsyncRetriever(Protocol):
    async def search(
        self, query: str, top_k: int = 10, filters: Any | None = None
    ) -> list[dict[str, Any]]:
        ...


class PassageRetriever:
    """Adapt an explicit chunk-capable retriever to the passage search protocol."""

    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    async def search(
        self, query: str, top_k: int = 10, filters: Any | None = None
    ) -> list[dict[str, Any]]:
        """Request chunk results; never fall back to paper-level deduplication."""
        return await self.retriever.search_chunks(query, top_k=top_k, filters=filters)


@dataclass(frozen=True)
class PassageSource:
    doi: str | None
    title: str | None
    authors: list[str] | None
    year: int | None
    bibkey: str | None
    source_url: str | None
    license_id: str | None


@dataclass(frozen=True)
class PassageMatch:
    chunk_id: str
    chunk_text: str
    score: float
    source: PassageSource
    kb_name: str | None
    collection_name: str | None = None
    content_sha256: str | None = None


def _validate_text(text: str) -> None:
    if not text or not text.strip():
        raise ValueError("input text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"input text exceeds {MAX_TEXT_CHARS} chars "
            "(caller must chunk longer inputs)"
        )


def _coerce_metadata(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "__dict__") and not isinstance(raw, dict):
        return dict(raw.__dict__)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _to_match(raw: dict[str, Any]) -> PassageMatch:
    meta = _coerce_metadata(raw.get("metadata"))
    paper_id = raw.get("paper_id") or meta.get("paper_id") or meta.get("doi") or ""
    kb = raw.get("kb_name") or meta.get("kb_name")
    text_hash = hashlib.blake2b((raw.get("text") or "").encode("utf-8"), digest_size=8).hexdigest()
    chunk_id = raw.get("chunk_id") or f"{kb}:{paper_id}:{text_hash}"
    source = PassageSource(
        doi=meta.get("doi"),
        title=meta.get("title"),
        authors=meta.get("authors"),
        year=meta.get("year"),
        bibkey=meta.get("bibkey"),
        source_url=meta.get("source_url") or meta.get("url"),
        license_id=meta.get("license_id") or meta.get("license"),
    )
    return PassageMatch(
        chunk_id=str(chunk_id),
        chunk_text=str(raw.get("text", "")),
        score=float(raw.get("score") or 0.0),
        source=source,
        kb_name=kb,
        collection_name=raw.get("collection_name"),
        content_sha256=hashlib.sha256(str(raw.get("text", "")).encode("utf-8")).hexdigest(),
    )


async def search_passages(
    retriever: _AsyncRetriever,
    *,
    text: str,
    k: int = 5,
    min_score: float | None = None,
) -> list[PassageMatch]:
    """Run a passage-level search against an already-constructed retriever.

    The retriever knows which KB(s) to query and how. This function only
    handles input validation, k-clamping, response normalisation, and the
    optional min_score filter.

    Note: when ``min_score`` is supplied, the returned list may contain
    fewer than ``k`` entries (or be empty) because the filter is applied
    after retrieval.
    """
    _validate_text(text)
    capped_k = min(max(k, 1), MAX_K)
    raw_results = await retriever.search(text, top_k=capped_k, filters=None)
    matches = [_to_match(r) for r in raw_results]
    if min_score is not None:
        matches = [m for m in matches if m.score >= min_score]
    return matches
