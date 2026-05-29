"""Anthropic-style contextual retrieval: LLM-generated per-chunk context.

Following the technique published in
https://www.anthropic.com/news/contextual-retrieval (Sep 2024), each chunk
gets a short LLM-generated sentence or two that situates it in its parent
document before being embedded. The contextual prefix typically lifts
retrieval recall on technical content by 30-40% at the cost of one LLM
call per chunk during ingest.

The prefix is **embedding-only**: it goes into the embedding model but
not into chunk.text, so synthesis prompts see only the original chunk
body. This keeps the technique transparent to downstream code (basic
mode, profound mode, agentic synthesis all use chunk.text unchanged).

Configuration lives under ``knowledge_base.contextual_retrieval``:
- ``contextual_retrieval``: master switch (default off)
- ``contextual_retrieval_model``: LLM model (default claude-haiku-4-5)
- ``contextual_retrieval_provider``: LLM provider (default anthropic)
- ``contextual_retrieval_max_chars``: cap on prefix length

Caching: keyed on (paper_id, chunk_index, doc_hash) and stored in
``data/contextual_cache/`` so subsequent ingests of the same paper don't
re-spend LLM tokens.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.retrieval.contextual")


CACHE_DIR = Path("./data/contextual_cache")

# Anthropic's published prompt (adapted slightly to make the contract
# explicit). We give the LLM the full document AND the focal chunk, ask
# for a 50-100-token contextual sentence, and forbid restating the
# chunk verbatim (which would defeat the purpose of paraphrasing).
_PROMPT_SYSTEM = """You write retrieval context sentences for a research paper search system.

Given the FULL DOCUMENT and ONE FOCAL CHUNK from it, produce a single short
contextual sentence (50-100 words) that situates the chunk within the
document. The sentence must:

1. Identify what section/topic the chunk discusses
2. Mention the paper's overall subject when relevant
3. Use specific entities (instruments, organisms, datasets, methods) when present
4. NOT quote the chunk verbatim — re-express in your own words
5. Be a single paragraph, no headers, no markdown

Return ONLY the contextual sentence."""

_PROMPT_USER_TEMPLATE = """<document>
{document}
</document>

<chunk>
{chunk}
</chunk>

Write the 50-100 word contextual sentence:"""


def _cache_key(paper_id: str, chunk_index: int, doc_sha: str) -> str:
    """Stable cache key per (paper, chunk, document version)."""
    return f"{paper_id}::{chunk_index}::{doc_sha[:16]}"


def _cache_path(key: str) -> Path:
    """Cache file path for a key — uses sha256 to keep filenames safe."""
    safe = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{safe}.json"


def _cache_load(key: str) -> str | None:
    """Return cached prefix or None."""
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("context") or None
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _cache_store(key: str, context: str) -> None:
    """Persist the generated context."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(key).write_text(
            json.dumps({"key": key, "context": context}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        logger.info("contextual_cache_write_failed", error=str(e))


def _document_sha(text: str) -> str:
    """sha256 of the document text — invalidates the cache when the
    document changes (e.g., re-extracted from a new PDF source)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


async def generate_chunk_context(
    *,
    paper_id: str,
    chunk_index: int,
    chunk_text: str,
    document_text: str,
    llm_client: Any,
    model: str = "claude-haiku-4-5",
    provider: str = "anthropic",
    max_chars: int = 400,
    use_cache: bool = True,
) -> str:
    """Generate (and cache) a contextual prefix for one chunk.

    Returns an empty string on failure so callers can simply concatenate
    without conditional logic. The caller is responsible for deciding
    whether to invoke this at all (gated by
    ``config.knowledge_base.contextual_retrieval``).
    """
    if not chunk_text or not document_text:
        return ""

    doc_sha = _document_sha(document_text)
    key = _cache_key(paper_id, chunk_index, doc_sha)

    if use_cache:
        cached = _cache_load(key)
        if cached is not None:
            logger.debug("contextual_cache_hit", paper_id=paper_id, chunk_index=chunk_index)
            return cached[:max_chars]

    # Cap document length to keep prompt size bounded. Anthropic's
    # benchmark used full-document context, but for very long papers we
    # truncate at ~50k chars; the recall lift saturates well before that.
    doc = document_text
    if len(doc) > 50_000:
        doc = doc[:50_000] + "\n...[truncated for context generation]"

    # The document is identical across every chunk of the same paper,
    # so we mark it as the cacheable prefix. Anthropic's 5-minute cache
    # TTL means all-chunks-of-paper-X within one ingestion ride the
    # cache for ~90% off on the document tokens, which dominate cost.
    from perspicacite.llm.client import build_cached_messages
    messages = build_cached_messages(
        system=_PROMPT_SYSTEM,
        cacheable_context=f"<document>\n{doc}\n</document>",
        user_message=(
            f"<chunk>\n{chunk_text}\n</chunk>\n\n"
            "Write the 50-100 word contextual sentence:"
        ),
        provider=provider,
    )
    try:
        text = await llm_client.complete(
            messages=messages,
            model=model,
            provider=provider,
            max_tokens=200,
            temperature=0.2,
            stage="contextual.prefix",
        )
    except Exception as e:
        logger.info(
            "contextual_generation_failed",
            paper_id=paper_id,
            chunk_index=chunk_index,
            error=str(e),
        )
        return ""

    # Sanitize: strip surrounding whitespace, collapse runs, cap.
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars - 1].rstrip() + "…"

    if use_cache:
        _cache_store(key, text)
    logger.info(
        "contextual_generated",
        paper_id=paper_id,
        chunk_index=chunk_index,
        chars=len(text),
    )
    return text


# ---------------------------------------------------------------------------
# Document-tier helpers (one prefix per paper, shared across all chunks)
# ---------------------------------------------------------------------------


_DOC_SUMMARY_PROMPT_SYSTEM = """You write retrieval context for a research-paper search system.

Given a research paper's full text, produce a single 50-100 word
contextual paragraph that captures: the paper's overall subject; the
methods or instruments used; the main finding; key entities (organisms,
datasets, software). The paragraph is prepended to every chunk of the
paper before embedding, so it should help a search query land on this
paper regardless of which chunk it matches.

Return ONLY the paragraph — no headers, no markdown."""


_DOC_SUMMARY_PROMPT_USER = """<paper>
{document}
</paper>

Write the 50-100 word retrieval-context paragraph:"""


def abstract_prefix_for_paper(
    *, paper_text: str, abstract: str | None, max_chars: int = 500,
) -> str:
    """Free document-tier context: returns the abstract when present,
    else a leading slice of the paper text. Same string is reused for
    every chunk of this paper, so zero LLM cost.
    """
    if abstract and abstract.strip():
        text = abstract.strip()
    elif paper_text:
        text = paper_text[:max_chars * 2].strip()
    else:
        return ""
    # Single-line + capped
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


async def generate_document_summary(
    *,
    paper_id: str,
    document_text: str,
    llm_client: Any,
    model: str = "claude-haiku-4-5",
    provider: str = "anthropic",
    max_chars: int = 500,
    use_cache: bool = True,
) -> str:
    """Generate (and cache) ONE 50-100 word retrieval-context paragraph
    for a paper. Same string is reused as the prefix for every chunk of
    the paper at embedding time. ~1/N the cost of full per-chunk
    contextual retrieval for N chunks.
    """
    if not document_text:
        return ""

    doc_sha = _document_sha(document_text)
    # Document-tier key omits chunk_index — one summary per (paper, doc version).
    key = _cache_key(paper_id, -1, doc_sha)
    if use_cache:
        cached = _cache_load(key)
        if cached is not None:
            logger.debug("contextual_doc_cache_hit", paper_id=paper_id)
            return cached[:max_chars]

    doc = document_text
    if len(doc) > 50_000:
        doc = doc[:50_000] + "\n...[truncated for context generation]"

    messages = [
        {"role": "system", "content": _DOC_SUMMARY_PROMPT_SYSTEM},
        {"role": "user", "content": _DOC_SUMMARY_PROMPT_USER.format(document=doc)},
    ]
    try:
        text = await llm_client.complete(
            messages=messages,
            model=model,
            provider=provider,
            max_tokens=200,
            temperature=0.2,
            stage="contextual.doc_summary",
        )
    except Exception as e:
        logger.info(
            "contextual_doc_summary_failed",
            paper_id=paper_id,
            error=str(e),
        )
        return ""

    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"

    if use_cache:
        _cache_store(key, text)
    logger.info(
        "contextual_doc_summary_generated",
        paper_id=paper_id,
        chars=len(text),
    )
    return text


async def generate_chunk_contexts_bulk(
    *,
    paper_id: str,
    chunks: list[Any],
    document_text: str,
    llm_client: Any,
    model: str = "claude-haiku-4-5",
    provider: str = "anthropic",
    max_chars: int = 400,
    use_cache: bool = True,
) -> list[str]:
    """Generate contexts for many chunks of one document.

    Sequential by default — most LLM providers rate-limit aggressive
    parallelism. If you need throughput, wrap this in asyncio.gather
    with a Semaphore.

    Returns one string per chunk, in order. Empty strings on failure.
    """
    out: list[str] = []
    for i, ch in enumerate(chunks):
        text = getattr(ch, "text", None) or ch.get("text") if isinstance(ch, dict) else None
        if not text:
            out.append("")
            continue
        idx = getattr(ch.metadata, "chunk_index", i) if hasattr(ch, "metadata") else i
        ctx = await generate_chunk_context(
            paper_id=paper_id,
            chunk_index=idx,
            chunk_text=text,
            document_text=document_text,
            llm_client=llm_client,
            model=model,
            provider=provider,
            max_chars=max_chars,
            use_cache=use_cache,
        )
        out.append(ctx)
    return out
