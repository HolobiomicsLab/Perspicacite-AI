"""Zotero → KB ingest: plan-then-execute. Worker (in same module) drives the unified pipeline."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from perspicacite.logging import get_logger
from perspicacite.models.papers import Author, Paper, PaperSource
from perspicacite.pipeline.download import retrieve_paper_content
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

if TYPE_CHECKING:
    from perspicacite.integrations.zotero import ZoteroClient

logger = get_logger("perspicacite.zotero_ingest")

# Max concurrent attachment/note lookups when summarizing a collection.
# Each item is one HTTP call; gathering an entire large collection at once
# (e.g. 727 items -> ~1450 requests) exhausts httpx's connection pool and
# raises PoolTimeout. Cap it well under httpx's default 100-connection pool.
_SUMMARIZE_CONCURRENCY = 8


class ZoteroKBPlanEntry(BaseModel):
    """One KB to be created/populated from a Zotero source."""

    kb_name: str
    source_collection_key: str | None
    source_collection_name: str | None
    item_count: int
    with_doi_count: int
    with_pdf_count: int
    with_notes_count: int


def _slugify(name: str) -> str:
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "kb"


async def _summarize_items(client: ZoteroClient, items: list[dict[str, Any]]) -> dict[str, int]:
    """Return {with_doi, with_pdf, with_notes} counts for a list of Zotero items.

    Attachment/note checks are one HTTP call per item. A bounded semaphore
    caps how many run at once so a large collection doesn't fire hundreds of
    simultaneous requests and exhaust httpx's connection pool (PoolTimeout).
    """
    with_doi = sum(1 for it in items if (it.get("data") or {}).get("DOI"))

    if not items:
        return {"with_doi": 0, "with_pdf": 0, "with_notes": 0}

    sem = asyncio.Semaphore(_SUMMARIZE_CONCURRENCY)

    async def _has_pdf(it: dict[str, Any]) -> bool:
        async with sem:
            atts = await client.get_item_attachments(it["key"])
        return any(
            (a.get("data") or {}).get("contentType") == "application/pdf"
            and (a.get("data") or {}).get("linkMode") in {"imported_file", "imported_url"}
            for a in atts
        )

    async def _has_note(it: dict[str, Any]) -> bool:
        async with sem:
            notes = await client.get_item_notes(it["key"])
        return any(n for n in notes)

    pdf_flags = await asyncio.gather(*(_has_pdf(it) for it in items))
    note_flags = await asyncio.gather(*(_has_note(it) for it in items))
    return {
        "with_doi": with_doi,
        "with_pdf": sum(1 for f in pdf_flags if f),
        "with_notes": sum(1 for f in note_flags if f),
    }


async def plan_kbs_from_zotero(
    client: ZoteroClient,
    *,
    top_level_collection_keys: list[str] | None = None,
    include_unfiled: bool = True,
    library_label: str = "Library",
) -> list[ZoteroKBPlanEntry]:
    """Return preview of what would be ingested — one entry per top-level
    collection (optionally rolled up across subcollections) plus an Unfiled
    bucket. Each entry counts items, items-with-DOI, items-with-PDF,
    items-with-notes."""
    out: list[ZoteroKBPlanEntry] = []
    tops = await client.list_top_level_collections()
    if top_level_collection_keys is not None:
        keys = set(top_level_collection_keys)
        tops = [c for c in tops if c.get("key") in keys]
    for c in tops:
        name = (c.get("data") or {}).get("name") or c["key"]
        items = await client.list_items_in_collection(c["key"], include_subcollections=True)
        summary = await _summarize_items(client, items)
        out.append(
            ZoteroKBPlanEntry(
                kb_name=f"{_slugify(library_label)}_{_slugify(name)}",
                source_collection_key=c["key"],
                source_collection_name=name,
                item_count=len(items),
                with_doi_count=summary["with_doi"],
                with_pdf_count=summary["with_pdf"],
                with_notes_count=summary["with_notes"],
            )
        )
    if include_unfiled:
        items = await client.list_top_level_items_without_collection()
        if items:
            summary = await _summarize_items(client, items)
            out.append(
                ZoteroKBPlanEntry(
                    kb_name=f"{_slugify(library_label)}_Unfiled",
                    source_collection_key=None,
                    source_collection_name=None,
                    item_count=len(items),
                    with_doi_count=summary["with_doi"],
                    with_pdf_count=summary["with_pdf"],
                    with_notes_count=summary["with_notes"],
                )
            )
    return out


def _item_to_paper(item: dict[str, Any]) -> Paper:
    """Convert a Zotero item to a perspicacite Paper. DOI-bearing items keep
    DOI as the canonical id; others fall back to the Zotero key."""
    data = item.get("data") or {}
    creators = data.get("creators") or []
    authors: list[Author] = []
    for cr in creators:
        first = cr.get("firstName") or ""
        last = cr.get("lastName") or cr.get("name") or ""
        full = (first + " " + last).strip() or last or first
        if full:
            authors.append(Author(name=full))
    doi = data.get("DOI") or None
    year: int | None = None
    date_str = str(data.get("date") or "")[:4]
    if date_str.isdigit():
        year = int(date_str)
    return Paper(
        id=doi or item.get("key") or "zotero:unknown",
        title=data.get("title") or "Untitled",
        authors=authors,
        doi=doi,
        year=year,
        journal=data.get("publicationTitle") or None,
        abstract=data.get("abstractNote") or None,
        # Closest existing source — Zotero items most often have DOIs and may
        # carry attached PDFs imported by the user.
        source=PaperSource.BIBTEX,
    )


async def _ensure_kb(*, name: str, app_state: Any) -> Any:
    """Return existing KB metadata or create a new one.

    Tries ``session_store.create_kb_metadata`` first (newer API surface used by
    tests/mocks).  Falls back to constructing a ``KnowledgeBase`` model and
    persisting via ``save_kb_metadata`` — the path used by the production
    session store today.
    """
    kb = await app_state.session_store.get_kb_metadata(name)
    if kb is not None:
        return kb
    create_fn = getattr(app_state.session_store, "create_kb_metadata", None)
    if create_fn is not None:
        created = await create_fn(name=name)
        if created is not None:
            return created
        # If the helper returned None, refetch.
        kb = await app_state.session_store.get_kb_metadata(name)
        if kb is not None:
            return kb
    # Production fallback: build the KB model ourselves and save it.
    from perspicacite.models.kb import (
        ChunkConfig,
        KnowledgeBase,
        chroma_collection_name_for_kb,
    )

    collection_name = chroma_collection_name_for_kb(name)
    # Create the underlying vector collection (idempotent).
    try:
        await app_state.vector_store.create_collection(collection_name)
    except Exception as exc:
        logger.warning("zotero_create_collection_failed", kb=name, error=str(exc))
    embedding_model = (
        getattr(app_state.embedding_provider, "model_name", None)
        if app_state.embedding_provider is not None
        else None
    ) or "text-embedding-3-small"
    new_kb = KnowledgeBase(
        name=name,
        description=None,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunk_config=ChunkConfig(),
    )
    await app_state.session_store.save_kb_metadata(new_kb)
    return new_kb


async def build_kbs_from_zotero(
    client: ZoteroClient,
    *,
    plan: list[ZoteroKBPlanEntry],
    app_state: Any,
    registry: Any,
    job_id: str,
) -> dict[str, Any]:
    """Execute a Zotero ingest plan. Emits per-item progress; finishes via registry.

    Per-item flow:
      1. dedup by paper_id (DOI when present, else Zotero key)
      2. fetch full text via the unified pipeline (uses DOI when available)
      3. fall back to attached PDF bytes via pdf_parser
      4. append any Zotero notes as a 'Notes' section
      5. accumulate in ``papers``; flush via ``DynamicKnowledgeBase.add_papers`` per KB
    """
    summary_per_kb: list[dict[str, Any]] = []
    try:
        for entry in plan:
            kb = await _ensure_kb(name=entry.kb_name, app_state=app_state)
            if entry.source_collection_key is None:
                items = await client.list_top_level_items_without_collection()
            else:
                items = await client.list_items_in_collection(
                    entry.source_collection_key, include_subcollections=True
                )
            papers: list[Paper] = []
            skipped = 0
            for idx, it in enumerate(items):
                paper = _item_to_paper(it)
                pid = paper.doi or paper.id
                if await app_state.vector_store.paper_exists(kb.collection_name, pid):
                    skipped += 1
                    await registry.publish(
                        job_id,
                        {
                            "type": "progress",
                            "kb": entry.kb_name,
                            "done": idx + 1,
                            "status": "skipped",
                        },
                    )
                    continue

                # Unified pipeline first (when DOI is known)
                if paper.doi and app_state.pdf_parser is not None:
                    try:
                        res = await retrieve_paper_content(
                            paper.doi, pdf_parser=app_state.pdf_parser
                        )
                        if res.success and res.full_text:
                            paper.full_text = res.full_text
                    except Exception as exc:
                        logger.warning(
                            "zotero_unified_pipeline_failed",
                            doi=paper.doi,
                            error=str(exc),
                        )

                # Fall back to attached PDF
                if not paper.full_text and app_state.pdf_parser is not None:
                    try:
                        atts = await client.get_item_attachments(it["key"])
                    except Exception as exc:
                        logger.warning(
                            "zotero_attachment_list_failed",
                            item=it.get("key"),
                            error=str(exc),
                        )
                        atts = []
                    for a in atts:
                        if (a.get("data") or {}).get("contentType") != "application/pdf":
                            continue
                        blob = await client.download_attachment_bytes(a["key"])
                        if not blob:
                            continue
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                            f.write(blob)
                            tmp = Path(f.name)
                        try:
                            parsed = await app_state.pdf_parser.parse(tmp)
                            if parsed.text:
                                paper.full_text = parsed.text
                                break
                        except Exception as exc:
                            logger.warning(
                                "zotero_pdf_parse_failed",
                                item=it.get("key"),
                                error=str(exc),
                            )
                        finally:
                            tmp.unlink(missing_ok=True)

                # Attach notes
                try:
                    notes = await client.get_item_notes(it["key"])
                except Exception as exc:
                    logger.debug("zotero notes fetch failed", error=str(exc))
                    notes = []
                if notes:
                    note_block = "\n\n# Notes\n\n" + "\n\n".join(notes)
                    paper.full_text = (paper.full_text or "") + note_block

                papers.append(paper)
                await registry.publish(
                    job_id,
                    {
                        "type": "progress",
                        "kb": entry.kb_name,
                        "done": idx + 1,
                        "status": "embedded",
                    },
                )

            added_chunks = 0
            if papers:
                dkb = DynamicKnowledgeBase(
                    vector_store=app_state.vector_store,
                    embedding_service=app_state.embedding_provider,
                )
                dkb.collection_name = kb.collection_name
                dkb._initialized = True
                added_chunks = await dkb.add_papers(papers, include_full_text=True)
                kb.paper_count += len(papers)
                kb.chunk_count += added_chunks
                await app_state.session_store.save_kb_metadata(kb)

            # Cycle A: on-disk capsule artifacts per added paper (metadata + resources + blocks).
            # Skip per-paper if capsule auto-build disabled.
            if app_state.config.capsule.auto_build_on_ingest and papers:
                from perspicacite.pipeline.capsule_builder import build_capsule
                for p in papers:
                    try:
                        await build_capsule(
                            paper=p, pdf_path=None, kb_name=entry.kb_name,
                            app_state=app_state, ingest_chunks=False,
                        )
                    except Exception as exc:
                        logger.warning("capsule_build_failed", paper_id=p.id, error=str(exc))

            summary_per_kb.append(
                {
                    "kb_name": entry.kb_name,
                    "added_papers": len(papers),
                    "added_chunks": added_chunks,
                    "skipped": skipped,
                }
            )

        result = {"per_kb": summary_per_kb}
        await registry.finish(job_id, result)
        return result
    except Exception as exc:
        logger.error("zotero_ingest_worker_failed", error=str(exc))
        await registry.fail(job_id, str(exc))
        raise
