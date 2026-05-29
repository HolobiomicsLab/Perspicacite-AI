"""Knowledge base CRUD + paper / bibtex / chunk routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from perspicacite.integrations.local_docs import (
    LocalDocsDisabledError,
    LocalDocsValidationError,
    expand_paths,
    ingest_local_documents,
    validate_local_path,
)
from perspicacite.models.kb import (
    ChunkConfig,
    KnowledgeBase,
    chroma_collection_name_for_kb,
)
from perspicacite.web.state import app_state

logger = logging.getLogger(__name__)


router = APIRouter()

# Strong-reference set that keeps fire-and-forget ingestion Tasks alive until
# they complete.  asyncio.create_task() returns a Task that Python's GC may
# collect if no other reference is held; storing it here prevents mid-flight
# cancellation.  The done_callback removes the Task once it finishes so the
# set does not grow unbounded.
_background_tasks: set = set()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class KBCreateRequest(BaseModel):
    """Request to create a knowledge base."""

    name: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9 _-]+$",
        min_length=1,
        max_length=100,
        description="KB name (spaces will be converted to underscores)",
    )
    description: str | None = None


class PaperData(BaseModel):
    """Paper data from chat results, for adding to KB."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    citations: int | None = None
    file: str | None = Field(default=None, description="Local PDF path (Zotero/Mendeley export)")


class KBAddPapersRequest(BaseModel):
    """Request to add papers to a knowledge base."""

    papers: list[PaperData]


class KBAddDOIsRequest(BaseModel):
    """Request to bulk-add papers from a list of DOIs."""

    dois: list[str] = Field(..., min_length=1)


class ExpandSimilarScoreRequest(BaseModel):
    direction: str = "both"          # "forward" | "backward" | "both"
    max_per_seed: int = 10
    method: str = "hybrid"           # "embedding" | "bm25" | "hybrid"


class _CalibrationLabel(BaseModel):
    score: float
    relevant: bool


class ExpandSimilarCutoffRequest(BaseModel):
    labels: list[_CalibrationLabel] = Field(default_factory=list)


class _ScoredCandidate(BaseModel):
    doi: str | None = None
    score: float


class ExpandSimilarCommitRequest(BaseModel):
    scored: list[_ScoredCandidate] = Field(default_factory=list)
    cutoff: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_bibtex_entries(bibtex_text: str) -> int:
    """Return the number of BibTeX entries in *bibtex_text* (best-effort)."""
    try:
        import bibtexparser

        return len(bibtexparser.loads(bibtex_text).entries)
    except Exception:
        return 0


async def _bibtex_ingest_worker(
    *,
    name: str,
    bibtex_text: str,
    job_id: str,
    registry,
) -> None:
    """Background worker for async BibTeX ingestion.

    Replicates the same logic as the sync handler but publishes per-paper
    progress events and finishes/fails via the JobRegistry.
    """
    try:
        import bibtexparser

        from perspicacite.models.papers import Author, PaperSource
        from perspicacite.pipeline.bibtex_kb import entries_to_papers
        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

        kb = await app_state.session_store.get_kb_metadata(name)
        if not kb:
            await registry.fail(job_id, f"Knowledge base '{name}' not found")
            return

        db = bibtexparser.loads(bibtex_text)
        papers = entries_to_papers(db.entries)
        if not papers:
            await registry.finish(job_id, {"added_papers": 0, "added_chunks": 0, "skipped": 0})
            return

        pdf_config = app_state.config.pdf_download if app_state.config else None
        pdf_kw = _get_pdf_fallback_kwargs(pdf_config)

        papers_to_add = []
        skipped = 0
        for i, paper in enumerate(papers):
            paper_id = paper.doi if paper.doi else paper.id
            exists = await app_state.vector_store.paper_exists(kb.collection_name, paper_id)
            if exists:
                skipped += 1
                await registry.publish(
                    job_id,
                    {"type": "progress", "done": i + 1, "doi": paper.doi, "status": "skipped"},
                )
                continue

            paper.source = PaperSource.BIBTEX

            # Local PDF check
            local_path = Path(paper.pdf_url) if paper.pdf_url else None
            if local_path and local_path.suffix.lower() == ".pdf" and local_path.exists():
                try:
                    parsed = await app_state.pdf_parser.parse(local_path)
                    if parsed.text:
                        paper.full_text = parsed.text
                        papers_to_add.append(paper)
                        await registry.publish(
                            job_id,
                            {
                                "type": "progress",
                                "done": i + 1,
                                "doi": paper.doi,
                                "status": "local_pdf",
                            },
                        )
                        continue
                except Exception as exc:
                    logger.warning(f"Local PDF parse failed for {paper.title[:50]}: {exc}")

            if paper.doi and app_state.pdf_parser:
                try:
                    result = await retrieve_paper_content(
                        paper.doi, pdf_parser=app_state.pdf_parser, **pdf_kw
                    )
                    if result.success and result.full_text:
                        paper.full_text = result.full_text
                        meta = result.metadata or {}
                        if meta.get("title") and not paper.title:
                            paper.title = meta["title"]
                        if meta.get("authors") and not paper.authors:
                            paper.authors = [Author(name=a) for a in meta["authors"]]
                        if result.abstract and not paper.abstract:
                            paper.abstract = result.abstract
                        await registry.publish(
                            job_id,
                            {
                                "type": "progress",
                                "done": i + 1,
                                "doi": paper.doi,
                                "status": "embedded",
                            },
                        )
                    else:
                        await registry.publish(
                            job_id,
                            {
                                "type": "progress",
                                "done": i + 1,
                                "doi": paper.doi,
                                "status": "no_full_text",
                            },
                        )
                except Exception as exc:
                    logger.warning(f"Content download failed for {paper.title[:50]}: {exc}")
                    await registry.publish(
                        job_id,
                        {"type": "progress", "done": i + 1, "doi": paper.doi, "status": "failed"},
                    )
            else:
                await registry.publish(
                    job_id,
                    {"type": "progress", "done": i + 1, "doi": paper.doi, "status": "no_doi"},
                )

            papers_to_add.append(paper)

        added = 0
        if papers_to_add:
            dkb = DynamicKnowledgeBase(
                vector_store=app_state.vector_store,
                embedding_service=app_state.embedding_provider,
            )
            dkb.collection_name = kb.collection_name
            dkb._initialized = True
            added = await dkb.add_papers(papers_to_add, include_full_text=True)
            kb.paper_count += len(papers_to_add)
            kb.chunk_count += added
            await app_state.session_store.save_kb_metadata(kb)

        # Cycle A: on-disk capsule artifacts per added paper (metadata + resources + blocks).
        # Skip per-paper if capsule auto-build disabled.
        if app_state.config.capsule.auto_build_on_ingest and papers_to_add:
            from perspicacite.pipeline.capsule_builder import build_capsule
            for p in papers_to_add:
                try:
                    await build_capsule(
                        paper=p, pdf_path=None, kb_name=name,
                        app_state=app_state, ingest_chunks=False,
                    )
                except Exception as exc:
                    logger.warning(f"capsule build failed for {p.id}: {exc}")

        added_with_full_text = sum(1 for p in papers_to_add if getattr(p, "full_text", None))
        added_metadata_only = len(papers_to_add) - added_with_full_text
        await registry.finish(
            job_id,
            {
                "added_papers": len(papers_to_add),
                "added_with_full_text": added_with_full_text,
                "added_metadata_only": added_metadata_only,
                "added_chunks": added,
                "skipped": skipped,
            },
        )
    except Exception as exc:
        logger.error(f"bibtex_ingest_worker failed: {exc}", exc_info=True)
        await registry.fail(job_id, str(exc))


async def _dois_ingest_worker(
    *,
    name: str,
    dois: list[str],
    job_id: str,
    registry,
) -> None:
    """Background worker for async DOI ingestion.

    Mirrors the sync add_dois_to_kb handler but publishes per-DOI progress
    events and finishes/fails via the JobRegistry.
    """
    try:
        from perspicacite.models.papers import Author, Paper, PaperSource
        from perspicacite.pipeline.download import retrieve_paper_content
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

        kb = await app_state.session_store.get_kb_metadata(name)
        if not kb:
            await registry.fail(job_id, f"Knowledge base '{name}' not found")
            return

        pdf_config = app_state.config.pdf_download if app_state.config else None
        pdf_kw = _get_pdf_fallback_kwargs(pdf_config)

        papers_to_add: list = []
        skipped: list = []
        failed: list = []
        metadata_only: list = []  # F-28/F-30
        dl: dict = {"attempted": 0, "success": 0, "failed": 0, "metadata_only": 0}

        for i, raw_doi in enumerate(dois):
            doi = (raw_doi or "").strip().replace("https://doi.org/", "")
            if not doi:
                await registry.publish(
                    job_id,
                    {"type": "progress", "done": i + 1, "doi": raw_doi, "status": "empty"},
                )
                continue

            if await app_state.vector_store.paper_exists(kb.collection_name, doi):
                skipped.append({"doi": doi})
                await registry.publish(
                    job_id,
                    {"type": "progress", "done": i + 1, "doi": doi, "status": "skipped"},
                )
                continue

            dl["attempted"] += 1
            try:
                result = await retrieve_paper_content(
                    doi, pdf_parser=app_state.pdf_parser, **pdf_kw
                )
            except Exception as exc:
                failed.append({"doi": doi, "reason": str(exc)})
                dl["failed"] += 1
                await registry.publish(
                    job_id,
                    {"type": "progress", "done": i + 1, "doi": doi, "status": "failed"},
                )
                continue

            if not result or not result.success:
                attempts = list(getattr(result, "attempts", []) or [])
                failed.append({
                    "doi": doi,
                    "reason": "; ".join(f"{a['source']}:{a['status']}" for a in attempts) or "no content",
                    "attempts": attempts,
                })
                dl["failed"] += 1
                await registry.publish(
                    job_id,
                    {"type": "progress", "done": i + 1, "doi": doi, "status": "no_content"},
                )
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
            )
            if result.full_text:
                paper.full_text = result.full_text
                dl["success"] += 1
                status = "embedded"
            else:
                dl["metadata_only"] += 1
                metadata_only.append({
                    "doi": doi,
                    "content_type": paper.content_type,
                    "attempts": list(getattr(result, "attempts", []) or []),
                })
                status = "no_full_text"
            papers_to_add.append(paper)
            await registry.publish(
                job_id,
                {"type": "progress", "done": i + 1, "doi": doi, "status": status},
            )

        added = 0
        added_with_full_text = sum(1 for p in papers_to_add if getattr(p, "full_text", None))
        added_metadata_only = len(papers_to_add) - added_with_full_text
        if papers_to_add:
            dkb = DynamicKnowledgeBase(
                vector_store=app_state.vector_store,
                embedding_service=app_state.embedding_provider,
            )
            dkb.collection_name = kb.collection_name
            dkb._initialized = True
            added = await dkb.add_papers(papers_to_add, include_full_text=True)
            kb.paper_count += len(papers_to_add)
            kb.chunk_count += added
            await app_state.session_store.save_kb_metadata(kb)

        # Cycle A: on-disk capsule artifacts per added paper (metadata + resources + blocks).
        # Skip per-paper if capsule auto-build disabled.
        if app_state.config.capsule.auto_build_on_ingest and papers_to_add:
            from perspicacite.pipeline.capsule_builder import build_capsule
            for p in papers_to_add:
                try:
                    await build_capsule(
                        paper=p, pdf_path=None, kb_name=name,
                        app_state=app_state, ingest_chunks=False,
                    )
                except Exception as exc:
                    logger.warning(f"capsule build failed for {p.id}: {exc}")

        await registry.finish(
            job_id,
            {
                "added_papers": len(papers_to_add),
                "added_with_full_text": added_with_full_text,
                "added_metadata_only": added_metadata_only,
                "added_chunks": added,
                "skipped_duplicates": len(skipped),
                "failed": failed,
                "metadata_only": metadata_only,
                "pdf_download": dl,
            },
        )
    except Exception as exc:
        logger.error(f"dois_ingest_worker failed: {exc}", exc_info=True)
        await registry.fail(job_id, str(exc))


def _get_pdf_fallback_kwargs(pdf_config) -> dict:
    """Build keyword args for retrieve_paper_content from PDFDownloadConfig."""
    if not pdf_config:
        return {}
    return {
        "alternative_endpoint": pdf_config.alternative_endpoint,
        "unpaywall_email": pdf_config.unpaywall_email,
        "wiley_tdm_token": pdf_config.wiley_tdm_token,
        "aaas_api_key": pdf_config.aaas_api_key,
        "rsc_api_key": pdf_config.rsc_api_key,
        "springer_api_key": pdf_config.springer_api_key,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/api/kb")
async def list_knowledge_bases():
    """List all knowledge bases."""
    if not app_state.session_store:
        return []
    kbs = await app_state.session_store.list_kbs()
    results = []
    for kb in kbs:
        stats = await app_state.vector_store.get_collection_stats(kb.collection_name)
        results.append(
            {
                "name": kb.name,
                "description": kb.description,
                "paper_count": stats.get("unique_papers", kb.paper_count),
                "chunk_count": stats.get("count", 0),
                "created_at": kb.created_at.isoformat() if kb.created_at else None,
            }
        )
    return results


@router.post("/api/kb")
async def create_knowledge_base(request: KBCreateRequest):
    """Create a new knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    # Sanitize KB name: replace spaces with underscores for storage
    kb_name = request.name.strip().replace(" ", "_")
    collection_name = chroma_collection_name_for_kb(kb_name)

    existing = await app_state.session_store.get_kb_metadata(kb_name)
    if existing:
        return {"error": f"Knowledge base '{kb_name}' already exists"}

    # Create collection (handles "already exists" gracefully)
    await app_state.vector_store.create_collection(collection_name)

    kb = KnowledgeBase(
        name=kb_name,
        description=request.description,
        collection_name=collection_name,
        embedding_model=app_state.embedding_provider.model_name,
        chunk_config=ChunkConfig(),
    )
    await app_state.session_store.save_kb_metadata(kb)
    logger.info(f"Created KB: {kb_name} (collection: {collection_name})")

    return {
        "name": kb.name,
        "description": kb.description,
        "collection_name": collection_name,
        "paper_count": 0,
        "chunk_count": 0,
    }


@router.get("/api/kb/{name}")
async def get_knowledge_base(name: str):
    """Get knowledge base details."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    stats = await app_state.vector_store.get_collection_stats(kb.collection_name)
    return {
        "name": kb.name,
        "description": kb.description,
        "paper_count": stats.get("unique_papers", kb.paper_count),
        "chunk_count": stats.get("count", 0),
        "embedding_model": kb.embedding_model,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
    }


@router.delete("/api/kb/{name}")
async def delete_knowledge_base(name: str):
    """Delete a knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    # Collection may not exist in ChromaDB
    with contextlib.suppress(Exception):
        await app_state.vector_store.delete_collection(kb.collection_name)

    # Delete metadata from SQLite
    import aiosqlite

    async with aiosqlite.connect(app_state.session_store.db_path) as db:
        await db.execute("DELETE FROM kb_metadata WHERE name = ?", (name,))
        await db.commit()

    logger.info(f"Deleted KB: {name}")
    return {"deleted": name}


@router.get("/api/kb/{name}/papers")
async def list_kb_papers(name: str):
    """List the papers in a knowledge base, derived from chunk metadata.

    Backs the KB detail page's Papers tab + overview counts. Returns a 200
    with an empty list (rather than 404) for an unknown/empty KB so the
    frontend's combined ``Promise.all`` load doesn't reject and blank the
    whole page.

    Papers are derived from ChromaDB chunk metadata (deduped by paper_id) —
    the authoritative record of what's actually embedded — rather than the
    ``kb_paper_references`` table, which isn't always populated (e.g. for KBs
    built through paths that don't record references).
    """
    if not app_state.session_store:
        return {"papers": [], "total": 0}
    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"papers": [], "total": 0}

    scan_cap = 20000
    try:
        coll = app_state.vector_store.client.get_collection(name=kb.collection_name)
        total_chunks = coll.count()
        got = coll.get(
            limit=min(total_chunks, scan_cap) if total_chunks else 0,
            include=["metadatas"],
        )
        metas = got.get("metadatas") or []
    except Exception as e:
        logger.warning(f"kb papers: collection scan failed for {name}: {e}")
        metas = []

    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in metas:
        m = m or {}
        pid = m.get("paper_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        authors = m.get("authors")
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        papers.append(
            {
                "paper_id": pid,
                "title": m.get("title"),
                "authors": authors or [],
                "year": m.get("year"),
                "doi": m.get("doi"),
                "added_at": None,
            }
        )
    return {"papers": papers, "total": len(papers)}


@router.post("/api/kb/{name}/papers")
async def add_papers_to_kb(name: str, request: KBAddPapersRequest):
    """Add papers to a knowledge base with deduplication and optional PDF download."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    from perspicacite.models.papers import Author, Paper, PaperSource
    from perspicacite.pipeline.download import retrieve_paper_content
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

    # Convert PaperData dicts to Paper models with deduplication check
    papers_to_add = []
    skipped_duplicates = []
    download_stats = {"attempted": 0, "success": 0, "failed": 0}

    pdf_config = app_state.config.pdf_download if app_state.config else None
    pdf_kw = _get_pdf_fallback_kwargs(pdf_config)

    for pd in request.papers:
        import hashlib

        paper_id = (
            pd.doi if pd.doi else f"generated:{hashlib.md5(pd.title.encode()).hexdigest()[:12]}"
        )

        # Check if paper already exists in this KB
        exists = await app_state.vector_store.paper_exists(kb.collection_name, paper_id)
        if exists:
            skipped_duplicates.append(
                {
                    "title": pd.title,
                    "paper_id": paper_id,
                    "doi": pd.doi,
                }
            )
            continue

        authors = [Author(name=a) for a in pd.authors]
        paper = Paper(
            id=paper_id,
            title=pd.title,
            authors=authors,
            year=pd.year,
            doi=pd.doi,
            abstract=pd.abstract,
            citation_count=pd.citations,
            source=PaperSource.USER_UPLOAD,
        )

        # Try local PDF first (e.g. from Zotero/Mendeley export)
        full_text = None
        if pd.file:
            local_path = Path(pd.file)
            if local_path.suffix.lower() == ".pdf" and local_path.exists():
                try:
                    parsed = await app_state.pdf_parser.parse(local_path)
                    if parsed.text:
                        full_text = parsed.text
                        download_stats["success"] += 1
                        logger.info(f"Parsed local PDF for: {pd.title[:50]}...")
                except Exception as e:
                    logger.warning(f"Local PDF parse failed for {pd.title[:50]}: {e}")

        # Try to download full text if DOI available and no local PDF
        if full_text is None and pd.doi and app_state.pdf_downloader and app_state.pdf_parser:
            download_stats["attempted"] += 1
            try:
                result = await retrieve_paper_content(
                    pd.doi, pdf_parser=app_state.pdf_parser, **pdf_kw
                )
                if result.success and result.full_text:
                    full_text = result.full_text
                    download_stats["success"] += 1
                    # Enrich paper metadata from discovery if original was placeholder
                    meta = result.metadata or {}
                    if meta.get("title") and paper.title.startswith("Reference"):
                        paper.title = meta["title"]
                    if meta.get("authors") and not paper.authors:
                        from perspicacite.models.papers import Author

                        paper.authors = [Author(name=a) for a in meta["authors"]]
                    if result.abstract and not paper.abstract:
                        paper.abstract = result.abstract
                    logger.info(f"Downloaded full text for: {paper.title[:50]}...")
                else:
                    download_stats["failed"] += 1
            except Exception as e:
                logger.warning(f"PDF download failed for {paper.title[:50]}: {e}")
                download_stats["failed"] += 1

        paper.full_text = full_text
        papers_to_add.append(paper)

    if not papers_to_add:
        logger.info(f"All {len(skipped_duplicates)} papers already exist in KB '{name}'")
        return {
            "added_papers": 0,
            "added_chunks": 0,
            "skipped_duplicates": len(skipped_duplicates),
            "kb": name,
        }

    # Use DynamicKnowledgeBase to add papers to the collection
    dkb = DynamicKnowledgeBase(
        vector_store=app_state.vector_store,
        embedding_service=app_state.embedding_provider,
    )
    # Override with the real KB collection
    dkb.collection_name = kb.collection_name
    dkb._initialized = True

    # Add papers with full text if available
    added = await dkb.add_papers(papers_to_add, include_full_text=True)

    # Update metadata counts only for new papers
    kb.paper_count += len(papers_to_add)
    kb.chunk_count += added
    await app_state.session_store.save_kb_metadata(kb)

    logger.info(
        f"Added {len(papers_to_add)} papers ({added} chunks) to KB '{name}', skipped {len(skipped_duplicates)} duplicates. PDF stats: {download_stats}"
    )
    return {
        "added_papers": len(papers_to_add),
        "added_chunks": added,
        "skipped_duplicates": len(skipped_duplicates),
        "duplicates": skipped_duplicates,
        "pdf_download": download_stats,
        "kb": name,
    }


@router.get("/api/kb/{name}/stats")
async def get_kb_stats(name: str):
    """Aggregate statistics for a KB, computed from ChromaDB metadata + the SQLite KB record."""
    if not app_state.session_store:
        return {"error": "System not initialized"}
    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}
    scan_cap = 20000
    try:
        coll = app_state.vector_store.client.get_collection(name=kb.collection_name)
        total_chunks = coll.count()
        got = coll.get(
            limit=min(total_chunks, scan_cap) if total_chunks else 0, include=["metadatas"]
        )
        metas = got.get("metadatas") or []
    except Exception as e:
        logger.warning(f"kb stats: collection scan failed for {name}: {e}")
        metas, total_chunks = [], 0
    by_year: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_content_type: dict[str, int] = {}
    by_journal: dict[str, int] = {}
    seen: set = set()
    for m in metas:
        m = m or {}
        pid = m.get("paper_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        year_key = str(m.get("year") or "unknown")
        by_year[year_key] = by_year.get(year_key, 0) + 1
        source_key = str(m.get("source") or "unknown")
        by_source[source_key] = by_source.get(source_key, 0) + 1
        ct_key = str(m.get("content_type") or "unknown")
        by_content_type[ct_key] = by_content_type.get(ct_key, 0) + 1
        j = (m.get("journal") or "").strip()
        if j:
            by_journal[j] = by_journal.get(j, 0) + 1
    top_journals = sorted(by_journal.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "name": kb.name,
        "paper_count": len(seen) or kb.paper_count,
        "chunk_count": total_chunks,
        "by_year": dict(sorted(by_year.items())),
        "by_source": by_source,
        "by_content_type": by_content_type,
        "top_journals": [{"journal": j, "count": c} for j, c in top_journals],
        "embedding_model": kb.embedding_model,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
        "scanned_chunks": len(metas),
        "scan_capped": total_chunks > scan_cap if total_chunks else False,
    }


@router.get("/api/kb/{name}/export")
async def kb_export(name: str, format: str = "obsidian-vault"):
    """Export a knowledge base as a downloadable zip archive.

    Currently supports ``format=obsidian-vault`` only, which produces a zip
    with one Markdown note per paper, one per conversation, and an Index.md.
    """
    from typing import Any as _Any

    from perspicacite.integrations.obsidian import build_obsidian_vault

    if format != "obsidian-vault":
        raise HTTPException(status_code=400, detail="unsupported format; use obsidian-vault")
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="System not initialized")

    kb = await app_state.session_store.get_kb_metadata(name)
    if kb is None:
        raise HTTPException(status_code=404, detail=f"Knowledge base '{name}' not found")

    # Gather per-paper metadata from ChromaDB.  ``list_paper_metadata`` returns
    # one merged dict per paper_id (title / DOI / year / authors).  Best-effort:
    # if the collection is not accessible we export a vault with no paper notes.
    papers: list[dict[str, _Any]] = []
    try:
        raw_meta = await app_state.vector_store.list_paper_metadata(kb.collection_name)
        for m in raw_meta:
            authors_raw = m.get("authors") or []
            if isinstance(authors_raw, str):
                import json as _json
                try:
                    authors_raw = _json.loads(authors_raw)
                except Exception:
                    authors_raw = [authors_raw]
            papers.append({
                "doi": m.get("doi") or m.get("paper_id"),
                "title": m.get("title"),
                "year": m.get("year"),
                "journal": m.get("journal"),
                "authors": authors_raw,
                "content_type": m.get("content_type"),
                "content_source": m.get("source") or m.get("content_source"),
                "abstract": m.get("abstract"),
            })
    except Exception as exc:
        logger.warning("kb_export_paper_enum_failed", kb=name, error=str(exc))
        papers = []

    # Conversations linked to this KB
    conv_dicts: list[dict[str, _Any]] = []
    try:
        convs = await app_state.session_store.list_conversations_by_kb(name)
        for c in convs:
            c_dict = c.model_dump()
            # Flatten messages to simple dicts for build_obsidian_vault
            msgs = []
            for msg in c_dict.get("messages") or []:
                msgs.append({
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "sources": msg.get("sources") or [],
                })
            c_dict["messages"] = msgs
            conv_dicts.append(c_dict)
    except Exception as exc:
        logger.warning("kb_export_conv_enum_failed", kb=name, error=str(exc))
        conv_dicts = []

    kb_dict = kb.model_dump()
    kb_dict["name"] = name

    blob = build_obsidian_vault(kb=kb_dict, papers=papers, conversations=conv_dicts)
    return Response(
        content=blob,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}-vault.zip"'},
    )


@router.get("/api/kb/{name}/chunks")
async def get_kb_chunks(
    name: str,
    limit: int = 20,
    offset: int = 0,
    paper_id: str | None = None,
):
    """Inspect raw chunks stored in a knowledge base (paginated)."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    limit = max(1, min(100, limit))
    offset = max(0, offset)

    coll = app_state.vector_store.client.get_collection(name=kb.collection_name)
    total = coll.count()

    where_filter = {"paper_id": paper_id} if paper_id else None
    result = coll.get(
        limit=limit,
        offset=offset,
        where=where_filter,
        include=["documents", "metadatas"],
    )

    chunks = []
    for i, chunk_id in enumerate(result["ids"]):
        meta = result["metadatas"][i] if result["metadatas"] else {}
        doc = result["documents"][i] if result["documents"] else ""
        chunks.append(
            {
                "id": chunk_id,
                "text": doc,
                "paper_id": meta.get("paper_id"),
                "chunk_index": meta.get("chunk_index"),
                "section": meta.get("section"),
                "title": meta.get("title"),
                "authors": meta.get("authors"),
                "year": meta.get("year"),
                "doi": meta.get("doi"),
                "source": meta.get("source"),
            }
        )

    return {
        "kb": name,
        "total_chunks": total,
        "offset": offset,
        "limit": limit,
        "returned": len(chunks),
        "chunks": chunks,
    }


@router.post("/api/kb/{name}/bibtex")
async def add_bibtex_to_kb(name: str, request: Request):
    """Upload a BibTeX file and add papers to a knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    try:
        body = await request.json()
        bibtex_content = body.get("bibtex", "")
    except Exception:
        return {"error": "Invalid request body"}

    if not bibtex_content.strip():
        return {"error": "BibTeX content is empty"}

    # Parse BibTeX entries using bibtexparser (same as CLI)
    import bibtexparser

    from perspicacite.models.papers import PaperSource
    from perspicacite.pipeline.bibtex_kb import entries_to_papers_with_diagnostics
    from perspicacite.pipeline.download import retrieve_paper_content
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

    # Use bibtexparser to parse the BibTeX content
    try:
        db = bibtexparser.loads(bibtex_content)
        entries = db.entries
        papers, dropped_entries = entries_to_papers_with_diagnostics(entries)
    except Exception as e:
        logger.error(f"BibTeX parsing failed: {e}")
        return {"error": f"Failed to parse BibTeX: {e!s}"}

    total_entries = len(entries)
    if not papers:
        return {
            "error": "No valid paper entries found in BibTeX file",
            "total_entries": total_entries,
            "failed": dropped_entries,
        }

    # Process papers with deduplication and PDF download
    papers_to_add = []
    skipped_existing: list[dict[str, Any]] = []
    metadata_only: list[dict[str, Any]] = []  # F-28/F-30
    download_stats = {
        "attempted": 0, "success": 0, "failed": 0,
        "local_pdf": 0, "metadata_only": 0,
    }

    pdf_config = app_state.config.pdf_download if app_state.config else None
    pdf_kw = _get_pdf_fallback_kwargs(pdf_config)

    for paper in papers:
        # Use DOI as ID if available, otherwise generate from title
        paper_id = paper.doi if paper.doi else paper.id

        # Check if paper already exists
        exists = await app_state.vector_store.paper_exists(kb.collection_name, paper_id)
        if exists:
            skipped_existing.append({"id": paper_id, "title": paper.title[:120]})
            continue

        # Ensure source is set to BIBTEX
        paper.source = PaperSource.BIBTEX

        # Try local PDF first (BibTeX ``file`` field mapped to pdf_url)
        local_path = Path(paper.pdf_url) if paper.pdf_url else None
        if local_path and local_path.suffix.lower() == ".pdf" and local_path.exists():
            try:
                parsed = await app_state.pdf_parser.parse(local_path)
                if parsed.text:
                    paper.full_text = parsed.text
                    download_stats["local_pdf"] += 1
                    papers_to_add.append(paper)
                    continue
            except Exception as e:
                logger.warning(f"Local PDF parse failed for {paper.title[:50]}: {e}")

        # Try to download full text if DOI available
        download_failed = False
        if paper.doi and app_state.pdf_parser:
            download_stats["attempted"] += 1
            try:
                result = await retrieve_paper_content(
                    paper.doi, pdf_parser=app_state.pdf_parser, **pdf_kw
                )
                if result.success and result.full_text:
                    paper.full_text = result.full_text
                    paper.content_type = getattr(result, "content_type", None)
                    download_stats["success"] += 1
                    # Enrich paper metadata from discovery
                    meta = result.metadata or {}
                    if meta.get("title") and not paper.title:
                        paper.title = meta["title"]
                    if meta.get("authors") and not paper.authors:
                        from perspicacite.models.papers import Author

                        paper.authors = [Author(name=a) for a in meta["authors"]]
                    if result.abstract and not paper.abstract:
                        paper.abstract = result.abstract
                elif result is not None and result.success:
                    # abstract-only success: tag content_type but no full_text
                    paper.content_type = getattr(result, "content_type", None)
                    if result.abstract and not paper.abstract:
                        paper.abstract = result.abstract
                    download_stats["metadata_only"] += 1
                    metadata_only.append({
                        "key": paper.doi,
                        "title": paper.title[:120],
                        "content_type": paper.content_type,
                        "attempts": list(getattr(result, "attempts", []) or []),
                        "reason": "abstract_only_from_discovery",
                    })
                else:
                    download_stats["failed"] += 1
                    download_failed = True
                    attempts = getattr(result, "attempts", []) if result is not None else []
                    dropped_entries.append({
                        "key": paper.doi,
                        "title": paper.title[:120],
                        "reason": "; ".join(
                            f"{a['source']}:{a['status']}" for a in attempts
                        ) if attempts else "download failed",
                        "attempts": list(attempts),
                    })
            except Exception as e:
                logger.warning(f"Content download failed for {paper.title[:50]}: {e}")
                download_stats["failed"] += 1
                download_failed = True
                dropped_entries.append({
                    "key": paper.doi,
                    "title": paper.title[:120],
                    "reason": str(e),
                })

        # Skip the paper entirely if DOI lookup completely failed AND the
        # BibTeX entry has no abstract/full_text to fall back on.
        if download_failed and not paper.abstract and not paper.full_text:
            continue

        # Track metadata-only adds from the BibTeX side too (no DOI / DOI failed
        # but BibTeX itself supplied title+authors; treat as metadata-only).
        if not paper.full_text and not any(
            entry["key"] == paper.doi for entry in metadata_only
        ):
            metadata_only.append({
                "key": paper.doi or paper.id,
                "title": paper.title[:120],
                "content_type": paper.content_type or "bibtex_metadata",
                "attempts": [],
                "reason": "bibtex_metadata_only" if not download_failed else "doi_lookup_failed_metadata_kept",
            })

        papers_to_add.append(paper)

    added_with_full_text = sum(1 for p in papers_to_add if getattr(p, "full_text", None))
    added_metadata_only_count = len(papers_to_add) - added_with_full_text

    if not papers_to_add:
        return {
            "message": "All papers already exist in KB",
            "added_papers": 0,
            "added_with_full_text": 0,
            "added_metadata_only": 0,
            "total_entries": total_entries,
            "skipped_duplicates": len(skipped_existing),
            "failed": dropped_entries,
            "metadata_only": metadata_only,
            "kb": name,
        }

    # Add papers to KB
    dkb = DynamicKnowledgeBase(
        vector_store=app_state.vector_store,
        embedding_service=app_state.embedding_provider,
    )
    dkb.collection_name = kb.collection_name
    dkb._initialized = True

    added = await dkb.add_papers(papers_to_add, include_full_text=True)

    # Update metadata
    kb.paper_count += len(papers_to_add)
    kb.chunk_count += added
    await app_state.session_store.save_kb_metadata(kb)

    logger.info(
        f"Added {len(papers_to_add)} papers from BibTeX ({added} chunks) to KB '{name}'. PDF stats: {download_stats}"
    )
    return {
        "added_papers": len(papers_to_add),
        "added_with_full_text": added_with_full_text,
        "added_metadata_only": added_metadata_only_count,
        "added_chunks": added,
        "total_entries": total_entries,
        "skipped_duplicates": len(skipped_existing),
        "failed": dropped_entries,
        "metadata_only": metadata_only,
        "pdf_download": download_stats,
        "kb": name,
    }


@router.post("/api/kb/{name}/bibtex/async")
async def add_bibtex_to_kb_async(name: str, request: Request):
    """Start an async BibTeX ingestion job. Returns {job_id, total}."""
    import asyncio as _asyncio

    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="jobs not configured")
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        body = await request.json()
        bibtex_content = body.get("bibtex", "")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request body") from None
    if not bibtex_content.strip():
        raise HTTPException(status_code=400, detail="BibTeX content is empty")

    total = _count_bibtex_entries(bibtex_content)
    job_id = await app_state.job_registry.create(kind="bibtex_ingest", total=total)
    task = _asyncio.create_task(
        _bibtex_ingest_worker(
            name=name,
            bibtex_text=bibtex_content,
            job_id=job_id,
            registry=app_state.job_registry,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id, "total": total}


@router.post("/api/kb/{name}/dois")
async def add_dois_to_kb(name: str, request: KBAddDOIsRequest):
    """Bulk-add papers to a knowledge base from a list of DOIs (synchronous)."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    if len(request.dois) > 200:
        raise HTTPException(status_code=400, detail="At most 200 DOIs per request")

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    from perspicacite.models.papers import Author, Paper, PaperSource
    from perspicacite.pipeline.download import retrieve_paper_content
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

    pdf_config = app_state.config.pdf_download if app_state.config else None
    pdf_kw = _get_pdf_fallback_kwargs(pdf_config)

    papers_to_add: list = []
    skipped: list = []
    failed: list = []
    metadata_only: list = []  # F-28/F-30: surface attempts on abstract-only adds
    dl = {"attempted": 0, "success": 0, "failed": 0, "metadata_only": 0}

    for raw_doi in request.dois:
        doi = (raw_doi or "").strip().replace("https://doi.org/", "")
        if not doi:
            continue

        if await app_state.vector_store.paper_exists(kb.collection_name, doi):
            skipped.append({"doi": doi})
            continue

        dl["attempted"] += 1
        try:
            result = await retrieve_paper_content(doi, pdf_parser=app_state.pdf_parser, **pdf_kw)
        except Exception as e:
            failed.append({"doi": doi, "reason": str(e)})
            dl["failed"] += 1
            continue

        if not result or not result.success:
            attempts = getattr(result, "attempts", []) if result is not None else []
            # Build a compact reason like "publisher_oa_pdf:miss; arxiv_pdf:miss"
            if attempts:
                reason = "; ".join(
                    f"{a['source']}:{a['status']}" + (f"({a.get('error','')})" if a.get("error") else "")
                    for a in attempts
                )
            else:
                reason = "no content"
            failed.append({"doi": doi, "reason": reason, "attempts": list(attempts)})
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
        )
        if result.full_text:
            paper.full_text = result.full_text
            dl["success"] += 1
        else:
            dl["metadata_only"] += 1
            attempts = list(getattr(result, "attempts", []) or [])
            metadata_only.append({
                "doi": doi,
                "content_type": getattr(result, "content_type", None),
                "attempts": attempts,
            })
        papers_to_add.append(paper)

    added_with_full_text = sum(1 for p in papers_to_add if getattr(p, "full_text", None))
    added_metadata_only = len(papers_to_add) - added_with_full_text

    if not papers_to_add:
        return {
            "added_papers": 0,
            "added_with_full_text": 0,
            "added_metadata_only": 0,
            "added_chunks": 0,
            "skipped_duplicates": len(skipped),
            "failed": failed,
            "metadata_only": metadata_only,
            "pdf_download": dl,
            "kb": name,
        }

    dkb = DynamicKnowledgeBase(
        vector_store=app_state.vector_store,
        embedding_service=app_state.embedding_provider,
    )
    dkb.collection_name = kb.collection_name
    dkb._initialized = True

    added = await dkb.add_papers(papers_to_add, include_full_text=True)

    kb.paper_count += len(papers_to_add)
    kb.chunk_count += added
    await app_state.session_store.save_kb_metadata(kb)

    logger.info(f"Added {len(papers_to_add)} papers from DOI list to KB '{name}' ({added} chunks)")
    return {
        "added_papers": len(papers_to_add),
        "added_with_full_text": added_with_full_text,
        "added_metadata_only": added_metadata_only,
        "added_chunks": added,
        "skipped_duplicates": len(skipped),
        "failed": failed,
        "metadata_only": metadata_only,
        "pdf_download": dl,
        "kb": name,
    }


@router.post("/api/kb/{name}/dois/async")
async def add_dois_to_kb_async(name: str, request: KBAddDOIsRequest):
    """Start an async DOI ingestion job. Returns {job_id, total} immediately."""
    import asyncio as _asyncio

    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="jobs not configured")
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="System not initialized")
    if len(request.dois) > 200:
        raise HTTPException(status_code=400, detail="At most 200 DOIs per request")

    total = len(request.dois)
    job_id = await app_state.job_registry.create(kind="doi_ingest", total=total)
    task = _asyncio.create_task(
        _dois_ingest_worker(
            name=name,
            dois=list(request.dois),
            job_id=job_id,
            registry=app_state.job_registry,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id, "total": total}


@router.get("/api/paper")
async def get_paper_detail(doi: str):
    """Discovery metadata + abstract + available content type for a DOI.

    Fast path: discovery + structured XML + abstract only via the unified pipeline.
    pdf_parser is intentionally not passed so the pipeline skips full PDF download
    and publisher API stages (no rate-limit pressure, response in ~1-2 s).
    """
    if not doi or not doi.strip():
        raise HTTPException(status_code=400, detail="doi query param required")
    doi = doi.strip().replace("https://doi.org/", "")
    from perspicacite.pipeline.download import retrieve_paper_content

    pdf_kw = _get_pdf_fallback_kwargs(app_state.config.pdf_download if app_state.config else None)
    try:
        result = await retrieve_paper_content(doi, pdf_parser=None, **pdf_kw)
    except Exception as e:
        return {"doi": doi, "error": str(e), "content_type": "none"}
    md = result.metadata or {}
    return {
        "doi": doi,
        "title": md.get("title"),
        "authors": md.get("authors") or [],
        "year": md.get("year"),
        "journal": md.get("journal"),
        "abstract": result.abstract or md.get("abstract"),
        "content_type": result.content_type,
        "content_source": result.content_source,
        "has_full_text": bool(result.full_text),
        "references_count": len(result.references) if result.references else 0,
    }


# ---------------------------------------------------------------------------
# Local document ingestion: multipart upload and server-side allow-listed paths
# ---------------------------------------------------------------------------


class AddLocalPathsRequest(BaseModel):
    paths: list[str]
    recursive: bool = True


# Strong-reference set for fire-and-forget local-docs ingestion tasks. Kept
# separate from `_background_tasks` so the two ingestion flows do not stomp on
# each other's lifecycle.
_local_tasks: set[asyncio.Task] = set()


@router.post("/api/kb/{name}/local-files")
async def add_local_files(
    name: str,
    files: list[UploadFile] = File(...),
) -> dict:
    """Ingest uploaded files (multipart) into the named KB."""
    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="Job registry not available")
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="perspicacite_upload_"))
    saved: list[Path] = []
    for uf in files:
        target = tmpdir / Path(uf.filename or "upload").name
        with target.open("wb") as out:
            while True:
                chunk = await uf.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        saved.append(target)
    job_id = await app_state.job_registry.create("local_docs_upload", total=len(saved))
    task = asyncio.create_task(
        ingest_local_documents(
            kb_name=name,
            paths=saved,
            app_state=app_state,
            registry=app_state.job_registry,
            job_id=job_id,
            recursive=False,
        )
    )
    _local_tasks.add(task)
    task.add_done_callback(_local_tasks.discard)
    return {"job_id": job_id, "sse_url": f"/api/jobs/{job_id}/events"}


@router.post("/api/kb/{name}/local-paths")
async def add_local_paths(name: str, payload: AddLocalPathsRequest) -> dict:
    """Ingest files identified by server-side paths into the named KB.

    Paths are validated against the configured allow-list. Disabled when no
    allow-list is configured. Symlink escapes are caught by re-validating each
    file after expansion.
    """
    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="Job registry not available")
    allowed = list(getattr(app_state.config.local_docs, "allowed_roots", []) or [])
    validated: list[Path] = []
    for raw in payload.paths:
        try:
            validated.append(validate_local_path(raw, allowed_roots=allowed))
        except LocalDocsDisabledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LocalDocsValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    expanded = expand_paths(validated, recursive=payload.recursive)
    # Re-validate every expanded file (covers symlink escapes from recursion).
    for f in expanded:
        try:
            validate_local_path(str(f), allowed_roots=allowed)
        except LocalDocsValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = await app_state.job_registry.create("local_docs_paths", total=len(expanded))
    task = asyncio.create_task(
        ingest_local_documents(
            kb_name=name,
            paths=expanded,
            app_state=app_state,
            registry=app_state.job_registry,
            job_id=job_id,
            recursive=False,
        )
    )
    _local_tasks.add(task)
    task.add_done_callback(_local_tasks.discard)
    return {"job_id": job_id, "sse_url": f"/api/jobs/{job_id}/events"}


@router.post("/api/kb/{name}/build-capsules")
async def build_capsules_for_kb_async(name: str, force: bool = False) -> dict:
    """Retro-build capsules for every paper in this KB. Returns job_id + sse_url."""
    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="Job registry not available")
    kb_meta = await app_state.session_store.get_kb_metadata(name)
    if kb_meta is None:
        raise HTTPException(status_code=404, detail=f"KB '{name}' not found")
    rows = await app_state.vector_store.list_paper_metadata(kb_meta.collection_name)
    job_id = await app_state.job_registry.create("capsule_build", total=len(rows))

    async def _runner():
        from perspicacite.pipeline.capsule_builder import (
            build_capsule,
            locate_cached_pdf,
            resolve_paper_from_metadata,
        )
        for i, row in enumerate(rows):
            paper = resolve_paper_from_metadata(row)
            pdf_path = locate_cached_pdf(row)
            try:
                res = await build_capsule(
                    paper=paper, pdf_path=pdf_path,
                    kb_name=name, app_state=app_state, force=force,
                )
                await app_state.job_registry.publish(job_id, {
                    "type": "progress", "done": i + 1, "paper": paper.id,
                    "status": res.get("status"),
                })
            except Exception as exc:
                await app_state.job_registry.publish(job_id, {
                    "type": "progress", "done": i + 1, "paper": paper.id,
                    "status": "errored", "error": str(exc),
                })
        await app_state.job_registry.finish(job_id, {"total": len(rows)})

    task = asyncio.create_task(_runner())
    _local_tasks.add(task)
    task.add_done_callback(_local_tasks.discard)
    return {"job_id": job_id, "sse_url": f"/api/jobs/{job_id}/events"}


@router.post("/api/kb/{name}/expand-similar/cutoff")
async def expand_similar_cutoff(name: str, payload: ExpandSimilarCutoffRequest) -> dict:
    """Best-fit keep/drop cutoff from the user's labels on the calibration
    samples. Fast + synchronous (no job)."""
    from perspicacite.search.screening import ScreenResult, cutoff_from_labels

    labeled = [
        (ScreenResult(item={}, score=lbl.score, kept=False), lbl.relevant)
        for lbl in payload.labels
    ]
    return {"cutoff": cutoff_from_labels(labeled)}


@router.post("/api/kb/{name}/expand-similar/score")
async def expand_similar_score(name: str, payload: ExpandSimilarScoreRequest) -> dict:
    """Phase 1: snowball + similarity-score candidates against the KB. Returns
    a job whose SSE ``done`` event carries {candidates, histogram, samples}."""
    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="Job registry not available")
    kb_meta = await app_state.session_store.get_kb_metadata(name)
    if kb_meta is None:
        raise HTTPException(status_code=404, detail=f"KB '{name}' not found")
    job_id = await app_state.job_registry.create("expand_similar_score", total=1)

    async def _runner():
        from perspicacite.pipeline.similarity_expansion import score_expansion_candidates
        try:
            report = await score_expansion_candidates(
                app_state=app_state,
                kb_name=name,
                direction=payload.direction,
                max_per_seed=payload.max_per_seed,
                method=payload.method,
            )
            await app_state.job_registry.finish(job_id, {
                "candidates": report.candidates,
                "histogram": report.histogram,
                "samples": report.samples,
                "seed_count": report.seed_count,
                "method": report.method,
            })
        except Exception as exc:
            await app_state.job_registry.fail(job_id, str(exc))

    task = asyncio.create_task(_runner())
    _local_tasks.add(task)
    task.add_done_callback(_local_tasks.discard)
    return {"job_id": job_id, "sse_url": f"/api/jobs/{job_id}/events"}


@router.post("/api/kb/{name}/expand-similar/commit")
async def expand_similar_commit(name: str, payload: ExpandSimilarCommitRequest) -> dict:
    """Phase 2: ingest the candidates scoring at/above ``cutoff`` into the KB.
    Returns a job whose SSE ``done`` event carries the ingest report."""
    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="Job registry not available")
    kb_meta = await app_state.session_store.get_kb_metadata(name)
    if kb_meta is None:
        raise HTTPException(status_code=404, detail=f"KB '{name}' not found")
    scored = [{"doi": c.doi, "score": c.score} for c in payload.scored]
    job_id = await app_state.job_registry.create("expand_similar_commit", total=len(scored))

    async def _runner():
        from perspicacite.pipeline.similarity_expansion import commit_expansion
        try:
            res = await commit_expansion(
                app_state=app_state, kb_name=name, scored=scored, cutoff=payload.cutoff
            )
            await app_state.job_registry.finish(job_id, res)
        except Exception as exc:
            await app_state.job_registry.fail(job_id, str(exc))

    task = asyncio.create_task(_runner())
    _local_tasks.add(task)
    task.add_done_callback(_local_tasks.discard)
    return {"job_id": job_id, "sse_url": f"/api/jobs/{job_id}/events"}


# ---------------------------------------------------------------------------
# External-resource fetch (Cycle C)
# ---------------------------------------------------------------------------


@router.post("/api/kb/{name}/paper/{paper_id:path}/fetch-resources")
async def fetch_paper_resources_async(name: str, paper_id: str, body: dict | None = None) -> dict:
    """Fetch external resources mined into a paper's capsule.

    Body: ``{"kinds": ["github","zenodo","doi"], "ingest": true, "force": false}``
    Returns: ``{"job_id": "...", "sse_url": "..."}``.
    """
    if app_state.job_registry is None:
        raise HTTPException(status_code=503, detail="Job registry not available")
    kb_meta = await app_state.session_store.get_kb_metadata(name)
    if kb_meta is None:
        raise HTTPException(status_code=404, detail=f"KB '{name}' not found")
    rows = await app_state.vector_store.list_paper_metadata(kb_meta.collection_name)
    # Accept paper_id in either bare-DOI form (matches KB storage) or
    # `doi:<doi>` form (matches the CLI --paper convention).
    norm_id = paper_id[4:] if paper_id.startswith("doi:") else paper_id
    row = next((r for r in rows if r.get("paper_id") == norm_id), None)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"paper '{paper_id}' not in KB '{name}'",
        )

    from perspicacite.pipeline.capsule_builder import (
        capsule_dir_for,
        resolve_paper_from_metadata,
    )
    from perspicacite.pipeline.external.fetch_orchestrator import (
        fetch_paper_resources,
    )

    payload = body or {}
    kinds = payload.get("kinds")
    ingest = bool(payload.get("ingest", True))
    force = bool(payload.get("force", False))

    paper = resolve_paper_from_metadata(row)
    cap_dir = capsule_dir_for(paper, root=app_state.config.capsule.root)
    paper._kb_name = name

    job_id = await app_state.job_registry.create("external_fetch", total=0)

    async def _runner():
        try:
            await fetch_paper_resources(
                paper=paper, capsule_dir=cap_dir, kinds=kinds,
                app_state=app_state, registry=app_state.job_registry,
                job_id=job_id, ingest=ingest, force=force,
            )
        except Exception as exc:
            await app_state.job_registry.fail(job_id, str(exc))

    task = asyncio.create_task(_runner())
    _local_tasks.add(task)
    task.add_done_callback(_local_tasks.discard)
    return {"job_id": job_id, "sse_url": f"/api/jobs/{job_id}/events"}


# ---------------------------------------------------------------------------
# Capsule figure serving (Cycle B)
# ---------------------------------------------------------------------------


@router.get("/api/capsule/{paper_id:path}/figures")
async def list_capsule_figures(paper_id: str, request: Request):
    """Return the parsed figures/index.json for the matching capsule."""
    cfg = request.app.state.app_state.config
    safe = paper_id.replace(":", "_").replace("/", "__")
    idx = cfg.capsule.root / safe / "figures" / "index.json"
    if not idx.is_file():
        raise HTTPException(status_code=404, detail="capsule not found")
    try:
        return JSONResponse(json.loads(idx.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/capsule/{paper_id:path}/figure/{fig_id}")
async def get_capsule_figure(paper_id: str, fig_id: str, request: Request):
    """Serve PNG bytes for a single figure inside a capsule."""
    cfg = request.app.state.app_state.config
    safe = paper_id.replace(":", "_").replace("/", "__")
    cap = cfg.capsule.root / safe
    idx = cap / "figures" / "index.json"
    if not idx.is_file():
        raise HTTPException(status_code=404, detail="capsule not found")
    try:
        records = json.loads(idx.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    match = next(
        (
            r
            for r in records
            if f"pdf_p{r.get('page', 0)}_i{r.get('index', 0)}" == fig_id
        ),
        None,
    )
    if not match:
        raise HTTPException(status_code=404, detail="figure not found")
    path = cap / "figures" / match["filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="figure file missing")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Claim graph export (indicium layer)
# ---------------------------------------------------------------------------


@router.get("/api/kb/{name}/claim-graph")
async def export_claim_graph(
    name: str,
    format: str = Query("nquads", description="nquads | turtle | jsonld | rocrate"),
) -> Response:
    """Download a KB's indicium claim graph in the requested serialisation format.

    Requires build_claim_graph to have been run for this KB.
    """
    try:
        import indicium  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="indicia extra not installed; run: uv sync --extra indicia",
        ) from None

    _valid = ("nquads", "turtle", "jsonld", "rocrate")
    if format not in _valid:
        raise HTTPException(status_code=400, detail=f"format must be one of {list(_valid)}")

    import io
    import json as _json
    import pathlib

    from perspicacite.indicium_layer.store import ClaimGraphStore

    data_dir = pathlib.Path("data/claim_graphs") / name
    if not data_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No claim graph found for KB '{name}'. Run build_claim_graph first.",
        )

    store = ClaimGraphStore(name, data_dir=data_dir, backend="oxigraph")

    def _rocrate_payload(name: str, jld: object) -> dict:
        return {
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
                    "name": f"Perspicacite claim graph: {name}",
                    "hasPart": [{"@id": "claim-graph.jsonld"}],
                },
            ] + (jld if isinstance(jld, list) else []),
        }

    def _cd(filename: str) -> dict:
        return {"Content-Disposition": f'attachment; filename="{filename}"'}

    try:
        if store._backend == "memory":
            assert store._g is not None
            if format == "nquads":
                raw = store._g.serialize(format="nquads")
                return Response(
                    content=raw,
                    media_type="application/n-quads",
                    headers=_cd(f"{name}-claim-graph.nq"),
                )
            elif format == "turtle":
                raw = store._g.serialize(format="turtle")
                return Response(
                    content=raw,
                    media_type="text/turtle",
                    headers=_cd(f"{name}-claim-graph.ttl"),
                )
            else:
                raw_jld = store._g.serialize(format="json-ld")
                jld = _json.loads(raw_jld) if isinstance(raw_jld, str) else raw_jld
                payload = jld if format == "jsonld" else _rocrate_payload(name, jld)
                ext = "jsonld" if format == "jsonld" else "rocrate.json"
                return Response(
                    content=_json.dumps(payload),
                    media_type="application/ld+json",
                    headers=_cd(f"{name}-claim-graph.{ext}"),
                )
        else:
            # oxigraph backend
            assert store._oxistore is not None
            if format == "nquads":
                buf = io.BytesIO()
                store._oxistore.dump(buf, "application/n-quads")
                return Response(
                    content=buf.getvalue(),
                    media_type="application/n-quads",
                    headers=_cd(f"{name}-claim-graph.nq"),
                )
            else:
                # Round-trip via N-Quads → rdflib for Turtle/JSON-LD/RO-Crate
                import rdflib

                buf = io.BytesIO()
                store._oxistore.dump(buf, "application/n-quads")
                g = rdflib.ConjunctiveGraph()
                g.parse(data=buf.getvalue().decode(), format="nquads")
                if format == "turtle":
                    raw = g.serialize(format="turtle")
                    return Response(
                        content=raw,
                        media_type="text/turtle",
                        headers=_cd(f"{name}-claim-graph.ttl"),
                    )
                else:
                    raw_jld = g.serialize(format="json-ld")
                    jld = _json.loads(raw_jld) if isinstance(raw_jld, str) else raw_jld
                    payload = jld if format == "jsonld" else _rocrate_payload(name, jld)
                    ext = "jsonld" if format == "jsonld" else "rocrate.json"
                    return Response(
                        content=_json.dumps(payload),
                        media_type="application/ld+json",
                        headers=_cd(f"{name}-claim-graph.{ext}"),
                    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}") from e
