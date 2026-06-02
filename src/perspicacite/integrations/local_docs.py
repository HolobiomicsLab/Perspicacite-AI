"""Local-document ingestion: path validate + chunk dispatch + KB write."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from perspicacite.integrations.capsule_reader import (
    ingest_capsule,
    is_capsule_dir,
)
from perspicacite.logging import get_logger
from perspicacite.models.documents import ChunkMetadata
from perspicacite.models.papers import Paper, PaperSource
from perspicacite.pipeline.chunking_dispatch import (
    _local_paper_id_for,
    chunk_document,
    infer_content_type,
)

logger = get_logger("perspicacite.local_docs")


class LocalDocsValidationError(ValueError):
    """Raised when a path fails validation."""


class LocalDocsDisabledError(RuntimeError):
    """Raised when local_docs.allowed_roots is empty (server-side path entry disabled)."""


def validate_local_path(raw_path: str, *, allowed_roots: list[Path]) -> Path:
    """Reject relative paths / '..' / outside allowed_roots. Return resolved Path."""
    if not allowed_roots:
        raise LocalDocsDisabledError(
            "local_docs.allowed_roots is empty — server-side path ingest is disabled"
        )
    if not os.path.isabs(raw_path):
        raise LocalDocsValidationError(f"path must be absolute: {raw_path}")
    if ".." in Path(raw_path).parts:
        raise LocalDocsValidationError(f"path must not contain '..': {raw_path}")
    p = Path(raw_path).resolve()
    if not p.exists():
        raise LocalDocsValidationError(f"path does not exist: {raw_path}")
    for root in allowed_roots:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise LocalDocsValidationError(
        f"path {raw_path} is not under any local_docs.allowed_roots"
    )


def expand_paths(paths: list[Path], *, recursive: bool) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            if recursive:
                out.extend(f for f in p.rglob("*") if f.is_file())
        else:
            out.append(p)
    return out


def _paper_for_file(path: Path) -> Paper:
    return Paper(
        id=_local_paper_id_for(path),
        title=path.name,
        source=PaperSource.LOCAL,
    )


def _extract_year_from_text(text: str | None) -> int | None:
    """Heuristic year extractor for URL/local-doc ingest (F-13).

    Looks for the first 4-digit year in [1990, current_year+1] within
    the first 4KB of the document. Catches:
      - ``Published: 2024`` / ``Date: 2024-01-15``
      - YouTube transcript headers (carry an upload date in the warning)
      - arXiv abstract pages (``[v1] 2023``)
      - Markdown front-matter with ``date: 2023-XX-XX``
    Returns ``None`` when no plausible year is present.
    """
    if not text:
        return None
    import re as _re
    from datetime import datetime as _dt
    max_year = _dt.now().year + 1
    sample = text[:4096]
    for m in _re.finditer(r"\b(19[9][0-9]|20[0-9]{2})\b", sample):
        y = int(m.group(1))
        if 1990 <= y <= max_year:
            return y
    return None


async def _read_text(path: Path, content_type: str, pdf_parser) -> str | None:
    if content_type == "pdf":
        if pdf_parser is None:
            return None
        parsed = await pdf_parser.parse(path)
        return parsed.text or None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("local_docs_read_failed", path=str(path), error=str(exc))
        return None
    return raw or None


async def _ingest_files(
    *,
    kb_name: str,
    files: list[Path],
    app_state,
    registry,
    job_id: str,
    external_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """When ``external_metadata`` is provided, every chunk is annotated with
    ``parent_paper_id`` / ``is_external=True`` / ``resource_refs`` for the
    Cycle C fetched-resource ingest path.

    .ipynb files are pre-processed via ``strip_notebook_outputs`` before
    chunking so embedded image blobs don't enter the KB.
    """
    try:
        kb = await app_state.session_store.get_kb_metadata(kb_name)
        if kb is None:
            await registry.fail(job_id, f"KB '{kb_name}' not found")
            return {}
        kb_cfg = app_state.config.knowledge_base
        total_chunks = 0
        for idx, fp in enumerate(files):
            content_type, language = infer_content_type(fp)
            paper = _paper_for_file(fp)
            text = await _read_text(fp, content_type, app_state.pdf_parser)
            # F-13: opportunistic year extraction from the document body so
            # KB-stats by_year and recency-weighted retrieval have signal
            # for URL- / file-ingested sources.
            year_guess = _extract_year_from_text(text)
            if year_guess is not None:
                paper.year = year_guess
            if not text:
                await registry.publish(job_id, {
                    "type": "progress", "done": idx + 1, "file": str(fp), "status": "empty",
                })
                continue
            # Cycle C: strip notebook outputs before chunking.
            if fp.suffix.lower() == ".ipynb":
                from perspicacite.pipeline.external.notebooks import (
                    strip_notebook_outputs,
                )
                text = strip_notebook_outputs(text)
            chunks = await chunk_document(
                text, paper,
                content_type=content_type, language=language, config=kb_cfg,
            )
            # R2 advanced: optionally augment with docling-extracted tables.
            if content_type == "pdf" and getattr(kb_cfg, "docling_extract_tables_figures", False):
                parser = app_state.pdf_parser
                pages = parser._page_count(fp)
                if parser._should_run_docling_extras(pages, kb_cfg):
                    pc = parser._run_docling_with_timeout(
                        fp, int(getattr(kb_cfg, "docling_timeout_s", 600))
                    )
                    if pc is not None and pc.tables:
                        from perspicacite.pipeline.chunking_dispatch import (
                            table_records_to_chunks,
                        )
                        chunks.extend(
                            table_records_to_chunks(
                                pc.tables, paper, start_index=len(chunks)
                            )
                        )
            # ChunkMetadata is frozen — recreate with source_file_path set,
            # plus optional external_metadata annotations (Cycle C).
            ext_parent = (external_metadata or {}).get("parent_paper_id")
            ext_resource_id = (external_metadata or {}).get("resource_id")
            for c in chunks:
                base = c.metadata.model_dump()
                base["source_file_path"] = str(fp.resolve())
                if external_metadata is not None:
                    base["is_external"] = True
                    if ext_parent:
                        base["parent_paper_id"] = ext_parent
                    if ext_resource_id:
                        existing = list(base.get("resource_refs") or [])
                        if ext_resource_id not in existing:
                            existing.append(ext_resource_id)
                        base["resource_refs"] = existing
                c.metadata = ChunkMetadata(**base)
            # Sub-project A: best-effort symbol-index sidecar write.
            try:
                _kb_dir = Path(app_state.config.capsule.root) / kb_name
                from perspicacite.pipeline.symbol_index import write_chunks_symbols
                write_chunks_symbols(kb_dir=_kb_dir, chunks=chunks)
            except Exception as _sym_exc:  # never break ingest on sidecar failure
                logger.warning(
                    "symbol_index_write_failed",
                    path=str(fp), error=str(_sym_exc)[:200],
                )
            if chunks:
                texts = [c.text for c in chunks]
                embeds = await app_state.embedding_provider.embed(texts)
                for c, e in zip(chunks, embeds, strict=True):
                    c.embedding = e
                await app_state.vector_store.add_documents(kb.collection_name, chunks)
                total_chunks += len(chunks)
            # Cycle A: on-disk capsule artifacts for PDFs (metadata + figures + blocks + resources).
            if content_type == "pdf" and app_state.config.capsule.auto_build_on_ingest:
                from perspicacite.pipeline.capsule_builder import build_capsule
                try:
                    paper.full_text = text
                    await build_capsule(
                        paper=paper, pdf_path=fp, kb_name=kb_name,
                        app_state=app_state, ingest_chunks=False,
                    )
                except Exception as exc:
                    logger.warning("capsule_build_failed", file=str(fp), error=str(exc))
            await registry.publish(job_id, {
                "type": "progress", "done": idx + 1, "file": str(fp),
                "status": "embedded", "chunks": len(chunks),
            })
        kb.chunk_count += total_chunks
        await app_state.session_store.save_kb_metadata(kb)
        result = {"added_chunks": total_chunks, "files": len(files)}
        await registry.finish(job_id, result)
        return result
    except Exception as exc:
        logger.error("local_docs_ingest_failed", error=str(exc))
        await registry.fail(job_id, str(exc))
        raise


async def ingest_local_documents(
    *,
    kb_name: str,
    paths: list[Path],
    app_state,
    registry,
    job_id: str,
    recursive: bool = True,
    external_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top-level entry used by routers, CLI, and MCP.

    Routes capsule directories (those containing a ``metadata.json`` with
    ``capsule_version``) through ``CapsuleReader.ingest_capsule``; other
    inputs go through the regular per-file ingest path.

    ``external_metadata`` (Cycle C): when provided, every chunk written by the
    per-file path is tagged with ``is_external=True`` plus
    ``parent_paper_id``/``resource_refs`` from the dict. Used by the
    fetch-resources orchestrator to mark fetched-repo / fetched-Zenodo
    content distinctly from primary paper content.
    """
    capsule_dirs = [p for p in paths if p.is_dir() and is_capsule_dir(p)]
    non_capsule_paths = [p for p in paths if p not in capsule_dirs]

    total_added = 0
    total_files = 0

    for cap in capsule_dirs:
        r = await ingest_capsule(
            capsule_dir=cap,
            kb_name=kb_name,
            app_state=app_state,
            registry=registry,
            job_id=job_id,
            finalize=False,
        )
        total_added += int(r.get("added_chunks", 0))
        total_files += int(r.get("files", 0))

    if non_capsule_paths:
        expanded = expand_paths(non_capsule_paths, recursive=recursive)
        r2 = await _ingest_files(
            kb_name=kb_name, files=expanded, app_state=app_state,
            external_metadata=external_metadata,
            registry=registry, job_id=job_id,
        )
        # _ingest_files already calls registry.finish — we returned via that
        # call's own finalize. When BOTH capsules and files are present we
        # accept that registry.finish fires twice (publish is idempotent in
        # tests; in production the second .finish is the authoritative one).
        total_added += int(r2.get("added_chunks", 0))
        total_files += int(r2.get("files", 0))
    elif capsule_dirs:
        # Only capsule dirs — finalize once now.
        await registry.finish(
            job_id, {"added_chunks": total_added, "files": total_files}
        )

    return {"added_chunks": total_added, "files": total_files}
