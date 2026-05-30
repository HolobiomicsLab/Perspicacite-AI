"""Per-paper capsule builder.

Orchestrates: PDF parse -> section split -> chunk per section -> tag -> embed ->
write Chroma + write capsule directory (metadata.json, figures/, text/,
resources.json). ASB-aligned schema; on-disk layout is byte-compatible with
ASB capsules (see docs/superpowers/specs/2026-05-13-capsule-multimodal-rag-design.md).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.documents import ChunkMetadata
from perspicacite.models.papers import Author, Paper, PaperSource
from perspicacite.pipeline.chunking_dispatch import chunk_document
from perspicacite.pipeline.external.accessions import mine_accessions
from perspicacite.pipeline.external.resources import (
    extract_doi_candidates,
    extract_github_repos,
    extract_zenodo_record_ids,
)
from perspicacite.pipeline.parsers.figures import RawFigure, extract_figures
from perspicacite.pipeline.parsers.section_splitter import split_sections

logger = get_logger("perspicacite.pipeline.capsule_builder")

CAPSULE_VERSION = "0.1"


def capsule_dir_for(paper: Paper, *, root: Path) -> Path:
    """Return the capsule directory for ``paper`` under ``root``.

    Paper IDs (e.g. ``doi:10.1234/abc`` or ``local:abc123``) are filesystem-
    sanitized: ``:`` becomes ``_`` and ``/`` becomes ``__``.
    """
    safe = paper.id.replace(":", "_").replace("/", "__")
    return root / safe


def write_metadata(
    capsule_dir: Path,
    *,
    paper: Paper,
    producer_version: str,
    source: str | None = None,
) -> None:
    """Write ``capsule_dir/metadata.json`` with the v0.1 Capsule schema."""
    capsule_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "capsule_version": CAPSULE_VERSION,
        "producer": "perspicacite",
        "producer_version": producer_version,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "paper_id": paper.id,
        "title": paper.title,
        "authors": [a.model_dump() for a in (paper.authors or [])],
        "year": getattr(paper, "year", None),
        "doi": getattr(paper, "doi", None),
        "source": source or (paper.source.value if paper.source else None),
        "task_id": None,
    }
    (capsule_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_figures(capsule_dir: Path, *, figures: list[RawFigure]) -> int:
    """Persist each ``RawFigure``'s bytes and emit ``figures/index.json``.

    Returns the number of figures written. Filenames follow ASB's
    ``fig_p<page:03d>_i<idx:02d>.<ext>`` convention (already set on each
    ``FigureRecord``).
    """
    fig_dir = capsule_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for raw in figures:
        rec = raw.record
        target = fig_dir / rec.filename
        target.write_bytes(raw.image_bytes)
        records.append(asdict(rec))

    (fig_dir / "index.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(records)


_FIG_MENTION_RE = re.compile(
    r"\b(?:fig(?:ure|\.)?|scheme)\s+([A-Za-z]?\d+[A-Za-z]?)\b",
    re.IGNORECASE,
)


def resolve_figure_refs(text: str, figures: list[RawFigure]) -> list[str]:
    """Return a deduped list of ``figure_id`` strings mentioned in ``text``.

    Only mentions whose ``figure_number`` exists in ``figures`` are kept.
    """
    if not text or not figures:
        return []
    by_number: dict[str, str] = {}
    for raw in figures:
        rec = raw.record
        if rec.figure_number:
            by_number.setdefault(
                rec.figure_number.lower(), f"pdf_p{rec.page}_i{rec.index}",
            )
    out: list[str] = []
    seen: set[str] = set()
    for m in _FIG_MENTION_RE.finditer(text):
        key = m.group(1).lower()
        fid = by_number.get(key)
        if fid and fid not in seen:
            seen.add(fid)
            out.append(fid)
    return out


def write_blocks(
    capsule_dir: Path, *, text: str,
    figures: list[RawFigure] | None = None,
    resources: list[dict] | None = None,
) -> int:
    """Section-split ``text`` and emit one paragraph-block per row into
    ``text/blocks.jsonl``.

    V1 block type is always ``paragraph``. Schema reserves ``heading`` /
    ``caption`` / ``table_latex`` / ``equation_latex`` for V2. ``char_span``
    is the offsets of the block content within ``text``.
    """
    text_dir = capsule_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    out_path = text_dir / "blocks.jsonl"

    if not text:
        out_path.write_text("", encoding="utf-8")
        return 0

    sm = split_sections(text)
    rows: list[dict] = []
    block_idx = 0
    for section, section_text in sm.sections.items():
        if not section_text.strip():
            continue
        for paragraph in _split_paragraphs(section_text):
            start = text.find(paragraph)
            end = start + len(paragraph) if start >= 0 else None
            rows.append({
                "block_id": f"b{block_idx:06d}",
                "page": None,
                "bbox": None,
                "type": "paragraph",
                "content": paragraph,
                "section": section,
                "char_span": [start, end] if start >= 0 else None,
                "figure_refs": resolve_figure_refs(paragraph, figures or []),
                "table_refs": [],
                "resource_refs": resolve_resource_refs(paragraph, resources or []),
            })
            block_idx += 1

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )

    # figure_mentions.jsonl — one row per (block_id, figure_id) pair
    mentions: list[dict] = []
    for r in rows:
        for fid in r.get("figure_refs", []):
            mentions.append({"block_id": r["block_id"], "figure_id": fid})
    (text_dir / "figure_mentions.jsonl").write_text(
        "\n".join(json.dumps(x) for x in mentions) + ("\n" if mentions else ""),
        encoding="utf-8",
    )

    return len(rows)


def _split_paragraphs(text: str) -> list[str]:
    """Split a section's text on blank lines; trim each paragraph."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _evidence_span(text: str, needle: str, radius: int = 60) -> str:
    idx = text.find(needle)
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    return text[start:end].replace("\n", " ").strip()


def write_resources(capsule_dir: Path, *, text: str) -> int:
    """Mine accessions + DOIs + GitHub + Zenodo from ``text``; write ``resources.json``.

    Returns count of records.
    """
    records: list[dict] = []
    for acc in mine_accessions(text or ""):
        records.append({
            "resource_id": f"{acc['kind']}:{acc['accession']}",
            "kind": acc["kind"],
            "identifier": acc["accession"],
            "url": acc["url"],
            "evidence_span": acc["evidence_span"],
            "char_span": None,
            "page": None,
            "block_id": None,
        })
    for repo in extract_github_repos(text or ""):
        records.append({
            "resource_id": f"github:{repo}",
            "kind": "github",
            "identifier": repo,
            "url": f"https://github.com/{repo}",
            "evidence_span": _evidence_span(text or "", f"github.com/{repo}"),
            "char_span": None,
            "page": None,
            "block_id": None,
        })
    for rec_id in extract_zenodo_record_ids(text or ""):
        records.append({
            "resource_id": f"zenodo:{rec_id}",
            "kind": "zenodo",
            "identifier": rec_id,
            "url": f"https://zenodo.org/record/{rec_id}",
            "evidence_span": _evidence_span(text or "", rec_id),
            "char_span": None,
            "page": None,
            "block_id": None,
        })
    for doi in extract_doi_candidates(text or ""):
        records.append({
            "resource_id": f"doi:{doi}",
            "kind": "doi",
            "identifier": doi,
            "url": f"https://doi.org/{doi}",
            "evidence_span": _evidence_span(text or "", doi),
            "char_span": None,
            "page": None,
            "block_id": None,
        })
    capsule_dir.mkdir(parents=True, exist_ok=True)
    (capsule_dir / "resources.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(records)


def resolve_resource_refs(text: str, resources: list[dict]) -> list[str]:
    """Return resource_ids whose ``identifier`` or ``url`` appears in ``text``."""
    if not text or not resources:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for r in resources:
        ident = r.get("identifier") or ""
        url = r.get("url") or ""
        if (ident and ident in text) or (url and url in text):
            rid = r["resource_id"]
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


async def build_capsule(
    *,
    paper: Paper,
    pdf_path: Path | None,
    kb_name: str,
    app_state,
    force: bool = False,
    producer_version: str = "0.0.0",
    ingest_chunks: bool = True,
) -> dict[str, Any]:
    """Build a capsule for ``paper`` and optionally ingest its chunks into ``kb_name``.

    Returns a dict with ``status`` (``built`` / ``skipped``), figure/chunk counts.
    Idempotent: no-op when ``capsule_dir/metadata.json`` exists with
    ``capsule_version >= app_state.config.capsule.min_version``, unless ``force``.

    ``ingest_chunks=False`` writes only the on-disk capsule (metadata, figures,
    blocks, resources) without chunking+embedding. The auto-build-on-ingest
    callers pass ``False`` because the host worker has already added chunks via
    its own pipeline; retro-build / CLI / MCP callers leave it ``True``.
    """
    cap_root = Path(app_state.config.capsule.root)
    cap = capsule_dir_for(paper, root=cap_root)
    meta_path = cap / "metadata.json"

    if not force and meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            if existing.get("capsule_version", "0.0") >= app_state.config.capsule.min_version:
                return {"status": "skipped", "capsule_dir": str(cap)}
        except Exception as exc:
            logger.debug("existing capsule metadata read failed", error=str(exc))  # fall through and rebuild

    # 1. Parse PDF if available
    text = ""
    if pdf_path is not None and pdf_path.exists():
        parsed = await app_state.pdf_parser.parse(pdf_path)
        text = (parsed.text or "") if parsed is not None else ""

    # 2. Figures (PDF only)
    figures: list[RawFigure] = []
    if pdf_path is not None and pdf_path.exists():
        figures = extract_figures(pdf_path)

    # 3. Mine resources first (blocks step uses them)
    cap.mkdir(parents=True, exist_ok=True)
    n_res = write_resources(cap, text=text)
    resources = json.loads((cap / "resources.json").read_text())

    # 4. Write figures + blocks (blocks references both figures and resources)
    n_figs = write_figures(cap, figures=figures)
    n_blocks = write_blocks(cap, text=text, figures=figures, resources=resources)

    # 4b. Supplementary Information manifest — list SI files from PMC JATS
    # XML when the paper is on PMC OA. Stored as supplementary/index.json
    # so downstream tooling can fetch on demand. We don't pull the bytes
    # here (some SI is large + opaque binary); fetch helper is a TODO.
    n_si = await _write_supplementary_manifest(cap, paper=paper)

    # 5. Metadata
    write_metadata(cap, paper=paper, producer_version=producer_version)

    # 6. Chunk per block + embed + write to Chroma (opt-out for auto-build callers)
    n_chunks = 0
    if text and ingest_chunks:
        n_chunks = await _ingest_chunks(
            paper=paper, blocks_path=cap / "text" / "blocks.jsonl",
            kb_name=kb_name, app_state=app_state,
        )

    return {
        "status": "built",
        "capsule_dir": str(cap),
        "figures": n_figs,
        "blocks": n_blocks,
        "resources": n_res,
        "supplementary": n_si,
        "chunks": n_chunks,
    }


async def _write_supplementary_manifest(cap: Path, *, paper: Paper) -> int:
    """Write ``<cap>/supplementary/index.json`` from the unified SI discovery
    (PMC JATS → Springer/Nature ESM → ACS supporting-info, in that order).

    Returns the number of SI entries across all sources. Records metadata
    only — bytes are pulled by ``fetch_supplementary`` on demand to keep
    every capsule build cheap.
    """
    doi = (paper.doi or "").strip()
    if not doi:
        return 0
    try:
        from perspicacite.pipeline.download.supplementary import discover_supplementary
        result = await discover_supplementary(doi)
    except Exception as exc:
        logger.debug("supplementary discovery failed", error=str(exc))
        return 0
    items = result.get("items") or []
    if not items:
        return 0
    si_dir = cap / "supplementary"
    si_dir.mkdir(parents=True, exist_ok=True)
    (si_dir / "index.json").write_text(
        json.dumps(
            {"items": items, "sources_tried": result.get("sources_tried", [])},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return len(items)


async def _ingest_chunks(
    *,
    paper: Paper,
    blocks_path: Path,
    kb_name: str,
    app_state,
) -> int:
    """Chunk each block via existing chunk_document(), tag with provenance,
    embed, and write to Chroma."""
    kb = await app_state.session_store.get_kb_metadata(kb_name)
    if kb is None:
        return 0
    kb_cfg = app_state.config.knowledge_base
    all_chunks = []
    for line in blocks_path.read_text().splitlines():
        if not line.strip():
            continue
        block = json.loads(line)
        chunks = await chunk_document(
            block["content"], paper,
            content_type="text", language=None, config=kb_cfg,
        )
        # Sub-project A: best-effort symbol-index sidecar write.
        try:
            _kb_dir = Path(app_state.config.capsule.root) / kb_name
            from perspicacite.pipeline.symbol_index import write_chunks_symbols
            write_chunks_symbols(kb_dir=_kb_dir, chunks=chunks)
        except Exception as _sym_exc:  # never break ingest on sidecar failure
            logger.warning("symbol_index_write_failed", error=str(_sym_exc)[:200])
        for c in chunks:
            md = c.metadata.model_dump()
            md.update({
                "source_section": block["section"],
                "page": block.get("page"),
                "char_span": tuple(block["char_span"]) if block.get("char_span") else None,
                "figure_refs": list(block.get("figure_refs", [])),
                "table_refs": list(block.get("table_refs", [])),
                "resource_refs": list(block.get("resource_refs", [])),
            })
            c.metadata = ChunkMetadata(**md)
        all_chunks.extend(chunks)

    if all_chunks:
        texts = [c.text for c in all_chunks]
        embeds = await app_state.embedding_provider.embed(texts)
        for c, e in zip(all_chunks, embeds, strict=True):
            c.embedding = e
        await app_state.vector_store.add_documents(kb.collection_name, all_chunks)
        kb.chunk_count += len(all_chunks)
        await app_state.session_store.save_kb_metadata(kb)
    return len(all_chunks)


_DEFAULT_PDF_CACHE = Path("./data/papers")


def resolve_paper_from_metadata(row: dict) -> Paper:
    """Reconstruct a minimal ``Paper`` from a vector-store metadata row."""
    authors_raw = row.get("authors") or ""
    authors: list[Author] = []
    if isinstance(authors_raw, str) and authors_raw.strip():
        for part in authors_raw.split(";"):
            name = part.strip()
            if name:
                authors.append(Author(name=name, family=name.split(",")[0].strip()))
    return Paper(
        id=row["paper_id"],
        title=row.get("title") or row["paper_id"],
        authors=authors,
        year=row.get("year"),
        doi=row.get("doi"),
        source=PaperSource.USER_UPLOAD if row.get("doi") else PaperSource.LOCAL,
    )


def locate_cached_pdf(row: dict, *, root: Path = _DEFAULT_PDF_CACHE) -> Path | None:
    """Best-effort: locate a cached PDF for this paper. Returns None if absent."""
    doi = row.get("doi")
    if doi:
        candidate = root / f"{doi.replace('/', '_')}.pdf"
        if candidate.exists():
            return candidate
    pid = row.get("paper_id") or ""
    if pid.startswith("local:"):
        candidate = root / f"{pid.replace(':', '_')}.pdf"
        if candidate.exists():
            return candidate
    return None
