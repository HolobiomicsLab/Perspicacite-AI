"""Disk cache for downloaded PDF bytes.

Until now :func:`retrieve_paper_content` parsed PDFs to text in memory
and threw the bytes away — meaning every re-ingest re-fetched from the
publisher, and downstream features (Zotero attachment upload,
``export-kb --with-pdfs``, future archival flows) had nothing to
attach. This module fixes both with a tiny content-addressed cache
keyed by DOI.

Layout:

    <cache_dir>/<sanitized-doi>.pdf       # bytes
    <cache_dir>/<sanitized-doi>.meta.json # {"doi", "source", "fetched_at",
                                          #  "size_bytes", "sha256"}

The sidecar metadata is not strictly required to serve cached bytes
(filename → DOI is recoverable), but it tells callers *which* source
won the priority race (publisher_oa_pdf vs wiley_pdf vs arxiv_pdf vs
…) without re-running the unified pipeline, and it gives forensic
"when did we fetch this" timestamps for provenance.

The cache is intentionally trivial: no expiry, no eviction, no
re-validation. PDFs of accepted scientific papers don't change. Users
who want to invalidate a single entry can just ``rm`` the two files
and re-ingest.

Sanitization replaces ``/`` and ``:`` with ``_`` so a DOI like
``10.1002/anie.202304040`` becomes ``10.1002_anie.202304040.pdf``,
which is round-trippable and matches what most BibTeX exporters / the
upcoming ``export-kb`` flow expect.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger

from .base import MIN_PLAUSIBLE_PDF_BYTES

logger = get_logger("perspicacite.pipeline.download.pdf_cache")


def _sanitize_doi(doi: str) -> str:
    """Make a DOI safe to use as a filename.

    DOIs contain ``/`` (and rarely ``:``) which collide with path
    separators. We map both to ``_``; the result is reversible only
    by convention (we don't try) but is stable and human-readable.
    """
    clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    return clean.replace("/", "_").replace(":", "_").replace("\\", "_")


def _paths(doi: str, cache_dir: Path) -> tuple[Path, Path]:
    key = _sanitize_doi(doi)
    return cache_dir / f"{key}.pdf", cache_dir / f"{key}.meta.json"


def get_cached_pdf(doi: str, cache_dir: str | Path) -> bytes | None:
    """Return cached bytes for ``doi`` or ``None`` when absent.

    Cheap path: stat → open → read. Doesn't validate that the bytes
    still start with ``%PDF`` — if the cache file is corrupted, the
    PDF parser downstream will reject it the same way it rejects a
    bad live download. The size floor is the downloader's, so a body
    the downloader would reject can never be re-served as a cache hit.
    """
    cache_dir = Path(cache_dir).expanduser()
    if not cache_dir.exists():
        return None
    pdf_path, _ = _paths(doi, cache_dir)
    try:
        if pdf_path.exists() and pdf_path.stat().st_size >= MIN_PLAUSIBLE_PDF_BYTES:
            data = pdf_path.read_bytes()
            logger.info(
                "pdf_cache_hit", doi=doi, size_bytes=len(data),
                path=str(pdf_path),
            )
            return data
    except OSError as exc:
        logger.warning("pdf_cache_read_failed", doi=doi, error=str(exc))
    return None


def store_pdf(
    doi: str,
    content: bytes,
    cache_dir: str | Path,
    *,
    source: str | None = None,
) -> Path | None:
    """Write ``content`` to the cache. Returns the PDF path, or ``None``
    on filesystem failure.

    ``source`` is the label from the unified-pipeline priority race
    (``publisher_oa_pdf`` / ``wiley_pdf`` / ``arxiv_pdf`` / etc.) and
    is preserved in the sidecar so callers can later report "this PDF
    came from Unpaywall in 2026-05".
    """
    if not content or len(content) < MIN_PLAUSIBLE_PDF_BYTES:
        return None
    cache_dir = Path(cache_dir).expanduser()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("pdf_cache_mkdir_failed", error=str(exc))
        return None
    pdf_path, meta_path = _paths(doi, cache_dir)
    try:
        pdf_path.write_bytes(content)
        meta: dict[str, Any] = {
            "doi": doi,
            "source": source,
            "fetched_at": int(time.time()),
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(
            "pdf_cache_store", doi=doi, source=source,
            size_bytes=len(content), path=str(pdf_path),
        )
        return pdf_path
    except OSError as exc:
        logger.warning("pdf_cache_write_failed", doi=doi, error=str(exc))
        # Best-effort cleanup so a partial write doesn't poison future reads
        for p in (pdf_path, meta_path):
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
        return None


def cached_pdf_path(doi: str, cache_dir: str | Path) -> Path | None:
    """Return the on-disk path of the cached PDF for ``doi``, or ``None``.

    Used by downstream tools (Zotero attachment upload, ``export-kb``)
    that need the actual file rather than the bytes.
    """
    pdf_path, _ = _paths(doi, Path(cache_dir).expanduser())
    if pdf_path.exists() and pdf_path.stat().st_size >= MIN_PLAUSIBLE_PDF_BYTES:
        return pdf_path
    return None


def read_cache_meta(doi: str, cache_dir: str | Path) -> dict[str, Any] | None:
    """Read the sidecar metadata, when present."""
    _, meta_path = _paths(doi, Path(cache_dir).expanduser())
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Rendered-markdown cache
# ---------------------------------------------------------------------------
# Re-rendering means re-downloading and re-parsing a PDF, which is the
# expensive half of ingestion. A builder that fetches the same paper on
# every run should pay that once.


def cached_markdown_path(doi: str, cache_dir: str | Path) -> Path:
    """Where the rendered markdown for ``doi`` lives. May not exist yet."""
    key = _sanitize_doi(doi)
    return Path(cache_dir).expanduser() / f"{key}.paper.md"


def get_cached_markdown(doi: str, cache_dir: str | Path) -> str | None:
    """Previously rendered markdown for ``doi``, or ``None``."""
    path = cached_markdown_path(doi, cache_dir)
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    return None


def store_markdown(doi: str, markdown: str, cache_dir: str | Path) -> Path:
    """Persist rendered markdown and return its path."""
    path = cached_markdown_path(doi, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


if __name__ == "__main__":
    import tempfile

    # Offline smoke checks: a temp dir only, never the real cache.
    SMOKE_DOI = "10.1234/smoke-check"
    SMOKE_PDF = b"%PDF-1.7 " + b"z" * MIN_PLAUSIBLE_PDF_BYTES
    SMOKE_TOO_SMALL = b"%PDF-1.7 tiny"

    with tempfile.TemporaryDirectory() as smoke_dir:
        assert get_cached_pdf(SMOKE_DOI, smoke_dir) is None
        assert store_pdf(SMOKE_DOI, SMOKE_TOO_SMALL, smoke_dir) is None
        assert get_cached_pdf(SMOKE_DOI, smoke_dir) is None
        assert store_pdf(SMOKE_DOI, SMOKE_PDF, smoke_dir, source="smoke") is not None
        assert get_cached_pdf(SMOKE_DOI, smoke_dir) == SMOKE_PDF
        assert cached_pdf_path(SMOKE_DOI, smoke_dir) is not None
        assert (read_cache_meta(SMOKE_DOI, smoke_dir) or {})["source"] == "smoke"
        assert _sanitize_doi("https://doi.org/10.1002/anie.1") == "10.1002_anie.1"
        assert get_cached_markdown(SMOKE_DOI, smoke_dir) is None
        store_markdown(SMOKE_DOI, "# smoke", smoke_dir)
        assert get_cached_markdown(SMOKE_DOI, smoke_dir) == "# smoke"
    print("pdf_cache.py smoke checks passed")
