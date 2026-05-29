"""Paper-hash + indicium schema-version delta detection for incremental builds."""

from __future__ import annotations

import hashlib

import indicium

from perspicacite.indicium_layer.manifest import Manifest


def compute_paper_hash(text: str) -> str:
    """Truncated sha256 of paper full text. Stable across builder versions."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def papers_needing_rebuild(
    manifest: Manifest,
    current_paper_texts: dict[str, str],
) -> list[str]:
    """Return paper IDs whose current hash differs from the manifest (new or changed)."""
    return [
        pid
        for pid, text in current_paper_texts.items()
        if manifest.paper_hashes.get(pid) != compute_paper_hash(text)
    ]


def schema_version_changed(manifest: Manifest) -> bool:
    """True when the manifest's schema version differs from the installed indicium."""
    return manifest.indicium_schema_version != indicium.__version__
