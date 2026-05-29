"""Unit tests for the ``external_link`` KB-log event kind (Task 9).

Two thin slices:

1. :class:`KBEvent` accepts ``event="external_link"`` (the ``EventKind``
   ``Literal`` must include it). The :class:`KBLogWriter` round-trips the
   event without schema-drift warnings.
2. :meth:`BundleManifest.collect_external_links` returns a deduped
   :class:`LinkBag` mined from README + ``docs/**/*.md``. Paper IDs go
   to ``papers``; non-paper URLs land in ``datasets`` / ``tools``.
"""
from __future__ import annotations

import json
from pathlib import Path

from perspicacite.pipeline.github.bundle import BundleManifest, LinkBag
from perspicacite.pipeline.kb_log import KBEvent, KBLogWriter

# ---------------------------------------------------------------------------
# event-kind extension
# ---------------------------------------------------------------------------


def test_kb_event_accepts_external_link_kind(tmp_path: Path) -> None:
    """``KBEvent(event="external_link", ...)`` constructs cleanly and
    survives a write → read round-trip with the URL preserved in
    ``extra``."""
    path = tmp_path / "kb.jsonl"
    w = KBLogWriter(path=path)
    w.append(
        KBEvent(
            event="external_link",
            kb_name="kb1",
            paper_id="",
            source_command="ingest_skill_bundle",
            extra={
                "url": "https://example.org/data.csv",
                "category": "dataset",
            },
        )
    )

    raw = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    payload = json.loads(raw[0])
    assert payload["event"] == "external_link"
    assert payload["extra"]["url"] == "https://example.org/data.csv"
    assert payload["extra"]["category"] == "dataset"
    assert payload["source_command"] == "ingest_skill_bundle"

    events = w.read_all()
    assert len(events) == 1
    assert events[0].event == "external_link"
    assert events[0].extra["url"] == "https://example.org/data.csv"


# ---------------------------------------------------------------------------
# BundleManifest.collect_external_links
# ---------------------------------------------------------------------------


def test_collect_external_links_returns_dedup_linkbag(tmp_path: Path) -> None:
    """README + docs URLs are merged into one LinkBag with no duplicates."""
    bundle = tmp_path / "bndl"
    bundle.mkdir()
    (bundle / "bundle.yml").write_text("name: demo\n", encoding="utf-8")
    (bundle / "README.md").write_text(
        "See https://figshare.com/foo and https://github.com/x/y for context.\n"
        "Also https://figshare.com/foo again (deduped).\n",
        encoding="utf-8",
    )
    docs = bundle / "docs"
    docs.mkdir()
    (docs / "intro.md").write_text(
        "Visit https://zenodo.org/record/1234 and "
        "https://github.com/x/y (same as README — should dedupe).\n",
        encoding="utf-8",
    )

    manifest = BundleManifest.from_directory(bundle)
    bag = manifest.collect_external_links()

    assert isinstance(bag, LinkBag)
    # GitHub URL bucketed to tools, figshare + zenodo to datasets.
    assert bag.tools == ["https://github.com/x/y"]
    assert set(bag.datasets) == {
        "https://figshare.com/foo",
        "https://zenodo.org/record/1234",
    }
    # No paper IDs in this fixture.
    assert bag.papers == []


def test_collect_external_links_empty_when_no_prose_files(tmp_path: Path) -> None:
    """A bundle with only bundle.yml + no README/docs → empty LinkBag."""
    bundle = tmp_path / "empty-bndl"
    bundle.mkdir()
    (bundle / "bundle.yml").write_text("name: empty\n", encoding="utf-8")

    manifest = BundleManifest.from_directory(bundle)
    bag = manifest.collect_external_links()

    assert bag.datasets == []
    assert bag.tools == []
    assert bag.papers == []


def test_collect_external_links_drops_paper_ids(tmp_path: Path) -> None:
    """DOIs / arxiv / PMC IDs surface in ``papers``, NOT in datasets."""
    bundle = tmp_path / "papers-bndl"
    bundle.mkdir()
    (bundle / "README.md").write_text(
        "Cite 10.1234/foo and arxiv:2204.12345.\n"
        "See https://example.org/dataset for data.\n",
        encoding="utf-8",
    )

    manifest = BundleManifest.from_directory(bundle)
    bag = manifest.collect_external_links()

    # Non-paper URL only.
    assert bag.datasets == ["https://example.org/dataset"]
    assert bag.tools == []
    # Paper refs surface in the papers field of the same LinkBag.
    kinds = {p.kind for p in bag.papers}
    assert "doi" in kinds
    assert "arxiv" in kinds
