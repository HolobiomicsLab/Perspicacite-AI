"""Unit tests for indicium_layer.invalidation."""

import pytest

pytest.importorskip("indicium", reason="indicium (indicia extra) not installed")

from perspicacite.indicium_layer.invalidation import (
    compute_paper_hash,
    papers_needing_rebuild,
    schema_version_changed,
)
from perspicacite.indicium_layer.manifest import Manifest


def test_compute_paper_hash_deterministic():
    a = compute_paper_hash("hello")
    b = compute_paper_hash("hello")
    c = compute_paper_hash("world")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_papers_needing_rebuild_detects_new_and_changed():
    manifest = Manifest(
        kb_name="kb",
        paper_hashes={"p1": compute_paper_hash("v1"), "p2": compute_paper_hash("v1")},
    )
    current = {
        "p1": "v1",  # unchanged
        "p2": "v2",  # changed
        "p3": "v1",  # new
    }
    out = sorted(papers_needing_rebuild(manifest, current))
    assert out == ["p2", "p3"]


def test_schema_version_changed_true_when_default():
    import indicium

    manifest = Manifest(kb_name="kb", indicium_schema_version="0.0.0+unknown")
    if indicium.__version__ != "0.0.0+unknown":
        assert schema_version_changed(manifest) is True


def test_schema_version_changed_false_when_match():
    import indicium

    manifest = Manifest(kb_name="kb", indicium_schema_version=indicium.__version__)
    assert schema_version_changed(manifest) is False
