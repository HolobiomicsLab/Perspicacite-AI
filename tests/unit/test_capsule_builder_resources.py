"""capsule_builder.write_resources mines accessions + URLs + GitHub + Zenodo."""

from __future__ import annotations

import json

from perspicacite.pipeline.capsule_builder import resolve_resource_refs, write_resources


def test_write_resources_emits_records(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    text = (
        "Data at PRIDE (PXD012345), code at https://github.com/foo/bar and "
        "https://zenodo.org/record/9876543 ; DOI 10.1234/abc."
    )
    n = write_resources(cap, text=text)
    payload = json.loads((cap / "resources.json").read_text())
    assert n == len(payload)
    kinds = {p["kind"] for p in payload}
    assert {"pride", "github", "zenodo", "doi"} <= kinds
    # GitHub identifier shape
    gh = next(p for p in payload if p["kind"] == "github")
    assert gh["identifier"] == "foo/bar"
    assert gh["resource_id"] == "github:foo/bar"


def test_resolve_resource_refs():
    res = [
        {"resource_id": "github:foo/bar", "kind": "github", "identifier": "foo/bar",
         "url": "https://github.com/foo/bar", "evidence_span": "", "char_span": None,
         "page": None, "block_id": None},
        {"resource_id": "pride:PXD012345", "kind": "pride", "identifier": "PXD012345",
         "url": "x", "evidence_span": "", "char_span": None, "page": None, "block_id": None},
    ]
    refs = resolve_resource_refs("see https://github.com/foo/bar for code", res)
    assert refs == ["github:foo/bar"]


def test_empty_text(tmp_path):
    cap = tmp_path / "cap"
    cap.mkdir()
    n = write_resources(cap, text="")
    assert n == 0
    assert json.loads((cap / "resources.json").read_text()) == []
