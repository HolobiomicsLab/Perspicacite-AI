"""Tests for ``perspicacite.pipeline.github.bundle``.

Covers the v1 ``bundle.yml`` parser plus the regex-based link extractor
used to surface inline DOIs / arXiv / PMC IDs from README + docs.

See:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
  section "bundle.yml minimal manifest (v1)" for the YAML schema.
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md``
  Task 3 for the four baseline tests + additions.
"""

from __future__ import annotations

import pytest

from perspicacite.pipeline.github.bundle import (
    DEFAULT_EXCLUDE_GLOBS,
    DEFAULT_INCLUDE_GLOBS,
    BundleManifest,
    ContentSpec,
    LinkBag,
    PaperRef,
    extract_links_from_text,
)

# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


def test_minimal_valid_yaml(tmp_path):
    p = tmp_path / "bundle.yml"
    p.write_text("name: scrna-qc\n")
    m = BundleManifest.parse(p)
    assert m.name == "scrna-qc"
    assert m.papers == []
    assert m.content.include == DEFAULT_INCLUDE_GLOBS
    assert m.content.exclude == DEFAULT_EXCLUDE_GLOBS
    assert m.readme_only is False


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "bundle.yml"
    p.write_text("name: x\nfuture_field: foo\nanother_unknown: [1, 2]\n")
    m = BundleManifest.parse(p)  # must not raise
    assert m.name == "x"
    # The raw payload still keeps the unknown keys for debugging.
    assert m.raw.get("future_field") == "foo"


def test_falls_back_to_readme_when_yaml_missing(tmp_path):
    (tmp_path / "README.md").write_text("# My skill\n\nIntro.")
    m = BundleManifest.from_directory(tmp_path)
    assert m.name == tmp_path.name
    assert m.readme_only is True
    # Defaults applied even in readme-only mode.
    assert m.content.include == DEFAULT_INCLUDE_GLOBS


def test_from_directory_uses_bundle_yml_when_present(tmp_path):
    (tmp_path / "README.md").write_text("# title\n")
    (tmp_path / "bundle.yml").write_text("name: explicit\n")
    m = BundleManifest.from_directory(tmp_path)
    assert m.name == "explicit"
    assert m.readme_only is False


def test_parse_full_manifest(tmp_path):
    """Honours all v1 fields: papers, content overrides, metadata."""
    p = tmp_path / "bundle.yml"
    p.write_text(
        "name: scrna-qc\n"
        "description: QC recipes\n"
        "version: 0.3.0\n"
        "domain: genomics\n"
        "authors:\n"
        "  - Alice\n"
        "  - Bob\n"
        "papers:\n"
        "  - doi: 10.1234/foo\n"
        "  - arxiv: '2204.12345'\n"
        "content:\n"
        "  include:\n"
        "    - README.md\n"
        "  exclude:\n"
        "    - tests/**\n"
    )
    m = BundleManifest.parse(p)
    assert m.name == "scrna-qc"
    assert m.description == "QC recipes"
    assert m.version == "0.3.0"
    assert m.domain == ["genomics"]
    assert m.authors == ["Alice", "Bob"]
    assert m.content.include == ["README.md"]
    assert m.content.exclude == ["tests/**"]
    assert PaperRef(kind="doi", value="10.1234/foo") in m.papers
    assert PaperRef(kind="arxiv", value="2204.12345") in m.papers


def test_parse_missing_name_raises(tmp_path):
    p = tmp_path / "bundle.yml"
    p.write_text("description: no name here\n")
    with pytest.raises(ValueError, match="name"):
        BundleManifest.parse(p)


def test_link_extraction_from_papers_section(tmp_path):
    p = tmp_path / "bundle.yml"
    p.write_text(
        "name: x\n"
        "papers:\n"
        "  - doi: 10.1234/foo\n"
        "  - arxiv: '2204.12345'\n"
        "  - pmc: 'PMC9123456'\n"
    )
    m = BundleManifest.parse(p)
    refs = m.collect_paper_refs()
    assert ("doi", "10.1234/foo") in refs
    assert ("arxiv", "2204.12345") in refs
    assert ("pmc", "PMC9123456") in refs


# ---------------------------------------------------------------------------
# extract_links_from_text — regex-based extractor
# ---------------------------------------------------------------------------


def test_extract_links_finds_dois_in_prose():
    text = "See 10.1234/foo and also https://doi.org/10.5678/bar.suffix"
    bag = extract_links_from_text(text)
    dois = {ref.value for ref in bag.papers if ref.kind == "doi"}
    assert "10.1234/foo" in dois
    assert "10.5678/bar.suffix" in dois


def test_extract_links_finds_arxiv():
    text = "Background: arXiv:2204.12345 explains the model. Also https://arxiv.org/abs/2305.01234"
    bag = extract_links_from_text(text)
    arxiv_ids = {ref.value for ref in bag.papers if ref.kind == "arxiv"}
    assert "2204.12345" in arxiv_ids
    assert "2305.01234" in arxiv_ids


def test_extract_links_finds_pmc():
    text = "Source PMC9123456 is the methods paper; see also PMC12345678."
    bag = extract_links_from_text(text)
    pmc_ids = {ref.value for ref in bag.papers if ref.kind == "pmc"}
    assert "PMC9123456" in pmc_ids
    assert "PMC12345678" in pmc_ids


def test_extract_links_classifies_github_url_as_tool():
    text = "Implementation at https://github.com/scverse/scanpy."
    bag = extract_links_from_text(text)
    assert "https://github.com/scverse/scanpy" in bag.tools


def test_extract_links_classifies_unknown_url_as_dataset():
    text = "Data at https://figshare.com/articles/dataset/12345"
    bag = extract_links_from_text(text)
    assert any("figshare.com" in u for u in bag.datasets)


def test_extract_links_pmc_url_extracts_id():
    text = "See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9123456/"
    bag = extract_links_from_text(text)
    pmc_ids = {ref.value for ref in bag.papers if ref.kind == "pmc"}
    assert "PMC9123456" in pmc_ids


def test_extract_links_empty_text_returns_empty_bag():
    bag = extract_links_from_text("")
    assert isinstance(bag, LinkBag)
    assert bag.papers == []
    assert bag.datasets == []
    assert bag.tools == []


def test_extract_links_doi_lowercases_prefix_only():
    """DOIs: prefix (10.XXXX) is case-insensitive per the standard so
    we lowercase it for stable dedup; suffix is case-preserved because
    some publisher stores distinguish suffix case."""
    text = "Reference 10.1234/FOO.BAR also 10.5678/MixedCase"
    bag = extract_links_from_text(text)
    dois = {ref.value for ref in bag.papers if ref.kind == "doi"}
    # Suffix preserved
    assert "10.1234/FOO.BAR" in dois
    assert "10.5678/MixedCase" in dois


def test_normalize_doi_only_lowercases_prefix():
    """Direct unit test for the helper — the prefix lowercases, the
    suffix is preserved verbatim."""
    from perspicacite.pipeline.github.bundle import _normalize_doi

    assert _normalize_doi("10.1234/Foo.Bar") == "10.1234/Foo.Bar"
    assert _normalize_doi("10.ABCD/foo") == "10.abcd/foo"
    assert _normalize_doi("10.AAA/BBB") == "10.aaa/BBB"
    # No '/' at all → fall back to lowercasing the whole thing.
    assert _normalize_doi("garbage-no-slash") == "garbage-no-slash"


def test_domain_accepts_list_and_scalar(tmp_path):
    """The YAML can be either a scalar or a list; both yield
    ``manifest.domain: list[str]`` for stable downstream consumers."""
    # Scalar form → single-element list
    p1 = tmp_path / "scalar" / "bundle.yml"
    p1.parent.mkdir()
    p1.write_text("name: a\ndomain: genomics\n")
    m1 = BundleManifest.parse(p1)
    assert m1.domain == ["genomics"]

    # List form → list
    p2 = tmp_path / "list" / "bundle.yml"
    p2.parent.mkdir()
    p2.write_text("name: b\ndomain:\n  - genomics\n  - single-cell\n")
    m2 = BundleManifest.parse(p2)
    assert m2.domain == ["genomics", "single-cell"]

    # Missing → empty list
    p3 = tmp_path / "missing" / "bundle.yml"
    p3.parent.mkdir()
    p3.write_text("name: c\n")
    m3 = BundleManifest.parse(p3)
    assert m3.domain == []


# ---------------------------------------------------------------------------
# Combined collection — YAML + README/docs mining
# ---------------------------------------------------------------------------


def test_collect_paper_refs_dedupes_across_yaml_and_readme(tmp_path):
    (tmp_path / "bundle.yml").write_text(
        "name: x\npapers:\n  - doi: 10.1234/foo\n"
    )
    (tmp_path / "README.md").write_text(
        "Main paper: 10.1234/foo — see also https://doi.org/10.5555/extra"
    )
    m = BundleManifest.from_directory(tmp_path)
    refs = m.collect_paper_refs()
    # DOI mentioned both in yaml and readme is present exactly once.
    doi_only = [(k, v) for (k, v) in refs if k == "doi"]
    assert doi_only.count(("doi", "10.1234/foo")) == 1
    # README-only DOI is also surfaced.
    assert ("doi", "10.5555/extra") in refs


def test_collect_paper_refs_readme_only_mode(tmp_path):
    """Falls-back-from-readme mode still mines paper IDs from README."""
    (tmp_path / "README.md").write_text(
        "Implements the method from 10.1038/foo — also see PMC9000000."
    )
    m = BundleManifest.from_directory(tmp_path)
    assert m.readme_only is True
    refs = m.collect_paper_refs()
    assert ("doi", "10.1038/foo") in refs
    assert ("pmc", "PMC9000000") in refs


def test_collect_paper_refs_mines_additional_docs(tmp_path):
    """Beyond README.md, any *.md under docs/ is mined."""
    (tmp_path / "bundle.yml").write_text("name: x\n")
    (tmp_path / "README.md").write_text("intro")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "background.md").write_text("Cited: 10.9999/from-docs")
    m = BundleManifest.from_directory(tmp_path)
    refs = m.collect_paper_refs()
    assert ("doi", "10.9999/from-docs") in refs


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_include_globs_cover_v1_targets():
    """The v1 plan locks in markdown / python / notebook / yaml coverage."""
    assert "**/*.md" in DEFAULT_INCLUDE_GLOBS
    assert "**/*.py" in DEFAULT_INCLUDE_GLOBS
    assert "**/*.ipynb" in DEFAULT_INCLUDE_GLOBS


def test_default_exclude_globs_drop_git_and_caches():
    assert any(g.startswith(".git") for g in DEFAULT_EXCLUDE_GLOBS)
    assert any("__pycache__" in g for g in DEFAULT_EXCLUDE_GLOBS)


def test_content_spec_defaults():
    cs = ContentSpec()
    assert cs.include == DEFAULT_INCLUDE_GLOBS
    assert cs.exclude == DEFAULT_EXCLUDE_GLOBS
