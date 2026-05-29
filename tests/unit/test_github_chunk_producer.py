"""Tests for ``perspicacite.pipeline.github.chunk_producer``.

The chunk producer is the file-classification layer of the 2026-05-15
GitHub / skill-bundle ingest pipeline. It takes a bundle directory +
manifest, walks the tree (delegating to :mod:`...walk`), and emits one
:class:`~perspicacite.models.papers.Paper` per matched file.

The Paper objects produced here are *fixtures*, not literature; the
downstream KB indexer treats them as ``content_kind="github_*"`` rows
so retrieval can filter chunks back to their bundle / file kind.

See:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md`` Task 4
"""

from __future__ import annotations

import json
from pathlib import Path

from perspicacite.models.papers import Author, PaperSource
from perspicacite.pipeline.github.bundle import BundleManifest, ContentSpec
from perspicacite.pipeline.github.chunk_producer import papers_from_directory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(
    name: str = "test-bundle",
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    authors: list[str] | None = None,
) -> BundleManifest:
    """Build a minimal manifest for tests.

    We construct directly (no YAML round-trip) so tests stay fast and
    aren't coupled to YAML serialisation quirks.
    """
    return BundleManifest(
        name=name,
        content=ContentSpec(
            include=include
            if include is not None
            else ["**/*.md", "**/*.py", "**/*.ipynb"],
            exclude=exclude if exclude is not None else [],
        ),
        authors=authors or [],
    )


def _write_notebook(path: Path, cells: list[dict]) -> None:
    """Materialise a minimal valid .ipynb JSON document at ``path``."""
    nb = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_chunk_producer_emits_markdown_paper(tmp_path: Path) -> None:
    """A README.md → one Paper with content_kind=github_markdown."""
    (tmp_path / "README.md").write_text("# Title\n\nIntro paragraph.\n")
    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha="abcd1234")
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Title"
    assert p.metadata["content_kind"] == "github_markdown"
    assert "Intro paragraph" in (p.full_text or "")
    assert p.source == PaperSource.SKILL_BUNDLE


def test_chunk_producer_markdown_title_falls_back_to_stem(tmp_path: Path) -> None:
    """When no H1 is present, the file stem becomes the title."""
    (tmp_path / "notes.md").write_text("Just some text, no headings.\n")
    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert len(papers) == 1
    assert papers[0].title == "notes"


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------


def test_chunk_producer_handles_notebook(tmp_path: Path) -> None:
    """A .ipynb with a markdown cell + a code cell + an image output
    yields a single Paper whose body contains the cell sources but
    NO base64 image data.
    """
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# QC notebook\n", "\n", "Some prose explaining QC.\n"],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 1,
            "source": "import scanpy as sc\nsc.pl.umap(adata)\n",
            "outputs": [
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": "iVBORw0KGgoAAAANSUhEUgAA" * 50,
                        "text/plain": ["<Figure size>"],
                    },
                    "metadata": {},
                }
            ],
        },
    ]
    nb_path = tmp_path / "qc.ipynb"
    _write_notebook(nb_path, cells)

    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha="cafe1234")
    assert len(papers) == 1
    p = papers[0]
    assert p.metadata["content_kind"] == "github_notebook"
    body = p.full_text or ""
    assert "QC notebook" in body
    assert "import scanpy" in body
    # No raw base64 image data should appear.
    assert "iVBORw0KGgoAAAANSUhEUgAA" not in body
    # Cell separators present.
    assert "# Cell" in body


def test_chunk_producer_notebook_accepts_string_source(tmp_path: Path) -> None:
    """nbformat allows ``cell.source`` to be a string OR list[str]; both work."""
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": "# A single string\n"}
    ]
    nb_path = tmp_path / "x.ipynb"
    _write_notebook(nb_path, cells)
    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert len(papers) == 1
    assert "A single string" in (papers[0].full_text or "")


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def test_chunk_producer_extracts_docstrings(tmp_path: Path) -> None:
    """A .py file → Paper body contains every docstring; function/class
    body code is omitted (avoids embedding noise per spec v1)."""
    src = (
        '"""Module-level docstring describing the module."""\n'
        "\n"
        "PI = 3.14  # NOT a docstring, must NOT appear\n"
        "\n"
        "def foo():\n"
        '    """foo function docstring."""\n'
        "    return 1 + 2  # body line MUST NOT appear\n"
        "\n"
        "def bar():\n"
        '    """bar function docstring."""\n'
        '    raise RuntimeError("hidden")  # body line MUST NOT appear\n'
        "\n"
        "class C:\n"
        '    """class C docstring."""\n'
        "    def method(self):\n"
        '        """C.method docstring."""\n'
        "        return self.x  # body line MUST NOT appear\n"
    )
    (tmp_path / "thing.py").write_text(src)

    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert len(papers) == 1
    p = papers[0]
    assert p.metadata["content_kind"] == "github_python"
    body = p.full_text or ""
    assert "Module-level docstring" in body
    assert "foo function docstring" in body
    assert "bar function docstring" in body
    assert "class C docstring" in body
    assert "C.method docstring" in body
    # No function bodies / module-level assignments.
    assert "1 + 2" not in body
    assert 'raise RuntimeError' not in body
    assert "PI = 3.14" not in body
    assert "self.x" not in body


def test_chunk_producer_python_no_docstrings_still_emits_paper(
    tmp_path: Path,
) -> None:
    """A .py file with zero docstrings still produces a Paper (empty body),
    not an error. Caller decides whether to keep it."""
    (tmp_path / "blank.py").write_text("x = 1\n")
    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert len(papers) == 1
    assert papers[0].metadata["content_kind"] == "github_python"


# ---------------------------------------------------------------------------
# Link mining
# ---------------------------------------------------------------------------


def test_chunk_producer_links_in_readme_attached_to_metadata(
    tmp_path: Path,
) -> None:
    """A README mentioning a DOI → Paper.metadata.mined_dois includes it."""
    (tmp_path / "README.md").write_text(
        "# Skill\n\nSee 10.1234/foo for background.\n"
    )
    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert len(papers) == 1
    dois = papers[0].metadata.get("mined_dois") or []
    assert "10.1234/foo" in dois


def test_chunk_producer_links_for_non_markdown_are_empty(tmp_path: Path) -> None:
    """The link miner runs only on markdown bodies. Python / notebook
    Paper.metadata still has the keys (for shape consistency) but the
    lists are empty.
    """
    (tmp_path / "x.py").write_text('"""See doi 10.9999/bar."""\n')
    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert len(papers) == 1
    # Python module: no DOIs mined even when one appears in the docstring.
    assert papers[0].metadata.get("mined_dois", []) == []


# ---------------------------------------------------------------------------
# Paper identity / metadata invariants
# ---------------------------------------------------------------------------


def test_chunk_producer_paper_id_is_stable(tmp_path: Path) -> None:
    """Re-running the producer on the same dir + same commit_sha must
    yield the same Paper.id for each file. KB-side dedup depends on this.
    """
    (tmp_path / "README.md").write_text("# T\n")
    manifest = _make_manifest(name="my-skill")
    p1 = papers_from_directory(tmp_path, manifest, commit_sha="abc123")
    p2 = papers_from_directory(tmp_path, manifest, commit_sha="abc123")
    assert [p.id for p in p1] == [p.id for p in p2]
    # And the id encodes the rel-path so two files in the same SHA don't collide.
    assert "README.md" in p1[0].id


def test_chunk_producer_paper_id_uses_HEAD_when_sha_none(tmp_path: Path) -> None:
    """When commit_sha is None we still need a deterministic id.
    Spec calls for ``HEAD`` as the placeholder.
    """
    (tmp_path / "README.md").write_text("# T\n")
    manifest = _make_manifest(name="my-skill")
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert "HEAD" in papers[0].id


def test_chunk_producer_source_skill_in_metadata(tmp_path: Path) -> None:
    """Every emitted Paper carries source_skill = manifest.name."""
    (tmp_path / "README.md").write_text("# T\n")
    (tmp_path / "x.py").write_text('"""doc"""\n')
    manifest = _make_manifest(name="scrna-qc")
    papers = papers_from_directory(tmp_path, manifest, commit_sha="aaa")
    assert all(p.metadata["source_skill"] == "scrna-qc" for p in papers)


def test_chunk_producer_authors_from_manifest(tmp_path: Path) -> None:
    """Manifest authors propagate onto every Paper as Author objects."""
    (tmp_path / "README.md").write_text("# T\n")
    manifest = _make_manifest(authors=["Alice", "Bob"])
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert papers[0].authors == [Author(name="Alice"), Author(name="Bob")]


def test_chunk_producer_metadata_carries_rel_path_and_commit_sha(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# G\n\nx.\n")
    manifest = _make_manifest()
    papers = papers_from_directory(tmp_path, manifest, commit_sha="deadbeef")
    assert len(papers) == 1
    assert papers[0].metadata["rel_path"] == "docs/guide.md"
    assert papers[0].metadata["commit_sha"] == "deadbeef"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_chunk_producer_skips_no_match_files(tmp_path: Path) -> None:
    """A file whose extension is not in the include globs is dropped."""
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n")
    (tmp_path / "README.md").write_text("# T\n")
    manifest = _make_manifest(include=["**/*.md"], exclude=[])
    papers = papers_from_directory(tmp_path, manifest, commit_sha=None)
    assert len(papers) == 1
    assert papers[0].metadata["rel_path"] == "README.md"


def test_chunk_producer_content_arg_overrides_manifest(tmp_path: Path) -> None:
    """An explicit ``content=ContentSpec(...)`` arg wins over manifest.content."""
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "README.md").write_text("# T\n")
    manifest = _make_manifest(include=["**/*.md"], exclude=[])
    # Override to include only .csv
    override = ContentSpec(include=["**/*.csv"], exclude=[])
    papers = papers_from_directory(
        tmp_path, manifest, commit_sha=None, content=override
    )
    # CSV produces a generic-text Paper, README is now filtered out.
    assert len(papers) == 1
    assert papers[0].metadata["rel_path"] == "data.csv"
    assert papers[0].metadata["content_kind"] == "github_text"
