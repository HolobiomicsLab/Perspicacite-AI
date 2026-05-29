"""Tests for ``perspicacite.pipeline.github.walk``.

The walker is the file-discovery layer of the 2026-05-15 GitHub /
skill-bundle ingest pipeline. It takes a root directory and a pair of
include/exclude glob lists (gitignore semantics) and returns the
ordered list of files that survive the filter.

See:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md`` Task 4

Why pathspec (gitwildmatch) and not :mod:`fnmatch`?
  Recursive ``**`` globbing isn't supported by ``fnmatch``; we rely on
  ``**/*.md`` working as "any .md anywhere under root". pathspec uses
  git's own pattern semantics so operators authoring ``bundle.yml``
  files behave the same way ``.gitignore`` already does.
"""

from __future__ import annotations

from pathlib import Path

from perspicacite.pipeline.github.walk import walk_filtered

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tree(root: Path, files: list[str]) -> None:
    """Materialise a list of relative paths under ``root`` as empty files."""
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")


def _rel(paths: list[Path], root: Path) -> set[str]:
    """Convert absolute paths into a set of root-relative POSIX strings.

    Tests assert on string sets (order-independent, OS-independent
    separator) so the walker's traversal order doesn't matter.
    """
    return {p.relative_to(root).as_posix() for p in paths}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_walker_respects_include_globs(tmp_path: Path) -> None:
    """Only files matching at least one include glob are returned."""
    _make_tree(
        tmp_path,
        [
            "README.md",
            "data.csv",
            "src/foo.py",
        ],
    )
    out = walk_filtered(tmp_path, include=["**/*.md", "**/*.py"], exclude=[])
    assert _rel(out, tmp_path) == {"README.md", "src/foo.py"}


def test_walker_respects_exclude_globs(tmp_path: Path) -> None:
    """Excluded paths are removed even when an include glob matches them."""
    _make_tree(
        tmp_path,
        [
            "src/foo.py",
            "tests/test_foo.py",
        ],
    )
    out = walk_filtered(
        tmp_path, include=["**/*.py"], exclude=["tests/**"]
    )
    assert _rel(out, tmp_path) == {"src/foo.py"}


def test_walker_excludes_git_dir(tmp_path: Path) -> None:
    """The default-exclude semantics drop ``.git`` metadata.

    The walker contract: pass ``.git/**`` in the ``exclude`` list and
    nothing under ``.git/`` should appear. (Default-exclude globs live
    in the manifest; the walker itself only honours what's passed in.)
    """
    _make_tree(
        tmp_path,
        [
            ".git/HEAD",
            ".git/config",
            "README.md",
        ],
    )
    out = walk_filtered(
        tmp_path,
        include=["**/*.md", "**/HEAD", "**/config"],
        exclude=[".git/**"],
    )
    assert _rel(out, tmp_path) == {"README.md"}


def test_walker_relative_path_semantics(tmp_path: Path) -> None:
    """Glob matching MUST use root-relative paths, not absolute paths.

    If we passed absolute paths into pathspec, an exclude like
    ``tests/**`` would match nothing because the absolute form starts
    with ``/private/var/...``. This is the most common bug when
    swapping pathspec into a walker — pin it with a regression test.
    """
    _make_tree(
        tmp_path,
        [
            "src/foo.py",
            "tests/test_foo.py",
        ],
    )
    out = walk_filtered(
        tmp_path, include=["src/**/*.py"], exclude=[]
    )
    # Only the src/-rooted file matches; the test file does not, because
    # ``src/**/*.py`` requires the relative path to start with ``src/``.
    # If we accidentally matched on absolute paths, ``src`` would appear
    # mid-string anywhere the temp dir contains the substring ``src``,
    # corrupting the result.
    assert _rel(out, tmp_path) == {"src/foo.py"}


def test_walker_returns_empty_on_no_match(tmp_path: Path) -> None:
    """Empty result is fine — caller decides whether that's an error."""
    _make_tree(tmp_path, ["README.md", "src/foo.py"])
    out = walk_filtered(
        tmp_path, include=["**/*.rs"], exclude=[]
    )
    assert out == []


def test_walker_returns_empty_on_missing_root(tmp_path: Path) -> None:
    """A non-existent root yields ``[]`` rather than raising.

    The caller (chunk_producer) treats an empty walk the same as a
    bundle with no matched files, which is a warning, not an error.
    """
    missing = tmp_path / "does-not-exist"
    assert walk_filtered(missing, include=["**/*"], exclude=[]) == []


def test_walker_skips_directories(tmp_path: Path) -> None:
    """Only files are returned, never directories.

    ``Path.rglob('*')`` yields both files and directories; the walker
    must filter to files. A glob like ``**/*`` would otherwise match
    every intermediate directory.
    """
    _make_tree(tmp_path, ["docs/intro.md"])
    out = walk_filtered(tmp_path, include=["**/*"], exclude=[])
    assert all(p.is_file() for p in out)
    assert _rel(out, tmp_path) == {"docs/intro.md"}
