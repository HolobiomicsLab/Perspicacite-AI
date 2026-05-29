"""File-tree walker honouring gitignore-style include/exclude globs.

Part of the 2026-05-15 GitHub-repo / skill-bundle ingest pipeline. The
walker is intentionally thin: it knows how to enumerate files, apply a
pair of gitwildmatch PathSpec objects, and return root-relative
:class:`pathlib.Path` results. Everything else (file classification,
chunking, Paper construction) lives in :mod:`.chunk_producer`.

Design references:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md`` Task 4

Why pathspec (gitwildmatch) instead of :mod:`fnmatch` or :meth:`Path.glob`?

1.  ``fnmatch`` doesn't grok recursive ``**``. Operators authoring
    ``bundle.yml`` files expect ``docs/**/*.md`` to mean "any markdown
    anywhere under docs/", which is the same expectation that
    ``.gitignore`` enforces.
2.  Combining include + exclude is awkward with raw :meth:`Path.glob`
    (callers end up open-coding set-difference). pathspec's two-PathSpec
    pattern is idiomatic and forward-compatible with operators porting
    ``.gitignore`` rules into their bundles.
3.  ``rglob('*')`` walks the whole tree; we then filter via the matched
    PathSpec. The cost is a single os.walk pass — fine for bundle
    directories which are typically <10k files.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pathspec

__all__ = ["walk_filtered"]


def walk_filtered(
    root: Path,
    *,
    include: list[str],
    exclude: list[str],
) -> list[Path]:
    """Walk ``root`` recursively and return matching files.

    A file is returned iff:

    1.  It matches **at least one** glob in ``include``.
    2.  It matches **no** glob in ``exclude``.

    Patterns use gitwildmatch (gitignore-style) semantics via the
    ``pathspec`` library — so ``**`` is recursive, ``*`` doesn't cross
    path separators, and a leading slash anchors to the repo root.

    Args
    ----
    root : Path
        The directory to walk. If ``root`` doesn't exist (or is a
        regular file rather than a directory) an empty list is returned;
        callers (bundle ingest) treat that the same as "no matched
        files", which is a warning, not an error.
    include : list[str]
        Globs to opt files in. Empty list means **no files match**
        (explicit-opt-in semantics; an empty include is almost always a
        misconfiguration, but we don't raise — caller can warn).
    exclude : list[str]
        Globs to opt files out. Empty list means "exclude nothing".

    Returns
    -------
    list[Path]
        Absolute paths (i.e. ``root / rel_path``) for matching files.
        Pathspec matching is performed on POSIX-style root-relative
        strings, so behaviour is OS-independent. Directories are never
        returned even when an include glob would match them.

    Notes
    -----
    *  The returned list isn't sorted — caller can sort if a stable
       order is required (downstream tests sort when they need it).
    *  Symlinks are followed via :meth:`Path.rglob`; we don't guard
       against cycles. For bundle directories that's acceptable; if
       this lands in a hostile-fs context, wrap with a cycle detector.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    include_spec = pathspec.PathSpec.from_lines("gitwildmatch", include)
    exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude)

    out: list[Path] = []
    for p in _iter_files(root):
        rel = p.relative_to(root).as_posix()
        # IMPORTANT: match on the *relative* path string. Passing the
        # absolute path would make ``tests/**`` fail to match
        # ``/tmp/.../tests/x.py`` (the pattern is anchored relative).
        if not include_spec.match_file(rel):
            continue
        if exclude_spec.match_file(rel):
            continue
        out.append(p)
    return out


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield every file under ``root``. Directories are skipped.

    ``rglob('*')`` would yield both files and directories; we filter to
    files because pathspec ``**/*`` happily matches directories too and
    the chunk_producer expects only readable files.
    """
    for p in root.rglob("*"):
        if p.is_file():
            yield p
