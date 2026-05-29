"""File-to-:class:`Paper` converter for GitHub / skill-bundle ingest.

This is the *content-extraction* layer of the 2026-05-15 ingest pipeline:
take a walked-and-filtered file, classify it by extension, extract the
useful prose/code/docstring content, and produce a
:class:`~perspicacite.models.papers.Paper` fixture the rest of the KB
indexing pipeline can consume.

We **don't** chunk here — the existing chunker (downstream) handles that
once it sees the Paper's ``full_text``. We **don't** embed here either.
This module is pure transformation: filesystem → Paper.

Design references:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md`` Task 4

Content-kind taxonomy (set on ``Paper.metadata.content_kind``):

=================== =========================================================
Kind                 Source
=================== =========================================================
``github_markdown`` ``.md`` files — README, docs prose.
``github_python``   ``.py`` files — module + class + function docstrings.
``github_notebook`` ``.ipynb`` files — concatenated cell sources.
``github_text``     Fallback for anything else that survived include globs
                    (``.txt``, ``.toml``, ``.yaml``, ``.json``...). Lazy
                    read; no further parsing.
=================== =========================================================

Notebook handling: we parse ``.ipynb`` as plain JSON with stdlib
:mod:`json`. ``nbformat`` is intentionally **not** added as a dependency
— the only thing we need is ``cells[*].source`` and ``cells[*].cell_type``,
which are nbformat-stable since v4.0.

Python handling: we use :mod:`ast` to extract the module docstring plus
all top-level function/class docstrings (and class-method docstrings),
joined with explicit ``# function: X`` separators. Source bodies are
deliberately **excluded** to avoid embedding noise — see spec
"Out of scope for v1: full code-symbol indexing".

Link mining: only ``.md`` files run through
:func:`extract_links_from_text`. Mined DOIs / arXiv IDs / PMC IDs are
attached to ``Paper.metadata`` under ``mined_dois`` / ``mined_arxiv`` /
``mined_pmc``. The top-level orchestrator (Task 5) feeds these into
``ingest_dois_into_kb`` for linked-paper resolution.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from perspicacite.models.papers import Author, Paper, PaperSource
from perspicacite.pipeline.github.bundle import (
    BundleManifest,
    ContentSpec,
    extract_links_from_text,
)
from perspicacite.pipeline.github.walk import walk_filtered

__all__ = ["papers_from_directory"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def papers_from_directory(
    root: Path,
    manifest: BundleManifest,
    commit_sha: str | None,
    *,
    content: ContentSpec | None = None,
) -> list[Paper]:
    """Walk ``root`` and produce one :class:`Paper` per matched file.

    Args
    ----
    root : Path
        The directory to walk. Typically a fetched bundle directory.
    manifest : BundleManifest
        Parsed ``bundle.yml``. Provides ``name`` (→ ``source_skill``),
        ``authors`` (→ ``Paper.authors``), and (when ``content`` is
        ``None``) the include/exclude globs.
    commit_sha : str | None
        Resolved commit SHA of the source repo. Embedded in
        ``Paper.id`` for stable cross-ingest dedup and stored in
        ``Paper.metadata.commit_sha``. ``None`` falls back to ``HEAD``
        as a sentinel.
    content : ContentSpec, optional
        Explicit override for the include/exclude globs. When passed,
        wins over ``manifest.content``. Mostly useful for the
        ``ingest_github_repo`` path where the user supplies CLI flags.

    Returns
    -------
    list[Paper]
        One Paper per surviving file. Order follows the walker's order
        (filesystem traversal); callers wanting a deterministic order
        should sort by ``Paper.metadata["rel_path"]``.

    Notes
    -----
    *  Files that fail to read (encoding errors, deleted between walk
       and read) are logged and skipped — one bad file shouldn't kill
       the whole ingest.
    *  Files whose extension doesn't match any of the four known
       classifiers fall back to ``_paper_from_generic_text`` rather
       than being silently dropped — they survived include globs for
       a reason.
    """
    spec = content if content is not None else manifest.content
    matched = walk_filtered(
        Path(root), include=spec.include, exclude=spec.exclude
    )

    org = _derive_org_from_manifest(manifest)
    repo = _derive_repo_from_manifest(manifest)
    sha_in_id = commit_sha or "HEAD"

    papers: list[Paper] = []
    for path in matched:
        rel_path = path.relative_to(root).as_posix()
        try:
            paper = _dispatch(
                path=path,
                rel_path=rel_path,
                manifest=manifest,
                commit_sha=commit_sha,
                org=org,
                repo=repo,
                sha_in_id=sha_in_id,
            )
        except (OSError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
            # Single-file failure must NOT abort the bundle: log + skip.
            # JSONDecodeError covers malformed .ipynb; SyntaxError covers
            # python files we can't ast.parse; OSError covers I/O.
            logger.warning(
                "github.chunk_producer: skipping %s (%s: %s)",
                rel_path,
                type(exc).__name__,
                exc,
            )
            continue
        if paper is not None:
            papers.append(paper)
    return papers


# ---------------------------------------------------------------------------
# Per-extension dispatch
# ---------------------------------------------------------------------------


def _dispatch(
    *,
    path: Path,
    rel_path: str,
    manifest: BundleManifest,
    commit_sha: str | None,
    org: str,
    repo: str,
    sha_in_id: str,
) -> Paper | None:
    """Pick the right handler based on the file's suffix.

    Returns ``None`` only for explicit drops (none today; reserved for
    future "this file looks malformed" paths so callers don't need to
    filter ``None`` aggressively).
    """
    ctx = _BuildContext(
        path=path,
        rel_path=rel_path,
        manifest=manifest,
        commit_sha=commit_sha,
        org=org,
        repo=repo,
        sha_in_id=sha_in_id,
    )
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _paper_from_markdown(ctx)
    if suffix == ".ipynb":
        return _paper_from_notebook(ctx)
    if suffix == ".py":
        return _paper_from_python(ctx)
    # Anything else that survived include globs (e.g. .yaml, .json) falls
    # back to "read it as text". This keeps the producer total, never
    # silently dropping a file the operator opted in to.
    return _paper_from_generic_text(ctx)


@dataclass(frozen=True)
class _BuildContext:
    """Per-file inputs shared across handlers.

    Bundling these into one object keeps each handler's signature short
    and makes it easy to add new metadata fields without rippling
    through every handler signature.
    """

    path: Path
    rel_path: str
    manifest: BundleManifest
    commit_sha: str | None
    org: str
    repo: str
    sha_in_id: str


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _paper_from_markdown(ctx: _BuildContext) -> Paper:
    """Markdown handler. Mines DOI/arXiv/PMC links from the body."""
    body = ctx.path.read_text(encoding="utf-8", errors="replace")
    title = _markdown_title(body) or ctx.path.stem
    abstract = _abstract_from_text(body)

    bag = extract_links_from_text(body)
    mined_dois = [r.value for r in bag.papers if r.kind == "doi"]
    mined_arxiv = [r.value for r in bag.papers if r.kind == "arxiv"]
    mined_pmc = [r.value for r in bag.papers if r.kind == "pmc"]

    return _build_paper(
        ctx=ctx,
        title=title,
        abstract=abstract,
        full_text=body,
        content_kind="github_markdown",
        extra_metadata={
            "mined_dois": mined_dois,
            "mined_arxiv": mined_arxiv,
            "mined_pmc": mined_pmc,
        },
    )


def _paper_from_notebook(ctx: _BuildContext) -> Paper:
    """Jupyter notebook handler.

    Parses ``.ipynb`` JSON, concatenates ``markdown`` and ``code`` cell
    sources with ``# Cell N`` separators, and drops cell **outputs**
    (image/png base64, large stdout dumps) so embedding stays signal-rich.
    """
    raw = ctx.path.read_text(encoding="utf-8", errors="replace")
    nb = json.loads(raw)
    cells = nb.get("cells") or []

    parts: list[str] = []
    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        kind = cell.get("cell_type")
        if kind not in ("markdown", "code"):
            continue
        source = _coerce_cell_source(cell.get("source"))
        if not source.strip():
            continue
        # Tag each cell so embedders can see boundaries, and so
        # retrieval surfaces "cell N" context.
        parts.append(f"# Cell {idx + 1} ({kind})\n\n{source}".rstrip())

    body = "\n\n".join(parts)
    title = ctx.path.stem
    abstract = _abstract_from_text(body)

    return _build_paper(
        ctx=ctx,
        title=title,
        abstract=abstract,
        full_text=body,
        content_kind="github_notebook",
    )


def _paper_from_python(ctx: _BuildContext) -> Paper:
    """Python source handler.

    Uses :mod:`ast` to extract:

    1.  Module-level docstring.
    2.  Top-level function/class docstrings.
    3.  Class-method docstrings.

    Function/class **bodies** are intentionally excluded — embedding
    raw source produces noisy retrieval results. The docstring-only
    surface is what v1 indexes; full-source indexing is a followup.
    """
    text = ctx.path.read_text(encoding="utf-8", errors="replace")
    # ast.parse raises SyntaxError on bad python; the dispatcher catches
    # it and skips the file rather than aborting the whole bundle.
    tree = ast.parse(text, filename=str(ctx.path))

    docs: list[str] = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docs.append(f"# module: {ctx.path.stem}\n\n{module_doc}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node)
            if d:
                docs.append(f"# function: {node.name}\n\n{d}")
        elif isinstance(node, ast.ClassDef):
            class_doc = ast.get_docstring(node)
            if class_doc:
                docs.append(f"# class: {node.name}\n\n{class_doc}")
            for inner in node.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    d = ast.get_docstring(inner)
                    if d:
                        docs.append(
                            f"# function: {node.name}.{inner.name}\n\n{d}"
                        )

    body = "\n\n".join(docs)
    title = ctx.path.stem
    abstract = _abstract_from_text(body) if body else None

    return _build_paper(
        ctx=ctx,
        title=title,
        abstract=abstract,
        full_text=body,
        content_kind="github_python",
    )


def _paper_from_generic_text(ctx: _BuildContext) -> Paper:
    """Lazy fallback for any other extension that survived include globs.

    No parsing, no link mining — just embed the raw text. The
    operator's include-glob choice is the implicit opt-in; we trust
    them not to point us at a binary blob.
    """
    body = ctx.path.read_text(encoding="utf-8", errors="replace")
    title = ctx.path.stem
    abstract = _abstract_from_text(body) if body else None
    return _build_paper(
        ctx=ctx,
        title=title,
        abstract=abstract,
        full_text=body,
        content_kind="github_text",
    )


# ---------------------------------------------------------------------------
# Shared Paper-construction helper
# ---------------------------------------------------------------------------


def _build_paper(
    *,
    ctx: _BuildContext,
    title: str,
    abstract: str | None,
    full_text: str,
    content_kind: str,
    extra_metadata: dict | None = None,
) -> Paper:
    """Materialise the Paper. Keeps the id/metadata shape consistent
    across handlers so KB-side dedup and retrieval filters work the
    same regardless of file kind.
    """
    paper_id = f"github:{ctx.org}/{ctx.repo}@{ctx.sha_in_id}:{ctx.rel_path}"
    metadata: dict[str, object] = {
        "content_kind": content_kind,
        "source_skill": ctx.manifest.name,
        "commit_sha": ctx.commit_sha,
        "rel_path": ctx.rel_path,
        # Always carry the empty-list shape so downstream filters can
        # rely on the key existing. Markdown handler overrides these.
        "mined_dois": [],
        "mined_arxiv": [],
        "mined_pmc": [],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Paper(
        id=paper_id,
        title=title,
        authors=[Author(name=n) for n in ctx.manifest.authors],
        abstract=abstract,
        full_text=full_text,
        year=None,
        doi=None,
        source=PaperSource.SKILL_BUNDLE,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _markdown_title(text: str) -> str | None:
    """Return the first H1 line as a title, or ``None`` if absent.

    Only ATX-style ``# `` headings are considered; setext-style
    (underline) headings are uncommon in modern README files and not
    worth the regex complexity for v1.
    """
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            # "# Title  " → "Title"
            return stripped[2:].strip() or None
    return None


def _abstract_from_text(text: str, max_chars: int = 500) -> str | None:
    """Pull a short abstract — first non-heading paragraph, capped.

    KB UIs need *something* to show as a preview; the README's first
    paragraph is the natural choice. We trim heading lines so the
    abstract isn't just "# Title".
    """
    if not text:
        return None
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        # Skip pure-heading paragraphs ("# ...", "## ..." etc.).
        non_heading = "\n".join(
            line for line in para.splitlines() if not line.lstrip().startswith("#")
        ).strip()
        if non_heading:
            return non_heading[:max_chars]
    # Everything was headings; fall back to first max_chars of raw text.
    return text[:max_chars].strip() or None


def _coerce_cell_source(source: object) -> str:
    """nbformat spec: ``cell.source`` is either ``str`` or ``list[str]``.

    Implementations differ: JupyterLab writes list-of-lines (each line
    keeps its trailing newline), nbconvert sometimes writes a single
    joined string. We accept both and return a single string with
    line-endings preserved.
    """
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    # Unknown shape: stringify defensively rather than crash.
    return str(source)


def _derive_org_from_manifest(manifest: BundleManifest) -> str:
    """Best-effort org name for the synthetic paper id.

    Bundles don't usually carry org info in the YAML; we fall back to
    ``"bundle"`` so the id template stays well-formed. The orchestrator
    (Task 5) overrides this when ingesting a GitHub repo URL where the
    org is known.
    """
    org = manifest.raw.get("org")
    return org if isinstance(org, str) else "bundle"


def _derive_repo_from_manifest(manifest: BundleManifest) -> str:
    """Likewise for the repo component of the id."""
    repo = manifest.raw.get("repo")
    return repo if isinstance(repo, str) else manifest.name
