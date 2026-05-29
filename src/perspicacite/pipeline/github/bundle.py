"""``bundle.yml`` manifest parser + inline-link extractor.

Part of the 2026-05-15 GitHub-repo / skill-bundle ingest pipeline. This
module is responsible for two pieces of context-extraction:

1.  **Manifest parsing.** Read the optional ``bundle.yml`` at the root
    of an agentic-science-builder skill-bundle directory, validate the
    minimal v1 schema, and surface paper references (DOIs, arXiv IDs,
    PMC IDs) declared by the bundle author.
2.  **Inline link mining.** Regex-scan README + docs prose for embedded
    paper IDs and URLs. This is how we catch citations the author
    didn't list explicitly in ``papers:``.

Design references:
- Spec: ``docs/superpowers/specs/2026-05-15-github-skill-bundle-ingest-design.md``
- Plan: ``docs/superpowers/plans/2026-05-15-github-skill-bundle-ingest.md`` (Task 3)

Forward-compat:
  The parser silently ignores unknown top-level keys so new v1.x fields
  added by the agentic-science-builder community don't break ingest.
  The raw parsed YAML is retained on :attr:`BundleManifest.raw` for
  debugging.

README-only fallback:
  If ``bundle.yml`` is missing entirely,
  :meth:`BundleManifest.from_directory` constructs a minimal manifest
  with :attr:`BundleManifest.readme_only` = ``True`` and the directory
  name as :attr:`BundleManifest.name`. Inline-link mining still runs on
  README + docs so DOIs in prose are surfaced even without a manifest.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default include globs when ``content.include`` is unset. v1 covers the
#: file types we actually know how to chunk: markdown docs, python
#: modules, notebooks, and yaml config (often a config-as-docs file).
DEFAULT_INCLUDE_GLOBS: list[str] = [
    "**/*.md",
    "**/*.py",
    "**/*.ipynb",
    "**/*.yaml",
    "**/*.yml",
]

#: Default exclude globs. Drops VCS metadata, JS deps, and Python
#: bytecode caches that have no business in a KB.
DEFAULT_EXCLUDE_GLOBS: list[str] = [
    ".git/**",
    "node_modules/**",
    "**/__pycache__/**",
]


# ---------------------------------------------------------------------------
# Regex patterns for inline-link mining
# ---------------------------------------------------------------------------

# DOI grammar (Crossref):
#   "10." <4-9 digits> "/" <suffix chars>
# Suffix chars per RFC: A-Z 0-9 plus a small punctuation set. We match
# case-insensitively because DOIs are case-insensitive.
_DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    re.IGNORECASE,
)

# arXiv: either bare "arxiv:" prefix or full URL. We deliberately do
# NOT match unanchored "\d{4}\.\d{4,5}" because that produces too many
# false positives (e.g. version numbers, software versions).
_ARXIV_PREFIX_RE = re.compile(r"\barxiv:\s*(\d{4}\.\d{4,5})\b", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(
    r"https?://arxiv\.org/abs/(\d{4}\.\d{4,5})", re.IGNORECASE
)

# PMC: identifier form is "PMC" + 6-8 digits.
_PMC_RE = re.compile(r"\bPMC\d{6,8}\b")
_PMC_URL_RE = re.compile(
    r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d{6,8})",
    re.IGNORECASE,
)

# Generic URL grabber. We post-classify the matches below.
_URL_RE = re.compile(r"https?://[^\s)>\"'`]+")

# DOI URL form — used to strip the "https://doi.org/" prefix.
_DOI_URL_PREFIX = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


PaperKind = Literal["doi", "arxiv", "pmc"]


@dataclass(frozen=True)
class PaperRef:
    """A single paper reference. ``kind`` matches the YAML key
    (``doi:`` / ``arxiv:`` / ``pmc:``) and ``value`` is the bare
    identifier (no URL prefix).
    """

    kind: PaperKind
    value: str


@dataclass
class LinkBag:
    """Container for the output of :func:`extract_links_from_text`.

    ``papers`` holds typed paper references (DOIs, arXiv, PMC). The
    free-form URL buckets (``datasets`` / ``tools``) are best-effort
    classification of anything that wasn't a paper ID — GitHub URLs go
    to ``tools``; everything else goes to ``datasets``.
    """

    papers: list[PaperRef] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


@dataclass
class ContentSpec:
    """Include / exclude globs for the file-walker (see Task 4).

    Defaults come from :data:`DEFAULT_INCLUDE_GLOBS` /
    :data:`DEFAULT_EXCLUDE_GLOBS`; the YAML can override either.
    """

    include: list[str] = field(default_factory=lambda: list(DEFAULT_INCLUDE_GLOBS))
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_GLOBS))


@dataclass
class BundleManifest:
    """Parsed ``bundle.yml`` (or README-only fallback).

    Attributes:
        name: Required. Becomes the KB suffix in per-skill mode.
        papers: Paper refs from the YAML ``papers:`` section. Inline
            mentions from README/docs are NOT folded in here — see
            :meth:`collect_paper_refs` for the combined view.
        content: Include/exclude globs (defaults applied when missing).
        readme_only: ``True`` when no ``bundle.yml`` was found and we
            fell back to the directory name. Downstream consumers use
            this to log a warning and skip strict-mode validation.
        description, domain, version, authors: Optional metadata.
        raw: The raw parsed YAML payload (kept for debugging /
            forward-compat). Empty dict in readme-only mode.
        directory: The directory the manifest was loaded from (when
            available). Used by :meth:`collect_paper_refs` to mine
            README + docs for inline citations.
    """

    name: str
    papers: list[PaperRef] = field(default_factory=list)
    content: ContentSpec = field(default_factory=ContentSpec)
    readme_only: bool = False
    description: str | None = None
    # ``domain`` is a list of research domains the bundle applies to.
    # Spec models it as a list; we kept a forward-compat path for
    # scalar-string YAML inputs (auto-wrapped into a single-element list).
    domain: list[str] = field(default_factory=list)
    version: str | None = None
    authors: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    directory: Path | None = None

    # ---- factories ------------------------------------------------------

    @classmethod
    def parse(cls, path: Path) -> BundleManifest:
        """Parse a ``bundle.yml`` file.

        Raises:
            ValueError: If the required ``name`` field is missing or
                blank.
            yaml.YAMLError: If the file isn't valid YAML.
        """
        raw_text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"bundle.yml at {path} must be a mapping at the top level"
            )

        name = data.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"bundle.yml at {path} is missing the required 'name' field"
            )

        papers = _parse_papers(data.get("papers"))
        content = _parse_content(data.get("content"))
        domain = _coerce_domain(data.get("domain"))
        authors = _coerce_str_list(data.get("authors"))

        return cls(
            name=name.strip(),
            papers=papers,
            content=content,
            readme_only=False,
            description=_coerce_str_or_none(data.get("description")),
            domain=domain,
            version=_coerce_str_or_none(data.get("version")),
            authors=authors,
            raw=data,
            directory=Path(path).parent,
        )

    @classmethod
    def from_directory(cls, directory: Path) -> BundleManifest:
        """Load a manifest from a directory.

        If ``<directory>/bundle.yml`` exists, behaves like :meth:`parse`.
        Otherwise constructs a minimal manifest with the directory name
        as ``name`` and ``readme_only=True``.
        """
        directory = Path(directory)
        bundle_yml = directory / "bundle.yml"
        if bundle_yml.is_file():
            return cls.parse(bundle_yml)
        # README-only fallback.
        return cls(
            name=directory.name,
            papers=[],
            content=ContentSpec(),
            readme_only=True,
            raw={},
            directory=directory,
        )

    # ---- queries --------------------------------------------------------

    def collect_paper_refs(self) -> set[tuple[str, str]]:
        """Return the union of ``papers:`` section entries + inline
        mentions mined from README + ``docs/**/*.md``.

        Returns a set of ``(kind, value)`` tuples so callers can use it
        for deduping / set algebra without importing :class:`PaperRef`.
        """
        refs: set[tuple[str, str]] = {(p.kind, p.value) for p in self.papers}
        for text in self._iter_prose_texts():
            bag = extract_links_from_text(text)
            for ref in bag.papers:
                refs.add((ref.kind, ref.value))
        return refs

    def collect_external_links(self) -> LinkBag:
        """Return a deduped :class:`LinkBag` of inline links mined from
        README + ``docs/**/*.md``.

        Companion to :meth:`collect_paper_refs`: that method returns the
        paper IDs the KB ingester routes through DOI-resolve; this one
        returns the *non-paper* URLs (``datasets`` / ``tools``) so the
        orchestrator can record them in the KB log as ``external_link``
        events. Paper refs surface in :attr:`LinkBag.papers` too, for
        callers that need a single mining sweep.

        Dedup is per-bucket and preserves first-seen order across the
        README + docs walk.
        """
        merged = LinkBag()
        seen_papers: set[tuple[str, str]] = set()
        seen_datasets: set[str] = set()
        seen_tools: set[str] = set()
        for text in self._iter_prose_texts():
            bag = extract_links_from_text(text)
            for ref in bag.papers:
                key = (ref.kind, ref.value)
                if key in seen_papers:
                    continue
                seen_papers.add(key)
                merged.papers.append(ref)
            for url in bag.datasets:
                if url in seen_datasets:
                    continue
                seen_datasets.add(url)
                merged.datasets.append(url)
            for url in bag.tools:
                if url in seen_tools:
                    continue
                seen_tools.add(url)
                merged.tools.append(url)
        return merged

    # ---- internals ------------------------------------------------------

    def _iter_prose_texts(self) -> Iterable[str]:
        """Yield README + docs markdown content from the bundle dir.

        Errors reading individual files are swallowed (best-effort
        mining shouldn't break ingest).
        """
        if self.directory is None or not self.directory.is_dir():
            return
        candidates: list[Path] = []
        for name in ("README.md", "README.MD", "Readme.md", "readme.md"):
            p = self.directory / name
            if p.is_file():
                candidates.append(p)
        docs = self.directory / "docs"
        if docs.is_dir():
            candidates.extend(sorted(docs.rglob("*.md")))
        for p in candidates:
            try:
                yield p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue


# ---------------------------------------------------------------------------
# YAML-payload coercion helpers
# ---------------------------------------------------------------------------


def _parse_papers(payload) -> list[PaperRef]:
    """Convert the ``papers:`` YAML payload into :class:`PaperRef` list.

    Accepts the v1 form::

        papers:
          - doi: "10.1234/foo"
          - arxiv: "2204.12345"
          - pmc: "PMC9123456"

    Anything that isn't a dict with one of the three known keys is
    silently dropped (forward-compat).
    """
    if not isinstance(payload, list):
        return []
    out: list[PaperRef] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        for kind in ("doi", "arxiv", "pmc"):
            if kind in item and item[kind] is not None:
                value = str(item[kind]).strip()
                if not value:
                    continue
                # Normalise DOI prefix to lowercase (case-insensitive
                # by the DOI standard); keep suffix case-preserved.
                if kind == "doi":
                    value = _normalize_doi(value)
                out.append(PaperRef(kind=kind, value=value))
                break  # one kind per list item
    return out


def _parse_content(payload) -> ContentSpec:
    """Convert the ``content:`` YAML payload into a :class:`ContentSpec`.

    Missing keys → defaults. Wrong types → defaults (with no error;
    forward-compat).
    """
    if not isinstance(payload, dict):
        return ContentSpec()
    include = payload.get("include")
    exclude = payload.get("exclude")
    return ContentSpec(
        include=(
            list(include)
            if isinstance(include, list) and include
            else list(DEFAULT_INCLUDE_GLOBS)
        ),
        exclude=(
            list(exclude)
            if isinstance(exclude, list) and exclude
            else list(DEFAULT_EXCLUDE_GLOBS)
        ),
    )


def _coerce_str_or_none(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return str(value)


def _coerce_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        return [value]
    return []


def _coerce_domain(value) -> list[str]:
    """``domain`` is documented as a list in the spec but operators
    sometimes write it as a single string. Accept both; ALWAYS return a
    list so downstream filters can match on individual domain entries
    (``"genomics" in manifest.domain``) without splitting strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return [str(value)]


def _normalize_doi(value: str) -> str:
    """Normalise a DOI per the standard: prefix (before the first ``/``)
    is case-insensitive — lower it for stable deduping; suffix is
    case-preserved per the DOI spec, since some publisher-side stores
    distinguish case-sensitive suffixes."""
    if "/" not in value:
        return value.lower()
    prefix, sep, suffix = value.partition("/")
    return f"{prefix.lower()}{sep}{suffix}"


# ---------------------------------------------------------------------------
# extract_links_from_text — the inline-link miner
# ---------------------------------------------------------------------------


def extract_links_from_text(text: str) -> LinkBag:
    """Mine inline DOIs / arXiv / PMC IDs and miscellaneous URLs from
    free-form prose.

    Used by :meth:`BundleManifest.collect_paper_refs` to surface
    citations the author didn't list in ``papers:``.

    Behaviour:
      * DOIs are lowercased (DOIs are case-insensitive identifiers).
      * arXiv matching requires the ``arXiv:`` prefix or a
        ``https://arxiv.org/abs/...`` URL — bare ``YYMM.NNNNN`` patterns
        are too ambiguous (could be a version string).
      * URLs are bucketed:
          - ``github.com/*`` → ``tools``
          - ``doi.org/*`` → DOI extracted to ``papers``
          - ``arxiv.org/abs/*`` → arXiv ID extracted to ``papers``
          - ``ncbi.nlm.nih.gov/pmc/*`` → PMC ID extracted to ``papers``
          - everything else → ``datasets`` (best-effort heuristic)
    """
    bag = LinkBag()
    if not text:
        return bag

    seen_papers: set[tuple[str, str]] = set()

    def _add_paper(kind: PaperKind, value: str) -> None:
        if kind == "doi":
            value = _normalize_doi(value)
        key = (kind, value)
        if key in seen_papers:
            return
        seen_papers.add(key)
        bag.papers.append(PaperRef(kind=kind, value=value))

    # --- DOIs (bare or in URLs) ----------------------------------------
    for m in _DOI_RE.finditer(text):
        doi = m.group(0)
        # Trim trailing punctuation that often glues itself to DOIs in
        # prose (e.g. "10.1234/foo." at end of sentence).
        doi = doi.rstrip(".,);:")
        _add_paper("doi", doi)

    # --- arXiv (prefix form + URL form) --------------------------------
    for m in _ARXIV_PREFIX_RE.finditer(text):
        _add_paper("arxiv", m.group(1))
    for m in _ARXIV_URL_RE.finditer(text):
        _add_paper("arxiv", m.group(1))

    # --- PMC (bare + URL form) -----------------------------------------
    for m in _PMC_URL_RE.finditer(text):
        _add_paper("pmc", m.group(1).upper())
    for m in _PMC_RE.finditer(text):
        _add_paper("pmc", m.group(0).upper())

    # --- URL classification --------------------------------------------
    seen_urls_dataset: set[str] = set()
    seen_urls_tool: set[str] = set()
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,);:'\"")
        lower = url.lower()
        if "github.com/" in lower:
            if url not in seen_urls_tool:
                seen_urls_tool.add(url)
                bag.tools.append(url)
            continue
        # doi.org, arxiv.org/abs, ncbi pmc are already handled by the
        # paper-ID regexes above; skip them here so they don't also
        # appear in `datasets`.
        if _DOI_URL_PREFIX.match(url):
            continue
        if "arxiv.org/abs/" in lower:
            continue
        if "ncbi.nlm.nih.gov/pmc/" in lower:
            continue
        if url not in seen_urls_dataset:
            seen_urls_dataset.add(url)
            bag.datasets.append(url)

    return bag


__all__ = [
    "DEFAULT_EXCLUDE_GLOBS",
    "DEFAULT_INCLUDE_GLOBS",
    "BundleManifest",
    "ContentSpec",
    "LinkBag",
    "PaperKind",
    "PaperRef",
    "extract_links_from_text",
]
