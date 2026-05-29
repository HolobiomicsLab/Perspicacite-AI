"""Export a KB as BibTeX + cached-PDF folder for Zotero / ZotFile import.

Mode C of the long-term preservation plan: a Zotero-free,
filesystem-only bridge. Run::

    perspicacite export-kb --kb my_kb --out ~/exports/my_kb --with-pdfs

and you get::

    ~/exports/my_kb/
        my_kb.bib                # one entry per paper
        papers/<sanitized-doi>.pdf  # cached bytes copied in

Drag ``my_kb.bib`` into Zotero (File → Import) and ZotFile auto-attaches
the PDF by filename match. Or just keep the folder under version control
— it's a portable, citation-manager-agnostic snapshot of the KB.

This module is a pure-output companion to :mod:`pipeline.search_to_kb`:
search_to_kb writes to the KB, export_kb reads from it. The two share
the same DOI sanitization (``_sanitize_doi``) so the PDF filenames
line up between PDF cache and export folder.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.pipeline.download.pdf_cache import (
    _sanitize_doi,
    cached_pdf_path,
)

logger = get_logger("perspicacite.pipeline.export_kb")


@dataclass
class ExportReport:
    kb_name: str
    out_dir: str
    papers: int = 0
    bibtex_entries: int = 0
    pdfs_copied: int = 0
    pdfs_missing: list[str] = field(default_factory=list)
    supplementary_copied: int = 0
    skipped_no_doi: int = 0
    bib_path: str | None = None
    csl_json_entries: int = 0
    ris_entries: int = 0
    csl_json_path: str | None = None
    ris_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bibtex_citation_key(paper: dict[str, Any]) -> str:
    """Make a stable BibTeX key.

    Prefer ``<firstAuthorLast><year>`` when both are available; fall
    back to a sanitized DOI; final fallback is ``paper_<paper_id>``.
    Zotero will rename on import if it conflicts, but a meaningful key
    helps when staying in plain BibTeX.
    """
    authors = paper.get("authors") or ""
    if isinstance(authors, list) and authors:
        first = authors[0]
        first_str = str(first).strip()
        last = first_str.split(",", 1)[0].strip() if "," in first_str else first_str.split(" ")[-1]
    elif isinstance(authors, str) and authors:
        first_str = authors.split(",", 1)[0].strip()
        last = first_str.split(" ")[-1] if " " in first_str else first_str
    else:
        last = ""
    year = paper.get("year") or ""
    if last and year:
        key = f"{last}{year}"
    elif paper.get("doi"):
        key = _sanitize_doi(str(paper["doi"]))
    else:
        key = f"paper_{paper.get('paper_id') or 'unknown'}"
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in key)


def _escape_bibtex(s: str) -> str:
    """Brace-protect BibTeX special chars in field values."""
    if not s:
        return ""
    s = s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("%", "\\%")
    return " ".join(s.split())


def _format_authors_bibtex(authors: Any) -> str:
    """BibTeX wants ``Last, First and Last, First`` joined by ``and``."""
    if not authors:
        return ""
    if isinstance(authors, str):
        parts = [a.strip() for a in authors.split(",") if a.strip()]
    elif isinstance(authors, list):
        parts = [str(a).strip() for a in authors if str(a).strip()]
    else:
        return ""
    formatted: list[str] = []
    for p in parts:
        if "," in p:
            formatted.append(p)
        else:
            toks = p.split()
            if len(toks) >= 2:
                formatted.append(f"{toks[-1]}, {' '.join(toks[:-1])}")
            else:
                formatted.append(p)
    return " and ".join(formatted)


def render_bibtex_entry(paper: dict[str, Any], file_path: Path | None = None) -> str:
    """Render one paper as a BibTeX ``@article`` entry.

    ``file_path``, when present, is emitted as a ``file = {…}`` field in
    the Zotero/JabRef/BetterBibTeX format (``description:absolute_path:PDF``)
    so Zotero's "Import BibTeX" sees and attaches the PDF.
    """
    key = _bibtex_citation_key(paper)
    fields: list[tuple[str, str]] = []
    if paper.get("title"):
        fields.append(("title", _escape_bibtex(str(paper["title"]))))
    authors = _format_authors_bibtex(paper.get("authors"))
    if authors:
        fields.append(("author", _escape_bibtex(authors)))
    if paper.get("journal"):
        fields.append(("journal", _escape_bibtex(str(paper["journal"]))))
    if paper.get("year"):
        fields.append(("year", str(paper["year"])))
    if paper.get("doi"):
        fields.append(("doi", str(paper["doi"])))
        fields.append(("url", f"https://doi.org/{paper['doi']}"))
    if paper.get("abstract"):
        fields.append(("abstract", _escape_bibtex(str(paper["abstract"]))))
    if file_path is not None:
        fields.append((
            "file",
            f":{file_path}:application/pdf",
        ))

    lines = [f"@article{{{key},"]
    for i, (k, v) in enumerate(fields):
        comma = "," if i < len(fields) - 1 else ""
        lines.append(f"  {k} = {{{v}}}{comma}")
    lines.append("}")
    return "\n".join(lines)


def _parse_authors(authors: Any) -> list[dict[str, str]]:
    """Normalise a paper's `authors` field to a list of CSL author dicts.

    Accepts:
    - ``list[str]`` of ``"Family, Given"`` entries.
    - ``str`` joined by ``" and "`` (BibTeX style).
    - ``list[dict]`` with ``family`` / ``given`` already present
      (pass-through after light validation).
    """
    if not authors:
        return []
    if isinstance(authors, str):
        names = [a.strip() for a in authors.split(" and ") if a.strip()]
    elif isinstance(authors, list):
        names = []
        for a in authors:
            if isinstance(a, str):
                names.append(a.strip())
            elif isinstance(a, dict) and ("family" in a or "given" in a):
                # Already CSL-shaped — keep as-is.
                names.append(a)  # type: ignore[arg-type]
    else:
        return []
    out: list[dict[str, str]] = []
    for n in names:
        if isinstance(n, dict):
            out.append(n)  # already shaped
            continue
        if "," in n:
            family, _, given = n.partition(",")
            out.append({"family": family.strip(), "given": given.strip()})
        else:
            # Single name — treat as family-only.
            out.append({"family": n, "given": ""})
    return out


def render_csl_json_entry(paper: dict[str, Any]) -> dict[str, Any]:
    """Render one paper as a CSL JSON item.

    Schema: https://github.com/citation-style-language/schema
    """
    item: dict[str, Any] = {
        "id": _bibtex_citation_key(paper),
        "type": "article-journal",
    }
    if paper.get("title"):
        item["title"] = str(paper["title"])
    authors = _parse_authors(paper.get("authors"))
    if authors:
        item["author"] = authors
    if paper.get("year"):
        try:
            year = int(paper["year"])
        except (TypeError, ValueError):
            year = None
        if year is not None:
            item["issued"] = {"date-parts": [[year]]}
    if paper.get("journal"):
        item["container-title"] = str(paper["journal"])
    if paper.get("doi"):
        item["DOI"] = str(paper["doi"])
        item["URL"] = f"https://doi.org/{paper['doi']}"
    if paper.get("abstract"):
        item["abstract"] = str(paper["abstract"])
    return item


def _ris_clean(value: str) -> str:
    """Collapse newlines so the line-oriented RIS format stays valid."""
    return " ".join(value.split())


def render_ris_entry(paper: dict[str, Any]) -> str:
    """Render one paper as a RIS record.

    Type is always ``JOUR`` (journal article) for now — extending to
    book / chapter / other types would mean carrying a structured
    `type` field in the paper metadata.
    """
    lines: list[str] = ["TY  - JOUR"]
    key = _bibtex_citation_key(paper)
    lines.append(f"ID  - {key}")
    if paper.get("title"):
        lines.append(f"T1  - {_ris_clean(str(paper['title']))}")
    for a in _parse_authors(paper.get("authors")):
        family = a.get("family", "")
        given = a.get("given", "")
        name = f"{family}, {given}".strip(", ").strip()
        if name:
            lines.append(f"AU  - {_ris_clean(name)}")
    if paper.get("year"):
        try:
            year = int(paper["year"])
            lines.append(f"PY  - {year}")
        except (TypeError, ValueError):
            pass
    if paper.get("journal"):
        lines.append(f"JO  - {_ris_clean(str(paper['journal']))}")
    if paper.get("doi"):
        lines.append(f"DO  - {paper['doi']}")
        lines.append(f"UR  - https://doi.org/{paper['doi']}")
    if paper.get("abstract"):
        lines.append(f"AB  - {_ris_clean(str(paper['abstract']))}")
    lines.append("ER  - ")
    return "\n".join(lines)


async def export_kb(
    *,
    app_state: Any,
    kb_name: str,
    out_dir: str | Path,
    with_pdfs: bool = True,
    with_supplementary: bool = False,
    overwrite: bool = False,
    formats: list[str] | None = None,
) -> ExportReport:
    """Render ``kb_name`` to a directory of BibTeX + (optional) PDFs.

    Args:
        app_state: Initialized :class:`AppState`.
        kb_name: KB to export.
        out_dir: Destination directory; created if missing.
        with_pdfs: Copy cached PDFs into ``<out_dir>/papers/``.
        with_supplementary: Copy supplementary files from
            ``data/capsules/<paper_id>/supplementary/files/`` into
            ``<out_dir>/supplementary/<paper_id>/``.
        overwrite: Replace the BibTeX file if it already exists. Files
            in ``papers/`` and ``supplementary/`` are always replaced.

    Returns:
        :class:`ExportReport` summarising what landed where.
    """
    _VALID = {"bibtex", "csl_json", "ris"}
    if formats is not None and not formats:
        raise ValueError("formats must be a non-empty list")
    formats = formats or ["bibtex"]
    for f in formats:
        if f not in _VALID:
            raise ValueError(f"unknown format: {f!r}; expected one of {sorted(_VALID)}")

    from perspicacite.models.kb import chroma_collection_name_for_kb

    kb_meta = await app_state.session_store.get_kb_metadata(kb_name)
    if not kb_meta:
        raise ValueError(f"KB '{kb_name}' not found")
    collection_name = (
        kb_meta.collection_name or chroma_collection_name_for_kb(kb_name)
    )

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if "bibtex" in formats:
        bib_path = out / f"{kb_name}.bib"
        if bib_path.exists() and not overwrite:
            raise FileExistsError(
                f"{bib_path} exists. Pass overwrite=True or remove the file."
            )
    else:
        bib_path = None

    pdf_cfg = app_state.config.pdf_download
    cache_dir = pdf_cfg.cache_dir if (pdf_cfg and pdf_cfg.cache_pdfs) else None

    papers_meta = await app_state.vector_store.list_paper_metadata(collection_name)
    report = ExportReport(
        kb_name=kb_name, out_dir=str(out), papers=len(papers_meta),
        bib_path=str(bib_path) if bib_path else None,
    )

    papers_subdir = out / "papers"
    if with_pdfs:
        papers_subdir.mkdir(exist_ok=True)
    si_subdir = out / "supplementary"
    if with_supplementary:
        si_subdir.mkdir(exist_ok=True)

    bib_entries: list[str] = []
    for paper in papers_meta:
        doi = (paper.get("doi") or "").strip()
        attached_path: Path | None = None
        if with_pdfs and cache_dir and doi:
            cached = cached_pdf_path(doi, cache_dir)
            if cached is not None:
                dest = papers_subdir / cached.name
                try:
                    shutil.copyfile(cached, dest)
                    attached_path = dest.resolve()
                    report.pdfs_copied += 1
                except OSError as exc:
                    logger.warning(
                        "export_kb_pdf_copy_failed",
                        doi=doi, error=str(exc),
                    )
            else:
                report.pdfs_missing.append(doi)
        if with_supplementary and doi:
            src = (
                Path(app_state.config.capsule.root)
                / doi.replace("/", "_") / "supplementary" / "files"
            )
            if src.exists():
                dest_dir = si_subdir / doi.replace("/", "_")
                dest_dir.mkdir(exist_ok=True)
                for f in src.glob("*"):
                    if f.is_file():
                        try:
                            shutil.copyfile(f, dest_dir / f.name)
                            report.supplementary_copied += 1
                        except OSError:
                            pass

        if not doi:
            report.skipped_no_doi += 1
        if "bibtex" in formats:
            bib_entries.append(render_bibtex_entry(paper, file_path=attached_path))

    if "bibtex" in formats and bib_path is not None:
        bib_path.write_text("\n\n".join(bib_entries) + "\n")
        report.bibtex_entries = len(bib_entries)

    if "csl_json" in formats:
        csl_path = out / f"{kb_name}.csl.json"
        if csl_path.exists() and not overwrite:
            raise FileExistsError(f"{csl_path} exists. Pass overwrite=True.")
        csl_items = [render_csl_json_entry(p) for p in papers_meta]
        csl_path.write_text(json.dumps(csl_items, indent=2))
        report.csl_json_entries = len(csl_items)
        report.csl_json_path = str(csl_path)

    if "ris" in formats:
        ris_path = out / f"{kb_name}.ris"
        if ris_path.exists() and not overwrite:
            raise FileExistsError(f"{ris_path} exists. Pass overwrite=True.")
        ris_records = [render_ris_entry(p) for p in papers_meta]
        ris_path.write_text("\n\n".join(ris_records) + "\n")
        report.ris_entries = len(ris_records)
        report.ris_path = str(ris_path)

    (out / "manifest.json").write_text(json.dumps(report.to_dict(), indent=2))
    logger.info(
        "export_kb_done", kb=kb_name, out=str(out),
        papers=report.papers, pdfs=report.pdfs_copied,
        supplementary=report.supplementary_copied,
    )
    return report
