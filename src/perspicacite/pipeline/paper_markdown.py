"""Render a retrieved paper as markdown for an external builder.

ASB's web source path accepts only ``text/markdown`` or ``text/html`` — a
PDF content-type raises outright — so a paper it cannot fetch itself has to
be handed over as markdown.

Rendered from the **retrieved document** (structured XML sections, or text
extracted from the PDF), never from knowledge-base chunks. Chunks are
embedded fragments sized for retrieval: reassembling them loses ordering
guarantees and silently drops whatever the chunker skipped, so a capsule
built from them would measure the chunker as much as the paper.
"""

from __future__ import annotations

from typing import Any

# Long reference lists add tokens without adding method detail.
MAX_REFERENCES = 100


def author_names(metadata: dict[str, Any]) -> str:
    """Comma-joined author names, tolerating strings or ``{"name": ...}``."""
    names = []
    for author in metadata.get("authors") or []:
        name = author if isinstance(author, str) else (author or {}).get("name")
        if name:
            names.append(str(name))
    return ", ".join(names)


def _byline(doi: str, metadata: dict[str, Any], source: str) -> list[str]:
    """Attribution block: who wrote it, where, and how we got it."""
    bits = [
        b for b in (
            author_names(metadata),
            metadata.get("journal"),
            str(metadata.get("year") or "") or None,
        ) if b
    ]
    lines = [" — ".join(bits), ""] if bits else []
    return lines + [f"DOI: {doi}", f"Retrieved via: {source}", ""]


def _body(result: Any) -> list[str]:
    """Section-labelled body when the source was structured, else flat text."""
    parts: list[str] = []
    for name, text in (result.sections or {}).items():
        if (text or "").strip():
            heading = str(name).replace("_", " ").strip().title()
            parts += [f"## {heading}", "", text.strip(), ""]
    if parts:
        return parts
    if result.full_text:
        return ["## Full text", "", result.full_text.strip(), ""]
    return []


def _references(result: Any) -> list[str]:
    """A bounded reference list; skipped entirely when there are none."""
    refs = result.references or []
    if not refs:
        return []
    lines = ["## References", ""]
    for ref in refs[:MAX_REFERENCES]:
        if isinstance(ref, str):
            lines.append(f"- {ref}")
            continue
        title = ref.get("title") or ref.get("unstructured") or ref.get("doi")
        if title:
            lines.append(f"- {title}")
    if len(refs) > MAX_REFERENCES:
        lines.append(f"- … {len(refs) - MAX_REFERENCES} further references omitted")
    return lines + [""]


def render_paper_markdown(doi: str, result: Any) -> str:
    """One paper as a markdown document.

    Raises ``ValueError`` when the retrieval produced neither full text nor
    an abstract — serving a title-only stub would build a hollow capsule
    that passes every gate and measures nothing.
    """
    if not (result.full_text or result.abstract):
        raise ValueError(f"no content retrieved for {doi}")
    metadata = result.metadata or {}
    parts = [f"# {metadata.get('title') or doi}", ""]
    parts += _byline(doi, metadata, result.content_source or "unknown")
    if result.abstract:
        parts += ["## Abstract", "", result.abstract.strip(), ""]
    parts += _body(result)
    parts += _references(result)
    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class _Fake:
        full_text: str | None = "Body text."
        abstract: str | None = "An abstract."
        sections: dict | None = None
        references: list | None = field(default_factory=list)
        content_source: str = "pmc"
        metadata: dict | None = field(default_factory=lambda: {
            "title": "A Paper", "authors": ["Ada L."], "journal": "J", "year": 2020})

    print(render_paper_markdown("10.1/x", _Fake()))
