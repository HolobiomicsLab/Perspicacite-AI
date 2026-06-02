"""Content-type-aware chunking dispatcher (markdown heading-aware, code language-aware).

Thin layer over the existing token chunker (``pipeline.chunking.chunk_text``).
Routes by ``content_type``:

- ``"markdown"`` with ``markdown_heading_aware`` flag -> heading-stack splitter
  that keeps fenced code blocks atomic.
- ``"code"`` with ``code_language_aware`` flag -> LangChain language-aware
  ``RecursiveCharacterTextSplitter.from_language``.
- ``"pdf"`` / ``"text"`` / any disabled flag -> fall through to the existing
  token chunker.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.kb import ChunkConfig
from perspicacite.models.papers import Paper

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".cs": "csharp",
}

_LANG_TO_LC: dict[str, Language] = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "go": Language.GO,
    "rust": Language.RUST,
    "java": Language.JAVA,
    "cpp": Language.CPP,
    "ruby": Language.RUBY,
    "swift": Language.SWIFT,
    "kotlin": Language.KOTLIN,
    "csharp": Language.CSHARP,
}


def table_records_to_chunks(tables, paper, start_index: int) -> list[DocumentChunk]:
    """Turn ``DoclingTable`` records into retrievable chunks tagged ``content_type='table'``.

    No-op when ``tables`` is empty (the fitz path), preserving today's behaviour.
    """
    chunks: list[DocumentChunk] = []
    for i, t in enumerate(tables):
        body = (f"{t.caption}\n\n{t.markdown}" if t.caption else t.markdown).strip()
        idx = start_index + i
        meta = ChunkMetadata(
            paper_id=getattr(paper, "paper_id", "unknown"),
            chunk_index=idx,
            content_type="table",
            page=getattr(t, "page", None),
            title=getattr(paper, "title", None),
            doi=getattr(paper, "doi", None),
            year=getattr(paper, "year", None),
        )
        chunks.append(
            DocumentChunk(id=f"{meta.paper_id}:table:{idx}", text=body, metadata=meta)
        )
    return chunks


def infer_content_type(path: Path) -> tuple[str, str | None]:
    """Map file extension to ``(content_type, language)``.

    ``content_type`` is one of ``{"pdf", "markdown", "code", "text"}``.
    ``language`` is non-None only when ``content_type == "code"``.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return ("pdf", None)
    if ext in {".md", ".mdx"}:
        return ("markdown", None)
    if ext in _EXT_TO_LANG:
        return ("code", _EXT_TO_LANG[ext])
    return ("text", None)


def _local_paper_id_for(path: Path) -> str:
    """Stable paper id for a local file: ``local:<sha1(abs path)[:12]>``."""
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return f"local:{h}"


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^```")


def _split_markdown_blocks(text: str) -> list[tuple[list[str], str]]:
    """Walk line-by-line; track heading stack; emit ``(heading_path, body)``.

    Fenced code blocks (```` ``` ````...```` ``` ````) are kept atomic and
    never split across a heading boundary.
    """
    lines = text.split("\n")
    stack: list[str] = []
    out: list[tuple[list[str], str]] = []
    buf: list[str] = []
    in_fence = False
    for ln in lines:
        if _FENCE_RE.match(ln.strip()):
            in_fence = not in_fence
            buf.append(ln)
            continue
        if in_fence:
            buf.append(ln)
            continue
        m = _HEADING_RE.match(ln)
        if m:
            if buf:
                body = "\n".join(buf).strip()
                if body:
                    out.append((list(stack), body))
                buf = []
            depth = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: depth - 1]
            stack.append(title)
            continue
        buf.append(ln)
    if buf:
        body = "\n".join(buf).strip()
        if body:
            out.append((list(stack), body))
    return out


def _chunk_size_overlap(config: Any) -> tuple[int, int]:
    """Pull ``chunk_size`` / ``chunk_overlap`` off either ``KnowledgeBaseConfig``
    or ``ChunkConfig`` (both expose those attributes)."""
    return (
        int(getattr(config, "chunk_size", 1000)),
        int(getattr(config, "chunk_overlap", 200)),
    )


def _chunk_markdown(text: str, paper: Paper, config: Any) -> list[DocumentChunk]:
    blocks = _split_markdown_blocks(text)
    base_id = paper.id
    chunk_size, chunk_overlap = _chunk_size_overlap(config)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[DocumentChunk] = []
    idx = 0
    for heading_path, body in blocks:
        for piece in splitter.split_text(body):
            md = ChunkMetadata(
                paper_id=base_id,
                chunk_index=idx,
                section=heading_path[-1] if heading_path else None,
                source=paper.source,
                title=paper.title,
                content_type="markdown",
                heading_path=heading_path,
            )
            chunks.append(DocumentChunk(id=f"{base_id}_md_{idx}", text=piece, metadata=md))
            idx += 1
    return chunks


def _chunk_code(text: str, paper: Paper, config: Any, *, language: str) -> list[DocumentChunk]:
    chunk_size, chunk_overlap = _chunk_size_overlap(config)
    lc_lang = _LANG_TO_LC.get(language)
    if lc_lang is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            lc_lang,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    base_id = paper.id
    chunks: list[DocumentChunk] = []
    for i, piece in enumerate(splitter.split_text(text)):
        md = ChunkMetadata(
            paper_id=base_id,
            chunk_index=i,
            source=paper.source,
            title=paper.title,
            content_type="code",
            language=language,
        )
        chunks.append(DocumentChunk(id=f"{base_id}_code_{i}", text=piece, metadata=md))
    return chunks


def _to_chunk_config(config: Any) -> ChunkConfig:
    """Adapt a ``KnowledgeBaseConfig``-like object to a ``ChunkConfig`` for the
    fallback ``chunk_text`` call. ``KnowledgeBaseConfig`` exposes
    ``chunking_method`` (not ``method``) and a restricted Literal that omits
    ``"section_aware"``; map them onto ``ChunkConfig`` safely.
    """
    if isinstance(config, ChunkConfig):
        return config
    method = getattr(config, "chunking_method", None) or getattr(config, "method", "token")
    # ChunkConfig accepts {"token","semantic","agentic","section_aware"}.
    if method not in {"token", "semantic", "agentic", "section_aware"}:
        method = "token"
    chunk_size, chunk_overlap = _chunk_size_overlap(config)
    return ChunkConfig(method=method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


async def chunk_document(
    text: str,
    paper: Paper,
    *,
    content_type: str,
    language: str | None,
    config: Any,
) -> list[DocumentChunk]:
    """Dispatch chunking by content type.

    Routing:
    - markdown + ``markdown_heading_aware``  → heading-stack splitter
    - code     + ``code_chunking != 'splitter'`` → ``chunking_code.chunk_code``
                                                    (falls back to splitter on None)
    - everything else → ``chunk_text``
    """
    if content_type == "markdown" and getattr(config, "markdown_heading_aware", True):
        return _chunk_markdown(text, paper, config)

    if content_type == "code" and language:
        mode = getattr(config, "code_chunking", "auto")
        if mode != "splitter":
            from perspicacite.pipeline.chunking_code import chunk_code
            cs, co = _chunk_size_overlap(config)
            result = chunk_code(text, paper, language=language,
                                file_path=getattr(paper, "source_file_path", None),
                                chunk_size=cs, chunk_overlap=co)
            if result is not None:
                return result
            # In "ast" mode this is unexpected — log it. In "auto" mode the
            # splitter fallback is the documented behaviour; stay quiet.
            if mode == "ast":
                from perspicacite.logging import get_logger
                get_logger("perspicacite.pipeline.chunking_dispatch").warning(
                    "code_chunking_ast_unavailable",
                    language=language,
                    paper_id=paper.id,
                    mode=mode,
                )
        return _chunk_code(text, paper, config, language=language)

    # Fallback: token chunker.
    from perspicacite.pipeline.chunking import chunk_text
    return await chunk_text(text, paper, _to_chunk_config(config))
