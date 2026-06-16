"""Code-aware chunking (sub-project A of the 2026-05-15 design).

Ports the AST/regex/notebook chunkers from AgenticScienceBuilder's
``script_linker.py`` and adds an optional Tree-sitter path for languages
without a Python ``ast`` equivalent. Each chunk carries symbol_name,
symbol_kind, line range, docstring, and module-level imports in metadata
so downstream symbol-index writes (``pipeline.symbol_index``) can derive
``SymbolRecord``s without re-parsing.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.papers import Paper

_DOCSTRING_MAX = 500

# When a class' source segment exceeds this many characters we additionally
# emit one chunk per top-level method (alongside the class-level chunk) so
# embeddings stay focused. Below the threshold the single class chunk is
# already small enough that splitting hurts retrieval more than it helps.
_METHOD_SUBCHUNK_THRESHOLD_CHARS = 1500


def _method_kind_from_decorators(node):
    """Return symbol_kind based on @classmethod / @staticmethod / @property decorators.

    Falls back to "method" for anything else (including custom decorators).
    """
    for dec in node.decorator_list:
        name = None
        if isinstance(dec, ast.Name):
            name = dec.id
        elif isinstance(dec, ast.Attribute):
            name = dec.attr
        if name == "classmethod":
            return "classmethod"
        if name == "staticmethod":
            return "staticmethod"
        if name == "property":
            return "property"
    return "method"


def _chunk_python_ast(
    text: str,
    paper: Paper,
    *,
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """AST-based Python chunker. One chunk per top-level def / class.

    Falls back to a single module chunk on SyntaxError or empty source
    (matches AgenticScienceBuilder/script_linker.py:55-128).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [_module_chunk(text, paper, file_path=file_path, imports=[])]

    lines = text.splitlines()

    # Collect top-level imports for annotation.
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])

    chunks: list[DocumentChunk] = []
    base_id = paper.id
    idx = 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            body_text = "\n".join(lines[start - 1 : end])
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            ds = ast.get_docstring(node)
            md = ChunkMetadata(
                paper_id=base_id,
                chunk_index=idx,
                source=paper.source,
                title=paper.title,
                content_type="code",
                language="python",
                source_file_path=file_path,
                symbol_name=node.name,
                symbol_kind=kind,
                parent_class=None,
                start_line=start,
                end_line=end,
                docstring=ds[:_DOCSTRING_MAX] if ds else None,
                imports=imports,
            )
            chunks.append(
                DocumentChunk(id=f"{base_id}_code_{idx}", text=body_text, metadata=md)
            )
            idx += 1

            # For large classes, additionally emit one chunk per top-level
            # method so embeddings stay focused on individual symbols. The
            # class-level chunk above remains for symbol-index browsing.
            if (
                isinstance(node, ast.ClassDef)
                and len(body_text) > _METHOD_SUBCHUNK_THRESHOLD_CHARS
            ):
                for sub in node.body:
                    if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    sub_start = sub.lineno
                    sub_end = getattr(sub, "end_lineno", sub_start) or sub_start
                    method_text = "\n".join(lines[sub_start - 1 : sub_end])
                    if not method_text.strip():
                        continue
                    sub_ds = ast.get_docstring(sub)
                    sub_md = ChunkMetadata(
                        paper_id=base_id,
                        chunk_index=idx,
                        source=paper.source,
                        title=paper.title,
                        content_type="code",
                        language="python",
                        source_file_path=file_path,
                        symbol_name=sub.name,
                        symbol_kind=_method_kind_from_decorators(sub),
                        parent_class=node.name,
                        start_line=sub_start,
                        end_line=sub_end,
                        docstring=sub_ds[:_DOCSTRING_MAX] if sub_ds else None,
                        imports=imports,
                    )
                    chunks.append(
                        DocumentChunk(
                            id=f"{base_id}_code_{idx}",
                            text=method_text,
                            metadata=sub_md,
                        )
                    )
                    idx += 1

    if not chunks:
        return [_module_chunk(text, paper, file_path=file_path, imports=imports)]
    return chunks


def _module_chunk(
    text: str, paper: Paper, *, file_path: str, imports: list[str]
) -> DocumentChunk:
    lines = text.splitlines()
    md = ChunkMetadata(
        paper_id=paper.id,
        chunk_index=0,
        source=paper.source,
        title=paper.title,
        content_type="code",
        language="python",
        source_file_path=file_path,
        symbol_name=file_path or "module",
        symbol_kind="module",
        start_line=1,
        end_line=max(len(lines), 1),
        docstring=None,
        imports=imports,
    )
    return DocumentChunk(id=f"{paper.id}_code_0", text=text, metadata=md)


_R_FUNCTION_RE = re.compile(
    r"^(?P<name>[A-Za-z_.][A-Za-z0-9_.]*)\s*<-\s*function\s*\(",
    re.MULTILINE,
)


def _chunk_r_regex(text: str, paper: Paper, *, file_path: str) -> list[DocumentChunk]:
    """Chunk R/Rmd source by ``name <- function(`` pattern.

    Each match starts a chunk that runs until the next match or EOF.
    Falls back to a single module chunk when no functions are found.
    """
    matches = list(_R_FUNCTION_RE.finditer(text))
    if not matches:
        md = ChunkMetadata(
            paper_id=paper.id,
            chunk_index=0,
            source=paper.source,
            title=paper.title,
            content_type="code",
            language="r",
            source_file_path=file_path,
            symbol_name=file_path or "module",
            symbol_kind="module",
            start_line=1,
            end_line=max(text.count("\n") + 1, 1),
            docstring=None,
            imports=[],
        )
        return [DocumentChunk(id=f"{paper.id}_code_0", text=text, metadata=md)]

    chunks: list[DocumentChunk] = []
    for i, m in enumerate(matches):
        start_line = text[: m.start()].count("\n") + 1
        end_char = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        end_line = text[:end_char].count("\n") + 1
        body = text[m.start() : end_char].rstrip()
        md = ChunkMetadata(
            paper_id=paper.id,
            chunk_index=i,
            source=paper.source,
            title=paper.title,
            content_type="code",
            language="r",
            source_file_path=file_path,
            symbol_name=m.group("name"),
            symbol_kind="function",
            start_line=start_line,
            end_line=end_line,
            docstring=None,
            imports=[],
        )
        chunks.append(DocumentChunk(id=f"{paper.id}_code_{i}", text=body, metadata=md))
    return chunks


def _chunk_notebook(text: str, paper: Paper, *, file_path: str) -> list[DocumentChunk]:
    """Chunk a Jupyter notebook by code cell.

    Each code cell becomes a chunk. Cells containing function/class defs
    are sub-split via ``_chunk_python_ast`` so individual functions are
    addressable. Cell outputs are stripped before parse (this is what
    the chunker sees — the disk-side stripping happens at fetch time in
    AgenticScienceBuilder's enrichment.py:_strip_notebook_outputs).

    Falls back to a single module chunk on JSON error or no code cells.
    """
    try:
        nb = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        md = ChunkMetadata(
            paper_id=paper.id, chunk_index=0, source=paper.source, title=paper.title,
            content_type="code", language="python", source_file_path=file_path,
            symbol_name=file_path or "module", symbol_kind="module",
            start_line=1, end_line=max(text.count("\n") + 1, 1),
            docstring=None, imports=[],
        )
        return [DocumentChunk(id=f"{paper.id}_code_0", text=text, metadata=md)]

    cells = nb.get("cells", []) if isinstance(nb, dict) else []
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if not code_cells:
        md = ChunkMetadata(
            paper_id=paper.id, chunk_index=0, source=paper.source, title=paper.title,
            content_type="code", language="python", source_file_path=file_path,
            symbol_name=file_path or "module", symbol_kind="module",
            start_line=1, end_line=1, docstring=None, imports=[],
        )
        return [DocumentChunk(id=f"{paper.id}_code_0", text="", metadata=md)]

    chunks: list[DocumentChunk] = []
    out_idx = 0
    for i, cell in enumerate(code_cells, start=1):
        src_lines = cell.get("source", [])
        cell_text = "".join(src_lines) if isinstance(src_lines, list) else str(src_lines)
        cell_name = f"{file_path}::cell_{i}"

        # Sub-chunk via AST if it has function/class defs.
        sub = _chunk_python_ast(cell_text, paper, file_path=cell_name,
                                chunk_size=1000, chunk_overlap=200)
        has_defs = any(c.metadata.symbol_kind in ("function", "class") for c in sub)
        if has_defs:
            # Re-index the sub-chunks so the parent paper's chunk_index is sequential.
            for s in sub:
                md = s.metadata.model_copy(update={"chunk_index": out_idx})
                chunks.append(DocumentChunk(id=f"{paper.id}_code_{out_idx}",
                                            text=s.text, metadata=md))
                out_idx += 1
        else:
            line_count = cell_text.count("\n") + 1
            md = ChunkMetadata(
                paper_id=paper.id, chunk_index=out_idx, source=paper.source,
                title=paper.title, content_type="code", language="python",
                source_file_path=file_path,
                symbol_name=cell_name, symbol_kind="cell",
                start_line=1, end_line=line_count, docstring=None, imports=[],
            )
            chunks.append(DocumentChunk(id=f"{paper.id}_code_{out_idx}",
                                        text=cell_text, metadata=md))
            out_idx += 1
    return chunks


# Optional Tree-sitter path. Activates when a parser-pack is importable;
# otherwise _chunk_treesitter returns None and the dispatcher falls back to the
# LangChain splitter for the same content. We prefer `tree_sitter_language_pack`
# (maintained, ships wheels for current CPython incl. 3.13/3.14); fall back to
# the legacy `tree_sitter_languages`.
def _resolve_ts_get_parser():
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        return get_parser
    except Exception:
        pass
    try:
        from tree_sitter_languages import get_parser  # type: ignore
        return get_parser
    except Exception:
        return None


_TS_GET_PARSER = _resolve_ts_get_parser()
HAS_TREE_SITTER = _TS_GET_PARSER is not None


_TS_NODE_TYPES = {
    # node_type → symbol_kind. Language-specific names are unified here.
    "function_declaration": "function",
    "function_definition": "function",
    "method_definition": "method",
    "method_declaration": "method",
    "class_declaration": "class",
    "class_definition": "class",
    "struct_specifier": "class",
    "struct_item": "class",
    "type_declaration": "class",
}


def _chunk_treesitter(
    text: str, paper: Paper, *, file_path: str, language: str
) -> list[DocumentChunk] | None:
    """Tree-sitter-backed chunker for non-Python languages.

    Returns None when the optional dep isn't installed OR when the parser
    cannot be obtained for ``language``; the caller falls back to the
    splitter. Never raises.
    """
    if not HAS_TREE_SITTER or _TS_GET_PARSER is None:
        return None
    try:
        parser = _TS_GET_PARSER(language)
    except Exception:
        return None
    # tree-sitter API churn: parse() wants bytes in some builds, str in others;
    # root_node is a property in py-tree-sitter but a method in some packs.
    try:
        try:
            tree = parser.parse(text.encode("utf-8"))
        except TypeError:
            tree = parser.parse(text)
    except Exception:
        return None
    root = tree.root_node
    if callable(root):
        root = root()

    lines = text.splitlines()
    src_bytes = text.encode("utf-8")
    chunks: list[DocumentChunk] = []
    idx = 0

    # --- Node-API compatibility shims. py-tree-sitter exposes
    # type/children/start_point/.text as PROPERTIES; tree-sitter-language-pack's
    # binding exposes kind/child_count/child(i)/start_position/byte-range as
    # zero-arg METHODS. `_v` resolves either form.
    def _v(a: Any):
        return a() if callable(a) else a

    def _kind(node: Any):
        k = getattr(node, "type", None)
        if k is None:
            k = getattr(node, "kind", None)
        return _v(k)

    def _children(node: Any) -> list:
        ch = _v(getattr(node, "children", None))
        if isinstance(ch, (list, tuple)):
            return list(ch)
        cc = _v(getattr(node, "child_count", None))
        if isinstance(cc, int):
            return [node.child(i) for i in range(cc)]
        return []

    def _row(pt: Any):
        pt = _v(pt)
        if pt is None:
            return None
        if hasattr(pt, "row"):
            return pt.row
        try:
            return pt[0]
        except Exception:
            return None

    def _rows(node: Any):
        sp = getattr(node, "start_point", None) or getattr(node, "start_position", None)
        ep = getattr(node, "end_point", None) or getattr(node, "end_position", None)
        return _row(sp), _row(ep)

    def _node_text(node: Any):
        sb, eb = _v(getattr(node, "start_byte", None)), _v(getattr(node, "end_byte", None))
        if isinstance(sb, int) and isinstance(eb, int):
            return src_bytes[sb:eb].decode("utf-8", "replace")
        t = _v(getattr(node, "text", None))
        if isinstance(t, bytes):
            return t.decode("utf-8", "replace")
        return str(t) if t is not None else None

    def _name(node: Any) -> str | None:
        getf = getattr(node, "child_by_field_name", None)
        if callable(getf):
            try:
                nm = getf("name")
                if nm is not None:
                    return _node_text(nm)
            except Exception:
                pass
        for child in _children(node):
            if _kind(child) in ("identifier", "type_identifier", "name"):
                return _node_text(child)
        for child in _children(node):  # name sometimes nests one level deeper
            nested = _name(child)
            if nested:
                return nested
        return None

    def _walk(node: Any) -> None:
        nonlocal idx
        for child in _children(node):
            kind = _TS_NODE_TYPES.get(_kind(child))
            if kind is not None:
                sr, er = _rows(child)
                if sr is None:
                    _walk(child)
                    continue
                start_row = sr + 1
                end_row = (er if er is not None else sr) + 1
                body = "\n".join(lines[start_row - 1 : end_row])
                name = _name(child) or f"{kind}_{idx}"
                md = ChunkMetadata(
                    paper_id=paper.id, chunk_index=idx, source=paper.source,
                    title=paper.title, content_type="code", language=language,
                    source_file_path=file_path, symbol_name=name, symbol_kind=kind,
                    start_line=start_row, end_line=end_row, docstring=None, imports=[],
                )
                chunks.append(DocumentChunk(id=f"{paper.id}_code_{idx}",
                                            text=body, metadata=md))
                idx += 1
                # Big class/struct → also emit per-method subchunks so a
                # 2000-line class doesn't collapse into one giant chunk
                # (mirrors the Python AST path's threshold behaviour).
                if kind == "class" and len(body) > _METHOD_SUBCHUNK_THRESHOLD_CHARS:
                    _walk(child)
            else:
                _walk(child)

    try:
        _walk(root)
    except Exception:  # any Node-API mismatch -> let caller fall back
        return None

    if not chunks:
        md = ChunkMetadata(
            paper_id=paper.id, chunk_index=0, source=paper.source, title=paper.title,
            content_type="code", language=language, source_file_path=file_path,
            symbol_name=file_path or "module", symbol_kind="module",
            start_line=1, end_line=max(len(lines), 1), docstring=None, imports=[],
        )
        return [DocumentChunk(id=f"{paper.id}_code_0", text=text, metadata=md)]
    return chunks


def chunk_code(
    text: str,
    paper: Paper,
    *,
    language: str,
    file_path: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk] | None:
    """Dispatch entry. Returns None when no backend applies (caller falls
    back to the splitter)."""
    fp = file_path or ""
    lang = (language or "").lower()
    if lang == "python":
        return _chunk_python_ast(text, paper, file_path=fp,
                                 chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if lang in ("r", "rmd"):
        return _chunk_r_regex(text, paper, file_path=fp)
    if lang == "ipynb":
        return _chunk_notebook(text, paper, file_path=fp)
    ts_langs = {"javascript", "typescript", "go", "rust", "java", "cpp",
                "ruby", "swift", "kotlin", "csharp"}
    if lang in ts_langs:
        return _chunk_treesitter(text, paper, file_path=fp, language=lang)
    return None
