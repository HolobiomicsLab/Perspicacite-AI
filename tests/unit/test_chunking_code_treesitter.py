# tests/unit/test_chunking_code_treesitter.py
import importlib.util

import pytest

from perspicacite.models.papers import Paper, PaperSource
from perspicacite.pipeline.chunking_code import (
    HAS_TREE_SITTER,
    _chunk_treesitter,
)


def _paper():
    return Paper(id="github:o/r@abc:f.go", title="t", abstract="", source=PaperSource.BIBTEX)


def test_constant_is_false_when_dep_missing():
    # Treat as a runtime probe — the constant equals importability. Mirror the
    # probe's own resolution order: either parser pack satisfies it.
    expected = (
        importlib.util.find_spec("tree_sitter_language_pack") is not None
        or importlib.util.find_spec("tree_sitter_languages") is not None
    )
    assert expected == HAS_TREE_SITTER


@pytest.mark.skipif(not HAS_TREE_SITTER, reason="no tree-sitter parser pack installed")
def test_go_function_extracted():
    src = (
        "package main\n\n"
        "func Hello(name string) string {\n"
        "    return \"hi \" + name\n"
        "}\n"
    )
    chunks = _chunk_treesitter(src, _paper(), file_path="f.go", language="go")
    assert chunks is not None
    names = [c.metadata.symbol_name for c in chunks]
    assert "Hello" in names


def test_returns_none_when_dep_unavailable_and_caller_falls_back():
    # When the dep isn't installed, _chunk_treesitter must return None
    # so the dispatcher can fall through to the splitter.
    if HAS_TREE_SITTER:
        pytest.skip("dep present; this guard is only meaningful when absent")
    out = _chunk_treesitter("func F() {}\n", _paper(), file_path="f.go", language="go")
    assert out is None
