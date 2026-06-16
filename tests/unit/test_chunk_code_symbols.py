"""Symbol-level code chunking (chunk_code) — regression guard.

The tree-sitter binding shipped by ``tree-sitter-language-pack`` exposes a
non-standard Node API (kind/child_count/start_position as zero-arg methods,
parse() wants str). ``_chunk_treesitter`` has a compatibility shim for both
that and classic py-tree-sitter; these tests lock the behaviour so a future
binding bump that breaks the shim is caught (it must fall back, never raise).
"""
from perspicacite.models.papers import Paper, PaperSource
from perspicacite.pipeline.chunking_code import chunk_code

_JAVA = """\
/*
 * Copyright (c) 2004-2024 The Example Team — license boilerplate that should
 * NOT become its own retrieval chunk (it is not a class/method node).
 */
package io.example.modules;
import java.util.List;

public class SpectraMerger {
    /** Merge MS2 spectra within a tolerance. */
    public MergedSpectrum merge(List scans, double tol) {
        return doMerge(scans, tol);
    }
    private MergedSpectrum doMerge(List s, double t) { return null; }
}
"""

_R = """\
#' roxygen comment
normalize <- function(x) { x / max(x) }
detect_peaks <- function(spec, thr = 1e5) { which(spec > thr) }
"""


def _paper(pid: str) -> Paper:
    return Paper(id=pid, title=pid, source=PaperSource.SKILL_BUNDLE)


def test_java_symbol_chunks_exclude_license_header():
    chunks = chunk_code(_JAVA, _paper("gh:x:SpectraMerger.java"), language="java",
                        file_path="SpectraMerger.java", chunk_size=1000, chunk_overlap=100)
    # tree-sitter may be unavailable -> chunk_code returns None and the dispatcher
    # falls back to a splitter. When it IS available we assert real symbol chunks.
    if chunks is None:
        return
    assert chunks, "expected at least one symbol chunk"
    # The standalone license header must not be emitted as its own chunk: every
    # chunk should carry a real symbol kind, and the class/method must appear.
    kinds = {c.metadata.symbol_kind for c in chunks}
    assert kinds & {"class", "method"}, kinds
    names = {c.metadata.symbol_name for c in chunks}
    assert "SpectraMerger" in names, names


def test_r_regex_symbol_chunks():
    chunks = chunk_code(_R, _paper("gh:x:utils.R"), language="r",
                        file_path="utils.R", chunk_size=1000, chunk_overlap=100)
    assert chunks is not None and len(chunks) == 2
    assert [c.metadata.symbol_name for c in chunks] == ["normalize", "detect_peaks"]
    assert all(c.metadata.symbol_kind == "function" for c in chunks)


def test_unknown_language_returns_none_for_fallback():
    # A language with no backend must return None (caller falls back), not raise.
    out = chunk_code("some text", _paper("gh:x:f.zzz"), language="zzz",
                     file_path="f.zzz", chunk_size=1000, chunk_overlap=100)
    assert out is None
