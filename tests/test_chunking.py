#!/usr/bin/env python3
"""Tests for text chunking strategies.

Tests the chunking dispatcher (rag/chunking.py) and advanced chunker
(pipeline/chunking_advanced.py) including token, semantic, and agentic methods.

Run: PYTHONPATH=src pytest tests/test_chunking.py -v
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Direct module loading to avoid eager import chains (chromadb etc.)
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent.parent / "src" / "perspicacite"


def _load_module(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, str(_BASE / rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load chunking modules directly
_chunking_mod = _load_module("perspicacite.rag.chunking", "rag/chunking.py")
_advanced_mod = _load_module(
    "perspicacite.pipeline.chunking_advanced", "pipeline/chunking_advanced.py"
)

SimpleChunker = _chunking_mod.SimpleChunker
AdvancedChunkerAdapter = _chunking_mod.AdvancedChunkerAdapter
create_chunker = _chunking_mod.create_chunker
split_into_sections = _advanced_mod.split_into_sections


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """Background

Elucidating the mechanisms that control the transcription of genes is one of the major goals of molecular biology. One current approach is to determine whether the regulatory sequences of a set of genes have significantly higher than expected affinity for a regulatory protein.

The same approach can also detect motifs that predict significantly lower than expected DNA binding by a protein, indicating that binding may be detrimental to the proper regulation of some or all of the genes.

Methods

Motif Affinity Functions

We define a motif affinity function as any function that takes a DNA sequence and a known motif, and returns a real-valued score representing the affinity of the motif for that sequence.

Association Functions

Given a motif affinity function, we can define an association function that maps a set of sequences to a significance score. This score indicates whether the motif is enriched in the set of sequences.

Results

We evaluated nine different motif enrichment analysis methods on a comprehensive benchmark dataset. Our results show that the methods differ substantially in their ability to detect enriched motifs, particularly when the signal is weak.

The best-performing methods use a variable-order threshold to define the set of genes, rather than requiring the user to specify a fixed threshold. This approach is more robust to noise in the data.

Discussion

Our findings suggest that the choice of motif enrichment method can have a significant impact on biological conclusions. We recommend that researchers carefully consider the method they use, and report which method was used in their publications.

Conclusions

We have presented a comprehensive comparison of motif enrichment analysis methods. Our results provide guidance for researchers in choosing the most appropriate method for their analysis.
"""

SHORT_TEXT = "This is a short text with only a few words."


@pytest.fixture
def cached_paper_path():
    """Path to a cached paper if available."""
    p = Path(__file__).parent.parent / "data" / "papers" / "PMC2868005.txt"
    if p.exists():
        return p
    return None


# ---------------------------------------------------------------------------
# SimpleChunker tests
# ---------------------------------------------------------------------------


class TestSimpleChunker:
    """Tests for the word-based SimpleChunker."""

    def test_basic_chunking(self):
        chunker = SimpleChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_text(SAMPLE_TEXT)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c.split()) <= 110  # allow slight overage from overlap

    def test_short_text_single_chunk(self):
        chunker = SimpleChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk_text(SHORT_TEXT)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunker = SimpleChunker()
        assert chunker.chunk_text("") == []

    def test_none_text(self):
        chunker = SimpleChunker()
        assert chunker.chunk_text(None) == []

    def test_exact_fit(self):
        words = ["word"] * 100
        text = " ".join(words)
        chunker = SimpleChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1

    def test_preserves_first_word(self):
        chunker = SimpleChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk_text(SAMPLE_TEXT)
        words_in = SAMPLE_TEXT.split()
        assert chunks[0].split()[0] == words_in[0]


# ---------------------------------------------------------------------------
# create_chunker factory tests
# ---------------------------------------------------------------------------


class TestCreateChunker:
    """Tests for the chunker factory."""

    def test_token_returns_simple(self):
        chunker = create_chunker(chunk_size=500, overlap=50, method="token")
        assert isinstance(chunker, SimpleChunker)

    def test_semantic_returns_advanced(self):
        chunker = create_chunker(chunk_size=500, overlap=50, method="semantic")
        assert isinstance(chunker, AdvancedChunkerAdapter)
        assert chunker.method == "semantic"

    def test_agentic_returns_advanced(self):
        chunker = create_chunker(chunk_size=500, overlap=50, method="agentic")
        assert isinstance(chunker, AdvancedChunkerAdapter)
        assert chunker.method == "agentic"

    def test_unknown_method_falls_back(self):
        chunker = create_chunker(chunk_size=500, overlap=50, method="nonexistent")
        assert isinstance(chunker, SimpleChunker)

    def test_default_method_is_token(self):
        chunker = create_chunker()
        assert isinstance(chunker, SimpleChunker)


# ---------------------------------------------------------------------------
# AdvancedChunkerAdapter tests
# ---------------------------------------------------------------------------


class TestAdvancedChunkerAdapter:
    """Tests for the async advanced chunker adapter."""

    @pytest.mark.asyncio
    async def test_semantic_chunking_async(self):
        chunker = AdvancedChunkerAdapter(method="semantic", chunk_size=500, overlap=50)
        chunks = await chunker.chunk_text_async(SAMPLE_TEXT)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c.strip()) > 0 for c in chunks)

    @pytest.mark.asyncio
    async def test_agentic_with_mock_llm(self):
        mock_llm = AsyncMock()

        async def mock_complete(prompt, **kwargs):
            return '{"spans": [{"start": 0, "end": 500}, {"start": 500, "end": 1000}]}'

        mock_llm.complete = mock_complete

        chunker = AdvancedChunkerAdapter(
            method="agentic", chunk_size=500, overlap=50, llm_client=mock_llm,
        )
        text = "Word. " * 300
        chunks = await chunker.chunk_text_async(text)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_agentic_fallback_on_llm_failure(self):
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM unavailable"))

        chunker = AdvancedChunkerAdapter(
            method="agentic", chunk_size=200, overlap=20, llm_client=mock_llm,
        )
        chunks = await chunker.chunk_text_async(SAMPLE_TEXT)
        assert len(chunks) >= 1

    def test_sync_raises_runtime_error(self):
        chunker = AdvancedChunkerAdapter(method="semantic")
        with pytest.raises(RuntimeError, match="requires async"):
            chunker.chunk_text(SAMPLE_TEXT)


# ---------------------------------------------------------------------------
# Section detection tests
# ---------------------------------------------------------------------------


class TestSectionDetection:
    """Tests for section splitting in chunking_advanced."""

    def test_split_into_sections(self):
        sections = split_into_sections(SAMPLE_TEXT)
        assert len(sections) >= 2
        section_names = [name for name, _ in sections]
        assert "Methods" in section_names
        assert "Results" in section_names
        assert "Discussion" in section_names

    def test_references_filtered(self):
        text_with_refs = SAMPLE_TEXT + "\nReferences\n[1] Smith et al. 2020.\n[2] Jones 2021.\n"
        sections = split_into_sections(text_with_refs)
        section_names = [name for name, _ in sections]
        assert "References" not in section_names

    def test_empty_text(self):
        sections = split_into_sections("")
        assert sections == []


# ---------------------------------------------------------------------------
# Integration: chunk a real cached paper (optional)
# ---------------------------------------------------------------------------


class TestWithCachedPaper:
    """Integration tests using a real cached paper from data/papers/."""

    @pytest.mark.skipif(
        not Path("data/papers/PMC2868005.txt").exists(),
        reason="No cached paper. Run europepmc download first.",
    )
    def test_token_chunker_real_paper(self):
        text = Path("data/papers/PMC2868005.txt").read_text()
        chunker = SimpleChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 5
        assert all(len(c.split()) <= 510 for c in chunks)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("data/papers/PMC2868005.txt").exists(),
        reason="No cached paper. Run europepmc download first.",
    )
    async def test_semantic_chunker_real_paper(self):
        text = Path("data/papers/PMC2868005.txt").read_text()
        chunker = AdvancedChunkerAdapter(method="semantic", chunk_size=500, overlap=50)
        chunks = await chunker.chunk_text_async(text)
        assert len(chunks) >= 3
        assert all(isinstance(c, str) for c in chunks)

    @pytest.mark.skipif(
        not Path("data/papers/PMC2868005_sections.json").exists(),
        reason="No cached sections available.",
    )
    def test_sections_file_valid(self):
        import json

        sections = json.loads(Path("data/papers/PMC2868005_sections.json").read_text())
        assert isinstance(sections, dict)
        assert len(sections) >= 5
        section_names = list(sections.keys())
        has_common = any(
            name in section_names
            for name in ["Background", "Methods", "Results", "Discussion", "Abstract"]
        )
        assert has_common, f"Expected common sections, got: {section_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
