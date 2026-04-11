"""Unit tests for Phase 4: Grounded citations, confidence scoring, UI trace events.

Tests pure functions and properties that require no LLM access:
- _verify_citations: citation marker validation and cleanup
- EvidenceFacet.confidence: blended confidence score
- EvidenceStore.overall_confidence / facet_confidences: aggregate scores

Run: PYTHONPATH=src pytest tests/unit/test_agentic_phase4.py -v
"""

import pytest

from perspicacite.rag.agentic.orchestrator import (
    AgenticOrchestrator,
    EvidenceFacet,
    EvidenceStore,
)


# ---------------------------------------------------------------------------
# _verify_citations
# ---------------------------------------------------------------------------


class TestVerifyCitations:
    """Tests for post-generation citation verification."""

    def _papers(self, n):
        """Create n dummy papers with title and doi."""
        return [
            {"title": f"Paper {i}", "doi": f"10.1/p{i}"}
            for i in range(1, n + 1)
        ]

    def test_all_valid_citations(self):
        answer = "FBMN is useful [1]. GNPS is also useful [2]."
        papers = self._papers(3)
        cleaned, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        assert cleaned == answer
        assert cmap["cited_count"] == 2
        assert len(cmap["cited"]) == 2
        assert cmap["cited"][0]["index"] == 1
        assert cmap["cited"][1]["index"] == 2
        assert len(cmap["uncited"]) == 1
        assert cmap["uncited"][0]["index"] == 3
        assert cmap["invalid_stripped"] == []
        assert cmap["total_papers"] == 3

    def test_invalid_citations_stripped(self):
        answer = "See [1] and [99] for details."
        papers = self._papers(3)
        cleaned, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        assert "[99]" not in cleaned
        assert "[1]" in cleaned
        assert cmap["invalid_stripped"] == [99]
        assert cmap["cited_count"] == 1

    def test_multiple_invalid_stripped(self):
        answer = "References [0] [1] [50] [100] are relevant."
        papers = self._papers(3)
        cleaned, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        assert "[0]" not in cleaned
        assert "[50]" not in cleaned
        assert "[100]" not in cleaned
        assert "[1]" in cleaned
        assert sorted(cmap["invalid_stripped"]) == [0, 50, 100]

    def test_double_space_collapse_after_strip(self):
        answer = "See [99] for more."
        papers = self._papers(1)
        cleaned, _ = AgenticOrchestrator._verify_citations(answer, papers)
        # "[99]" removed → "See  for more." → "See for more." (double space collapsed)
        assert "  " not in cleaned

    def test_no_citations_in_answer(self):
        answer = "FBMN is a method for molecular networking."
        papers = self._papers(3)
        cleaned, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        assert cleaned == answer
        assert cmap["cited_count"] == 0
        assert len(cmap["uncited"]) == 3

    def test_all_papers_cited(self):
        answer = "[1] [2] [3]"
        papers = self._papers(3)
        _, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        assert cmap["cited_count"] == 3
        assert len(cmap["uncited"]) == 0

    def test_empty_papers_no_crash(self):
        answer = "Some answer with [1] reference."
        cleaned, cmap = AgenticOrchestrator._verify_citations(answer, [])
        assert cmap["total_papers"] == 0
        assert cmap["cited_count"] == 0
        assert "[1]" not in cleaned
        assert cmap["invalid_stripped"] == [1]

    def test_empty_answer(self):
        papers = self._papers(3)
        cleaned, cmap = AgenticOrchestrator._verify_citations("", papers)
        assert cleaned == ""
        assert cmap["cited_count"] == 0
        assert len(cmap["uncited"]) == 3

    def test_citation_map_includes_title_and_doi(self):
        answer = "As shown in [1]."
        papers = [{"title": "FBMN Paper", "doi": "10.1/fbmn"}]
        _, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        assert cmap["cited"][0]["title"] == "FBMN Paper"
        assert cmap["cited"][0]["doi"] == "10.1/fbmn"

    def test_uncited_includes_title(self):
        answer = "Only [1] is cited."
        papers = self._papers(3)
        _, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        assert len(cmap["uncited"]) == 2
        assert cmap["uncited"][0]["title"] == "Paper 2"

    def test_repeated_valid_citation(self):
        answer = "[1] is good. As noted in [1], also [1]."
        papers = self._papers(2)
        _, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        # Same citation counted once in the set
        assert cmap["cited_count"] == 1

    def test_repeated_invalid_stripped(self):
        answer = "[99] says X. Also [99] says Y."
        papers = self._papers(1)
        cleaned, cmap = AgenticOrchestrator._verify_citations(answer, papers)
        # Both [99] occurrences stripped
        assert "[99]" not in cleaned
        assert cmap["invalid_stripped"] == [99]


# ---------------------------------------------------------------------------
# EvidenceFacet.confidence
# ---------------------------------------------------------------------------


class TestEvidenceFacetConfidence:
    """Tests for the blended confidence score on EvidenceFacet."""

    def _facet(self, entries=None):
        f = EvidenceFacet(query="test", step_ids=set())
        if entries:
            f.entries = entries
            for e in entries:
                k = f._entry_key(e)
                if k:
                    f._seen_keys.add(k)
        return f

    def test_empty_entries_returns_zero(self):
        f = self._facet()
        assert f.confidence == 0.0

    def test_single_entry_default_relevance(self):
        """1 entry, no relevance_score → default 3, no full text → count + rel only."""
        f = self._facet([{"title": "P1"}])
        # count_score = min(1/5, 1) = 0.2
        # rel_score = (3 - 1) / 4 = 0.5  (default relevance = 3)
        # ft_score = 0.0
        # confidence = 0.45*0.2 + 0.35*0.5 + 0.20*0.0 = 0.09 + 0.175 = 0.265
        assert f.confidence == pytest.approx(0.265, abs=0.001)

    def test_five_entries_saturates_count(self):
        entries = [{"title": f"P{i}"} for i in range(5)]
        f = self._facet(entries)
        # count_score = min(5/5, 1) = 1.0
        # rel_score = 0.5 (default 3)
        # ft_score = 0.0
        # confidence = 0.45*1.0 + 0.35*0.5 + 0.20*0.0 = 0.45 + 0.175 = 0.625
        assert f.confidence == pytest.approx(0.625, abs=0.001)

    def test_high_relevance_boosts_score(self):
        entries = [{"title": "P1", "relevance_score": 5}]
        f = self._facet(entries)
        # count_score = 0.2
        # rel_score = (5 - 1) / 4 = 1.0
        # ft_score = 0.0
        # confidence = 0.45*0.2 + 0.35*1.0 + 0.20*0.0 = 0.09 + 0.35 = 0.44
        assert f.confidence == pytest.approx(0.44, abs=0.001)

    def test_low_relevance_reduces_score(self):
        entries = [{"title": "P1", "relevance_score": 1}]
        f = self._facet(entries)
        # count_score = 0.2
        # rel_score = (1 - 1) / 4 = 0.0
        # ft_score = 0.0
        # confidence = 0.45*0.2 + 0.35*0.0 + 0.20*0.0 = 0.09
        assert f.confidence == pytest.approx(0.09, abs=0.001)

    def test_full_text_availability_boosts(self):
        entries = [
            {"title": "P1", "pdf_downloaded": True},
            {"title": "P2", "full_text": "some text"},
        ]
        f = self._facet(entries)
        # count_score = min(2/5, 1) = 0.4
        # rel_score = 0.5 (default 3)
        # ft_score = 2/2 = 1.0
        # confidence = 0.45*0.4 + 0.35*0.5 + 0.20*1.0 = 0.18 + 0.175 + 0.20 = 0.555
        assert f.confidence == pytest.approx(0.555, abs=0.001)

    def test_partial_full_text(self):
        entries = [
            {"title": "P1", "pdf_downloaded": True},
            {"title": "P2"},
            {"title": "P3"},
        ]
        f = self._facet(entries)
        # count_score = min(3/5, 1) = 0.6
        # rel_score = 0.5
        # ft_score = 1/3 ≈ 0.333
        # confidence = 0.45*0.6 + 0.35*0.5 + 0.20*0.333 = 0.27 + 0.175 + 0.0667 ≈ 0.5117
        assert f.confidence == pytest.approx(0.5117, abs=0.002)

    def test_many_entries_saturates_count_at_5(self):
        entries = [{"title": f"P{i}"} for i in range(10)]
        f = self._facet(entries)
        # count_score = min(10/5, 1) = 1.0 (capped)
        # Same as 5 entries for count component
        assert f.confidence == pytest.approx(0.625, abs=0.001)

    def test_mixed_relevance_averages(self):
        entries = [
            {"title": "P1", "relevance_score": 5},
            {"title": "P2", "relevance_score": 1},
        ]
        f = self._facet(entries)
        # avg_rel = 3.0 → rel_score = 0.5
        # count_score = 0.4, ft_score = 0.0
        # confidence = 0.45*0.4 + 0.35*0.5 = 0.18 + 0.175 = 0.355
        assert f.confidence == pytest.approx(0.355, abs=0.001)


# ---------------------------------------------------------------------------
# EvidenceStore.overall_confidence / facet_confidences
# ---------------------------------------------------------------------------


class TestEvidenceStoreConfidence:
    """Tests for aggregate confidence methods on EvidenceStore."""

    def test_empty_store_returns_zero(self):
        es = EvidenceStore()
        assert es.overall_confidence() == 0.0

    def test_single_facet_confidence(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        for i in range(5):
            es.facets["fbmn"].entries.append({"title": f"P{i}"})
        # Matches the facet confidence calculation
        assert es.overall_confidence() == es.facets["fbmn"].confidence

    def test_multiple_facets_returns_mean(self):
        es = EvidenceStore()
        es.register_facet("a", "A")
        es.register_facet("b", "B")
        # Facet a: 5 entries (covered, high count), no relevance/ft
        for i in range(5):
            es.facets["a"].entries.append({"title": f"A{i}"})
        # Facet b: 1 entry (partial)
        es.facets["b"].entries.append({"title": "B0"})
        conf_a = es.facets["a"].confidence
        conf_b = es.facets["b"].confidence
        expected = (conf_a + conf_b) / 2
        assert es.overall_confidence() == pytest.approx(expected, abs=0.001)

    def test_facet_confidences_returns_per_key(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        es.facets["fbmn"].entries.append({"title": "P1"})
        es.facets["gnps"].entries.append({"title": "P2"})
        result = es.facet_confidences()
        assert set(result.keys()) == {"fbmn", "gnps"}
        assert result["fbmn"] == es.facets["fbmn"].confidence
        assert result["gnps"] == es.facets["gnps"].confidence

    def test_all_gap_facets_returns_zero(self):
        es = EvidenceStore()
        es.register_facet("a", "A")
        es.register_facet("b", "B")
        # No entries on either
        assert es.overall_confidence() == 0.0

    def test_three_facets_mean(self):
        es = EvidenceStore()
        for key in ("a", "b", "c"):
            es.register_facet(key, key.upper())
        # Different entry counts per facet
        for i in range(5):
            es.facets["a"].entries.append({"title": f"A{i}"})
        for i in range(2):
            es.facets["b"].entries.append({"title": f"B{i}"})
        # c: 0 entries (gap)
        expected = (
            es.facets["a"].confidence
            + es.facets["b"].confidence
            + es.facets["c"].confidence
        ) / 3
        assert es.overall_confidence() == pytest.approx(expected, abs=0.001)
