"""Unit tests for Phase 1 agentic module features.

Tests pure functions and dataclass methods that require no LLM access:
- EvidenceStore: dedup, truncation, key generation
- heuristic_query_complexity: composite pattern detection
- ResearchPlanner static methods: query cleaning, decomposition, subqueries
- AgenticOrchestrator: paper dedup, author normalization, parallel batching

Run: PYTHONPATH=src pytest tests/unit/test_agentic_phase1.py -v
"""

import pytest

from perspicacite.rag.agentic.intent import (
    heuristic_query_complexity,
    HEURISTIC_WEAK_COMPLEXITY_TAGS,
)
from perspicacite.rag.agentic.orchestrator import (
    AgenticOrchestrator,
    EvidenceFacet,
    EvidenceStore,
)
from perspicacite.rag.agentic.planner import (
    Plan,
    ResearchPlanner,
    Step,
    StepType,
)


# ---------------------------------------------------------------------------
# EvidenceStore
# ---------------------------------------------------------------------------


class TestEvidenceStore:
    """Tests for EvidenceStore dedup and formatting (updated for faceted Phase 2 API)."""

    def test_empty_store(self):
        es = EvidenceStore()
        assert es.all_entries == []
        assert es.to_prompt_block() == ""

    def test_add_single_hit(self):
        es = EvidenceStore()
        es.add_kb_hits([{"title": "Paper A", "doi": "10.1/a", "excerpt": "text"}])
        assert len(es.all_entries) == 1
        assert es.all_entries[0]["title"] == "Paper A"

    def test_dedup_by_doi(self):
        es = EvidenceStore()
        es.add_kb_hits([{"title": "Paper A", "doi": "10.1/a", "excerpt": "x"}])
        es.add_kb_hits([{"title": "Paper A (duplicate)", "doi": "10.1/a", "excerpt": "y"}])
        assert len(es.all_entries) == 1

    def test_dedup_by_title_fallback(self):
        es = EvidenceStore()
        es.add_kb_hits([{"title": "Same Title Here", "doi": "", "excerpt": "x"}])
        es.add_kb_hits([{"title": "Same Title Here", "doi": "", "excerpt": "y"}])
        assert len(es.all_entries) == 1

    def test_different_papers_not_deduped(self):
        es = EvidenceStore()
        es.add_kb_hits([{"title": "Paper A", "doi": "10.1/a"}])
        es.add_kb_hits([{"title": "Paper B", "doi": "10.1/b"}])
        assert len(es.all_entries) == 2

    def test_entry_key_prefers_doi(self):
        f = EvidenceFacet(query="test")
        key = f._entry_key({"doi": "10.1/a", "title": "Anything"})
        assert key == "doi:10.1/a"

    def test_entry_key_title_fallback(self):
        f = EvidenceFacet(query="test")
        key = f._entry_key({"doi": "", "title": "Some Title"})
        assert key.startswith("title:")

    def test_entry_key_empty(self):
        f = EvidenceFacet(query="test")
        key = f._entry_key({"doi": "", "title": ""})
        assert key == ""

    def test_to_prompt_block_truncation(self):
        es = EvidenceStore()
        # Add enough hits to exceed max_chars so truncation triggers
        for i in range(10):
            es.add_kb_hits([{"title": f"Paper {i}", "doi": f"10.1/{i}", "excerpt": "x" * 300}])
        block = es.to_prompt_block(max_chars=300)
        assert len(block) <= 304  # 300 + "\n…"
        assert block.endswith("…")

    def test_to_prompt_block_max_entries(self):
        es = EvidenceStore()
        for i in range(20):
            es.add_kb_hits([{"title": f"Paper {i}", "doi": f"10.1/{i}"}])
        block = es.to_prompt_block(max_entries_per_facet=5)
        # Should only include last 5 entries per facet
        assert block.count("- Paper") == 5

    def test_to_prompt_block_includes_doi(self):
        es = EvidenceStore()
        es.add_kb_hits([{"title": "Paper A", "doi": "10.1/a", "excerpt": ""}])
        block = es.to_prompt_block()
        assert "DOI: 10.1/a" in block


# ---------------------------------------------------------------------------
# heuristic_query_complexity
# ---------------------------------------------------------------------------


class TestHeuristicQueryComplexity:
    """Tests for regex-based composite query detection."""

    @pytest.mark.parametrize(
        "query,expected_tag",
        [
            ("FBMN vs GNPS", "vs"),
            ("FBMN vs. GNPS", "vs"),
            ("lotus versus openalex", "versus"),
            ("compare method A and B", "compare"),
            ("comparison of two approaches", "comparison"),
            ("pros and cons of deep learning", "pros_cons"),
            ("advantages and disadvantages of RAG", "adv_disadv"),
            ("difference between BERT and GPT", "difference_between"),
            ("differences between LC-MS and GC-MS", "differences_between"),
            ("trade-offs of large models", "tradeoffs"),
            ("tradeoffs of small models", "tradeoffs"),
            ("effect of temperature on extraction yield", "effect_on"),
            ("solvent and its effect on recovery", "and_effect_on"),
        ],
    )
    def test_composite_patterns(self, query, expected_tag):
        complexity, tag = heuristic_query_complexity(query)
        assert complexity == "composite", f"Expected composite for: {query}"
        assert tag == expected_tag

    @pytest.mark.parametrize(
        "query",
        [
            "what is feature-based molecular networking",
            "tell me about CRISPR gene editing",
            "how does FBMN work",
            "applications of metabolomics",
            "review of mass spectrometry methods",
        ],
    )
    def test_simple_queries(self, query):
        complexity, tag = heuristic_query_complexity(query)
        assert complexity == "simple", f"Expected simple for: {query}"

    def test_empty_query(self):
        complexity, tag = heuristic_query_complexity("")
        assert complexity == "simple"

    def test_none_query(self):
        complexity, tag = heuristic_query_complexity(None)
        assert complexity == "simple"

    def test_weak_complexity_tags(self):
        """effect_on is a weak signal that shouldn't override LLM saying simple."""
        assert "effect_on" in HEURISTIC_WEAK_COMPLEXITY_TAGS


# ---------------------------------------------------------------------------
# ResearchPlanner._clean_query_for_search
# ---------------------------------------------------------------------------


class TestCleanQueryForSearch:
    """Tests for conversational preamble stripping."""

    @pytest.mark.parametrize(
        "input_query,expected",
        [
            ("what is CRISPR gene editing", "CRISPR gene editing"),
            ("what are the applications of ML", "the applications of ML"),
            ("tell me about metabolomics", "metabolomics"),
            ("how does FBMN work", "FBMN work"),
            ("how do plants grow", "plants grow"),
            ("explain quantum computing", "quantum computing"),
            ("describe the process", "the process"),
            ("can you tell me about RAG", "RAG"),
            ("i want to learn about Python", "Python"),
            ("i want to know about NLP", "NLP"),
            ("i'd like to know about docking", "docking"),
        ],
    )
    def test_strips_prefixes(self, input_query, expected):
        assert ResearchPlanner._clean_query_for_search(input_query) == expected

    def test_no_prefix_unchanged(self):
        q = "CRISPR gene editing applications"
        assert ResearchPlanner._clean_query_for_search(q) == q

    def test_strips_whitespace(self):
        assert ResearchPlanner._clean_query_for_search("  what is FBMN  ") == "FBMN"

    def test_case_insensitive_prefix(self):
        assert ResearchPlanner._clean_query_for_search("What Is FBMN") == "FBMN"

    def test_only_first_prefix_stripped(self):
        """Only the matching prefix is removed, not recursive."""
        result = ResearchPlanner._clean_query_for_search("what is explain this")
        assert result == "explain this"


# ---------------------------------------------------------------------------
# ResearchPlanner._decompose_query
# ---------------------------------------------------------------------------


class TestDecomposeQuery:
    """Tests for 'X and its Y' query splitting."""

    def test_basic_decompose(self):
        """'X and its Y' with base > 5 chars decomposes."""
        result = ResearchPlanner._decompose_query("metabolomics and its applications")
        assert len(result) == 2
        assert result[0] == "metabolomics"
        assert result[1] == "metabolomics applications"

    def test_decompose_their(self):
        result = ResearchPlanner._decompose_query("methods and their limitations")
        assert len(result) == 2
        assert result[0] == "methods"
        assert result[1] == "methods limitations"

    def test_decompose_the(self):
        result = ResearchPlanner._decompose_query("CRISPR and the ethics")
        # "CRISPR" is 6 chars, > 5 threshold → decomposes
        assert len(result) == 2
        assert result[0] == "CRISPR"
        assert result[1] == "CRISPR ethics"

    def test_short_base_no_decompose(self):
        """Base topic <= 5 chars → no decomposition."""
        result = ResearchPlanner._decompose_query("FBMN and its application")
        # "FBMN" is 4 chars, not > 5
        assert len(result) == 1

    def test_short_aspect_no_decompose(self):
        """Aspect <= 3 chars → no decomposition."""
        result = ResearchPlanner._decompose_query("metabolomics and its use")
        # "use" is 3 chars, not > 3
        assert len(result) == 1

    def test_no_and_returns_single(self):
        result = ResearchPlanner._decompose_query("feature-based molecular networking")
        assert result == ["feature-based molecular networking"]

    def test_plain_and_also_decomposes(self):
        """Plain 'X and Y' (no pronoun) also matches — the pronoun group is optional."""
        result = ResearchPlanner._decompose_query("metabolomics and proteomics")
        # Regex: \band\b(?:\s+(?:its|their|the))?\s+ matches "and " without pronoun
        assert len(result) == 2
        assert result[0] == "metabolomics"
        assert result[1] == "metabolomics proteomics"


# ---------------------------------------------------------------------------
# ResearchPlanner._composite_subqueries
# ---------------------------------------------------------------------------


class TestCompositeSubqueries:
    """Tests for comparison/multi-entity query splitting."""

    def test_vs_split(self):
        result = ResearchPlanner._composite_subqueries("FBMN vs GNPS workflows")
        assert result == ["FBMN", "GNPS workflows"]

    def test_versus_split(self):
        result = ResearchPlanner._composite_subqueries("BERT versus GPT models")
        assert result == ["BERT", "GPT models"]

    def test_vs_dot_split(self):
        result = ResearchPlanner._composite_subqueries("LC-MS vs. GC-MS analysis")
        assert result == ["LC-MS", "GC-MS analysis"]

    def test_and_split_fallback(self):
        """When no vs/versus, try 'X and its Y' decomposition first."""
        result = ResearchPlanner._composite_subqueries("FBMN and its application")
        assert len(result) == 2

    def test_plain_and_split(self):
        """Plain 'X and Y' without pronoun → _decompose_query catches it first."""
        result = ResearchPlanner._composite_subqueries("metabolomics and proteomics")
        # _decompose_query matches "and " → returns [base, "base aspect"]
        assert result == ["metabolomics", "metabolomics proteomics"]

    def test_no_split_returns_single(self):
        result = ResearchPlanner._composite_subqueries("feature-based molecular networking")
        assert result == ["feature-based molecular networking"]

    def test_empty_string(self):
        result = ResearchPlanner._composite_subqueries("")
        assert result == [""]

    def test_none_like_input(self):
        result = ResearchPlanner._composite_subqueries("  ")
        # whitespace-only strips to empty, returns the original stripped
        assert len(result) >= 1

    def test_short_vs_fragments(self):
        """Very short fragments (< 3 chars) should not split."""
        result = ResearchPlanner._composite_subqueries("a vs b")
        # Both "a" and "b" are <= 2 chars, should not split
        assert len(result) == 1

    def test_max_3_results(self):
        """Output capped at 3 subqueries."""
        # This is a guard; current logic produces at most 2
        result = ResearchPlanner._composite_subqueries("X versus Y")
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# AgenticOrchestrator._normalize_authors
# ---------------------------------------------------------------------------


class TestNormalizeAuthors:
    """Tests for author list normalization."""

    def test_list_of_strings(self):
        assert AgenticOrchestrator._normalize_authors(["Alice", "Bob"]) == ["Alice", "Bob"]

    def test_comma_separated_string(self):
        assert AgenticOrchestrator._normalize_authors("Alice, Bob, Carol") == [
            "Alice",
            "Bob",
            "Carol",
        ]

    def test_single_string(self):
        assert AgenticOrchestrator._normalize_authors("Alice") == ["Alice"]

    def test_none(self):
        assert AgenticOrchestrator._normalize_authors(None) == []

    def test_empty_list(self):
        assert AgenticOrchestrator._normalize_authors([]) == []

    def test_empty_string(self):
        assert AgenticOrchestrator._normalize_authors("") == []

    def test_strips_whitespace(self):
        assert AgenticOrchestrator._normalize_authors("  Alice  ,  Bob  ") == [
            "Alice",
            "Bob",
        ]

    def test_filters_empty_entries(self):
        # Empty strings are filtered by `if a`, but whitespace-only strings
        # pass through `if a` and then `.strip()` makes them ""
        result = AgenticOrchestrator._normalize_authors(["Alice", "", "  ", "Bob"])
        assert "Alice" in result
        assert "Bob" in result
        # Whitespace-only entries become empty strings after strip
        assert result.count("") == 1


# ---------------------------------------------------------------------------
# AgenticOrchestrator._dedupe_paper_dicts
# ---------------------------------------------------------------------------


class TestDedupePaperDicts:
    """Tests for paper deduplication by DOI/title fingerprint."""

    def _make_orchestrator(self):
        """Create a minimal orchestrator instance (no real deps needed)."""
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def test_dedup_by_doi(self):
        orch = self._make_orchestrator()
        papers = [
            {"title": "Paper A", "doi": "10.1/a", "abstract": "short"},
            {"title": "Paper A (preprint)", "doi": "10.1/a", "abstract": "longer abstract here", "cited_by_count": 5},
        ]
        result = orch._dedupe_paper_dicts(papers)
        assert len(result) == 1
        # Should keep the higher-quality version (longer abstract)
        assert result[0]["abstract"] == "longer abstract here"

    def test_dedup_by_title_fingerprint(self):
        """Long similar titles (>40 chars alphanumeric) should dedup."""
        orch = self._make_orchestrator()
        papers = [
            {"title": "Feature-Based Molecular Networking for Mass Spectrometry Data Analysis"},
            {"title": "Feature-Based Molecular Networking for Mass Spectrometry Data Analysis", "doi": "10.1/x"},
        ]
        result = orch._dedupe_paper_dicts(papers)
        assert len(result) == 1

    def test_different_papers_kept(self):
        orch = self._make_orchestrator()
        papers = [
            {"title": "Paper A", "doi": "10.1/a"},
            {"title": "Paper B", "doi": "10.1/b"},
        ]
        result = orch._dedupe_paper_dicts(papers)
        assert len(result) == 2

    def test_removes_titleless_unknowns(self):
        """Papers with no title and unknown dedup key are dropped."""
        orch = self._make_orchestrator()
        papers = [
            {"doi": ""},
            {"title": "Paper A", "doi": "10.1/a"},
        ]
        result = orch._dedupe_paper_dicts(papers)
        assert len(result) == 1
        assert result[0]["title"] == "Paper A"

    def test_prefers_journal_over_biorxiv(self):
        """Journal version wins over bioRxiv when deduped by title fingerprint."""
        orch = self._make_orchestrator()
        # Title must be > 40 chars (alphanumeric) for fingerprint dedup
        long_title = "Feature-Based Molecular Networking for Mass Spectrometry Data Analysis in Natural Products"
        papers = [
            {"title": long_title, "doi": "10.1101/2024.01.01", "abstract": "short"},
            {"title": long_title, "doi": "10.1038/s41586-024-12345", "abstract": "short"},
        ]
        result = orch._dedupe_paper_dicts(papers)
        assert len(result) == 1
        # Journal version (non-bioRxiv) should win
        assert "1038" in result[0]["doi"]


# ---------------------------------------------------------------------------
# AgenticOrchestrator._normalize_doi_for_dedupe
# ---------------------------------------------------------------------------


class TestNormalizeDoi:
    """Tests for DOI normalization in dedup."""

    def test_plain_doi(self):
        assert AgenticOrchestrator._normalize_doi_for_dedupe("10.1/a") == "10.1/a"

    def test_https_doi_org(self):
        assert AgenticOrchestrator._normalize_doi_for_dedupe("https://doi.org/10.1/a") == "10.1/a"

    def test_http_dx_doi(self):
        assert AgenticOrchestrator._normalize_doi_for_dedupe("http://dx.doi.org/10.1/a") == "10.1/a"

    def test_doi_prefix(self):
        assert AgenticOrchestrator._normalize_doi_for_dedupe("doi:10.1/a") == "10.1/a"

    def test_none(self):
        assert AgenticOrchestrator._normalize_doi_for_dedupe(None) == ""

    def test_empty(self):
        assert AgenticOrchestrator._normalize_doi_for_dedupe("") == ""

    def test_case_insensitive(self):
        result = AgenticOrchestrator._normalize_doi_for_dedupe("HTTPS://DOI.ORG/10.1/A")
        assert result == "10.1/a"


# ---------------------------------------------------------------------------
# AgenticOrchestrator._paper_quality_tuple
# ---------------------------------------------------------------------------


class TestPaperQualityTuple:
    """Tests for paper quality scoring used in dedup."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def test_better_abstract_wins(self):
        orch = self._make_orchestrator()
        low = orch._paper_quality_tuple({"abstract": "short"})
        high = orch._paper_quality_tuple({"abstract": "a" * 500})
        assert high > low

    def test_more_citations_wins(self):
        orch = self._make_orchestrator()
        low = orch._paper_quality_tuple({"cited_by_count": 1})
        high = orch._paper_quality_tuple({"cited_by_count": 100})
        assert high > low

    def test_newer_wins(self):
        orch = self._make_orchestrator()
        low = orch._paper_quality_tuple({"year": 2020})
        high = orch._paper_quality_tuple({"year": 2024})
        assert high > low

    def test_journal_over_biorxiv(self):
        orch = self._make_orchestrator()
        biorxiv = orch._paper_quality_tuple({"doi": "10.1101/2024.01.01"})
        journal = orch._paper_quality_tuple({"doi": "10.1038/s41586-024-12345"})
        assert journal > biorxiv


# ---------------------------------------------------------------------------
# AgenticOrchestrator._get_next_parallel_batch
# ---------------------------------------------------------------------------


class TestGetNextParallelBatch:
    """Tests for step dependency resolution and parallel batching."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _plan(self, steps):
        return Plan(steps=steps, reasoning="test", estimated_steps=len(steps))

    def test_first_step_no_deps(self):
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="s1", type=StepType.KB_SEARCH, description="", tool="kb_search", tool_input={}, depends_on=[]),
            Step(id="s2", type=StepType.ANSWER, description="", depends_on=["s1"]),
        ])
        batch = orch._get_next_parallel_batch(plan, [], {})
        assert len(batch) == 1
        assert batch[0].id == "s1"

    def test_parallel_kb_steps(self):
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search", tool_input={}, depends_on=[]),
            Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search", tool_input={}, depends_on=[]),
            Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1", "kb2"]),
        ])
        batch = orch._get_next_parallel_batch(plan, [], {})
        assert len(batch) == 2
        assert {s.id for s in batch} == {"kb1", "kb2"}

    def test_completed_step_skipped(self):
        orch = self._make_orchestrator()
        s1 = Step(id="s1", type=StepType.KB_SEARCH, description="", tool="kb_search", tool_input={}, depends_on=[])
        plan = self._plan([
            s1,
            Step(id="s2", type=StepType.ANSWER, description="", depends_on=["s1"]),
        ])
        batch = orch._get_next_parallel_batch(plan, [s1], {})
        assert len(batch) == 1
        assert batch[0].id == "s2"

    def test_unmet_deps_not_ready(self):
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="s1", type=StepType.KB_SEARCH, description="", tool="kb_search", tool_input={}, depends_on=[]),
            Step(id="s2", type=StepType.ANSWER, description="", depends_on=["s1"]),
        ])
        # s1 not completed, s2 should not be ready
        batch = orch._get_next_parallel_batch(plan, [], {})
        # s1 is ready (no deps), not s2
        assert len(batch) == 1
        assert batch[0].id == "s1"

    def test_all_completed_returns_empty(self):
        orch = self._make_orchestrator()
        s1 = Step(id="s1", type=StepType.ANSWER, description="")
        plan = self._plan([s1])
        batch = orch._get_next_parallel_batch(plan, [s1], {})
        assert batch == []

    def test_non_search_parallel_not_batched(self):
        """Only parallel search steps (KB + LITERATURE) are batched; other types return first ready."""
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="a1", type=StepType.ANALYZE, description="", depends_on=[]),
            Step(id="a2", type=StepType.ANALYZE, description="", depends_on=[]),
        ])
        batch = orch._get_next_parallel_batch(plan, [], {})
        # Non-search parallel steps: only first ready returned
        assert len(batch) == 1
        assert batch[0].id == "a1"


# ---------------------------------------------------------------------------
# AgenticOrchestrator._get_recent_found_papers
# ---------------------------------------------------------------------------


class TestGetRecentFoundPapers:
    """Tests for structured paper retrieval for quality assessment."""

    def _make_orchestrator(self):
        orch = AgenticOrchestrator.__new__(AgenticOrchestrator)
        orch._found_papers = []
        return orch

    def test_no_papers(self):
        orch = self._make_orchestrator()
        orch._found_papers = []
        result = orch._get_recent_found_papers()
        assert result == []

    def test_no_attribute(self):
        orch = AgenticOrchestrator.__new__(AgenticOrchestrator)
        # No _found_papers attribute at all
        assert not hasattr(orch, "_found_papers")
        result = orch._get_recent_found_papers()
        assert result == []

    def test_returns_papers(self):
        orch = self._make_orchestrator()
        orch._found_papers = [
            {"title": "Paper A", "abstract": "Abstract A", "source": "kb_search", "_step_id": "s1"},
            {"title": "Paper B", "abstract": "Abstract B", "source": "kb_search", "_step_id": "s1"},
        ]
        result = orch._get_recent_found_papers(limit=5)
        assert len(result) == 2
        assert result[0]["title"] == "Paper A"

    def test_step_id_filter(self):
        orch = self._make_orchestrator()
        orch._found_papers = [
            {"title": "Paper A", "abstract": "A", "source": "kb_search", "_step_id": "s1"},
            {"title": "Paper B", "abstract": "B", "source": "literature_search", "_step_id": "s2"},
        ]
        result = orch._get_recent_found_papers(step_ids=["s1"])
        assert len(result) == 1
        assert result[0]["title"] == "Paper A"

    def test_step_id_filter_fallback(self):
        """If step_id filter yields nothing, fall back to all papers."""
        orch = self._make_orchestrator()
        orch._found_papers = [
            {"title": "Paper A", "abstract": "A", "source": "kb_search", "_step_id": "s1"},
        ]
        result = orch._get_recent_found_papers(step_ids=["nonexistent"])
        # Falls back to all papers when filter matches nothing
        assert len(result) == 1

    def test_limit(self):
        orch = self._make_orchestrator()
        orch._found_papers = [
            {"title": f"Paper {i}", "abstract": f"Abstract {i}", "source": "kb_search"}
            for i in range(10)
        ]
        result = orch._get_recent_found_papers(limit=3)
        assert len(result) == 3

    def test_dedup_by_title(self):
        orch = self._make_orchestrator()
        orch._found_papers = [
            {"title": "Same Paper", "abstract": "A", "source": "kb_search"},
            {"title": "Same Paper", "abstract": "B", "source": "kb_search"},
        ]
        result = orch._get_recent_found_papers(limit=5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# AgenticOrchestrator._maybe_upgrade_single_kb_to_composite_parallel
# ---------------------------------------------------------------------------


class TestMaybeUpgradeSingleKbToCompositeParallel:
    """Tests for upgrading a single KB step to parallel composite steps."""

    def _make_orchestrator(self):
        orch = AgenticOrchestrator.__new__(AgenticOrchestrator)
        return orch

    def test_upgrades_single_kb_step(self):
        orch = self._make_orchestrator()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN vs GNPS"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1"]),
            ],
            reasoning="test",
            estimated_steps=2,
        )
        orch._maybe_upgrade_single_kb_to_composite_parallel(plan, "FBMN vs GNPS", "test_kb")
        # Single KB step should be replaced with 2 composite steps
        kb_steps = [s for s in plan.steps if s.type == StepType.KB_SEARCH]
        assert len(kb_steps) == 2
        # Answer step should depend on both new steps
        ans = [s for s in plan.steps if s.type == StepType.ANSWER][0]
        new_ids = {s.id for s in kb_steps}
        assert set(ans.depends_on) == new_ids

    def test_no_upgrade_for_simple_query(self):
        orch = self._make_orchestrator()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "feature-based molecular networking"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1"]),
            ],
            reasoning="test",
            estimated_steps=2,
        )
        orch._maybe_upgrade_single_kb_to_composite_parallel(plan, "feature-based molecular networking", "test_kb")
        # Should NOT upgrade — not a composite query
        kb_steps = [s for s in plan.steps if s.type == StepType.KB_SEARCH]
        assert len(kb_steps) == 1

    def test_no_upgrade_when_multiple_kb_steps(self):
        orch = self._make_orchestrator()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "A"}, depends_on=[]),
                Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "B"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1", "kb2"]),
            ],
            reasoning="test",
            estimated_steps=3,
        )
        orch._maybe_upgrade_single_kb_to_composite_parallel(plan, "A vs B", "test_kb")
        # Already has multiple KB steps, should not change
        kb_steps = [s for s in plan.steps if s.type == StepType.KB_SEARCH]
        assert len(kb_steps) == 2

    def test_preserves_top_k(self):
        orch = self._make_orchestrator()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "A vs B", "top_k": 15}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1"]),
            ],
            reasoning="test",
            estimated_steps=2,
        )
        orch._maybe_upgrade_single_kb_to_composite_parallel(plan, "A vs B", "test_kb")
        kb_steps = [s for s in plan.steps if s.type == StepType.KB_SEARCH]
        for s in kb_steps:
            assert s.tool_input.get("top_k") == 15
