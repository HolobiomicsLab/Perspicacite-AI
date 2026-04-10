"""Unit tests for Phase 2: Faceted evidence store + gap-driven replan.

Tests:
- EvidenceFacet: status, dedup, entry management
- EvidenceStore: facet registration, add_hits, gap_summary, to_prompt_block
- _register_evidence_facets: plan → facet mapping
- _facet_key_for_step: step → facet key resolution
- _accumulate_lit_evidence: literature → facet flow
- _evaluate_progress: gap-driven decision logic (with mocked quality assessor)

Run: PYTHONPATH=src .venv/bin/pytest tests/unit/test_agentic_phase2.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from perspicacite.rag.agentic.orchestrator import (
    AgenticOrchestrator,
    AgentSession,
    EvidenceFacet,
    EvidenceStore,
)
from perspicacite.rag.agentic.planner import Plan, Step, StepType


# ---------------------------------------------------------------------------
# EvidenceFacet
# ---------------------------------------------------------------------------


class TestEvidenceFacet:
    """Tests for single facet status and dedup."""

    def test_status_gap(self):
        f = EvidenceFacet(query="FBMN")
        assert f.status == "gap"

    def test_status_partial(self):
        f = EvidenceFacet(query="FBMN")
        f.entries = [{"title": "A"}, {"title": "B"}]
        assert f.status == "partial"

    def test_status_covered(self):
        f = EvidenceFacet(query="FBMN")
        f.entries = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        assert f.status == "covered"

    def test_entry_key_doi(self):
        f = EvidenceFacet(query="test")
        assert f._entry_key({"doi": "10.1/a", "title": "X"}) == "doi:10.1/a"

    def test_entry_key_title_fallback(self):
        f = EvidenceFacet(query="test")
        key = f._entry_key({"doi": "", "title": "Some Title"})
        assert key.startswith("title:")

    def test_entry_key_empty(self):
        f = EvidenceFacet(query="test")
        assert f._entry_key({}) == ""

    def test_add_entry_dedup(self):
        f = EvidenceFacet(query="test")
        assert f._add_entry({"doi": "10.1/a", "title": "Paper A"})
        assert not f._add_entry({"doi": "10.1/a", "title": "Paper A dup"})
        assert len(f.entries) == 1

    def test_add_entry_different_papers(self):
        f = EvidenceFacet(query="test")
        assert f._add_entry({"doi": "10.1/a", "title": "A"})
        assert f._add_entry({"doi": "10.1/b", "title": "B"})
        assert len(f.entries) == 2


# ---------------------------------------------------------------------------
# EvidenceStore — facet management
# ---------------------------------------------------------------------------


class TestEvidenceStoreFacets:
    """Tests for facet registration and lookup."""

    def test_register_facet_new(self):
        es = EvidenceStore()
        f = es.register_facet("fbmn", "FBMN")
        assert "fbmn" in es.facets
        assert f.query == "FBMN"

    def test_register_facet_idempotent(self):
        es = EvidenceStore()
        f1 = es.register_facet("fbmn", "FBMN")
        f2 = es.register_facet("fbmn", "FBMN again")
        assert f1 is f2
        assert len(es.facets) == 1

    def test_facet_for_step_found(self):
        es = EvidenceStore()
        f = es.register_facet("fbmn", "FBMN")
        f.step_ids.append("step1")
        assert es.facet_for_step("step1") is f

    def test_facet_for_step_not_found(self):
        es = EvidenceStore()
        assert es.facet_for_step("nonexistent") is None

    def test_all_entries(self):
        es = EvidenceStore()
        es.register_facet("a", "A")
        es.register_facet("b", "B")
        es.facets["a"].entries = [{"title": "Paper A"}]
        es.facets["b"].entries = [{"title": "Paper B"}]
        entries = es.all_entries
        assert len(entries) == 2

    def test_gap_summary(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        es.facets["fbmn"].entries = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        es.facets["gnps"].entries = []  # gap
        summary = es.gap_summary()
        assert summary["fbmn"] == "covered"
        assert summary["gnps"] == "gap"


# ---------------------------------------------------------------------------
# EvidenceStore — add_hits with global dedup
# ---------------------------------------------------------------------------


class TestEvidenceStoreAddHits:
    """Tests for cross-facet evidence routing and dedup."""

    def test_add_hits_to_facet(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.add_hits(
            [{"title": "Paper A", "doi": "10.1/a"}],
            step_id="step1",
            facet_key="fbmn",
        )
        assert len(es.facets["fbmn"].entries) == 1
        assert "step1" in es.facets["fbmn"].step_ids

    def test_add_hits_auto_creates_facet(self):
        es = EvidenceStore()
        es.add_hits(
            [{"title": "Paper A", "doi": "10.1/a"}],
            step_id="step1",
            facet_key="unknown_facet",
        )
        assert "unknown_facet" in es.facets

    def test_cross_facet_same_paper_allowed(self):
        """Same paper can appear in multiple facets (per-facet dedup only)."""
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        paper = {"title": "Review Paper", "doi": "10.1/review"}

        es.add_hits([paper], step_id="s1", facet_key="fbmn")
        es.add_hits([paper], step_id="s2", facet_key="gnps")

        # Per-facet dedup only — paper enters both facets
        assert len(es.facets["fbmn"].entries) == 1
        assert len(es.facets["gnps"].entries) == 1

    def test_step_id_recorded(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.add_hits([], step_id="step1", facet_key="fbmn")
        assert "step1" in es.facets["fbmn"].step_ids

    def test_step_id_not_duplicated(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.add_hits([], step_id="step1", facet_key="fbmn")
        es.add_hits([], step_id="step1", facet_key="fbmn")
        assert es.facets["fbmn"].step_ids.count("step1") == 1

    def test_add_kb_hits_backward_compat(self):
        es = EvidenceStore()
        es.add_kb_hits([{"title": "A", "doi": "10.1/a"}], step_id="s1", facet_key="main")
        assert len(es.facets["main"].entries) == 1


# ---------------------------------------------------------------------------
# EvidenceStore — to_prompt_block (faceted)
# ---------------------------------------------------------------------------


class TestEvidenceToPromptBlock:
    """Tests for per-facet prompt rendering."""

    def test_empty_store(self):
        es = EvidenceStore()
        assert es.to_prompt_block() == ""

    def test_single_facet_gap(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        block = es.to_prompt_block()
        assert "[GAP]" in block
        assert "FBMN" in block
        assert "(no evidence yet)" in block

    def test_single_facet_covered(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        for i in range(3):
            es.facets["fbmn"].entries.append({"title": f"Paper {i}", "doi": f"10.1/{i}"})
        block = es.to_prompt_block()
        assert "[COVERED]" in block
        assert "Paper 0" in block

    def test_partial_facet(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.facets["fbmn"].entries.append({"title": "Paper A"})
        block = es.to_prompt_block()
        assert "[PARTIAL]" in block

    def test_multiple_facets(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        es.facets["fbmn"].entries = [{"title": "A"}] * 3  # covered
        block = es.to_prompt_block()
        assert "[COVERED]" in block
        assert "[GAP]" in block

    def test_truncation(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        for i in range(20):
            es.facets["fbmn"].entries.append(
                {"title": f"Paper {i}", "doi": f"10.1/{i}", "excerpt": "x" * 300}
            )
        block = es.to_prompt_block(max_chars=500)
        assert len(block) <= 504
        assert block.endswith("…")

    def test_max_entries_per_facet(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        for i in range(20):
            es.facets["fbmn"].entries.append({"title": f"Paper {i}", "doi": f"10.1/{i}"})
        block = es.to_prompt_block(max_entries_per_facet=3)
        # Should show at most 3 paper entries
        assert block.count("- Paper") == 3

    def test_includes_doi_in_output(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.facets["fbmn"].entries.append({"title": "Paper A", "doi": "10.1/a"})
        block = es.to_prompt_block()
        assert "DOI: 10.1/a" in block

    def test_includes_excerpt(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.facets["fbmn"].entries.append({"title": "Paper A", "excerpt": "Key finding here"})
        block = es.to_prompt_block()
        assert "Key finding here" in block


# ---------------------------------------------------------------------------
# _register_evidence_facets
# ---------------------------------------------------------------------------


class TestRegisterEvidenceFacets:
    """Tests for plan → facet mapping."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _make_session(self):
        s = AgentSession.__new__(AgentSession)
        s.evidence = None
        return s

    def test_registers_search_steps(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "GNPS"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1", "kb2"]),
            ],
            reasoning="test",
            estimated_steps=3,
        )
        orch._register_evidence_facets(session, plan)
        assert len(session.evidence.facets) == 2

    def test_skips_non_search_steps(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        plan = Plan(
            steps=[
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=[]),
            ],
            reasoning="test",
            estimated_steps=1,
        )
        orch._register_evidence_facets(session, plan)
        # No search steps → "main" fallback facet
        assert "main" in session.evidence.facets

    def test_step_ids_recorded(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
            ],
            reasoning="test",
            estimated_steps=1,
        )
        orch._register_evidence_facets(session, plan)
        key = "fbmn"
        assert "kb1" in session.evidence.facets[key].step_ids

    def test_same_query_shares_facet(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
            ],
            reasoning="test",
            estimated_steps=2,
        )
        orch._register_evidence_facets(session, plan)
        assert len(session.evidence.facets) == 1
        facet = list(session.evidence.facets.values())[0]
        assert "kb1" in facet.step_ids
        assert "kb2" in facet.step_ids

    def test_idempotent_reregistration(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
            ],
            reasoning="test",
            estimated_steps=1,
        )
        orch._register_evidence_facets(session, plan)
        orch._register_evidence_facets(session, plan)
        assert len(session.evidence.facets) == 1


# ---------------------------------------------------------------------------
# _facet_key_for_step
# ---------------------------------------------------------------------------


class TestFacetKeyForStep:
    """Tests for step → facet key resolution."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _make_session(self):
        s = AgentSession.__new__(AgentSession)
        s.evidence = None
        return s

    def test_resolves_registered_facet(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
            ],
            reasoning="test",
            estimated_steps=1,
        )
        orch._register_evidence_facets(session, plan)
        step = plan.steps[0]
        key = orch._facet_key_for_step(session, step)
        assert key == "fbmn"

    def test_fallback_to_step_query(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        session.evidence = EvidenceStore()
        step = Step(id="unknown", type=StepType.KB_SEARCH, description="",
                    tool="kb_search", tool_input={"query": "GNPS"}, depends_on=[])
        key = orch._facet_key_for_step(session, step)
        assert key == "gnps"

    def test_fallback_main_for_empty_query(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        session.evidence = EvidenceStore()
        step = Step(id="s1", type=StepType.KB_SEARCH, description="",
                    tool="kb_search", tool_input={}, depends_on=[])
        key = orch._facet_key_for_step(session, step)
        assert key == "main"


# ---------------------------------------------------------------------------
# _accumulate_lit_evidence
# ---------------------------------------------------------------------------


class TestAccumulateLitEvidence:
    """Tests for literature search → facet flow."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _make_session(self):
        s = AgentSession.__new__(AgentSession)
        s.evidence = None
        return s

    def test_accumulates_to_matching_facet(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        # Register a facet for step "lit1"
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.facets["fbmn"].step_ids.append("lit1")
        session.evidence = es

        papers = [
            {"title": "Paper A", "doi": "10.1/a", "abstract": "About FBMN"},
        ]
        orch._accumulate_lit_evidence(papers, step_id="lit1", session=session)
        assert len(es.facets["fbmn"].entries) == 1

    def test_no_session_noop(self):
        orch = self._make_orchestrator()
        orch._accumulate_lit_evidence([{"title": "A"}], step_id="s1", session=None)

    def test_no_evidence_noop(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        session.evidence = None
        orch._accumulate_lit_evidence([{"title": "A"}], step_id="s1", session=session)

    def test_empty_papers_noop(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        session.evidence = EvidenceStore()
        orch._accumulate_lit_evidence([], step_id="s1", session=session)

    def test_extracts_abstract_as_excerpt(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.facets["fbmn"].step_ids.append("lit1")
        session.evidence = es

        papers = [{"title": "Paper A", "doi": "", "abstract": "A" * 800}]
        orch._accumulate_lit_evidence(papers, step_id="lit1", session=session)
        entry = es.facets["fbmn"].entries[0]
        assert len(entry["excerpt"]) == 600  # truncated to 600


# ---------------------------------------------------------------------------
# _evaluate_progress — gap-driven decisions
# ---------------------------------------------------------------------------


class TestEvaluateProgress:
    """Tests for gap-driven evaluation with mocked quality assessor."""

    def _make_orchestrator(self):
        orch = AgenticOrchestrator.__new__(AgenticOrchestrator)
        orch.early_exit_confidence = 0.85
        orch.quality_assessor = MagicMock()
        orch.quality_assessor.assess = AsyncMock(return_value=(False, [], 0.0))
        return orch

    def _make_session(self):
        s = AgentSession.__new__(AgentSession)
        s.evidence = None
        return s

    def _plan_with_answer(self, *search_steps):
        steps = list(search_steps) + [
            Step(id="ans", type=StepType.ANSWER, description="", depends_on=[]),
        ]
        return Plan(steps=steps, reasoning="test", estimated_steps=len(steps))

    @pytest.mark.asyncio
    async def test_no_remaining_steps_returns_answer(self):
        orch = self._make_orchestrator()
        s1 = Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                  tool_input={"query": "FBMN"}, depends_on=[])
        plan = self._plan_with_answer(s1)
        session = self._make_session()
        orch._register_evidence_facets(session, plan)

        result = await orch._evaluate_progress(
            "test query", plan, completed_steps=plan.steps, step_results={}, session=session,
        )
        assert result["decision"] == "answer"

    @pytest.mark.asyncio
    async def test_gap_facets_with_remaining_search_triggers_replan(self):
        orch = self._make_orchestrator()
        # Plan with 2 search steps + answer; only kb1 completed, kb2 still remaining
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "GNPS"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1", "kb2"]),
            ],
            reasoning="test",
            estimated_steps=3,
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)
        # FBMN facet has no entries → gap; GNPS facet also gap

        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS", plan, completed_steps=[kb1], step_results={}, session=session,
        )
        assert result["decision"] == "replan"
        assert len(result["gap_facets"]) > 0

    @pytest.mark.asyncio
    async def test_covered_facets_no_replan(self):
        orch = self._make_orchestrator()
        orch.quality_assessor.assess = AsyncMock(return_value=(True, [], 0.90))
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1"]),
            ],
            reasoning="test",
            estimated_steps=2,
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)
        # Add enough entries to make FBMN "covered"
        for i in range(3):
            session.evidence.facets["fbmn"].entries.append(
                {"title": f"Paper {i}", "doi": f"10.1/{i}"}
            )
        orch._found_papers = [
            {"title": "Paper 0", "abstract": "about FBMN", "source": "kb_search", "_step_id": "kb1"},
        ]

        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN", plan, completed_steps=[kb1], step_results={"kb1": "some result"},
            session=session,
        )
        # No gap facets, quality assessor says sufficient → answer
        assert result["decision"] == "answer"

    @pytest.mark.asyncio
    async def test_quality_sufficient_but_facets_not_covered_continues(self):
        orch = self._make_orchestrator()
        orch.quality_assessor.assess = AsyncMock(return_value=(True, [], 0.90))
        # 3 search steps: kb1 (FBMN), kb2 (GNPS), lit1 (fallback) — lit1 is remaining
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "GNPS"}, depends_on=[]),
                Step(id="lit1", type=StepType.LITERATURE_SEARCH, description="",
                     tool="literature_search", tool_input={"query": "GNPS"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1", "kb2", "lit1"]),
            ],
            reasoning="test",
            estimated_steps=4,
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)
        # FBMN covered, GNPS gap
        for i in range(3):
            session.evidence.facets["fbmn"].entries.append({"title": f"P {i}", "doi": f"10.1/f{i}"})

        orch._found_papers = [
            {"title": "Paper FBMN", "abstract": "about FBMN", "source": "kb_search", "_step_id": "kb1"},
        ]
        # kb1 and kb2 completed, lit1 still remaining
        kb1, kb2 = plan.steps[0], plan.steps[1]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS", plan, completed_steps=[kb1, kb2],
            step_results={"kb1": "FBMN results", "kb2": "GNPS sparse"}, session=session,
        )
        # Quality says sufficient but GNPS facet is gap → should NOT answer
        assert result["decision"] != "answer"

    @pytest.mark.asyncio
    async def test_quality_insufficient_with_gaps_replans(self):
        orch = self._make_orchestrator()
        orch.quality_assessor.assess = AsyncMock(
            return_value=(False, ["missing GNPS data"], 0.4)
        )
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1"]),
            ],
            reasoning="test",
            estimated_steps=2,
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)
        orch._found_papers = [
            {"title": "P1", "abstract": "about FBMN", "source": "kb_search", "_step_id": "kb1"},
        ]

        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS", plan, completed_steps=[kb1],
            step_results={"kb1": "FBMN results"}, session=session,
        )
        assert result["decision"] == "replan"
        assert len(result["missing_aspects"]) > 0

    @pytest.mark.asyncio
    async def test_returns_structured_dict(self):
        orch = self._make_orchestrator()
        plan = self._plan_with_answer(
            Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "FBMN"}, depends_on=[]),
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)

        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "test", plan, completed_steps=[kb1], step_results={}, session=session,
        )
        assert "decision" in result
        assert "gap_facets" in result
        assert "missing_aspects" in result
        assert "evaluation_text" in result

    @pytest.mark.asyncio
    async def test_substantial_results_answer_fallback(self):
        """When all facets covered + quality sufficient + 3+ steps → answer."""
        orch = self._make_orchestrator()
        orch.quality_assessor.assess = AsyncMock(return_value=(True, [], 0.90))
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "GNPS"}, depends_on=[]),
                Step(id="lit1", type=StepType.LITERATURE_SEARCH, description="",
                     tool="literature_search", tool_input={"query": "FBMN GNPS"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="",
                     depends_on=["kb1", "kb2", "lit1"]),
            ],
            reasoning="test",
            estimated_steps=4,
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)
        # All facets covered
        for key in session.evidence.facets:
            for i in range(3):
                session.evidence.facets[key].entries.append(
                    {"title": f"P{i}", "doi": f"10.1/{key}/{i}"}
                )
        orch._found_papers = [
            {"title": "P1", "abstract": "FBMN content", "source": "kb_search", "_step_id": "kb1"},
            {"title": "P2", "abstract": "GNPS content", "source": "kb_search", "_step_id": "kb2"},
        ]

        # 3 steps completed with substantial results, only answer remaining
        step_results = {
            "kb1": "Found 5 papers about FBMN with detailed content...",
            "kb2": "Found 3 papers about GNPS with analysis...",
            "lit1": "Found 8 papers in literature search...",
        }
        completed = plan.steps[:3]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS", plan, completed_steps=completed,
            step_results=step_results, session=session,
        )
        # Only answer step remains → no remaining steps check should trigger
        # Actually remaining_steps = [ans], so no remaining search → falls through
        # All facets covered, quality sufficient, 3+ completed → answer
        assert result["decision"] == "answer"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPhase2EdgeCases:
    """Edge cases and integration-style tests for Phase 2."""

    def test_empty_facet_query_uses_main(self):
        es = EvidenceStore()
        es.add_hits([{"title": "A"}], step_id="s1", facet_key="")
        # Empty facet key should still create a facet
        assert len(es.facets) >= 1

    def test_facet_key_case_insensitive(self):
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.add_hits([{"title": "A", "doi": "10.1/a"}], step_id="s1", facet_key="FBMN")
        # Keys are lowercased in _register_evidence_facets but not in add_hits
        # "FBMN" != "fbmn" → creates separate facet
        assert "FBMN" in es.facets
        assert "fbmn" in es.facets

    def test_prompt_block_handles_missing_fields(self):
        es = EvidenceStore()
        es.register_facet("test", "test")
        es.facets["test"].entries.append({})  # No title, doi, excerpt
        block = es.to_prompt_block()
        assert "?" in block  # Default title is "?"

    def test_full_lifecycle(self):
        """Simulate a complete query lifecycle: register → add → gap check → replan."""
        es = EvidenceStore()

        # Register facets for composite query
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")

        # Add hits for FBMN only
        for i in range(3):
            es.add_hits(
                [{"title": f"FBMN Paper {i}", "doi": f"10.1/f{i}"}],
                step_id="kb1",
                facet_key="fbmn",
            )

        # Check gap status
        summary = es.gap_summary()
        assert summary["fbmn"] == "covered"
        assert summary["gnps"] == "gap"

        # Prompt block shows both facets
        block = es.to_prompt_block()
        assert "[COVERED]" in block
        assert "[GAP]" in block
