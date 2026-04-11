"""Unit tests for Phase 3: Execution engine + answer quality features.

Tests pure functions and dataclass methods that require no LLM access:
- _build_facet_overview: multi-facet prompt rendering
- _evaluate_progress B9 fix: covered_by_remaining gap check
- _get_next_parallel_batch: mixed KB + LIT parallel batching (DAG scheduler)

Run: PYTHONPATH=src pytest tests/unit/test_agentic_phase3.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from perspicacite.rag.agentic.orchestrator import (
    AgenticOrchestrator,
    AgentSession,
    EvidenceFacet,
    EvidenceStore,
)
from perspicacite.rag.agentic.planner import Plan, Step, StepType


# ---------------------------------------------------------------------------
# _build_facet_overview
# ---------------------------------------------------------------------------


class TestBuildFacetOverview:
    """Tests for per-facet overview rendering in the answer prompt."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _make_session(self):
        s = AgentSession.__new__(AgentSession)
        s.evidence = None
        return s

    def test_no_evidence_returns_empty(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        assert orch._build_facet_overview(session) == ""

    def test_single_facet_returns_empty(self):
        """Only useful for composite (multi-facet) queries."""
        orch = self._make_orchestrator()
        session = self._make_session()
        session.evidence = EvidenceStore()
        session.evidence.register_facet("main", "main")
        assert orch._build_facet_overview(session) == ""

    def test_two_facets_returns_overview(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        es.facets["fbmn"].entries = [{"title": "Paper A"}]
        session.evidence = es

        result = orch._build_facet_overview(session)
        assert "Research facets investigated:" in result
        assert '"FBMN"' in result
        assert '"GNPS"' in result
        assert "[GAP]" in result  # GNPS has no entries
        assert "[PARTIAL]" in result  # FBMN has 1 entry

    def test_status_labels_correct(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("gap_facet", "GapTopic")
        es.register_facet("partial_facet", "PartialTopic")
        es.register_facet("covered_facet", "CoveredTopic")
        # gap: 0 entries
        # partial: 1 entry
        es.facets["partial_facet"].entries.append({"title": "P1"})
        # covered: 3+ entries
        for i in range(3):
            es.facets["covered_facet"].entries.append({"title": f"C{i}"})
        session.evidence = es

        result = orch._build_facet_overview(session)
        assert "[GAP]" in result
        assert "[PARTIAL]" in result
        assert "[COVERED]" in result

    def test_entry_count_displayed(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        for i in range(5):
            es.facets["fbmn"].entries.append({"title": f"Paper {i}"})
        session.evidence = es

        result = orch._build_facet_overview(session)
        assert "5 source(s)" in result

    def test_titles_shown_up_to_four(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        for i in range(6):
            es.facets["fbmn"].entries.append({"title": f"FBMN Paper {i}"})
        session.evidence = es

        result = orch._build_facet_overview(session)
        # Only first 4 titles shown per facet
        assert "FBMN Paper 0" in result
        assert "FBMN Paper 3" in result
        assert "FBMN Paper 4" not in result

    def test_titles_truncated_at_80_chars(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("a", "A")
        es.register_facet("b", "B")
        long_title = "X" * 200
        es.facets["a"].entries.append({"title": long_title})
        session.evidence = es

        result = orch._build_facet_overview(session)
        # Title in output should be truncated to 80 chars
        assert "X" * 80 in result
        assert "X" * 81 not in result

    def test_no_entries_shows_none(self):
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("fbmn", "FBMN")
        es.register_facet("gnps", "GNPS")
        # Both have 0 entries
        session.evidence = es

        result = orch._build_facet_overview(session)
        assert "(none)" in result

    def test_empty_title_uses_empty_string(self):
        """Entries without title produce empty string, filtered by 'if t'."""
        orch = self._make_orchestrator()
        session = self._make_session()
        es = EvidenceStore()
        es.register_facet("a", "A")
        es.register_facet("b", "B")
        es.facets["a"].entries.append({"title": ""})
        session.evidence = es

        result = orch._build_facet_overview(session)
        assert "(none)" in result  # empty title filtered out


# ---------------------------------------------------------------------------
# _evaluate_progress — B9 fix: covered_by_remaining
# ---------------------------------------------------------------------------


class TestEvaluateProgressCoveredByRemaining:
    """Tests for the B9 fix: gap facets with remaining steps that already
    target them should NOT trigger replan.
    """

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

    @pytest.mark.asyncio
    async def test_gap_facet_with_remaining_step_targeting_it_continues(self):
        """If remaining plan has a step that targets a gap facet, should NOT replan."""
        orch = self._make_orchestrator()
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
        # FBMN facet gets entries (covered), GNPS stays gap
        for i in range(3):
            session.evidence.facets["fbmn"].entries.append(
                {"title": f"P{i}", "doi": f"10.1/f{i}"}
            )

        orch._found_papers = [
            {"title": "P0", "abstract": "FBMN content", "source": "kb_search", "_step_id": "kb1"},
        ]

        # kb1 completed, kb2 still remaining (targets GNPS gap facet)
        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS", plan, completed_steps=[kb1],
            step_results={"kb1": "FBMN results"}, session=session,
        )
        # Should NOT replan — kb2 already targets the gap
        assert result["decision"] != "replan"

    @pytest.mark.asyncio
    async def test_gap_facet_with_no_remaining_search_steps_continues(self):
        """If no remaining SEARCH steps target a gap facet, the gap check is
        skipped (only ANSWER remains) and the loop lets the answer step run.
        The chat() loop will pick up the ANSWER step next iteration regardless.
        """
        orch = self._make_orchestrator()
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
        # FBMN has no entries → gap
        orch._found_papers = []

        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN", plan, completed_steps=[kb1],
            step_results={"kb1": "sparse results"}, session=session,
        )
        # No remaining search steps → gap check skipped, falls through
        # Quality assessor returns insufficient → but no gaps trigger replan
        # Decision is "continue" which lets the ANSWER step run next
        assert result["decision"] == "continue"

    @pytest.mark.asyncio
    async def test_mixed_gap_and_covered_with_partial_remaining(self):
        """Multiple facets: one covered, one gap with remaining step, one gap without."""
        orch = self._make_orchestrator()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "GNPS"}, depends_on=[]),
                Step(id="kb3", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "metabolomics"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="",
                     depends_on=["kb1", "kb2", "kb3"]),
            ],
            reasoning="test",
            estimated_steps=4,
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)
        # FBMN covered
        for i in range(3):
            session.evidence.facets["fbmn"].entries.append(
                {"title": f"P{i}", "doi": f"10.1/f{i}"}
            )
        # GNPS: gap, but kb2 (remaining) targets it
        # metabolomics: gap, but kb3 (remaining) targets it
        orch._found_papers = [
            {"title": "P0", "abstract": "FBMN content", "source": "kb_search", "_step_id": "kb1"},
        ]

        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS vs metabolomics", plan, completed_steps=[kb1],
            step_results={"kb1": "FBMN results"}, session=session,
        )
        # Both gap facets have remaining steps → should NOT replan
        assert result["decision"] != "replan"

    @pytest.mark.asyncio
    async def test_gap_facet_with_lit_step_targeting_it_continues(self):
        """Remaining LITERATURE_SEARCH step targeting a gap facet also prevents replan."""
        orch = self._make_orchestrator()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="lit1", type=StepType.LITERATURE_SEARCH, description="",
                     tool="literature_search", tool_input={"query": "GNPS"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1", "lit1"]),
            ],
            reasoning="test",
            estimated_steps=3,
        )
        session = self._make_session()
        orch._register_evidence_facets(session, plan)
        # FBMN covered
        for i in range(3):
            session.evidence.facets["fbmn"].entries.append(
                {"title": f"P{i}", "doi": f"10.1/f{i}"}
            )
        # GNPS gap, but lit1 targets it
        orch._found_papers = [
            {"title": "P0", "abstract": "FBMN", "source": "kb_search", "_step_id": "kb1"},
        ]

        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS", plan, completed_steps=[kb1],
            step_results={"kb1": "results"}, session=session,
        )
        assert result["decision"] != "replan"

    @pytest.mark.asyncio
    async def test_uncovered_gap_facet_replans_even_with_other_covered(self):
        """If remaining steps only cover SOME gaps, uncovered ones still trigger replan."""
        orch = self._make_orchestrator()
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
        # FBMN gap, GNPS gap
        # kb2 targets GNPS but NOT FBMN → FBMN is uncovered

        orch._found_papers = []
        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN vs GNPS", plan, completed_steps=[kb1],
            step_results={"kb1": "sparse"}, session=session,
        )
        # FBMN is gap with no remaining step → replan
        assert result["decision"] == "replan"

    @pytest.mark.asyncio
    async def test_no_session_no_crash(self):
        """B9 logic should not crash when session is None."""
        orch = self._make_orchestrator()
        plan = Plan(
            steps=[
                Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                     tool_input={"query": "FBMN"}, depends_on=[]),
                Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1"]),
            ],
            reasoning="test",
            estimated_steps=2,
        )
        kb1 = plan.steps[0]
        result = await orch._evaluate_progress(
            "FBMN", plan, completed_steps=[kb1],
            step_results={"kb1": "results"}, session=None,
        )
        # Should not crash; decision is whatever the logic decides
        assert "decision" in result


# ---------------------------------------------------------------------------
# _get_next_parallel_batch — mixed KB + LIT batching
# ---------------------------------------------------------------------------


class TestParallelBatchMixedTypes:
    """Tests for DAG scheduler batching mixed search step types."""

    def _make_orchestrator(self):
        return AgenticOrchestrator.__new__(AgenticOrchestrator)

    def _plan(self, steps):
        return Plan(steps=steps, reasoning="test", estimated_steps=len(steps))

    def test_mixed_kb_and_lit_parallel(self):
        """KB and LIT steps with no deps should both be in the same batch."""
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "A"}, depends_on=[]),
            Step(id="lit1", type=StepType.LITERATURE_SEARCH, description="",
                 tool="literature_search", tool_input={"query": "B"}, depends_on=[]),
            Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1", "lit1"]),
        ])
        batch = orch._get_next_parallel_batch(plan, [], {})
        assert len(batch) == 2
        assert {s.id for s in batch} == {"kb1", "lit1"}

    def test_three_parallel_search_steps(self):
        """Three search steps (2 KB + 1 LIT) all batched together."""
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "A"}, depends_on=[]),
            Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "B"}, depends_on=[]),
            Step(id="lit1", type=StepType.LITERATURE_SEARCH, description="",
                 tool="literature_search", tool_input={"query": "C"}, depends_on=[]),
            Step(id="ans", type=StepType.ANSWER, description="",
                 depends_on=["kb1", "kb2", "lit1"]),
        ])
        batch = orch._get_next_parallel_batch(plan, [], {})
        assert len(batch) == 3
        assert {s.id for s in batch} == {"kb1", "kb2", "lit1"}

    def test_answer_step_always_alone(self):
        """ANSWER step is never batched with other steps."""
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "A"}, depends_on=[]),
            Step(id="ans", type=StepType.ANSWER, description="", depends_on=["kb1"]),
        ])
        # kb1 completed, only answer remains
        kb1 = plan.steps[0]
        batch = orch._get_next_parallel_batch(plan, [kb1], {})
        assert len(batch) == 1
        assert batch[0].type == StepType.ANSWER

    def test_dep_chain_prevents_premature_batching(self):
        """Steps with unmet deps are not included in the batch."""
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "A"}, depends_on=[]),
            Step(id="lit1", type=StepType.LITERATURE_SEARCH, description="",
                 tool="literature_search", tool_input={"query": "B"}, depends_on=["kb1"]),
            Step(id="ans", type=StepType.ANSWER, description="",
                 depends_on=["kb1", "lit1"]),
        ])
        # Only kb1 should be ready (lit1 depends on kb1)
        batch = orch._get_next_parallel_batch(plan, [], {})
        assert len(batch) == 1
        assert batch[0].id == "kb1"

    def test_second_wave_after_first_completes(self):
        """After first batch completes, second wave becomes ready."""
        orch = self._make_orchestrator()
        plan = self._plan([
            Step(id="kb1", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "A"}, depends_on=[]),
            Step(id="kb2", type=StepType.KB_SEARCH, description="", tool="kb_search",
                 tool_input={"query": "B"}, depends_on=[]),
            Step(id="lit1", type=StepType.LITERATURE_SEARCH, description="",
                 tool="literature_search", tool_input={"query": "C"}, depends_on=["kb1", "kb2"]),
            Step(id="ans", type=StepType.ANSWER, description="",
                 depends_on=["lit1"]),
        ])
        kb1, kb2 = plan.steps[0], plan.steps[1]
        batch = orch._get_next_parallel_batch(plan, [kb1, kb2], {})
        assert len(batch) == 1
        assert batch[0].id == "lit1"
