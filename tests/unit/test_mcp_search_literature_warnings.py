"""Unit tests for search_literature warnings surface + enrichment opt-out."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.mcp import server as mcp_server
from perspicacite.search.query_optimizer import OptimizationResult
from perspicacite.search.scilex_adapter import SciLExAdapter

_OPT_PASSTHROUGH = OptimizationResult(
    searched_query="q", enabled=False, applied=False,
    context_used=False, fallback_reason=None,
)


@pytest.fixture
def app_state():
    state = MagicMock()
    state.initialized = True
    state.config.search.query_optimization.enabled = False
    state.config.search.query_optimization.timeout_s = 5.0
    state.config.search.query_optimization.max_context_chars = 300
    state.config.llm.default_provider = "anthropic"
    state.config.llm.default_model = "claude-haiku-4-5"
    state.config.llm.models = {}
    state.config.llm.providers_per_stage = {}
    state.llm_client = MagicMock()
    state.vector_store = MagicMock()
    with patch.object(mcp_server, "mcp_state", state):
        yield state


def _stub_aggregator_with_scilex(dropped: list[str], papers=None) -> MagicMock:
    """Aggregator with a SciLEx provider that reports dropped_apis."""
    scilex = MagicMock(spec=SciLExAdapter)
    scilex._last_dropped_apis = dropped
    scilex._last_quota_warning = None

    agg = MagicMock()
    agg.available = True
    agg.search = AsyncMock(return_value=papers or [])
    agg._providers = [scilex]
    agg.last_errors_by_database = {}
    return agg


@pytest.mark.asyncio
async def test_search_literature_returns_dropped_apis_warning(app_state):
    """When SciLEx drops google_scholar, response has a warning entry."""
    agg = _stub_aggregator_with_scilex(dropped=["google_scholar"])
    with patch(
        "perspicacite.search.domain_aggregator.build_aggregator",
        return_value=agg,
    ), patch(
        "perspicacite.search.query_optimizer.optimize_query",
        AsyncMock(return_value=_OPT_PASSTHROUGH),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        AsyncMock(side_effect=lambda p: p),
    ):
        raw = await mcp_server.search_literature(
            query="q", databases=["semantic_scholar", "google_scholar"],
        )
    data = json.loads(raw)
    warnings = data.get("warnings", [])
    assert any(
        w.get("kind") == "unknown_apis_dropped" and "google_scholar" in w.get("apis", [])
        for w in warnings
    ), f"Expected unknown_apis_dropped warning, got: {warnings}"


@pytest.mark.asyncio
async def test_search_literature_no_warning_when_all_known(app_state):
    """No warnings when all requested APIs are SciLEx-supported."""
    agg = _stub_aggregator_with_scilex(dropped=[])
    with patch(
        "perspicacite.search.domain_aggregator.build_aggregator",
        return_value=agg,
    ), patch(
        "perspicacite.search.query_optimizer.optimize_query",
        AsyncMock(return_value=_OPT_PASSTHROUGH),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        AsyncMock(side_effect=lambda p: p),
    ):
        raw = await mcp_server.search_literature(query="q")
    data = json.loads(raw)
    assert data.get("warnings", []) == []


@pytest.mark.asyncio
async def test_search_literature_skips_enrich_when_disabled(app_state):
    """When enrich=False, enrich_papers is never called."""
    agg = _stub_aggregator_with_scilex(dropped=[])
    mock_enrich = AsyncMock(side_effect=lambda p: p)
    with patch(
        "perspicacite.search.domain_aggregator.build_aggregator",
        return_value=agg,
    ), patch(
        "perspicacite.search.query_optimizer.optimize_query",
        AsyncMock(return_value=_OPT_PASSTHROUGH),
    ), patch(
        "perspicacite.pipeline.enrichment.crossref_enrich.enrich_papers",
        mock_enrich,
    ):
        await mcp_server.search_literature(query="q", enrich=False)
    mock_enrich.assert_not_called()
