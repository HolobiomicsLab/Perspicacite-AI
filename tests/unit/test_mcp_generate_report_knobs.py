"""Tests for new knobs on generate_report MCP tool.

Verifies that the four new knobs (``screen_method``, ``screen_threshold``,
``max_papers_to_download``, ``databases``) plumbed through the MCP tool
arrive on the RAGRequest, are clamped/validated, and default to None.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import perspicacite.mcp.server as mcp_server
from perspicacite.mcp.server import MCPState, generate_report
from perspicacite.models.rag import RAGMode, StreamEvent
from perspicacite.rag.engine import RAGEngine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _real_config() -> MagicMock:
    cfg = MagicMock()
    cfg.llm.default_provider = "deepseek"
    cfg.llm.default_model = "deepseek-chat"
    return cfg


def _kb_meta(name: str = "kb", model: str = "text-embedding-3-small") -> SimpleNamespace:
    return SimpleNamespace(name=name, embedding_model=model, collection_name=f"coll_{name}")


def _make_state() -> MCPState:
    state = MCPState()
    state.initialized = True
    state.config = _real_config()
    state.llm_client = MagicMock()
    state.embedding_provider = MagicMock()
    state.vector_store = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None
    meta = _kb_meta("kb")
    ss_mock = MagicMock()
    ss_mock.get_kb_metadata = AsyncMock(return_value=meta)
    state.session_store = ss_mock
    return state


def _install_capturing_engine(captured: list[Any]):
    """Swap the module-level RAGEngine for a capturing subclass.

    Returns a cleanup callable.
    """

    class _CapturingRAGEngine(RAGEngine):
        async def query_stream(
            self, req, *, message_id=None, conversation_id=None
        ) -> AsyncIterator[StreamEvent]:
            captured.append(req)
            yield StreamEvent(event="content", data='{"delta": "ok"}')
            yield StreamEvent(event="done", data="{}")

    import perspicacite.rag.engine as _engine_mod

    original_cls = _engine_mod.RAGEngine
    _engine_mod.RAGEngine = _CapturingRAGEngine  # type: ignore[assignment]

    def _cleanup() -> None:
        _engine_mod.RAGEngine = original_cls

    return _cleanup


@pytest.mark.asyncio
async def test_knobs_pass_through_to_rag_request() -> None:
    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_capturing_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            await generate_report(
                query="x",
                kb_name="kb",
                mode="advanced",
                screen_method="rerank",
                screen_threshold=0.4,
                max_papers_to_download=12,
                databases=["arxiv", "crossref"],
            )
    finally:
        cleanup()

    assert captured, "query_stream was never invoked"
    req = captured[0]
    assert getattr(req, "screen_method", None) == "rerank"
    assert getattr(req, "screen_threshold", None) == 0.4
    assert getattr(req, "max_papers_to_download", None) == 12
    assert getattr(req, "databases", None) == ["arxiv", "crossref"]


@pytest.mark.asyncio
async def test_invalid_threshold_is_clamped() -> None:
    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_capturing_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            await generate_report(
                query="x",
                kb_name="kb",
                screen_threshold=1.5,
                max_papers_to_download=999,
            )
    finally:
        cleanup()

    assert captured, "query_stream was never invoked"
    req = captured[0]
    assert req.screen_threshold == 1.0
    assert req.max_papers_to_download == 50


@pytest.mark.asyncio
async def test_unknown_databases_dropped() -> None:
    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_capturing_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            await generate_report(
                query="x",
                kb_name="kb",
                databases=["arxiv", "nonsense_db", "pubmed"],
            )
    finally:
        cleanup()

    assert captured, "query_stream was never invoked"
    req = captured[0]
    assert "arxiv" in req.databases
    assert "pubmed" in req.databases
    assert "nonsense_db" not in req.databases


@pytest.mark.asyncio
async def test_default_knobs_are_none() -> None:
    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_capturing_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            await generate_report(query="x", kb_name="kb")
    finally:
        cleanup()

    assert captured, "query_stream was never invoked"
    req = captured[0]
    assert getattr(req, "screen_method", None) is None
    assert getattr(req, "screen_threshold", None) is None
    assert getattr(req, "max_papers_to_download", None) is None
    assert getattr(req, "databases", None) is None


@pytest.mark.asyncio
async def test_agentic_mode_routes_to_agentic_rag_mode() -> None:
    """Regression guard: mode="agentic" must resolve to RAGMode.AGENTIC.

    The AGENTIC mode delegates to the AgenticOrchestrator; this verifies the
    string→enum wiring on the MCP boundary stays intact so the orchestrator
    path remains reachable from generate_report.
    """
    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_capturing_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            await generate_report(query="x", kb_name="kb", mode="agentic")
    finally:
        cleanup()

    assert captured, "query_stream was never invoked"
    req = captured[0]
    assert req.mode == RAGMode.AGENTIC


@pytest.mark.asyncio
async def test_unknown_screen_method_resets_to_none() -> None:
    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_capturing_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            await generate_report(
                query="x",
                kb_name="kb",
                screen_method="invalid_method_xyz",
            )
    finally:
        cleanup()

    assert captured, "query_stream was never invoked"
    req = captured[0]
    # Unknown methods are reset to None (fall-back to server default).
    assert req.screen_method is None


def _install_telemetry_emitting_engine(captured: list[Any]):
    """Capturing engine that also drives the request's telemetry_sink.

    Used to verify that ResponseMetadataCollector aggregates events emitted
    during the RAG run and merges them into the final JSON response.
    """

    class _EmittingRAGEngine(RAGEngine):
        async def query_stream(
            self, req, *, message_id=None, conversation_id=None
        ) -> AsyncIterator[StreamEvent]:
            captured.append(req)
            sink = getattr(req, "telemetry_sink", None)
            if sink is not None:
                sink.append({"kind": "tokens", "in": 100, "out": 50})
                sink.append(
                    {
                        "kind": "cost_estimate",
                        "usd": 0.01,
                        "model": "deepseek/deepseek-chat",
                    }
                )
                sink.append(
                    {
                        "kind": "query_rephrased",
                        "original": "x",
                        "rewritten": "x refined",
                        "reason": "expansion",
                    }
                )
                sink.append(
                    {
                        "kind": "provider_progress",
                        "phase": "done",
                        "query": "x refined",
                        "by_provider": {"arxiv": 3},
                        "total": 3,
                    }
                )
            yield StreamEvent(event="content", data='{"delta": "ok"}')
            yield StreamEvent(event="done", data="{}")

    import perspicacite.rag.engine as _engine_mod

    original_cls = _engine_mod.RAGEngine
    _engine_mod.RAGEngine = _EmittingRAGEngine  # type: ignore[assignment]

    def _cleanup() -> None:
        _engine_mod.RAGEngine = original_cls

    return _cleanup


@pytest.mark.asyncio
async def test_generate_report_embeds_usage_in_response() -> None:
    """End-to-end: telemetry events surface as response extras."""
    import json as _json

    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_telemetry_emitting_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            raw = await generate_report(query="x", kb_name="kb")
    finally:
        cleanup()

    payload = _json.loads(raw)
    assert payload.get("success") is True
    # Usage aggregated from tokens + cost_estimate events.
    assert payload.get("usage", {}).get("tokens_in") == 100
    assert payload["usage"]["tokens_out"] == 50
    assert payload["usage"]["model"] == "deepseek/deepseek-chat"
    assert payload["usage"]["cost_usd_estimate"] == pytest.approx(0.01, rel=1e-3)
    # Query rephrasing surfaced.
    assert payload.get("query_rephrasings") == [
        {"original": "x", "refined": "x refined", "reason": "expansion"}
    ]
    # Attempts surfaced.
    assert payload.get("attempts") == [
        {"query": "x refined", "provider_counts": {"arxiv": 3}, "hit_count": 3}
    ]


@pytest.mark.asyncio
async def test_generate_report_omits_extras_when_no_telemetry() -> None:
    """No telemetry events → no spurious keys in response."""
    import json as _json

    state = _make_state()
    captured: list[Any] = []
    cleanup = _install_capturing_engine(captured)
    try:
        with patch.object(mcp_server, "mcp_state", state):
            raw = await generate_report(query="x", kb_name="kb")
    finally:
        cleanup()

    payload = _json.loads(raw)
    assert payload.get("success") is True
    assert "usage" not in payload
    assert "attempts" not in payload
    assert "query_rephrasings" not in payload
