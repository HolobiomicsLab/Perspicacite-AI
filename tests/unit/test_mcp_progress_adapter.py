"""Unit tests for MCPProgressAdapter."""
from unittest.mock import AsyncMock

import pytest

from perspicacite.mcp.progress_adapter import MCPProgressAdapter


@pytest.mark.asyncio
async def test_query_rephrased_event():
    ctx = AsyncMock()
    adapter = MCPProgressAdapter(ctx)
    await adapter.on_event({
        "kind": "query_rephrased",
        "original": "what is X",
        "rewritten": "X",
    })
    ctx.report_progress.assert_called_once()
    args = ctx.report_progress.call_args.kwargs
    assert "Rewrote search query" in args["message"]


@pytest.mark.asyncio
async def test_batch_progress_updates_counters():
    ctx = AsyncMock()
    adapter = MCPProgressAdapter(ctx)
    await adapter.on_event({
        "kind": "batch_progress",
        "stage": "abstract_analysis",
        "current": 3, "total": 10,
    })
    args = ctx.report_progress.call_args.kwargs
    assert args["progress"] == 3
    assert args["total"] == 10
    assert "abstract_analysis: 3/10" in args["message"]


@pytest.mark.asyncio
async def test_throttling_drops_rapid_events(monkeypatch):
    ctx = AsyncMock()
    adapter = MCPProgressAdapter(ctx)
    # First event passes through.
    await adapter.on_event({
        "kind": "query_rephrased", "original": "a", "rewritten": "b",
    })
    # Second event same instant — must be dropped.
    await adapter.on_event({
        "kind": "query_rephrased", "original": "c", "rewritten": "d",
    })
    assert ctx.report_progress.call_count == 1


@pytest.mark.asyncio
async def test_unknown_kind_silently_ignored():
    ctx = AsyncMock()
    adapter = MCPProgressAdapter(ctx)
    await adapter.on_event({"kind": "not_a_real_event"})
    ctx.report_progress.assert_not_called()


@pytest.mark.asyncio
async def test_ctx_error_swallowed():
    ctx = AsyncMock()
    ctx.report_progress.side_effect = RuntimeError("transport down")
    adapter = MCPProgressAdapter(ctx)
    # Must not raise.
    await adapter.on_event({
        "kind": "query_rephrased", "original": "a", "rewritten": "b",
    })

"""Coverage extension for phase_progress / tokens / cost_estimate events."""
import json

import pytest


class _Ctx:
    def __init__(self):
        self.calls = []

    async def report_progress(self, *, progress, total, message):
        self.calls.append((progress, total, message))


async def _wait_throttle(adapter):
    """Force the adapter to bypass its 1-second min-spacing for tests."""
    adapter._last_emit_t = 0.0


async def test_phase_progress_emits_human_and_meta():
    ctx = _Ctx()
    adapter = MCPProgressAdapter(ctx)
    await _wait_throttle(adapter)

    await adapter.on_event(
        {"kind": "phase_progress", "phase": "retrieve", "state": "running"}
    )

    assert len(ctx.calls) == 1
    _, _, msg = ctx.calls[0]
    assert "retrieve" in msg.lower()
    assert "running" in msg.lower()
    # META JSON tail for structured-data consumers
    assert "\nMETA:" in msg
    meta = json.loads(msg.split("\nMETA:", 1)[1])
    assert meta == {"kind": "phase_progress", "phase": "retrieve", "state": "running"}


async def test_tokens_emits_running_totals():
    ctx = _Ctx()
    adapter = MCPProgressAdapter(ctx)
    await _wait_throttle(adapter)

    await adapter.on_event(
        {
            "kind": "tokens",
            "in": 1200,
            "out": 350,
            "cumulative_in": 5400,
            "cumulative_out": 1100,
        }
    )

    assert len(ctx.calls) == 1
    _, _, msg = ctx.calls[0]
    assert "tokens" in msg.lower()
    meta = json.loads(msg.split("\nMETA:", 1)[1])
    assert meta["in"] == 1200
    assert meta["cumulative_out"] == 1100


async def test_cost_estimate_emits_usd():
    ctx = _Ctx()
    adapter = MCPProgressAdapter(ctx)
    await _wait_throttle(adapter)

    await adapter.on_event(
        {
            "kind": "cost_estimate",
            "usd": 0.034,
            "model": "deepseek/deepseek-chat",
        }
    )

    assert len(ctx.calls) == 1
    _, _, msg = ctx.calls[0]
    assert "0.034" in msg or "$0.034" in msg
    meta = json.loads(msg.split("\nMETA:", 1)[1])
    assert meta["usd"] == 0.034
    assert meta["model"] == "deepseek/deepseek-chat"


async def test_unknown_event_kind_is_silent():
    ctx = _Ctx()
    adapter = MCPProgressAdapter(ctx)
    await _wait_throttle(adapter)

    await adapter.on_event({"kind": "this_is_not_a_real_event", "data": 42})
    assert ctx.calls == []
