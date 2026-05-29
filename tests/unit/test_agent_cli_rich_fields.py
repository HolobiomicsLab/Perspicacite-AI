"""Tests for F2 + F4 (audit 2026-05-15): agent_cli rich result fields.

- F2: budget tracker receives token + cost data even without a
  provenance collector.
- F4: ``cost_usd_path`` / ``cache_read_tokens_path`` /
  ``cache_creation_tokens_path`` extract the corresponding values
  from the CLI's JSON payload.
"""
from __future__ import annotations

import json

import pytest

from perspicacite.llm.agent_cli import AgentCLIClient

CLAUDE_JSON_PAYLOAD = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 3846,
        "result": "Hello, world.",
        "session_id": "abc",
        "total_cost_usd": 0.02999325,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 229,
            "cache_creation_input_tokens": 23077,
            "cache_read_input_tokens": 12345,
        },
    }
)


def _make_client() -> AgentCLIClient:
    return AgentCLIClient(
        executable="claude",
        provider_label="claude_cli",
        output_format="json",
        result_json_path="result",
        usage_input_tokens_path="usage.input_tokens",
        usage_output_tokens_path="usage.output_tokens",
        cost_usd_path="total_cost_usd",
        cache_read_tokens_path="usage.cache_read_input_tokens",
        cache_creation_tokens_path="usage.cache_creation_input_tokens",
    )


def test_parse_output_full_extracts_text_tokens_cost_and_cache():
    cli = _make_client()
    text, in_t, out_t, details = cli._parse_output_full(CLAUDE_JSON_PAYLOAD)
    assert text == "Hello, world."
    assert in_t == 2
    assert out_t == 229
    assert details["cost_usd"] == pytest.approx(0.02999325)
    assert details["cache_read_tokens"] == 12345
    assert details["cache_creation_tokens"] == 23077


def test_parse_output_full_handles_missing_cost_path():
    cli = AgentCLIClient(
        executable="claude",
        provider_label="claude_cli",
        output_format="json",
        result_json_path="result",
        usage_input_tokens_path="usage.input_tokens",
        usage_output_tokens_path="usage.output_tokens",
        # no cost_usd_path
    )
    text, _in_t, _out_t, details = cli._parse_output_full(CLAUDE_JSON_PAYLOAD)
    assert text == "Hello, world."
    assert details == {}


def test_parse_output_full_handles_malformed_json():
    cli = _make_client()
    _text, in_t, out_t, details = cli._parse_output_full("not json at all")
    assert in_t == 0 and out_t == 0
    assert details == {}


def test_parse_output_with_usage_still_returns_2tuple_for_back_compat():
    """The legacy ``_parse_output_with_usage`` signature must keep
    returning ``(text, in, out)`` — callers expecting three items
    shouldn't break."""
    cli = _make_client()
    out = cli._parse_output_with_usage(CLAUDE_JSON_PAYLOAD)
    assert len(out) == 3
    assert out[0] == "Hello, world."
    assert out[1] == 2
    assert out[2] == 229


@pytest.mark.asyncio
async def test_complete_pushes_cost_to_budget_tracker(monkeypatch, tmp_path):
    """End-to-end: a mocked CLI call lands cost in the BudgetTracker."""
    from perspicacite.llm.budget import BudgetTracker, set_budget_tracker

    cli = _make_client()
    tracker = BudgetTracker(action="warn")
    set_budget_tracker(tracker)

    # Mock asyncio subprocess to return our canned JSON.
    class _FakeProc:
        returncode = 0

        async def communicate(self, input: bytes | None = None):
            return (CLAUDE_JSON_PAYLOAD.encode(), b"")

        async def wait(self):
            return 0

        def kill(self):
            pass

    import asyncio
    async def _fake_exec(*args, **kwargs):
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    try:
        text = await cli.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="sonnet",
            stage="test",
        )
        assert text == "Hello, world."
        s = tracker.summary()
        # summary() rounds to 6 decimal places.
        assert s["usd"] == pytest.approx(0.02999325, abs=1e-6)
        assert s["tokens_in"] == 2
        assert s["tokens_out"] == 229
        # has_unknown_costs must be False — we used record_cost, not the
        # PRICING_TABLE lookup.
        assert s["has_unknown_costs"] is False
    finally:
        set_budget_tracker(None)


@pytest.mark.asyncio
async def test_complete_falls_back_to_token_pricing_when_no_cost_path(monkeypatch):
    """When ``cost_usd_path`` is not configured, agent_cli should still
    feed token counts to the tracker via .record() — the budget tracker
    then estimates cost from PRICING_TABLE."""
    from perspicacite.llm.budget import BudgetTracker, set_budget_tracker

    # No cost_usd_path here.
    cli = AgentCLIClient(
        executable="claude",
        provider_label="claude_cli",
        output_format="json",
        result_json_path="result",
        usage_input_tokens_path="usage.input_tokens",
        usage_output_tokens_path="usage.output_tokens",
    )
    tracker = BudgetTracker(action="warn")
    set_budget_tracker(tracker)

    class _FakeProc:
        returncode = 0
        async def communicate(self, input=None):
            return (CLAUDE_JSON_PAYLOAD.encode(), b"")
        async def wait(self):
            return 0
        def kill(self):
            pass

    import asyncio
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec",
        lambda *a, **k: _make_awaitable_proc(_FakeProc()),
    )

    try:
        text = await cli.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="sonnet",
        )
        assert text == "Hello, world."
        s = tracker.summary()
        assert s["tokens_in"] == 2
        assert s["tokens_out"] == 229
        # has_unknown_costs depends on whether (claude_cli, sonnet) is in
        # PRICING_TABLE. Either way the call should not have raised.
    finally:
        set_budget_tracker(None)


def _make_awaitable_proc(p):
    """Helper: wrap a fake process so monkeypatch returns a coroutine."""
    async def _runner(*a, **k):
        return p
    return _runner()
