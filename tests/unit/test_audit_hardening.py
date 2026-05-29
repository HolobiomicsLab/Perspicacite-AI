"""Tests for the F1 / F3 / F9 fixes from the 2026-05-15 audit.

- F1: AsyncLLMClient must NOT retry on AuthError (deterministic-fail).
- F3: ``suggested_action`` distinguishes missing/invalid keys from
  quota-exceeded.
- F9: ``litellm.suppress_debug_info`` is set at module load.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# F3 — suggested_action hint plumbing
# ---------------------------------------------------------------------------

def test_suggested_action_missing_key_does_not_say_wait_for_quota():
    from perspicacite.llm.errors import suggested_action
    msg = suggested_action("anthropic", hint="missing_or_invalid_key")
    assert "wait" not in msg.lower()
    assert "quota" not in msg.lower()
    assert "api key" in msg.lower()
    assert "missing or invalid" in msg.lower()


def test_suggested_action_quota_keeps_wait_wording():
    from perspicacite.llm.errors import suggested_action
    msg = suggested_action("anthropic", hint="quota_exceeded")
    # anthropic falls through _SUGGESTED_ACTIONS — the existing message
    # already says something about quota / reset.
    assert "quota" in msg.lower() or "reset" in msg.lower() or "rate" in msg.lower()


def test_suggested_action_unknown_hint_falls_through():
    from perspicacite.llm.errors import suggested_action
    a = suggested_action("anthropic")
    b = suggested_action("anthropic", hint="unknown")
    c = suggested_action("anthropic", hint=None)
    assert a == b == c


# ---------------------------------------------------------------------------
# F3 — _auth_hint sniffer
# ---------------------------------------------------------------------------

def test_auth_hint_detects_invalid_key():
    from perspicacite.llm.client import _auth_hint
    cases = [
        'AnthropicException - {"error":{"message":"invalid x-api-key"}}',
        "OpenAI: invalid api key",
        "Unauthorized: missing api key in request",
        "api key not found",
    ]
    for c in cases:
        assert _auth_hint(c) == "missing_or_invalid_key", c


def test_auth_hint_detects_quota():
    from perspicacite.llm.client import _auth_hint
    cases = [
        "credit balance is too low",
        "billing limit reached",
        "usage limit exceeded",
        "monthly quota exhausted",
    ]
    for c in cases:
        assert _auth_hint(c) == "quota_exceeded", c


def test_auth_hint_unknown_falls_back():
    from perspicacite.llm.client import _auth_hint
    assert _auth_hint("some other 401 reason") == "unknown"


# ---------------------------------------------------------------------------
# F1 — deterministic-fail predicate
# ---------------------------------------------------------------------------

def test_is_deterministic_fail_for_auth_error():
    from perspicacite.llm.client import _is_deterministic_fail
    from perspicacite.llm.errors import AuthError

    assert _is_deterministic_fail(AuthError("x", provider="anthropic"))


def test_is_deterministic_fail_for_budget_breach():
    from perspicacite.llm.budget import BudgetExceededError
    from perspicacite.llm.client import _is_deterministic_fail

    assert _is_deterministic_fail(BudgetExceededError("over budget"))


def test_is_deterministic_fail_for_litellm_auth_class_name():
    from perspicacite.llm.client import _is_deterministic_fail

    class AuthenticationError(Exception):
        pass

    assert _is_deterministic_fail(AuthenticationError("invalid x-api-key"))


def test_is_deterministic_fail_for_invalid_key_pattern_in_generic_exc():
    from perspicacite.llm.client import _is_deterministic_fail

    e = RuntimeError(
        'litellm.AuthenticationError: AnthropicException - '
        '{"type":"error","error":{"message":"invalid x-api-key"}}'
    )
    assert _is_deterministic_fail(e)


def test_is_deterministic_fail_negative_for_random_error():
    from perspicacite.llm.client import _is_deterministic_fail
    assert not _is_deterministic_fail(RuntimeError("network blip"))
    assert not _is_deterministic_fail(TimeoutError("503 service unavailable"))


# ---------------------------------------------------------------------------
# F9 — LiteLLM banner suppression
# ---------------------------------------------------------------------------

def test_litellm_banner_suppressed_at_import():
    # Importing the client module flips the global flag.
    import litellm

    import perspicacite.llm.client  # noqa: F401

    assert getattr(litellm, "suppress_debug_info", False) is True


# ---------------------------------------------------------------------------
# F1 — end-to-end retry behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_does_not_retry_on_auth_error(monkeypatch):
    """A failing API call with an auth message should fire once, not three times."""
    from perspicacite.config.schema import LLMConfig, LLMProviderConfig
    from perspicacite.llm.client import AsyncLLMClient
    from perspicacite.llm.errors import AuthError

    cfg = LLMConfig(
        default_provider="anthropic",
        default_model="claude-3-haiku",
        cache_enabled=False,
        providers={"anthropic": LLMProviderConfig(base_url="https://x")},
    )
    client = AsyncLLMClient(cfg)

    calls = {"n": 0}

    class _AuthFail(Exception):
        pass
    _AuthFail.__name__ = "AuthenticationError"

    async def fake_acompletion(*args, **kwargs):
        calls["n"] += 1
        raise _AuthFail(
            'AnthropicException - {"error":{"message":"invalid x-api-key"}}'
        )

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(AuthError):
        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            cache=False,
        )
    assert calls["n"] == 1, f"expected 1 attempt, got {calls['n']}"


@pytest.mark.asyncio
async def test_complete_does_not_retry_on_cancellation(monkeypatch):
    """asyncio.CancelledError must propagate after ONE attempt, never be retried.

    Tenacity catches BaseException, so a predicate that only excludes
    deterministic-fail errors would treat CancelledError as retryable and
    swallow it — defeating every asyncio.wait_for/timeout in the codebase
    (extraction tools' 80s cap especially). Cancellation must bubble out.
    """
    import asyncio

    from perspicacite.config.schema import LLMConfig, LLMProviderConfig
    from perspicacite.llm.client import AsyncLLMClient

    cfg = LLMConfig(
        default_provider="anthropic",
        default_model="claude-3-haiku",
        cache_enabled=False,
        providers={"anthropic": LLMProviderConfig(base_url="https://x")},
    )
    client = AsyncLLMClient(cfg)

    calls = {"n": 0}

    async def fake_acompletion(*args, **kwargs):
        calls["n"] += 1
        raise asyncio.CancelledError()

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(asyncio.CancelledError):
        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            cache=False,
        )
    assert calls["n"] == 1, f"expected 1 attempt (no retry), got {calls['n']}"
