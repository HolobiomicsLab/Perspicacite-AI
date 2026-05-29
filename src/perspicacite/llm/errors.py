"""LLM error hierarchy + rate-limit detection helpers.

See docs/superpowers/specs/2026-05-14-rate-limit-detection-design.md.

The patterns below are intentionally permissive — false positives
(treating a non-rate-limit error as a rate limit) are mild because
the tenacity retry chain will retry either way. The exception type
just changes the message the orchestrator surfaces and which
provider Wave 3.2's fallback chain will skip next.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Base class for Perspicacité LLM errors."""


@dataclass
class _RateLimitHit:
    retry_after_seconds: int | None


class RateLimitError(LLMError):
    """Provider declined the call due to rate / quota limits."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        retry_after_seconds: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds


class TimeoutError(LLMError):
    """The LLM call timed out (subprocess or API)."""


class AuthError(LLMError):
    """Provider auth failed (401, missing creds, expired session)."""

    def __init__(self, message: str, *, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider


# (compiled pattern, retry_seconds_extractor). Extractors return None
# when no usable retry hint is available. First match wins.
_RATE_LIMIT_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], int | None]]] = [
    # Claude Code: "Rate limit reached. Try again in 1h 23m."
    (
        re.compile(r"rate\s*limit\s*reached.*?try\s*again\s*in\s*"
                   r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?",
                   re.IGNORECASE | re.DOTALL),
        lambda m: (
            (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60)
            or None
        ),
    ),
    # Claude Code: "Usage limit exceeded"
    (
        re.compile(r"usage\s*limit\s*exceeded", re.IGNORECASE),
        lambda m: None,
    ),
    # HTTP 429 from anywhere
    (
        re.compile(r"\b429\b|too\s*many\s*requests", re.IGNORECASE),
        lambda m: None,
    ),
    # Generic "rate limit"
    (
        re.compile(r"\brate.?limit", re.IGNORECASE),
        lambda m: None,
    ),
    # Codex / OpenAI: "quota exceeded"
    (
        re.compile(r"quota\s*(exceeded|exhausted)", re.IGNORECASE),
        lambda m: None,
    ),
]


def detect_rate_limit(text: str) -> _RateLimitHit | None:
    """Return a hit (with optional retry hint) if ``text`` matches any
    known rate-limit pattern. ``None`` otherwise."""
    if not text:
        return None
    for pattern, extractor in _RATE_LIMIT_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                seconds = extractor(m)
            except Exception:
                seconds = None
            return _RateLimitHit(retry_after_seconds=seconds)
    return None


_AUTH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bauthentication\s*error\b", re.IGNORECASE),
    re.compile(r"\b(api[_\s]?key)\b.*\b(missing|not\s*set|required|invalid)\b",
               re.IGNORECASE),
    re.compile(r"\b(missing|not\s*set)\b.*\b(api[_\s]?key)\b", re.IGNORECASE),
    re.compile(r"environment\s*variable.*\bnot\s*set\b", re.IGNORECASE),
    re.compile(r"please\s+run\s+['\"]?\w+\s+login", re.IGNORECASE),
    re.compile(r"\b401\b|\bunauthorized\b", re.IGNORECASE),
]


def detect_auth_error(text: str) -> bool:
    """Return True when ``text`` looks like an auth failure."""
    if not text:
        return False
    return any(p.search(text) for p in _AUTH_PATTERNS)


_SUGGESTED_ACTIONS: dict[str, str] = {
    "anthropic": (
        "Wait for the Anthropic quota reset, or route this stage through "
        "a fallback provider via `llm.providers_per_stage` (DeepSeek / "
        "OpenAI / Gemini)."
    ),
    "openai": (
        "Wait for the OpenAI quota reset, or route through Anthropic / "
        "DeepSeek / Gemini via `llm.providers_per_stage`."
    ),
    "deepseek": (
        "DeepSeek rate-limited. Wait for reset or fall back to another "
        "provider via `llm.providers_per_stage`."
    ),
    "gemini": (
        "Gemini rate-limited. Wait for reset or fall back via "
        "`llm.providers_per_stage`."
    ),
    "claude_cli": (
        "Your Claude Pro/Max subscription is rate-limited. Switch to the "
        "direct Anthropic API (`providers_per_stage` → `anthropic`) or "
        "wait for the quota reset."
    ),
    "agent_cli": (
        "The agent CLI's subscription is rate-limited. Wait for reset "
        "or fall back to another agent_cli / direct API."
    ),
    "ollama": (
        "Local Ollama returned a transient error. Check the server logs "
        "(`localhost:11434`)."
    ),
}


def suggested_action(provider: str, *, hint: str | None = None) -> str:
    """Return a human-readable next-step message.

    ``hint`` lets callers distinguish auth-failure sub-modes (F3, audit
    2026-05-15) so the message is accurate:

    - ``"missing_or_invalid_key"`` → the user's API key is missing /
      typo'd / revoked. "Wait for quota reset" is wrong.
    - ``"quota_exceeded"`` / ``"unknown"`` / ``None`` → keep the
      rate-limit / quota wording.
    """
    if hint == "missing_or_invalid_key":
        return (
            "API key is missing or invalid. Set the appropriate "
            "`*_API_KEY` env var (or use the matching `config.<provider>.example.yml` "
            "preset), or switch this stage to `claude_cli` / another provider "
            "via `llm.providers_per_stage`."
        )
    return _SUGGESTED_ACTIONS.get(
        provider,
        "Wait for the quota reset or configure a fallback provider via "
        "`llm.providers_per_stage`.",
    )
