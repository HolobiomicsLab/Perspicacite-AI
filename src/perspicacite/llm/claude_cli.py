"""Claude Code CLI subprocess LLM provider — preset over ``agent_cli``.

Historically a standalone module; now a thin convenience wrapper that
constructs an :class:`~perspicacite.llm.agent_cli.AgentCLIClient` with
Claude Code's flags baked in. Existing imports
(``from perspicacite.llm.claude_cli import ClaudeCLIClient``) keep
working unchanged.

See ``agent_cli.py`` for the underlying implementation and
``config.claude_code.example.yml`` for a config-only setup.
"""

from __future__ import annotations

from perspicacite.llm.agent_cli import AgentCLIClient

# Claude Code's CLI shape (as of 2.x):
#   claude -p --output-format json --no-session-persistence
#          --model {sonnet|haiku|opus}
#          --append-system-prompt "..."
#          < stdin
# Returns JSON: {"type": "result", "result": "<assistant text>", ...}
_CLAUDE_CODE_DEFAULTS: dict = {
    "executable": "claude",
    "provider_label": "claude_cli",
    "prompt_via": "stdin",
    "extra_args": [
        "-p",
        "--output-format", "json",
        "--no-session-persistence",
    ],
    "system_flag": "--append-system-prompt",
    "model_flag": "--model",
    "output_format": "json",
    "result_json_path": "result",
    "model_aliases": {
        "sonnet": "sonnet",
        "haiku": "haiku",
        "opus": "opus",
        # Fuzzy contains-match collapses any of these to the alias:
        "claude-sonnet": "sonnet",
        "claude-haiku":  "haiku",
        "claude-opus":   "opus",
    },
    "env_extra": {
        "CLAUDE_CODE_USER_AGENT": "Perspicacite/2.0",
    },
}


def ClaudeCLIClient(  # noqa: N802 — back-compat factory named after the retired class
    *,
    executable: str = "claude",
    timeout: float = 180.0,
    cwd: str | None = None,
    env_extra: dict[str, str] | None = None,
    usage_input_tokens_path: str | None = None,
    usage_output_tokens_path: str | None = None,
    cost_usd_path: str | None = None,
    cache_read_tokens_path: str | None = None,
    cache_creation_tokens_path: str | None = None,
) -> AgentCLIClient:
    """Backwards-compatible factory returning an :class:`AgentCLIClient`
    pre-configured for Claude Code.

    The original :class:`ClaudeCLIClient` was its own class; now it's
    just an :class:`AgentCLIClient` with Claude-specific defaults.
    Keeping the function name + ``__init__``-style kwargs means
    existing callers don't change.
    """
    kw = dict(_CLAUDE_CODE_DEFAULTS)
    kw["executable"] = executable
    kw["timeout"] = timeout
    if cwd is not None:
        kw["cwd"] = cwd
    if env_extra:
        merged = dict(kw["env_extra"])
        merged.update(env_extra)
        kw["env_extra"] = merged
    if usage_input_tokens_path is not None:
        kw["usage_input_tokens_path"] = usage_input_tokens_path
    if usage_output_tokens_path is not None:
        kw["usage_output_tokens_path"] = usage_output_tokens_path
    # F4 (audit 2026-05-15): forward rich result fields.
    if cost_usd_path is not None:
        kw["cost_usd_path"] = cost_usd_path
    if cache_read_tokens_path is not None:
        kw["cache_read_tokens_path"] = cache_read_tokens_path
    if cache_creation_tokens_path is not None:
        kw["cache_creation_tokens_path"] = cache_creation_tokens_path
    return AgentCLIClient(**kw)


__all__ = ["ClaudeCLIClient"]
