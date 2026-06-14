"""Provider x stage matrix audit — Wave 1.2 of framework-hardening roadmap.

For each available LLM provider, verify:
- Liveness: AsyncLLMClient.complete returns a non-empty string.
- Stage routing: configuring models.<stage>/providers_per_stage.<stage>
  causes that pair to be dispatched.

Marked @pytest.mark.live; default unit suite skips them. Opt-in:
    pytest tests/integration -m live -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from perspicacite.config.schema import LLMConfig, LLMProviderConfig
from perspicacite.llm.client import AsyncLLMClient, resolve_stage_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGES = [
    "routing",
    "screening",
    "rephrase",
    "contextual",
    "synthesis_basic",
    "synthesis_heavy",
]

# Tiny prompt — elicits a 1-10 char response, minimises cost and latency.
LIVENESS_MESSAGES = [
    {"role": "system", "content": "Reply only with the word OK."},
    {"role": "user", "content": "Say OK."},
]


# ---------------------------------------------------------------------------
# Prerequisite helpers
# ---------------------------------------------------------------------------


def _has_env(var: str) -> bool:
    return bool(os.environ.get(var, "").strip())


def _binary_on_path(name: str) -> bool:
    return shutil.which(name) is not None


def _ollama_running_with_models() -> tuple[bool, str]:
    """Return (available, first_model_name).

    Pings http://localhost:11434/api/tags. Returns (False, "") when the
    server is down or has no models pulled.
    """
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "3", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False, ""
        import json
        data = json.loads(result.stdout)
        models = [m["name"] for m in data.get("models", [])]
        if not models:
            return False, ""
        # Prefer a small model if available.
        for small in ("phi3:mini", "phi3", "llama3:8b", "llama3", "mistral"):
            for m in models:
                if small in m:
                    return True, m
        return True, models[0]
    except Exception:
        return False, ""


def _skip_if_unavailable(provider: str) -> None:
    """Call at the top of each liveness test; calls pytest.skip() if
    the provider cannot be used on this machine right now.

    The reason string is shown in the test output so the audit report
    can explain skips without manual inspection.
    """
    if provider == "anthropic":
        if not _has_env("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not in environment — provider unavailable")

    elif provider == "openai":
        if not _has_env("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not in environment — provider unavailable")

    elif provider == "deepseek":
        if not _has_env("DEEPSEEK_API_KEY"):
            pytest.skip("DEEPSEEK_API_KEY not in environment — provider unavailable")

    elif provider == "gemini":
        if not _has_env("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not in environment — provider unavailable")

    elif provider == "ollama":
        ok, _ = _ollama_running_with_models()
        if not ok:
            pytest.skip(
                "Ollama not reachable at localhost:11434 or no models pulled — "
                "run `ollama serve` and `ollama pull <model>` first"
            )

    elif provider == "claude_cli":
        if not _binary_on_path("claude"):
            pytest.skip("claude binary not found on PATH — install Claude Code CLI")
        # Use ~/.claude/config.json presence as auth proxy; the task brief
        # says .credentials.json may not exist depending on auth flow.
        # Accepting either config.json (always written by Claude Code) or a
        # session token as "authenticated enough to attempt a call".
        claude_home = Path.home() / ".claude"
        if not (claude_home / "config.json").exists() and not (
            claude_home / ".credentials.json"
        ).exists():
            pytest.skip(
                "Neither ~/.claude/config.json nor ~/.claude/.credentials.json found — "
                "claude CLI does not appear to be configured"
            )

    elif provider == "agent_cli":
        if not _binary_on_path("codex"):
            pytest.skip("codex binary not found on PATH — install OpenAI Codex CLI")
        codex_auth = Path.home() / ".codex" / "auth.json"
        if not codex_auth.exists():
            pytest.skip(
                f"~/.codex/auth.json not found ({codex_auth}) — "
                "run `codex login` to authenticate"
            )
        # Note: `codex exec` works fine via subprocess pipe stdin — verified
        # live with `echo ... | codex exec --skip-git-repo-check --sandbox
        # read-only --ephemeral --output-last-message FILE`. No TTY needed.


# ---------------------------------------------------------------------------
# LLMConfig factories
# ---------------------------------------------------------------------------


def _make_config_for(provider: str, model: str, **kwargs: Any) -> LLMConfig:
    """Build a minimal LLMConfig for the given provider+model pair.

    Ollama needs a base_url; agent-CLI providers need an executable.
    All other providers use the defaults that ship with LLMConfig.
    """
    # Start with the default provider map so we don't miss any keys the
    # client code expects.
    base_providers: dict[str, LLMProviderConfig] = {
        "anthropic": LLMProviderConfig(base_url="https://api.anthropic.com", timeout=60),
        "openai": LLMProviderConfig(base_url="https://api.openai.com/v1", timeout=60),
        "deepseek": LLMProviderConfig(base_url="https://api.deepseek.com", timeout=60),
        "gemini": LLMProviderConfig(base_url="https://generativelanguage.googleapis.com", timeout=60),
        "ollama": LLMProviderConfig(base_url="http://localhost:11434", timeout=60),
        # agent-CLI providers — must have `executable` set so
        # _is_agent_cli_provider() returns True.
        "claude_cli": LLMProviderConfig(
            executable="claude",
            timeout=180,
            max_retries=1,
        ),
        # Codex preset — mirrors configs/llm/codex.yml. Verified live
        # in commit 7f1e7d7 (~16 s round-trip on this machine).
        "agent_cli": LLMProviderConfig(
            executable="codex",
            timeout=300,
            max_retries=1,
            prompt_via="stdin",
            extra_args=[
                "exec",
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--ephemeral",
            ],
            model_flag="--model",
            output_format="text",
            output_file_flag="--output-last-message",
        ),
    }

    providers = {k: v for k, v in base_providers.items()}

    return LLMConfig(
        default_provider=provider,
        default_model=model,
        providers=providers,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Liveness tests — one real API call per provider
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_liveness_anthropic():
    """Anthropic: verify a real API call returns a non-empty string."""
    _skip_if_unavailable("anthropic")
    import asyncio

    config = _make_config_for("anthropic", "claude-haiku-4-5")
    client = AsyncLLMClient(config)
    result = asyncio.get_event_loop().run_until_complete(
        client.complete(
            messages=LIVENESS_MESSAGES,
            provider="anthropic",
            model="claude-haiku-4-5",
            max_tokens=10,
            temperature=0.0,
            cache=False,           # Wave 2.1: never cache liveness responses
        )
    )
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result.strip()) > 0, "Response was empty"


@pytest.mark.live
def test_liveness_openai():
    """OpenAI: verify a real API call returns a non-empty string."""
    _skip_if_unavailable("openai")
    import asyncio

    config = _make_config_for("openai", "gpt-4o-mini")
    client = AsyncLLMClient(config)
    result = asyncio.get_event_loop().run_until_complete(
        client.complete(
            messages=LIVENESS_MESSAGES,
            provider="openai",
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0.0,
            cache=False,           # Wave 2.1: never cache liveness responses
        )
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.live
def test_liveness_deepseek():
    """DeepSeek: verify a real API call returns a non-empty string."""
    _skip_if_unavailable("deepseek")
    import asyncio

    config = _make_config_for("deepseek", "deepseek-chat")
    client = AsyncLLMClient(config)
    result = asyncio.get_event_loop().run_until_complete(
        client.complete(
            messages=LIVENESS_MESSAGES,
            provider="deepseek",
            model="deepseek-chat",
            max_tokens=10,
            temperature=0.0,
            cache=False,           # Wave 2.1: never cache liveness responses
        )
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.live
def test_liveness_gemini():
    """Gemini: verify a real API call returns a non-empty string."""
    _skip_if_unavailable("gemini")
    import asyncio

    config = _make_config_for("gemini", "gemini-1.5-flash")
    client = AsyncLLMClient(config)
    result = asyncio.get_event_loop().run_until_complete(
        client.complete(
            messages=LIVENESS_MESSAGES,
            provider="gemini",
            model="gemini-1.5-flash",
            max_tokens=10,
            temperature=0.0,
            cache=False,           # Wave 2.1: never cache liveness responses
        )
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.live
def test_liveness_ollama():
    """Ollama: verify a real local API call returns a non-empty string."""
    _skip_if_unavailable("ollama")
    import asyncio

    ok, model = _ollama_running_with_models()
    assert ok, "Ollama vanished between skip check and test body"

    config = _make_config_for("ollama", model)
    client = AsyncLLMClient(config)
    result = asyncio.get_event_loop().run_until_complete(
        client.complete(
            messages=LIVENESS_MESSAGES,
            provider="ollama",
            model=model,
            max_tokens=10,
            temperature=0.0,
            cache=False,           # Wave 2.1: never cache liveness responses
        )
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.live
def test_liveness_claude_cli():
    """claude_cli: verify the Claude Code subprocess returns a non-empty string."""
    _skip_if_unavailable("claude_cli")
    import asyncio

    config = _make_config_for("claude_cli", "haiku")
    client = AsyncLLMClient(config)
    result = asyncio.get_event_loop().run_until_complete(
        client.complete(
            messages=LIVENESS_MESSAGES,
            provider="claude_cli",
            model="haiku",
            max_tokens=10,
            temperature=0.0,
            cache=False,           # Wave 2.1: never cache liveness responses
        )
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


@pytest.mark.live
def test_liveness_agent_cli_codex():
    """agent_cli (codex): verify the Codex subprocess returns a non-empty string."""
    _skip_if_unavailable("agent_cli")
    import asyncio

    config = _make_config_for("agent_cli", "gpt-5.5")
    client = AsyncLLMClient(config)
    result = asyncio.get_event_loop().run_until_complete(
        client.complete(
            messages=LIVENESS_MESSAGES,
            provider="agent_cli",
            model="gpt-5.5",
            max_tokens=10,
            temperature=0.0,
            cache=False,           # Wave 2.1: never cache liveness responses
        )
    )
    assert isinstance(result, str)
    assert len(result.strip()) > 0


# ---------------------------------------------------------------------------
# Stage-routing tests — no real API calls
# ---------------------------------------------------------------------------


def _make_config_with_stage_override(
    stage: str,
    stage_provider: str,
    stage_model: str,
    default_provider: str = "anthropic",
    default_model: str = "claude-haiku-4-5",
) -> LLMConfig:
    """Build a config that pins a single stage to a specific (provider, model).

    The *default* pair is the fallback for every other stage.
    """
    providers_per_stage = {stage: stage_provider}
    models = {stage: stage_model}
    return _make_config_for(
        default_provider,
        default_model,
        providers_per_stage=providers_per_stage,
        models=models,
    )


# Parametrize over all six stages plus a handful of provider/model combos
# that are representative but don't require API keys (routing is pure config).
_STAGE_ROUTING_CASES = [
    # (stage, pinned_provider, pinned_model)
    ("routing", "openai", "gpt-4o-mini"),
    ("screening", "deepseek", "deepseek-chat"),
    ("rephrase", "gemini", "gemini-1.5-flash"),
    ("contextual", "ollama", "llama3:8b"),
    ("synthesis_basic", "anthropic", "claude-haiku-4-5"),
    ("synthesis_heavy", "anthropic", "claude-sonnet-4-5"),
]


@pytest.mark.live
@pytest.mark.parametrize("stage,pinned_provider,pinned_model", _STAGE_ROUTING_CASES)
def test_stage_routing_resolve(
    stage: str,
    pinned_provider: str,
    pinned_model: str,
) -> None:
    """resolve_stage_model returns the pinned (provider, model) for a configured stage.

    This test is purely a config-resolver check — no real API calls are made.
    It verifies that the dict lookups in ``resolve_stage_model`` work correctly
    for each of the six stage names.
    """

    class _FakeConfig:
        """Minimal config shim that mimics the shape resolve_stage_model reads."""

        class _LLM:
            default_provider = "anthropic"
            default_model = "claude-haiku-4-5"
            models: dict[str, str] = {}
            providers_per_stage: dict[str, str] = {}

        llm = _LLM()

    fake = _FakeConfig()
    fake.llm.models = {stage: pinned_model}
    fake.llm.providers_per_stage = {stage: pinned_provider}

    provider_out, model_out = resolve_stage_model(fake, stage)
    assert provider_out == pinned_provider, (
        f"Stage '{stage}': expected provider '{pinned_provider}', got '{provider_out}'"
    )
    assert model_out == pinned_model, (
        f"Stage '{stage}': expected model '{pinned_model}', got '{model_out}'"
    )


@pytest.mark.live
def test_stage_routing_fallback_to_default() -> None:
    """resolve_stage_model falls back to (default_provider, default_model) for
    stages not pinned in the config."""

    class _FakeConfig:
        class _LLM:
            default_provider = "deepseek"
            default_model = "deepseek-chat"
            models: dict[str, str] = {}
            providers_per_stage: dict[str, str] = {}

        llm = _LLM()

    fake = _FakeConfig()
    # Pin only "synthesis_heavy"; all other stages should fall back.
    fake.llm.models = {"synthesis_heavy": "claude-sonnet-4-5"}
    fake.llm.providers_per_stage = {"synthesis_heavy": "anthropic"}

    for stage in STAGES:
        provider_out, model_out = resolve_stage_model(fake, stage)
        if stage == "synthesis_heavy":
            assert provider_out == "anthropic"
            assert model_out == "claude-sonnet-4-5"
        else:
            assert provider_out == "deepseek", (
                f"Stage '{stage}' should fall back to 'deepseek', got '{provider_out}'"
            )
            assert model_out == "deepseek-chat", (
                f"Stage '{stage}' should fall back to 'deepseek-chat', got '{model_out}'"
            )


@pytest.mark.live
def test_stage_routing_dispatch_capture() -> None:
    """AsyncLLMClient.complete dispatches the correct (provider, model) pair for
    a stage-configured call, as captured via monkeypatching litellm.acompletion.

    No real API key is required. Uses AsyncMock to capture what model string
    LiteLLM would have been called with, then verifies the per-stage override
    was applied.
    """
    import asyncio

    # Build the LLMConfig pinning "routing" stage to openai/gpt-4o-mini.
    llm_config = _make_config_for(
        "anthropic",
        "claude-haiku-4-5",
        providers_per_stage={"routing": "openai"},
        models={"routing": "gpt-4o-mini"},
    )

    client = AsyncLLMClient(llm_config)

    # resolve_stage_model expects a Config-shaped object with a `.llm` attribute,
    # NOT a bare LLMConfig.  Build a lightweight shim.
    class _ConfigShim:
        llm = llm_config

    shim = _ConfigShim()

    class MockResponse(dict):
        class Choice:
            class Message:
                content = "OK"
            message = Message()

        def __init__(self) -> None:
            super().__init__(usage={"prompt_tokens": 1, "completion_tokens": 1})
            self.choices = [self.Choice()]

    captured_model: list[str] = []

    async def mock_acompletion(**kwargs: Any) -> MockResponse:
        captured_model.append(kwargs.get("model", ""))
        return MockResponse()

    import litellm
    original = litellm.acompletion
    litellm.acompletion = mock_acompletion  # type: ignore[assignment]
    try:
        # Resolve stage to get the (provider, model) pair the orchestrator
        # would pass down.
        stage = "routing"
        provider, model = resolve_stage_model(shim, stage)
        assert provider == "openai", f"Expected 'openai', got '{provider}'"
        assert model == "gpt-4o-mini", f"Expected 'gpt-4o-mini', got '{model}'"

        # Actually call complete with that resolved pair.
        asyncio.get_event_loop().run_until_complete(
            client.complete(
                messages=LIVENESS_MESSAGES,
                provider=provider,
                model=model,
                max_tokens=10,
            )
        )
    finally:
        litellm.acompletion = original  # type: ignore[assignment]

    assert captured_model, "mock_acompletion was never called"
    assert captured_model[0] == "openai/gpt-4o-mini", (
        f"LiteLLM was called with model='{captured_model[0]}', "
        "expected 'openai/gpt-4o-mini'"
    )
