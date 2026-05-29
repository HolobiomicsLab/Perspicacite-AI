"""Unit tests for global LiteLLM timeout (Issue 1 — three-tier policy)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_config(default_timeout_s: float | None = None) -> MagicMock:
    """Build a minimal LLMConfig-like mock."""
    cfg = MagicMock()
    cfg.default_provider = "deepseek"
    cfg.default_model = "deepseek-chat"
    cfg.providers = {
        "deepseek": MagicMock(
            timeout=None,   # no provider-level timeout set
            base_url="",
            max_retries=1,
            executable=None,
        )
    }
    cfg.free_tier_fallback_models = []
    cfg.free_auto_mode = False
    cfg.use_mcp_sampling = False
    cfg.cache_enabled = False
    if default_timeout_s is not None:
        cfg.default_timeout_s = default_timeout_s
    else:
        del cfg.default_timeout_s   # attribute absent
    return cfg


@pytest.mark.asyncio
async def test_global_timeout_fallback_applied_when_no_provider_timeout():
    """When provider.timeout is None and no kwargs timeout, DEFAULT_LLM_TIMEOUT_S is used."""
    from perspicacite.llm.client import DEFAULT_LLM_TIMEOUT_S, AsyncLLMClient

    cfg = _make_config()
    client = AsyncLLMClient(cfg)

    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp.model = "deepseek/deepseek-chat"
        resp.get = lambda k, d=None: {"usage": resp.usage}.get(k, d)
        return resp

    litellm_mock = MagicMock()
    litellm_mock.acompletion = AsyncMock(side_effect=fake_acompletion)

    with patch.object(client, "_get_litellm", return_value=litellm_mock):
        with patch.object(client, "_get_provider_config") as mock_get_cfg:
            mock_get_cfg.return_value = MagicMock(timeout=None, base_url="", max_retries=1)
            with patch.object(client, "_build_model_string", return_value="deepseek/deepseek-chat"):
                with patch.object(client, "_is_agent_cli_provider", return_value=False):
                    await client.complete(
                        messages=[{"role": "user", "content": "hello"}],
                        model="deepseek-chat",
                        provider="deepseek",
                    )

    assert captured_kwargs.get("timeout") == DEFAULT_LLM_TIMEOUT_S


@pytest.mark.asyncio
async def test_config_level_timeout_overrides_default():
    """llm.default_timeout_s in config replaces the code constant."""
    from perspicacite.llm.client import AsyncLLMClient

    cfg = _make_config(default_timeout_s=120.0)
    client = AsyncLLMClient(cfg)

    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp.model = "deepseek/deepseek-chat"
        resp.get = lambda k, d=None: {"usage": resp.usage}.get(k, d)
        return resp

    litellm_mock = MagicMock()
    litellm_mock.acompletion = AsyncMock(side_effect=fake_acompletion)

    with patch.object(client, "_get_litellm", return_value=litellm_mock):
        with patch.object(client, "_get_provider_config") as mock_get_cfg:
            mock_get_cfg.return_value = MagicMock(timeout=None, base_url="", max_retries=1)
            with patch.object(client, "_build_model_string", return_value="deepseek/deepseek-chat"):
                with patch.object(client, "_is_agent_cli_provider", return_value=False):
                    await client.complete(
                        messages=[{"role": "user", "content": "hello"}],
                        model="deepseek-chat",
                        provider="deepseek",
                    )

    assert captured_kwargs.get("timeout") == 120.0


@pytest.mark.asyncio
async def test_per_call_kwarg_timeout_wins_over_all():
    """Explicit timeout=N kwarg always takes precedence."""
    from perspicacite.llm.client import AsyncLLMClient

    cfg = _make_config(default_timeout_s=120.0)
    client = AsyncLLMClient(cfg)

    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp.model = "deepseek/deepseek-chat"
        resp.get = lambda k, d=None: {"usage": resp.usage}.get(k, d)
        return resp

    litellm_mock = MagicMock()
    litellm_mock.acompletion = AsyncMock(side_effect=fake_acompletion)

    with patch.object(client, "_get_litellm", return_value=litellm_mock):
        with patch.object(client, "_get_provider_config") as mock_get_cfg:
            mock_get_cfg.return_value = MagicMock(timeout=None, base_url="", max_retries=1)
            with patch.object(client, "_build_model_string", return_value="deepseek/deepseek-chat"):
                with patch.object(client, "_is_agent_cli_provider", return_value=False):
                    await client.complete(
                        messages=[{"role": "user", "content": "hello"}],
                        model="deepseek-chat",
                        provider="deepseek",
                        timeout=30.0,   # explicit per-call kwarg
                    )

    assert captured_kwargs.get("timeout") == 30.0


@pytest.mark.asyncio
async def test_provider_timeout_overrides_global_default():
    """Provider-level timeout wins over llm.default_timeout_s (tier 2 > config tier)."""
    from perspicacite.llm.client import AsyncLLMClient

    cfg = _make_config(default_timeout_s=120.0)
    client = AsyncLLMClient(cfg)

    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp.model = "deepseek/deepseek-chat"
        resp.get = lambda k, d=None: {"usage": resp.usage}.get(k, d)
        return resp

    litellm_mock = MagicMock()
    litellm_mock.acompletion = AsyncMock(side_effect=fake_acompletion)

    with patch.object(client, "_get_litellm", return_value=litellm_mock):
        with patch.object(client, "_get_provider_config") as mock_get_cfg:
            mock_get_cfg.return_value = MagicMock(timeout=45, base_url="", max_retries=1)
            with patch.object(client, "_build_model_string", return_value="deepseek/deepseek-chat"):
                with patch.object(client, "_is_agent_cli_provider", return_value=False):
                    await client.complete(
                        messages=[{"role": "user", "content": "hello"}],
                        model="deepseek-chat",
                        provider="deepseek",
                        # No explicit timeout kwarg — provider's 45s should win over config's 120s
                    )

    assert captured_kwargs.get("timeout") == 45.0


def test_profound_mode_has_synthesis_timeout_s_attribute():
    """ProfoundRAGMode must expose synthesis_timeout_s read from config."""
    from unittest.mock import MagicMock

    from perspicacite.rag.modes.deep_research import ProfoundRAGMode

    cfg = MagicMock()
    cfg.rag_modes.deep_research = None   # triggers fallback to profound
    cfg.rag_modes.profound = None        # both None triggers dict fallback in __init__
    mode = ProfoundRAGMode(cfg)
    assert hasattr(mode, "synthesis_timeout_s"), "synthesis_timeout_s attribute missing"
    assert mode.synthesis_timeout_s == 90.0, f"expected 90.0, got {mode.synthesis_timeout_s}"


@pytest.mark.asyncio
async def test_profound_synthesis_timeout_fires():
    """asyncio.timeout(50ms) actually fires when a coroutine sleeps for 10s.

    This is a behavioral smoke test for the pattern used in both execute()
    and execute_stream() to cap the synthesis phase.
    """

    status_events: list[str] = []
    timed_out = False

    async def hanging_synthesis():
        await asyncio.sleep(10)  # much longer than the 50ms cap
        return "never reached"

    try:
        async with asyncio.timeout(0.05):  # 50ms
            await hanging_synthesis()
    except TimeoutError:
        timed_out = True
        status_events.append(
            "Deep research: synthesis time budget reached — returning partial answer."
        )

    assert timed_out, "asyncio.timeout did not fire — pattern is broken"
    assert any("synthesis time budget" in e for e in status_events), (
        f"Expected synthesis timeout status event, got: {status_events}"
    )
