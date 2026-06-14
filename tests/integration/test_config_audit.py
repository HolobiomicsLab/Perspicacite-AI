"""Config loading audit — Wave 1.4 of framework-hardening roadmap.

Verifies every shipped preset (config.example.yml + configs/) parses cleanly,
stage-resolution fall-through behaves correctly, backward compat
is preserved, and the new agent_cli LLMProviderConfig fields parse.

Marked @pytest.mark.config; runs in the default suite (fast, no I/O
beyond small YAML reads, no LLM calls).
"""

from __future__ import annotations

import glob
import textwrap
from pathlib import Path

import pytest
import yaml

from perspicacite.config.schema import Config, LLMConfig, LLMProviderConfig
from perspicacite.llm.client import AsyncLLMClient, resolve_stage_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent

STAGES = [
    "routing",
    "screening",
    "rephrase",
    "contextual",
    "synthesis_basic",
    "synthesis_heavy",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_llm_config(
    *,
    default_provider: str = "anthropic",
    default_model: str = "claude-haiku-4-5",
    models: dict | None = None,
    providers_per_stage: dict | None = None,
) -> LLMConfig:
    """Build an LLMConfig with one dummy provider entry to satisfy validation."""
    return LLMConfig(
        default_provider=default_provider,
        default_model=default_model,
        models=models or {},
        providers_per_stage=providers_per_stage or {},
        providers={
            default_provider: LLMProviderConfig(
                base_url="https://api.example.com",
                timeout=60,
            )
        },
    )


def _wrap_config(llm: LLMConfig) -> object:
    """Wrap an LLMConfig in a tiny namespace so resolve_stage_model finds it."""

    class _Cfg:
        pass

    c = _Cfg()
    c.llm = llm
    return c


# ---------------------------------------------------------------------------
# 1. YAML preset parsing — parametrized over every config.*.example.yml
# ---------------------------------------------------------------------------

_YAML_PRESETS = sorted(
    glob.glob(str(REPO_ROOT / "config.example.yml"))
    + glob.glob(str(REPO_ROOT / "configs" / "llm" / "*.yml"))
    + glob.glob(str(REPO_ROOT / "configs" / "embedders" / "*.yml"))
)


@pytest.mark.config
@pytest.mark.parametrize("yaml_path", _YAML_PRESETS, ids=lambda p: Path(p).name)
def test_yaml_preset_parses(yaml_path: str) -> None:
    """Every shipped preset (config.example.yml + configs/llm + configs/embedders) parses cleanly into Config."""
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{yaml_path} did not parse to a dict"

    cfg = Config(**data)

    # The default_provider declared in the file must exist in cfg.llm.providers
    assert cfg.llm.default_provider in cfg.llm.providers, (
        f"{Path(yaml_path).name}: default_provider={cfg.llm.default_provider!r} "
        f"not found in providers keys={list(cfg.llm.providers)}"
    )


# ---------------------------------------------------------------------------
# 2. Stage resolution — falls back to default
# ---------------------------------------------------------------------------


@pytest.mark.config
def test_stage_resolution_falls_back_to_default() -> None:
    """When models/providers_per_stage are empty, every stage gets the default."""
    llm = _minimal_llm_config(
        default_provider="anthropic",
        default_model="claude-haiku-4-5",
    )
    cfg = _wrap_config(llm)
    for stage in STAGES:
        provider, model = resolve_stage_model(cfg, stage)
        assert provider == "anthropic", f"stage={stage}: expected provider 'anthropic', got {provider!r}"
        assert model == "claude-haiku-4-5", f"stage={stage}: expected model 'claude-haiku-4-5', got {model!r}"


# ---------------------------------------------------------------------------
# 3. Stage resolution — model override
# ---------------------------------------------------------------------------


@pytest.mark.config
def test_stage_resolution_uses_model_override() -> None:
    """models[stage] pins the model while the provider stays at default."""
    llm = _minimal_llm_config(
        default_provider="anthropic",
        default_model="claude-sonnet-4-5",
        models={"routing": "claude-haiku-4-5"},
    )
    cfg = _wrap_config(llm)

    # Overridden stage returns the pinned model with the default provider
    provider, model = resolve_stage_model(cfg, "routing")
    assert provider == "anthropic"
    assert model == "claude-haiku-4-5"

    # Non-overridden stages still fall back to the global default
    for stage in STAGES:
        if stage == "routing":
            continue
        p, m = resolve_stage_model(cfg, stage)
        assert m == "claude-sonnet-4-5", f"stage={stage}: expected default model, got {m!r}"
        assert p == "anthropic", f"stage={stage}: expected default provider, got {p!r}"


# ---------------------------------------------------------------------------
# 4. Stage resolution — provider override
# ---------------------------------------------------------------------------


@pytest.mark.config
def test_stage_resolution_uses_provider_override() -> None:
    """providers_per_stage[stage] pins the provider while model stays at default."""
    llm = LLMConfig(
        default_provider="anthropic",
        default_model="claude-haiku-4-5",
        providers_per_stage={"screening": "ollama"},
        providers={
            "anthropic": LLMProviderConfig(base_url="https://api.anthropic.com"),
            "ollama": LLMProviderConfig(base_url="http://localhost:11434"),
        },
    )
    cfg = _wrap_config(llm)

    provider, model = resolve_stage_model(cfg, "screening")
    assert provider == "ollama", f"expected provider 'ollama', got {provider!r}"
    assert model == "claude-haiku-4-5", f"expected default model, got {model!r}"

    # Other stages still use the default provider
    for stage in STAGES:
        if stage == "screening":
            continue
        p, m = resolve_stage_model(cfg, stage)
        assert p == "anthropic", f"stage={stage}: expected 'anthropic', got {p!r}"


# ---------------------------------------------------------------------------
# 5. Stage resolution — combine both overrides
# ---------------------------------------------------------------------------


@pytest.mark.config
def test_stage_resolution_combines_overrides() -> None:
    """Both models and providers_per_stage pinning the same stage combine correctly."""
    llm = LLMConfig(
        default_provider="anthropic",
        default_model="claude-sonnet-4-5",
        models={"synthesis_heavy": "llama3.3:70b"},
        providers_per_stage={"synthesis_heavy": "ollama"},
        providers={
            "anthropic": LLMProviderConfig(base_url="https://api.anthropic.com"),
            "ollama": LLMProviderConfig(base_url="http://localhost:11434"),
        },
    )
    cfg = _wrap_config(llm)

    provider, model = resolve_stage_model(cfg, "synthesis_heavy")
    assert provider == "ollama", f"expected 'ollama', got {provider!r}"
    assert model == "llama3.3:70b", f"expected 'llama3.3:70b', got {model!r}"

    # All other stages fall back to global defaults
    for stage in STAGES:
        if stage == "synthesis_heavy":
            continue
        p, m = resolve_stage_model(cfg, stage)
        assert p == "anthropic"
        assert m == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# 6. Backward compat — minimal pre-tiering YAML (no models / providers_per_stage)
# ---------------------------------------------------------------------------

_MINIMAL_YAML = textwrap.dedent("""\
    llm:
      default_provider: anthropic
      default_model: claude-sonnet-4-5
      providers:
        anthropic:
          base_url: "https://api.anthropic.com"
          timeout: 120
""")


@pytest.mark.config
def test_minimal_pre_tiering_config_compat() -> None:
    """A minimal pre-tiering YAML (no models/providers_per_stage) parses and resolves."""
    data = yaml.safe_load(_MINIMAL_YAML)
    llm = LLMConfig(**data["llm"])

    assert llm.default_provider == "anthropic"
    assert llm.default_model == "claude-sonnet-4-5"
    assert llm.models == {}
    assert llm.providers_per_stage == {}

    cfg = _wrap_config(llm)
    for stage in STAGES:
        provider, model = resolve_stage_model(cfg, stage)
        assert provider == "anthropic", f"stage={stage}: got {provider!r}"
        assert model == "claude-sonnet-4-5", f"stage={stage}: got {model!r}"


# ---------------------------------------------------------------------------
# 7. Agent-CLI config parses and is detected by _is_agent_cli_provider
# ---------------------------------------------------------------------------

_AGENT_CLI_YAML = textwrap.dedent("""\
    llm:
      default_provider: agent_cli
      default_model: gpt-5.5
      providers:
        agent_cli:
          base_url: ""
          executable: "codex"
          prompt_via: "stdin"
          extra_args:
            - "exec"
            - "--skip-git-repo-check"
          output_file_flag: "--output-last-message"
          output_format: "text"
          timeout: 300
          max_retries: 1
""")


@pytest.mark.config
def test_agent_cli_config_parses_and_detects() -> None:
    """A YAML with agent_cli fields parses cleanly; _is_agent_cli_provider returns True."""
    data = yaml.safe_load(_AGENT_CLI_YAML)
    llm = LLMConfig(**data["llm"])

    # Field-level assertions
    agent_cfg = llm.providers["agent_cli"]
    assert isinstance(agent_cfg, LLMProviderConfig)
    assert agent_cfg.executable == "codex"
    assert agent_cfg.prompt_via == "stdin"
    assert agent_cfg.output_file_flag == "--output-last-message"
    assert "exec" in agent_cfg.extra_args
    assert agent_cfg.output_format == "text"

    # Detection — build a real AsyncLLMClient (no LLM calls made)
    client = AsyncLLMClient(config=llm)
    assert client._is_agent_cli_provider("agent_cli") is True

    # Sanity: a plain provider key is not detected as agent_cli
    llm2 = LLMConfig(
        default_provider="anthropic",
        default_model="claude-haiku-4-5",
        providers={
            "anthropic": LLMProviderConfig(base_url="https://api.anthropic.com"),
        },
    )
    client2 = AsyncLLMClient(config=llm2)
    assert client2._is_agent_cli_provider("anthropic") is False
