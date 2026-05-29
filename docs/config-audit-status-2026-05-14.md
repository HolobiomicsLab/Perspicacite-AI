# Config Loading Audit — Wave 1.4

**Date:** 2026-05-14
**Branch:** main
**Test file:** `tests/integration/test_config_audit.py`
**Marker:** `@pytest.mark.config`

---

## YAML Preset Audit

| Preset file | Outcome | default_provider |
|---|---|---|
| `config/example.yml` | PASS | `anthropic` |
| `config/providers/claude-code.yml` | PASS | `claude_cli` |
| `config/providers/codex.yml` | PASS | `agent_cli` |
| `config/providers/hermes.yml` | PASS | `agent_cli` |
| `config/providers/openclaw.yml` | PASS | `agent_cli` |
| `config/providers/ollama.yml` | PASS | `ollama` |

All 6 presets parse into `Config` without validation errors. For each, `default_provider` is present in `cfg.llm.providers`.

Note: `config/example.yml` uses a flat name pattern (no middle segment) and is discovered via a separate `glob.glob("config/example.yml")` call alongside the `config.*.example.yml` glob.

---

## Stage-Resolution Coverage Matrix

| Scenario | Stages tested | Result |
|---|---|---|
| All defaults (empty `models` + `providers_per_stage`) | all 6 | PASS |
| `models["routing"] = "claude-haiku-4-5"` override | routing + remaining 5 | PASS |
| `providers_per_stage["screening"] = "ollama"` override | screening + remaining 5 | PASS |
| Combined `models` + `providers_per_stage` on `synthesis_heavy` | synthesis_heavy + remaining 5 | PASS |

Stages covered: `routing`, `screening`, `rephrase`, `contextual`, `synthesis_basic`, `synthesis_heavy`.

---

## Backward Compatibility

A minimal pre-tiering YAML string (only `default_provider`, `default_model`, one provider entry — no `models` or `providers_per_stage` keys) parses into `LLMConfig` with both dicts defaulting to `{}`. `resolve_stage_model` returns the global default pair for every stage.

**Outcome: PASS**

---

## Agent-CLI Detection

A YAML snippet with `executable: "codex"`, `prompt_via: "stdin"`, `extra_args: ["exec", "--skip-git-repo-check"]`, and `output_file_flag: "--output-last-message"` parses into `LLMConfig` cleanly. `AsyncLLMClient._is_agent_cli_provider("agent_cli")` returns `True`. A plain non-agent-CLI provider (`"anthropic"`, no `executable`) returns `False`.

**Outcome: PASS**

---

## Reproducer

```bash
source .venv/bin/activate
pytest tests/integration/test_config_audit.py -m config -v \
  --timeout=10 --timeout-method=signal --no-header
```

---

## Results Summary

- **12 tests collected**, **12 passed**, **0 failed**, **0 skipped**
- **Total time: 0.13 s**
- No LLM calls, no HTTP, no embeddings, no ChromaDB.
