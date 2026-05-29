# Provider x Stage Matrix Status — 2026-05-14

Wave 1.2 of the framework-hardening roadmap. Audit of all configured LLM
providers on this machine, verifying liveness (real API call) and stage-routing
dispatch (config-resolver, no network).

## Environment

| Field | Value |
|---|---|
| Date | 2026-05-14 |
| Host | NOTHIASs-MacBook-Air.local (Darwin arm64 25.4.0) |
| Python | 3.13 (venv at `.venv/`) |
| Test file | `tests/integration/test_provider_matrix.py` |
| Run command | `pytest tests/integration/test_provider_matrix.py -m live -v --no-header --timeout=60 --timeout-method=signal` |

## Liveness results

| Provider | Outcome | Default model | Reason |
|---|---|---|---|
| `anthropic` | SKIPPED | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` not in environment |
| `openai` | SKIPPED | `gpt-4o-mini` | `OPENAI_API_KEY` not in environment |
| `deepseek` | SKIPPED | `deepseek-chat` | `DEEPSEEK_API_KEY` not in environment |
| `gemini` | SKIPPED | `gemini-1.5-flash` | `GOOGLE_API_KEY` not in environment |
| `ollama` | SKIPPED | first available model | Ollama not reachable at localhost:11434 |
| `claude_cli` | **PASS** | `haiku` | `claude` binary found; `~/.claude/config.json` present; real call returned non-empty string |
| `agent_cli` (codex) | **PASS** | `gpt-5.5` | `codex` binary found, `~/.codex/auth.json` present; full preset flags from `config/providers/codex.yml` (`exec --skip-git-repo-check --sandbox read-only --ephemeral --output-last-message`); ~8 s round-trip |

## Stage-routing results

All six stage-routing tests passed. These are pure config-resolver checks
(no real API calls); they verify that `resolve_stage_model` correctly returns
the pinned `(provider, model)` pair for each stage name.

| Stage | Pinned provider | Pinned model | Outcome |
|---|---|---|---|
| `routing` | `openai` | `gpt-4o-mini` | PASS |
| `screening` | `deepseek` | `deepseek-chat` | PASS |
| `rephrase` | `gemini` | `gemini-1.5-flash` | PASS |
| `contextual` | `ollama` | `llama3:8b` | PASS |
| `synthesis_basic` | `anthropic` | `claude-haiku-4-5` | PASS |
| `synthesis_heavy` | `anthropic` | `claude-sonnet-4-5` | PASS |

Additional routing tests:

| Test | Outcome |
|---|---|
| `test_stage_routing_fallback_to_default` — unpinned stages fall back to `(default_provider, default_model)` | PASS |
| `test_stage_routing_dispatch_capture` — mock-captures the `litellm.acompletion` model string end-to-end | PASS |

## Summary

| Metric | Value |
|---|---|
| Total collected | 15 |
| Passed | 10 |
| Skipped | 5 |
| Failed | 0 |
| Wall time | 13.4 s |

Liveness PASS: 2 (`claude_cli`, `agent_cli`/codex)
Liveness SKIP: 5 (4 missing API keys, 1 Ollama not running)
Stage-routing PASS: 8 of 8 (all stages + fallback + dispatch-capture)

## Notes

- The `DeprecationWarning: There is no current event loop` on `test_liveness_claude_cli`
  is benign — the event loop is created implicitly on Python 3.13. A future cleanup
  could switch to `asyncio.run()` instead of `get_event_loop().run_until_complete()`.
- `agent_cli` (codex) works fine via subprocess pipe stdin — no TTY required.
  Verified live with the full preset from `config/providers/codex.yml`:
  `exec --skip-git-repo-check --sandbox read-only --ephemeral
  --output-last-message <tempfile>`. Earlier draft of this test included an
  incorrect `sys.stdin.isatty()` skip; removed.

- To test the four API-key-gated providers, export the relevant key(s) and re-run:

  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  pytest tests/integration/test_provider_matrix.py::test_liveness_anthropic -m live -v
  ```

## How to reproduce (full suite)

```bash
source .venv/bin/activate
pytest tests/integration/test_provider_matrix.py -m live -v \
  --no-header --timeout=60 --timeout-method=signal
```
