# Agent-CLI routing — caveats

Practical notes on the `agent_cli` / `claude_cli` providers (the
subprocess-based LLM routing path). Captured from live testing
during the May 2026 rollout. Keep this in sync as upstream CLIs evolve.

See also:
- [`configs/llm/claude_code.yml`](../configs/llm/claude_code.yml)
- [`configs/llm/codex.yml`](../configs/llm/codex.yml)
- [`configs/llm/openclaw.yml`](../configs/llm/openclaw.yml)
- [`configs/llm/hermes.yml`](../configs/llm/hermes.yml)
- [`src/perspicacite/llm/agent_cli.py`](../src/perspicacite/llm/agent_cli.py)

## What "agent CLI" routing means

Each LLM call inside Perspicacité spawns the configured CLI binary as
a subprocess, feeds it the prompt (stdin or `--prompt`-style flag),
reads the assistant text back (stdout or `--output-last-message FILE`),
and continues. The user's CLI subscription pays for the inference —
no API key sits in Perspicacité's config.

## Cross-cutting caveats (apply to every agent-CLI preset)

### Rate limits are shared with your interactive session
The same subscription window that powers your `claude` or `codex`
terminal sessions powers Perspicacité's internal calls. A heavy
multi-paper ingest with contextual retrieval can issue hundreds of
calls and freeze you out of the CLI interactively for hours.

**Mitigations:**
- Enable `pdf_download.cache_pdfs: true` so re-ingest reuses on-disk PDFs.
- Use per-stage tiering (`llm.models` / `llm.providers_per_stage`) to
  keep hot-loop stages on a cheap path (Haiku, Ollama) and reserve
  the agent CLI for synthesis only.
- For unattended / production use, prefer direct API + prompt caching
  (`config.example.yml` + `ANTHROPIC_API_KEY`).

### No prompt caching
Agent CLIs don't expose `cache_control` (Anthropic's prompt-cache
markers) over their stdin/argv interface. The 90% discount on
repeated prefixes that the direct API provides is unavailable here.
The `build_cached_messages` helper detects the non-Anthropic provider
and emits plain strings, so this is silent — but it's real cost on
hot prefixes like contextual retrieval.

### No per-call temperature or `max_tokens`
The CLIs don't expose these flags in non-interactive mode. The
parameters are accepted by `AgentCLIClient.complete()` for API
compatibility but ignored. Whatever defaults the CLI ships with apply.

### No streaming
The subprocess returns the whole completion at once. Synthesis UI
paths that expect token-by-token streaming will see a single chunk.

### No structured tool-use
The CLI's response is plain text. Tool-using stages (agentic mode)
that depend on Anthropic's `tool_use` blocks won't work through this
route. Use direct API for agentic mode.

## Token usage in provenance (Wave 2.3+)

`AgentCLIClient` can extract input / output token counts from the
CLI's JSON output when `usage_input_tokens_path` and
`usage_output_tokens_path` are set in the provider config. Today:

| CLI | Status |
|---|---|
| Claude Code | ✅ Live — preset wires `usage.input_tokens` / `usage.output_tokens` |
| Codex | ❌ Out — `--output-last-message` returns plain text. Would require switching to `--json` event-stream parsing (followup). |
| OpenClaw | ❓ Unverified — set paths in config if known |
| Hermes | ❓ Unverified — set paths in config if known |

When paths are unset or the CLI doesn't surface usage, provenance
records `prompt_tokens=0, completion_tokens=0` (today's behaviour, no
regression). Budget caps (Wave 2.4) treat those as "unknown cost".

## Per-CLI caveats

### Claude Code (`claude_cli`) — verified live ✅

Driven via `claude -p --output-format json --no-session-persistence
--model {sonnet|haiku|opus} --append-system-prompt "<system>" < stdin`.
Returns JSON `{"result": "<text>", ...}`, which we parse via
`result_json_path: "result"`.

| Property | Value |
|---|---|
| Latency (small prompt, Haiku) | ~5–15 s |
| Model selection | `--model {sonnet,haiku,opus}` |
| System prompt | `--append-system-prompt` |
| Output | JSON to stdout, single `result` key |
| Auth | `claude login` (subscription) |

**Specific caveats:**
- `--no-session-persistence` keeps Perspicacité runs out of your
  session picker. Without it, every call would clutter `claude
  resume`.
- Model aliases (`sonnet`/`haiku`/`opus`) collapse the full
  `claude-sonnet-4-5-*` names via fuzzy substring match in
  `ClaudeCLIClient` — set `default_model: "claude-sonnet-4-5"` and
  it still resolves to `sonnet`.
- Sampling (`sampling/createMessage` MCP protocol) is not yet
  implemented in Claude Code CLI as of May 2026
  ([anthropics/claude-code#1785](https://github.com/anthropics/claude-code/issues/1785)).
  `llm.use_mcp_sampling: true` falls through silently on this client.

### OpenAI Codex (`agent_cli` → `codex`) — verified live ✅

Driven via `codex exec --skip-git-repo-check --sandbox read-only
--ephemeral --model <NAME> --output-last-message <tempfile> < stdin`.
The final assistant message is read from the tempfile (cleanest path
— stdout is polluted with banner / token counter / session id).

| Property | Value |
|---|---|
| Verified version | codex-cli 0.130.0 |
| Latency (small prompt, gpt-5.5) | ~6 s direct, ~16 s via `AsyncLLMClient` |
| Model selection | `--model <NAME>` |
| System prompt | None — prepended to the body in `_flatten_messages` |
| Output | `--output-last-message FILE` (text) |
| Auth | `codex login` (ChatGPT subscription) |

**Specific caveats:**
- **Codex is full agent machinery, not a completion endpoint.**
  Each call spins up the sandbox, tool loop, and memory layer. Per-call
  latency is ~5–15 s even for trivial prompts. This is fine for
  synthesis (one or two calls per report) but expensive for
  hot-loop stages — routing/screening on hundreds of papers will be
  slow *and* burn a lot of Codex's session window.
- `--sandbox read-only` is the cheapest sandbox mode; we pass it to
  prevent any model-generated shell commands from touching disk.
  We just want a completion, not an agent run.
- `--skip-git-repo-check` is required when Perspicacité runs outside
  a git repository (e.g., from `/tmp` or as a service).
- `--ephemeral` skips persisting session files — without it, every
  call writes to `~/.codex/sessions/`, which clutters
  `codex resume` and burns disk.
- Stdout contains a banner (workdir, model, provider, sandbox mode,
  session id, token counter) plus user/assistant transcript. The
  `--output-last-message` flag is the only clean extraction path;
  scraping stdout is brittle.

### OpenClaw (`agent_cli` → `openclaw`) — best-effort, untested

Driven via `openclaw agent --message "<prompt>"`. Model is selected
inside `~/.openclaw/openclaw.json`, not on the CLI.

**Specific caveats:**
- Multi-channel agent platform (WhatsApp, Telegram, Slack, etc.),
  not primarily a single-shot completion CLI. The `agent --message`
  invocation works for one-shot prompts but its output shape may
  vary across releases.
- No system-prompt flag — system message gets prepended to the body.
- No `--model` flag — selector is in OpenClaw's own config file.
- Output: assumed plain text. If your version supports `--json`,
  flip `output_format: "json"` and set `result_json_path` to the key
  that holds the assistant text.
- A `--thinking high` knob can be added globally via `extra_args`.

### Hermes (`agent_cli` → `hermes`) — best-effort, untested

Driven via `hermes ask "<prompt>"` (assumed — verify against your
installed version).

**Specific caveats:**
- Persistent-memory autonomous agent, not a single-shot completion
  CLI. Upstream docs don't yet publish a stable non-interactive flag
  schema.
- The model is managed by Hermes's own config (set during
  `hermes setup`), not by a CLI flag.
- **Simpler alternative for Hermes models:** the Hermes family is
  published on Ollama as `hermes-3:70b` etc. Use
  [`configs/llm/ollama.yml`](../configs/llm/ollama.yml) with
  `default_model: "hermes-3:70b"` — fully supported today, no CLI
  dependency.

## Recommended per-stage tiering

Heavy synthesis benefits from the agent CLI (subscription cost = $0).
Hot loops (routing, screening, contextual retrieval) hit the rate
limits hardest and benefit least from agent-tier quality.

A pragmatic mix:

```yaml
llm:
  default_provider: "claude_cli"
  default_model:    "sonnet"

  providers_per_stage:
    # Hot loops — keep these cheap and fast
    routing:    "claude_cli"
    screening:  "claude_cli"
    rephrase:   "claude_cli"
    contextual: "claude_cli"
    # Synthesis — quality matters more than speed
    synthesis_basic:  "claude_cli"
    synthesis_heavy:  "claude_cli"

  models:
    routing:    "haiku"
    screening:  "haiku"
    rephrase:   "haiku"
    contextual: "haiku"
    synthesis_basic:  "sonnet"
    synthesis_heavy:  "sonnet"      # or "opus" if your plan includes it
```

For Codex (which has no fast "haiku-tier" model), the right mix is
often: agent CLI only on synthesis, Ollama or direct API on hot loops:

```yaml
llm:
  default_provider: "ollama"
  default_model:    "llama3.2:3b"   # routing, screening, rephrase, contextual

  providers_per_stage:
    synthesis_basic:  "agent_cli"   # → codex for the final report
    synthesis_heavy:  "agent_cli"

  models:
    synthesis_basic:  "gpt-5"
    synthesis_heavy:  "gpt-5.5"
```

## When to NOT use agent-CLI routing

- **Production / unattended deployments.** Subscription rate limits
  weren't designed for batch workloads; you'll get throttled and
  block your own interactive use.
- **Heavy contextual retrieval ingests.** Hundreds of per-chunk calls
  multiplied by per-call agent latency = hours. Direct API + prompt
  caching is dramatically cheaper *and* faster.
- **Agentic mode.** Needs structured tool-use that the CLI text
  interface can't carry. Use direct API.
- **Streaming UI.** Subprocess output is buffered.

Use it for: solo / interactive use, small-to-medium ingests, "use
my Pro subscription instead of paying API rates again" scenarios.

## Adding a new agent CLI

No Python changes needed — write a YAML preset:

1. Run `<your-cli> --help` to find the flag schema.
2. Copy the closest existing preset (Codex is the most general).
3. Set:
   - `executable`: binary name
   - `extra_args`: subcommand + any required flags
   - `prompt_via`: `"stdin"` (cleaner) or `"arg"`
   - `system_flag`: if the CLI has one; else `null`
   - `model_flag`: `"--model"` or equivalent; `null` if model is set
     in the CLI's own config
   - `output_file_flag`: e.g. `"--output-last-message"` if available;
     else `null` (stdout is used)
   - `output_format`: `"text"` or `"json"` (with `result_json_path`)
4. Live-test by piping a tiny prompt through the CLI directly first
   to see the output shape, *then* wire it through `AsyncLLMClient`.
5. Add caveats to this doc.
