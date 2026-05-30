# INSTALL_AGENT — Perspicacité-AI

Step-by-step install guide for a Claude Code-style agent (or any copilot in a
shell) to bring Perspicacité up on a fresh machine semi-autonomously. Each
step ends with a **verify** the agent must run and match against the expected
output before continuing.

This is the operational sibling of [`README.md`](README.md) and
[`docs/getting-started.md`](docs/getting-started.md) — same flow, more
defensive, explicit verification at every checkpoint.

## Operating contract

- Every step has an explicit verify. Stop on first mismatch; report; do not
  improvise.
- Stay in the repo root for every step. The CLI loads `.env` from
  `Path.cwd()` (and parents) at startup.
- Perspicacité has a strict LLM-key preflight that fails fast if no provider
  key is reachable. The dev-only escape hatch is
  `PERSPICACITE_ALLOW_MISSING_LLM_KEYS=1`.
- **Non-interactive shells (nohup, agents, systemd, CI) do not source `~/.zshrc`.**
  Variables exported there are invisible. Two reliable ways to launch from a
  non-interactive context: (a) put keys in `.env` at the repo root — the CLI
  auto-loads them; or (b) launch via `bash -c '. ~/.zshrc; uv run perspicacite -c config.yml serve'`
  to explicitly source the rc file.

---

## Step 0 — Preconditions

```bash
python3 --version          # need 3.12+ for hatchling+pydantic2 toolchain
which uv && uv --version   # need uv ≥ 0.5; install via `brew install uv` or `pipx install uv`
git --version
```

If `uv` is missing:

```bash
brew install uv   # macOS
# or
curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux
```

---

## Step 1 — Clone

```bash
cd ~/git
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git
cd Perspicacite-AI
```

**Verify:**

```bash
test -f pyproject.toml && test -f config/example.yml && echo OK
```

---

## Step 2 — Install Python deps

```bash
uv sync
```

This creates `.venv/`, installs Perspicacité in editable mode plus all
declared deps (FastAPI, ChromaDB, sentence-transformers, litellm, fastmcp,
etc.). First run downloads ~2 GB; takes 2–5 min.

**Optional extras:**

```bash
uv pip install -e ".[scilex]"      # multi-database literature search
uv pip install -e ".[cookies]"     # institutional PDF access
uv pip install -e ".[code-parsing]" # tree-sitter chunker
uv pip install -e ".[html-ingest]"  # URL-to-KB ingestion
uv pip install -e ".[youtube-ingest]" # YouTube captions
uv pip install -e ".[browser]"     # headless Chromium fallback
```

**Verify:**

```bash
uv run perspicacite version
# Expected: "Perspicacité v2.0.0" (or current)
```

---

## Step 3 — Configure

```bash
cp config/example.yml config.yml
cp .env.example .env
```

Edit `.env` to add at least one provider key. The CLI loads `.env` from the
current working directory at startup; shell-exported variables override.

```bash
# choose the provider that matches config.yml's llm.default_provider
ANTHROPIC_API_KEY=sk-ant-...
# or
DEEPSEEK_API_KEY=sk-...
# or
OPENROUTER_API_KEY=sk-or-v1-...   # config.yml ships with openrouter as default
```

Edit `config.yml` only if you need a non-default provider. The key fields:

```yaml
llm:
  default_provider: "deepseek"   # one of: anthropic, deepseek, openai, openrouter

pdf_download:
  unpaywall_email: "your@email.com"   # required for OA PDF discovery
```

**Verify:**

```bash
# Key present in .env without echoing the secret
grep -E "^(ANTHROPIC|DEEPSEEK|OPENROUTER|OPENAI)_API_KEY=" .env | wc -l
# Expected: 1 or more
```

> **Embeddings note.** `config.yml` defaults to `embedding_model:
> text-embedding-3-small` (OpenAI). Without `OPENAI_API_KEY`, the embedding
> layer transparently falls back to `all-MiniLM-L6-v2`
> (sentence-transformers, local, ~80 MB). The fallback works fine but adds
> ~3 s to first boot for model download/load. For full-speed embeddings,
> add `OPENAI_API_KEY` to `.env` *in addition to* your chat-provider key, or
> set `knowledge_base.embedding_model: "all-MiniLM-L6-v2"` in `config.yml`
> to skip OpenAI entirely.

---

## Step 4 — First serve

```bash
uv run perspicacite -c config.yml serve
```

The startup runs a preflight that fails fast if the provider key is missing.
A clean start looks like:

```
🚀 Starting Perspicacité v2.0.0
   Server: http://0.0.0.0:8000
   MCP: http://0.0.0.0:8000
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify (in another shell):**

```bash
curl -s http://localhost:8000/api/health | head -c 200
# Expected: {"status":"healthy","initialized":true,...}
```

`Ctrl+C` to stop. The MCP endpoint is at the same port + `/mcp`.

---

## Step 5 — Smoke a knowledge base (optional)

Once the server is running, a 30-second smoke:

```bash
# in another shell — create an empty KB
uv run perspicacite -c config.yml create-kb smoke --description "first smoke"
uv run perspicacite -c config.yml list-kb
```

**Verify:**

```bash
uv run perspicacite -c config.yml list-kb | grep smoke
```

To populate by DOI:

```bash
curl -sN -X POST http://localhost:8000/api/kb/smoke/dois/async \
  -H "Content-Type: application/json" \
  -d '{"dois": ["10.1038/s41586-023-06924-6"]}'
```

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `LLM preflight failed: default_provider='X' but X_API_KEY is not set` | no provider key in shell or `.env` | add key to `.env` (CLI auto-loads) or `export X_API_KEY=...` |
| Server starts but `/api/health` never responds | startup hung (e.g. sentence-transformers downloading first-run model) | watch `logs/web_app_*.log`; first start takes 30–60 s |
| `pdf_cookies_missing` warning | paywalled-PDF fetches will fail | (warning only) run `perspicacite import-browser-cookies` if you need them |
| `httpx` resolver conflict on `uv sync` | scilex pins httpx<0.28 vs fastmcp >=0.28 | already resolved via `[tool.uv] override-dependencies` — if you see it, run `uv sync --no-cache` |
| `Address already in use` on :8000 | another process holds the port | `lsof -nP -i :8000` to find owner; either kill it or `uv run perspicacite -c config.yml serve --port 8001` |
| Want to dev without keys | preflight blocks | `PERSPICACITE_ALLOW_MISSING_LLM_KEYS=1 uv run perspicacite -c config.yml serve` |

---

## After install

- 32 MCP tools at `http://localhost:8000/mcp` — full catalog: [`docs/reference/mcp-tools.md`](docs/reference/mcp-tools.md)
- REST API: [`docs/reference/rest-api.md`](docs/reference/rest-api.md)
- CLI reference: [`docs/reference/cli.md`](docs/reference/cli.md)
- Concepts (KBs, RAG modes, capsules, provenance): [`docs/concepts/`](docs/concepts/)
- Recipe book for common workflows: [`docs/recipe-book-2026-05-15.md`](docs/recipe-book-2026-05-15.md)
