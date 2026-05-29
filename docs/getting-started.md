# Getting Started

This guide takes you from a bare clone to a working knowledge base with your first
answered research question. Estimated time: under ten minutes.

---

## Requirements

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** — recommended package manager
  (`pip install uv` or `brew install uv`)
- At least one LLM API key (DeepSeek is the default; cheapest to start with)
- Optional: a `.bib` file with some papers to import

---

## 1. Install

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git
cd Perspicacite-AI
uv sync
```

To enable multi-database literature search (Semantic Scholar, OpenAlex, PubMed, arXiv,
HAL, DBLP), install the SciLEx extra:

```bash
uv pip install -e ".[scilex]"
```

SciLEx is optional — all KB-side tools work without it.

---

## 2. Configure

```bash
cp config/example.yml config.yml
cp .env.example .env
```

Edit `.env` and add at least one API key (the CLI auto-loads `.env` from
the current working directory on startup):

```bash
# DeepSeek (default, cheapest)
DEEPSEEK_API_KEY=sk-...

# Or Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Or OpenAI
OPENAI_API_KEY=sk-...
```

> Shell-exported variables (e.g. `export ANTHROPIC_API_KEY=...` in `~/.zshrc`)
> take precedence over `.env`. Either source is fine — pick whichever fits
> your secret-management style.
>
> For offline / mocked-LLM development, set
> `PERSPICACITE_ALLOW_MISSING_LLM_KEYS=1` to bypass the startup preflight.
>
> **Embeddings:** the default `embedding_model: text-embedding-3-small`
> requires `OPENAI_API_KEY`. Without it, embeddings transparently fall
> back to `all-MiniLM-L6-v2` (local sentence-transformers). For
> full-speed embeddings, add `OPENAI_API_KEY` *in addition to* your
> chat-provider key, or set `knowledge_base.embedding_model:
> "all-MiniLM-L6-v2"` in `config.yml`.

The minimal `config.yml` block to edit:

```yaml
llm:
  default_provider: "deepseek"
  default_model: "deepseek-chat"

pdf_download:
  unpaywall_email: "your@email.com"   # required for open-access PDF discovery
```

All other config values have sensible defaults. See
[`docs/reference/config.md`](reference/config.md) for the full schema.

---

## 3. Start the server

```bash
./dev.sh
```

This starts both processes together and stops them both on Ctrl+C:

- **`:8000`** — Python/FastAPI backend (REST API + MCP server)  
- **`:3000`** — Next.js frontend ← **open this in your browser**

Frontend dependencies are installed automatically on the first run.

> **The backend takes ~1 minute to boot** — it loads ML models (PyTorch,
> sentence-transformers) on startup. The UI will show connection errors until
> it is ready; just wait.

The MCP server is available at **http://localhost:8000/mcp** (same port as the
backend, `/mcp` path). Use `--no-mcp` to disable it, `--no-ui` for headless
API-only mode.

**Need separate terminals?** You can start the two processes independently:

```bash
# Terminal 1
uv run perspicacite -c config.yml serve   # backend on :8000

# Terminal 2
cd frontend && npm install && npm run dev  # frontend on :3000 (npm install: first time only)
```

---

## 4. Create your first knowledge base

### Option A: from a BibTeX file (recommended for existing paper collections)

```bash
uv run perspicacite -c config.yml create-kb my-kb --from-bibtex refs.bib --description "My first KB"
```

This downloads full text where available, chunks the content, embeds it, and indexes
it into ChromaDB. A 20-paper BibTeX file typically takes 60-120 seconds depending on
PDF availability.

### Option B: from the web UI

1. Open **http://localhost:3000**
2. Click **"+ Create new KB"** in the left sidebar
3. Enter a name and optional description
4. Drag and drop a `.bib` file, then click **"Create from BibTeX"**

### Option C: empty KB, add papers by DOI later

```bash
uv run perspicacite -c config.yml create-kb my-kb --description "Diamond sensors"
```

Then add papers:

```bash
curl -X POST http://localhost:8000/api/kb/my-kb/dois/async \
  -H "Content-Type: application/json" \
  -d '{"dois": ["10.1038/s41586-023-06924-6", "10.1103/PhysRevLett.131.013001"]}'
```

Poll job progress:

```bash
curl -sN http://localhost:8000/api/jobs/<job_id>/events
```

---

## 5. Ask your first question

### Web UI

Select your KB from the left sidebar, type a question, choose a RAG mode, and press
Enter. For a first test, **Basic** mode is fast and requires no additional LLM calls
beyond synthesis.

### CLI

```bash
uv run perspicacite -c config.yml query "what methods are used to detect magnetometry?" \
  --kb my-kb --mode basic
```

### REST API

```bash
curl -sN -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "what methods are used?", "kb_name": "my-kb", "mode": "basic", "stream": true}'
```

---

## What to read next

- [concepts/knowledge-bases.md](concepts/knowledge-bases.md) — understand KB storage
  and multi-KB routing
- [concepts/rag-modes.md](concepts/rag-modes.md) — choose the right mode for your
  question type
- [guides/ingest-bibtex.md](guides/ingest-bibtex.md) — BibTeX import in depth
- [guides/search-to-kb.md](guides/search-to-kb.md) — build a KB from a literature
  search without a pre-existing `.bib`
- [reference/config.md](reference/config.md) — full configuration reference
