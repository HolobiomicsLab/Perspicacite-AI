<div align="center">

# Perspicacité — AI-Powered Scientific Literature Research Assistant

*Local-first RAG system for searching, understanding, and organizing academic literature*

**6 RAG modes** &nbsp;·&nbsp; **Unified content pipeline** &nbsp;·&nbsp; **Hybrid vector+BM25 retrieval** &nbsp;·&nbsp; **MCP server** &nbsp;·&nbsp; **REST API**

[![Paper](https://img.shields.io/badge/Paper-ISWC--C%202025-blue?style=flat-square)](https://iswc2025.semanticweb.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

</div>

---

**Perspicacité** (French for "insight") helps scientists, researchers, and students search, understand, and organize academic literature using AI grounded in real research papers. It works entirely on your machine — only LLM inference calls leave your environment.

## Documentation

- [docs/index.md](docs/index.md) — documentation home with links to all sections
- [docs/VISION.md](docs/VISION.md) — framework vision, design philosophy, architecture overview
- [docs/getting-started.md](docs/getting-started.md) — install, configure, first KB, first question

---

## Features

- **Multi-database search** — Semantic Scholar, OpenAlex, PubMed, arXiv, HAL, DBLP via SciLEx
- **Unified content pipeline** — PMC JATS XML, arXiv HTML, OA PDFs, publisher APIs, and institutional-access via browser-cookie replay; quality-priority routing
- **PDF extraction** — fast PyMuPDF text on every ingest, plus optional [docling](docs/pdf-extraction-docling.md) layout extraction for structured tables/figures (advanced, off by default)
- **6 RAG modes** — Basic, Advanced, Profound, Agentic, Literature Survey, Contradiction; per-stage LLM tiering (Haiku routing/screening, Sonnet synthesis)
- **Knowledge base management** — BibTeX import, DOI bulk-add, local document ingest, Zotero-collection import; async ingestion with SSE progress streaming
- **Citation-graph expansion** — forward + backward snowball over OpenAlex; automatic Semantic Scholar fallback for arXiv-seeded papers (see [docs/concepts/citation-graph.md](docs/concepts/citation-graph.md))
- **Honest sourcing** — `PaperSource` enum records the true origin of every paper (`OPENALEX`, `PUBMED`, `ARXIV`, `CROSSREF`, `SEMANTIC_SCHOLAR`, `BIBTEX`, `LOCAL`) — no generic `WEB_SEARCH` catch-all (see [docs/reference/paper-source-enum.md](docs/reference/paper-source-enum.md))
- **MCP server** — 32 tools at `/mcp` for integration with Mimosa-AI, Claude Code, SmolAgents, Codex
- **REST API** — full JSON API with SSE streaming for async jobs
- **Provenance tracking** — per-answer retrieval trace stored in SQLite; exportable as RO-Crate 1.1 zip
- **Auto KB routing** — `kb_name: "auto"` scores all KBs against your query (BM25 or LLM) and fans across the top-N in parallel
- **Capsule enrichment** — per-paper figures, references, code snippets, and supplementary files indexed alongside main text
- **Long-term preservation** — PDF byte cache, Zotero attachment push (4-step file-upload protocol), BibTeX + PDF folder export, Obsidian vault export
- **Flexible LLM routing** — direct API (Anthropic with prompt caching, OpenAI, DeepSeek, Gemini), Ollama (local), agent-CLI subprocess (Claude Code, Codex, OpenClaw, Hermes), MCP sampling
- **Local-first** — data stays on your machine; no telemetry

---

## Quick Start

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git
cd Perspicacite-AI
uv sync
```

### Configure

```bash
cp config.example.yml config.yml
cp .env.example .env
# Edit .env — add at least one LLM API key
```

| Provider | Key variable | Get a key at |
|----------|-------------|--------------|
| **DeepSeek** (default) | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/) |
| **OpenAI** | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/) |
| **Anthropic** | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/) |

Also set `pdf_download.unpaywall_email` in `config.yml` for open-access PDF discovery.

### Run

```bash
./dev.sh
```

`dev.sh` starts both processes with a single command and stops both on Ctrl+C:

- **`:8000`** — Python/FastAPI backend (REST API + MCP server)
- **`:3000`** — Next.js frontend ← **open this in your browser**

Frontend dependencies (`node_modules`) are installed automatically on the first
run; subsequent starts skip that step.

> **Heads-up — the backend is slow to start.** It loads ML models (PyTorch,
> sentence-transformers) on boot, so expect roughly a minute before it is ready
> (longer on the very first run). On Windows-mounted / WSL filesystems it is
> slower still — keeping the repo on a **native Linux filesystem** speeds
> startup up substantially. The UI shows connection errors until the backend
> finishes booting; just wait for it.

#### Web UI — Next.js on `:3000` (primary)

**http://localhost:3000** is the main UI: streaming chat, CNRS visual identity,
configurable database picker, KB management, and the similarity-expansion
workflow. It proxies `/api/*` to the backend at `:8000` (overridable via
`PERSPICACITE_BACKEND_URL`). **Use this for all everyday research.**

#### Running the two processes separately

If you need to start them in separate terminals (e.g. to attach a debugger):

```bash
# Terminal 1 — backend: REST API on :8000, MCP at /mcp
uv run perspicacite -c config.yml serve

# Terminal 2 — frontend: http://localhost:3000
cd frontend
npm install          # first time only; dev.sh handles this automatically
npm run dev

# To point the frontend at a different backend (e.g. the OpenAI-embedding port):
PERSPICACITE_BACKEND_URL=http://localhost:8002 npm run dev
```

#### Legacy server-rendered UI on `:8000` (deprecated)

`http://localhost:8000` still serves a Jinja-rendered fallback UI that requires
no Node.js. It is **deprecated** — new features are only added to the Next.js
frontend — and will be removed in a future release. Use it only as a last resort
when Node.js is unavailable. The historical pre-fork snapshot is preserved on the
`backup/gui-vanilla-jinja` branch.

#### Optional: Google Scholar via Playwright

Google Scholar is opt-in because it relies on either the [SerpApi](https://serpapi.com/)
managed scraper (recommended, requires `SERPAPI_API_KEY`; free tier ~100/mo)
or headless Chromium via Playwright.

To enable headless Chromium scraping:

```bash
uv sync --extra browser              # installs the playwright Python package
uv run playwright install chromium   # downloads the browser binary (~150 MB)
```

Then in `config.yml`:

```yaml
search:
  enabled_providers:
    # ...
    - google_scholar

google_scholar:
  enabled: true
  headless: true
```

Without both steps the server logs `google_scholar_playwright_missing` and
Scholar silently returns zero results. SerpApi-only setups can skip the
`uv sync --extra browser` and `playwright install` commands — set
`SERPAPI_API_KEY` instead and Scholar will route through SerpApi.

> Keys can live in either the shell environment (`export ANTHROPIC_API_KEY=...`)
> or in `.env` at the project root — the CLI loads `.env` on startup. Shell
> vars take precedence over `.env`. For offline / mocked-LLM development,
> bypass the preflight with `PERSPICACITE_ALLOW_MISSING_LLM_KEYS=1`.

The MCP server is also accessible at `http://localhost:8000/mcp` (streamable HTTP). MCP clients connect there to call tools like `search_knowledge_base`, `generate_report`, and `ingest_asb_run`. See [docs/MCP.md](docs/MCP.md) for the envelope contract and per-tool latency expectations.

### First knowledge base

```bash
# From a BibTeX file
uv run perspicacite -c config.yml create-kb my-kb --from-bibtex refs.bib

# Or from a live literature search (requires SciLEx extra)
uv pip install -e ".[scilex]"
uv run perspicacite -c config.yml search-to-kb --query "diamond magnetometry" --kb sensors --max-results 20
```

Full walkthrough: [docs/getting-started.md](docs/getting-started.md).

---

## Documentation

| Section | What it covers |
|---------|---------------|
| [docs/concepts/](docs/concepts/) | Knowledge bases, RAG modes, capsules, provenance, citation graph |
| [docs/guides/](docs/guides/) | BibTeX import, search-to-KB, citation expansion, Zotero, Obsidian, institutional PDFs |
| [docs/reference/](docs/reference/) | CLI flags, REST endpoints, MCP tools, config schema, PaperSource enum |
| [docs/development/](docs/development/) | Contributing, architecture tour, testing, design workflow |

---

## RAG Modes

Six modes covering different cost/depth trade-offs. See [docs/concepts/rag-modes.md](docs/concepts/rag-modes.md) for when to use each.

| Mode | Best for | Speed |
|------|----------|-------|
| **Basic** | Quick answers from a well-curated KB | Fast |
| **Advanced** | Broader KB search with query expansion and WRRF fusion | Medium |
| **Profound** | Complex multi-faceted questions (3 retrieval cycles) | Slower |
| **Agentic** | Questions requiring live discovery beyond the KB | Variable |
| **Literature Survey** | Mapping a research field (checkpoint/resume supported) | Slowest |
| **Contradiction** | Comparing conflicting findings across papers | Medium |

---

## MCP Server

32 tools at `http://localhost:8000/mcp`. Key tools:

`search_literature` · `get_paper_content` · `search_knowledge_base` · `generate_report` · `add_dois_to_kb` · `build_kb_from_search` · `expand_kb_via_citations` · `push_to_zotero` · `build_capsules_for_kb` · `export_kb`

Full tool catalog: [docs/reference/mcp-tools.md](docs/reference/mcp-tools.md).

For use with Mimosa-AI, Claude Code, SmolAgents, or any MCP-compatible agent, point at `http://localhost:8000/mcp` (streamable-HTTP transport).

---

## REST API

Full JSON API with SSE streaming. Key endpoints:

`POST /api/chat` · `GET /api/kb` · `POST /api/kb/{name}/bibtex/async` · `POST /api/kb/{name}/dois/async` · `GET /api/jobs/{id}/events` · `GET /api/conversations/{id}/export`

Full endpoint list: [docs/reference/rest-api.md](docs/reference/rest-api.md).

---

## CLI Commands

```bash
perspicacite serve                  # Start server (web UI + MCP)
perspicacite create-kb NAME         # Create KB (from BibTeX with --from-bibtex)
perspicacite add-to-kb NAME         # Add papers to existing KB
perspicacite list-kb                # List all KBs
perspicacite query QUESTION         # Ask a question against a KB
perspicacite search-to-kb           # Build KB from a literature search
perspicacite expand-kb              # Grow KB via citation graph
perspicacite export-kb              # Export KB (BibTeX / Obsidian vault)
perspicacite screen-papers          # Score candidates by relevance (no server needed)
perspicacite pubmed-search          # Search PubMed to BibTeX (no server needed)
perspicacite import-browser-cookies # Export session cookies for institutional PDF access
perspicacite check-cookies          # Check cookie freshness
perspicacite build-capsule          # Build per-paper capsule (figures, SI, code)
perspicacite build-capsules         # Build capsules for all papers in a KB
perspicacite version                # Show installed version
```

Full flags and usage: [docs/reference/cli.md](docs/reference/cli.md).

---

## Configuration

Copy `config.example.yml` and edit. Key sections:

```yaml
llm:
  default_provider: "deepseek"
  default_model: "deepseek-chat"

knowledge_base:
  embedding_model: "text-embedding-3-small"
  chunk_size: 1000

pdf_download:
  unpaywall_email: "your@email.com"

mcp:
  enabled: true
```

Full schema: [docs/reference/config.md](docs/reference/config.md).

Ready-made presets live in [`configs/`](configs/README.md): LLM-provider presets under
`configs/llm/` (Ollama, Claude Code, Codex, Hermes, OpenClaw, OpenRouter-free) and
embedding/retrieval presets under `configs/embedders/` (bge-m3, openai-large, specter2, …).
Point at one with `-c`, e.g. `uv run perspicacite -c configs/embedders/bge_m3.yml serve`.

---

## Privacy & Data

- **Your data stays local** — KBs stored in ChromaDB and SQLite on your machine
- **API calls only** — queries sent to your configured LLM provider; no other data leaves your environment
- **No tracking** — no usage analytics

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the CLA workflow and contribution guidelines.
Dev-loop setup, test structure, and architecture: [docs/development/](docs/development/).

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

---

## Citation

<p align="center">
<b>Citation:</b> <em>An AI Pipeline for Scientific Literacy and Discovery: a Demonstration of Perspicacité-AI Integration with Knowledge Graphs</em><br>
L. Pradi, T. Jiang, M. Feraud, M. Bekbergenova, Y. Taghzouti, L.-F. Nothias — <em>ISWC-C 2025</em>
</p>

```bibtex
@inproceedings{pradi2025perspicacite,
  title     = {An AI Pipeline for Scientific Literacy and Discovery: a Demonstration of Perspicacit\'{e}-AI Integration with Knowledge Graphs},
  author    = {Pradi, Lucas and Jiang, Tao and Feraud, Matthieu and Bekbergenova, Madina and Taghzouti, Yousouf and Nothias, Louis-Felix},
  booktitle = {ISWC-C 2025},
  year      = {2025}
}
```

```bibtex
@softwareversion{scilex2026,
  title  = {SciLEx, Science Literature Exploration Toolkit},
  author = {Ringwald, C\'{e}lian and Navet, Benjamin},
  url    = {https://github.com/Wimmics/SciLEx},
  year   = {2026}
}
```

---

## Acknowledgments

- **[SciLEx](https://github.com/Wimmics/SciLEx)** — literature exploration toolkit powering multi-database search
- **[ChromaDB](https://www.trychroma.com/)** — local vector storage
- **[OpenAlex](https://openalex.org/)** and **[Semantic Scholar](https://www.semanticscholar.org/)** — academic search and citation graphs
- **[Unpaywall](https://unpaywall.org/)** — open access discovery
