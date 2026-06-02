# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Perspicacité is a local-first, AI-powered scientific literature research assistant. It exposes a FastAPI web app, a REST API, and an MCP server (40+ tools) that can be consumed by external agents (e.g., Mimosa-AI). Users query academic databases, build personal knowledge bases (KBs) from BibTeX or DOIs, and answer research questions using one of six RAG modes.

## Environment

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync --dev           # install all dependencies including dev extras
cp .env.example .env    # then add at least one LLM API key
```

Configuration lives in `config.yml` (git-ignored; copy from `config.example.yml`). The app reads it at startup via `perspicacite.config.loader.load_config()`.

## Commands

```bash
# Run the server (web UI + MCP at /mcp)
uv run perspicacite -c config.yml serve

# With hot-reload for development
uv run perspicacite -c config.yml serve --reload

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Unit tests (no external services needed)
uv run pytest tests/unit/ -v

# Single test file
uv run pytest tests/unit/test_config.py -v

# Single test by name
uv run pytest tests/unit/test_config.py::test_default_config -v

# Skip tests requiring live API keys
uv run pytest tests/unit/ -m "not live" -v

# Live MCP tests (requires running server on port 8000)
uv run python tests/test_mcp_live.py --all --port 8000
uv run python tests/test_mcp_live.py --test search

# Phase 2 CLI subcommands (no server needed)
uv run perspicacite screen-papers --input refs.bib --candidates cand.bib --output out.bib
uv run perspicacite pubmed-search --query "microbiome" --max-results 50 --output hits.bib
```

## Architecture

### Startup and State

`AppState` in [src/perspicacite/web/state.py](src/perspicacite/web/state.py) is a process-wide singleton. On FastAPI lifespan startup it initializes every subsystem in order: LLM client → embedding provider → ChromaDB vector store → tool registry → `AgenticOrchestrator` → `RAGEngine` → SQLite session store → PDF downloader/parser. All routers import the module-level `app_state` instance.

### RAG Modes

`RAGEngine` ([src/perspicacite/rag/engine.py](src/perspicacite/rag/engine.py)) is the single entry point for all RAG work. It holds a dict of `{RAGMode → BaseRAGMode handler}` and dispatches `execute()` / `execute_stream()` to the correct one. The six modes are:

| Mode | Key behaviour |
|------|--------------|
| `basic` | Single-pass vector retrieval, no rerank |
| `advanced` | Query expansion, WRRF fusion scoring, BM25+vector hybrid, rerank |
| `deep_research` | Multi-cycle (up to 3 iterations) with planning + reflection (formerly `profound`) |
| `agentic` | Intent-classified, tool-using, up to 5 iterations |
| `literature_survey` | Broad search → theme clustering → AI paper recommendations |
| `contradiction` | Multi-paper claim clustering → agreement / disagreement / open-question brief; degrades gracefully (<3 papers → normal answer + note) |

All modes extend `BaseRAGMode` ([src/perspicacite/rag/modes/base.py](src/perspicacite/rag/modes/base.py)) and must implement `execute()` and `execute_stream()`.

**Recency-weighted retrieval:** `RAGRequest` accepts an optional `recency_weight` (float 0..1) and `recency_half_life_years` field. `retrieval/recency.py` `apply_recency_weighting()` applies exponential decay by `year`; it is a no-op when `recency_weight` is None or 0. Wired into all six RAG modes. See `retrieval/recency.py` → `apply_recency_weighting()`.

**Multi-KB query:** `RAGRequest.kb_names: list[str] | None` lets a caller fan a query across multiple knowledge bases simultaneously. `retrieval/multi_kb.py` `MultiKBRetriever` queries each KB's ChromaDB collection in parallel, merges results by score, deduplicates by `paper_id`, and tags each chunk with `kb_name`. `check_embedding_compat(kb_metas)` refuses to proceed (returns an error string) when the requested KBs were embedded with different models. `BaseRAGMode._build_kb_retriever(request, vector_store, embedding_provider)` returns a `MultiKBRetriever` when `len(kb_names) > 1`, else a `DynamicKnowledgeBase`. Wired into `basic`, `contradiction`, `advanced`, `deep_research`, and `agentic` modes. `literature_survey` accepts `kb_names` and now retrieves semantically similar papers from ALL provided KBs before the survey (pre-filtering already-known papers out of the broad search), and stores final-recommendation DOIs as lightweight reference rows in `kb_paper_references` for every KB beyond the first. These references can be ingested later via `add_dois_to_kb`. `RAGEngine` receives an optional `session_store` kwarg and injects it into `LiteratureSurveyRAGMode`. `SourceReference.kb_name: str | None` tags each source reference with its originating KB. The chat router runs `check_embedding_compat` before streaming and emits an error SSE event on mismatch; the `generate_report` and `search_knowledge_base` MCP tools also accept an optional `kb_names` list with the same compat check.

**Chat router:** `web/routers/chat.py` `RAG_MODE_MAP` includes `"contradiction"`. `generate_report` MCP tool (`mcp/server.py`) accepts `mode="contradiction"`, `recency_weight: float = 0.0`, and `kb_names: list[str] | None`.

The `AgenticOrchestrator` ([src/perspicacite/rag/agentic/orchestrator.py](src/perspicacite/rag/agentic/orchestrator.py)) handles the `agentic` mode in detail: intent classification → planning → multi-step execution with map-reduce paper summarisation (capped at `MAP_REDUCE_MAX_PAPERS = 8`) → optional replan (max `MAX_REPLANS = 3`).

### Content Retrieval Pipeline

`retrieve_paper_content()` in [src/perspicacite/pipeline/download/unified.py](src/perspicacite/pipeline/download/unified.py) is the single function for fetching paper content from a DOI. It tries sources in priority order:

1. **Discovery** — OpenAlex + Unpaywall (gets PMCID, arXiv ID, OA URLs, abstract); Crossref fills any missing title/authors/year/journal/abstract without overwriting discovery values (`pipeline/download/crossref.py`).
2. **Alternative endpoint** — user-configured mirror (optional)
3. **Structured full text** — PMC JATS XML → arXiv HTML → bioRxiv/medRxiv JATS XML (preserves sections + references; `pipeline/download/biorxiv.py`; `content_source` = `"biorxiv"`/`"medrxiv"`)
4. **PDF full text** — OA URL → arXiv PDF → Unpaywall → publisher APIs (ACS, RSC, AAAS, Springer, Wiley, Elsevier)
5. **Abstract only** — from discovery metadata (bioRxiv/medRxiv abstract used as fallback when discovery returns none)
6. **Discard** — returns `PaperContent(success=False)`

Publisher API keys are passed as kwargs; missing keys skip that source gracefully. Check `content_type` in the result: `"structured"` > `"full_text"` > `"abstract"` > `"none"`.

### PDF extraction backends

PDF **text** is always extracted with PyMuPDF (`fitz`) → `pdfplumber` fallback (`pipeline/parsers/pdf.py`) — fast, default, on every ingest. **Structured tables + figures** are an opt-in advanced layer via **docling**, off by default (`knowledge_base.docling_extract_tables_figures`, guarded by `docling_max_pages` / `docling_timeout_s`; needs `uv sync --extra docling`). When enabled on the local-file ingest path, docling adds `content_type="table"` chunks on top of the fitz text; if docling is absent / oversized / times out, the text is unaffected. docling is CPU-bound (~min/page) and the MPS/GPU path is unusable on Apple Silicon. Full details: [docs/pdf-extraction-docling.md](docs/pdf-extraction-docling.md).

### Retrieval

`ChromaVectorStore` ([src/perspicacite/retrieval/chroma_store.py](src/perspicacite/retrieval/chroma_store.py)) wraps ChromaDB. KB collections are named via `chroma_collection_name_for_kb()` from `models/kb.py`. The hybrid retriever ([src/perspicacite/retrieval/hybrid.py](src/perspicacite/retrieval/hybrid.py)) combines ChromaDB cosine scores with BM25Okapi scores; weights default to 0.5/0.5 but can optionally be determined by the LLM at query time. `MultiKBRetriever` ([src/perspicacite/retrieval/multi_kb.py](src/perspicacite/retrieval/multi_kb.py)) fans a query across multiple KB collections, merges by score, deduplicates by `paper_id`, and tags results with `kb_name`; use `check_embedding_compat(kb_metas)` to validate that all queried KBs share the same embedding model before retrieval.

### Web App

FastAPI app is defined in [src/perspicacite/web/app.py](src/perspicacite/web/app.py). Routers live under [src/perspicacite/web/routers/](src/perspicacite/web/routers/): `chat`, `conversations`, `health`, `kb`, `survey`. The single-page UI is served from `templates/index.html`; CSS and JS are in `static/css/` (6 files) and `static/js/` (10 files). After editing static assets, hard-refresh the browser (Cmd+Shift+R) to bypass cache.

**Phase 5 additions:**
- `GET /api/kb/{name}/stats` — KB statistics: paper/chunk counts, by-year histogram, by-source and by-content-type breakdowns, top journals, embedding model.
- `GET /api/paper?doi=...` — live-fetches discovery metadata + abstract for a DOI via the unified pipeline; reports which `content_type` is available without adding the paper to any KB.
- `GET /api/conversations/search?q=...` — full-text search across all conversations; uses SQLite FTS5 over message content with a LIKE fallback. `SessionStore.init_db()` creates the FTS5 shadow table idempotently; `add_message()` keeps it in sync.
- `GET /api/conversations/{id}/export?format=markdown` — downloadable Markdown rendering of a conversation (Q&A turns + cited sources/references footer).
- New static JS files: `static/js/kb_stats.js` (KB stats tab) and `static/js/paper_detail.js` (paper-detail slide-over panel + pipeline-step badges on chat source cards).
- `MANUAL_QA.md` (git-tracked, repo root) — human click-through checklist for Phase 5 features.

### MCP Server

Defined in [src/perspicacite/mcp/server.py](src/perspicacite/mcp/server.py) using `fastmcp`. It has its own `MCPState` singleton (separate from `AppState`) and is mounted at `/mcp` by the CLI. Tool usage patterns are documented in [docs/perspicacite_skills.md](docs/perspicacite_skills.md) (and the live `get_usage_guide` tool is the source of truth).

The claim/standardization tools — `extract_claims_from_passages` and `export_astra` — require the optional **`indicia`** extra (`uv sync --extra indicia`), which pulls in the local [`indicium`](../indicium) standard package (typed claims: Bucur 5-slot SuperPattern + ECO/CiTO/SEPIO; SHACL-validated). They degrade with a clear error when the extra isn't installed.

For using Perspicacité over MCP intelligently (query shaping, tool/mode choice), follow [.claude/skills/perspicacite-mcp/SKILL.md](.claude/skills/perspicacite-mcp/SKILL.md) or call the `get_usage_guide` tool for the live source of truth.

### Configuration

Pydantic models in [src/perspicacite/config/schema.py](src/perspicacite/config/schema.py) define the full config hierarchy: `Config → {ServerConfig, MCPConfig, LLMConfig, KnowledgeBaseConfig, RAGModesConfig, SciLexConfig, PDFDownloadConfig, ...}`. LLM access goes through LiteLLM (`llm/client.py`), supporting DeepSeek (default), OpenAI, Anthropic, and MiniMax.

### Persistence

- **ChromaDB** (`./chroma_db/`) — vector chunks per KB collection
- **SQLite** (`./data/perspicacite.db` via `aiosqlite`) — KB metadata, conversations, sessions
- **File cache** (`./data/papers/`) — cached reference JSON per DOI

## Key Design Patterns

- **`structlog`** is used project-wide (`from perspicacite.logging import get_logger`). Log events use keyword arguments, not f-strings: `logger.info("event_name", key=val)`.
- **`AsyncLLMClient`** (`llm/client.py`) wraps LiteLLM. Always call it via `await client.complete(messages=[...])`.
- All RAG mode `execute_stream()` methods yield `StreamEvent` objects; errors are yielded as `StreamEvent(event="error", ...)` rather than raised.
- `tests/unit/` — pure unit tests (no external services, use pytest markers). Top-level `tests/test_*.py` — integration/live tests requiring running services or API keys; exclude with `-m "not live"`.

## Domain Rule Files

Topic-specific development rules live in `docs/rules/` (git-ignored). Read the relevant file before modifying a subsystem:

- [docs/rules/rag_development.md](docs/rules/rag_development.md) — Adding/modifying RAG modes, streaming contract, retrieval stack, prompts, conversation context
- [docs/rules/content_pipeline.md](docs/rules/content_pipeline.md) — Download pipeline, adding publisher modules, PaperContent structure, DOI handling pitfalls
- [docs/rules/api_web.md](docs/rules/api_web.md) — AppState singleton, router conventions, SSE streaming, static assets, MCP co-hosting, session store
- [docs/rules/testing.md](docs/rules/testing.md) — Test layout, markers, shared fixtures, mocking patterns, coverage

## Self-Use via MCP

`.mcp.json` (git-ignored) configures Claude Code to call Perspicacite's own 10 MCP tools when the server is running. This is useful during development to test KB operations, search literature for context, or verify pipeline output:

```bash
# Start the server, then Claude Code can call tools like:
# perspicacite:search_literature, perspicacite:get_paper_content,
# perspicacite:search_knowledge_base, perspicacite:generate_report,
# perspicacite:screen_papers, perspicacite:add_dois_to_kb
uv run perspicacite -c config.yml serve
```

The MCP server is at `http://localhost:8000/mcp`. Tool reference: [docs/perspicacite_skills.md](docs/perspicacite_skills.md).

## Development Log

Meaningful changes are recorded in `AGENT_LOG.md` (git-ignored). Add an entry when making architectural changes, adding subsystems, or completing a significant feature.
