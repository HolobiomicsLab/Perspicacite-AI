# Contributing

This file covers the practical dev-loop for contributors. For licensing requirements
and the CLA workflow, see the top-level
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).

---

## Dev-loop setup

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git
cd Perspicacite-AI

# Install with all dev + test extras
uv sync
uv pip install -e ".[scilex,cookies]"

# Copy config and add at least one LLM key
cp config/example.yml config.yml
cp .env.example .env
```

---

## Running tests

```bash
# Unit tests only (no external services, no API keys)
uv run pytest tests/unit/ -v

# Skip tests requiring live API keys
uv run pytest tests/unit/ -m "not live" -v

# Live integration tests (requires MISTRAL_API_KEY and/or SS API key)
uv run pytest tests/integration/ -v

# All tests with coverage
uv run pytest --cov=src/perspicacite --cov-report=term-missing
```

See [`development/testing.md`](testing.md) for detailed test-gate documentation.

---

## Linting and type checking

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

All three must pass before a PR can be merged (enforced by CI).

---

## Making a change

1. Create a branch from `main`.
2. Make your change with focused, reviewable commits. Each commit should be
   independently meaningful.
3. Run tests, lint, and type check locally.
4. Open a PR using the PR template in `.github/PULL_REQUEST_TEMPLATE.md`.
5. If this is your first external contribution, sign the Individual Contributor
   Agreement (the CLA bot will prompt you in the PR).

---

## Project structure

```
src/perspicacite/
  cli.py                     # all CLI subcommands
  config/schema.py           # Pydantic configuration model
  mcp/server.py              # MCP server with 32 tools
  models/papers.py           # Paper, PaperSource, Author models
  models/kb.py               # KnowledgeBase model
  pipeline/
    download/                # content retrieval pipeline
      discovery.py           # OpenAlex + Unpaywall discovery
      unified.py             # retrieve_paper_content() entry point
      europepmc.py           # PMC JATS XML fetcher
      arxiv.py               # arXiv HTML + PDF
      biorxiv.py             # bioRxiv/medRxiv JATS XML
      crossref.py            # Crossref metadata enrichment
    parsers/pdf.py           # PyMuPDF-based parser
    bibtex_kb.py             # BibTeX → KB pipeline
  rag/
    engine.py                # RAGEngine (routes to mode handlers)
    modes/                   # basic, advanced, profound, agentic, literature_survey, contradiction
    tools/                   # tool registry, KB search, LOTUS
  retrieval/
    multi_kb.py              # MultiKBRetriever — fan-out across multiple KBs
    recency.py               # apply_recency_weighting() — exponential decay by year
  search/
    scilex_adapter.py        # multi-database search via SciLEx
    screening.py             # screen_papers() BM25 + LLM relevance scoring
    pubmed.py                # PubMedSearchAdapter (Biopython Entrez)
  web/                       # FastAPI app, routers, AppState singleton
  integrations/              # Zotero, citation graph, snowball
tests/
  unit/                      # fast tests with no external dependencies
  integration/               # tests requiring live API keys
  audit/                     # symbol-index and end-to-end audit harnesses
```

---

## Documentation

Docs live under `docs/` and are organized into:
- `docs/concepts/` — background reading
- `docs/guides/` — step-by-step walkthroughs
- `docs/reference/` — exact interface specifications
- `docs/development/` — contributor docs (this section)
- `docs/superpowers/` — internal design specs and implementation plans

When you add a CLI flag, REST endpoint, or MCP tool, update the corresponding
reference doc. When you add a feature, consider whether it warrants a guide entry.

---

## Related topics

- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — CLA and legal requirements
- [development/testing.md](testing.md) — test-gate details
- [development/architecture.md](architecture.md) — code tour
- [development/superpowers-workflow.md](superpowers-workflow.md) — design workflow
