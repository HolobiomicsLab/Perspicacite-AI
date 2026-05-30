# Testing

Perspicacité's test suite is organized into three tiers, each with different
dependencies and gate conditions.

---

## Test tiers

### Unit tests (`tests/unit/`)

No external services, no API keys, no running server. Run fast; always run in CI.

```bash
uv run pytest tests/unit/ -v
```

Unit tests cover:
- Config parsing and validation (`config/schema.py`)
- Paper model validation and PaperSource enum
- BibTeX parser edge cases
- Chunking strategy logic
- BM25 scoring
- Snowball deduplication and filter logic
- Provider client mocking

### Integration tests (`tests/integration/`)

Require at least one live API key. Marked with `@pytest.mark.live`.

```bash
# Run all integration tests (requires keys set in .env)
uv run pytest tests/integration/ -v

# Skip tests requiring live keys (safe for offline dev)
uv run pytest tests/ -m "not live" -v
```

**API key gates:**

| Test group | Required key / condition |
|-----------|--------------------------|
| Embedding tests | `OPENAI_API_KEY` or a local embedding model |
| LLM synthesis tests | `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, or `OPENAI_API_KEY` |
| Semantic Scholar fallback | `SS_API_KEY` (optional; tests run without it but at lower rate limit) |
| arXiv fetch | No key needed (arXiv is open) |
| PMC JATS fetch | No key needed (Europe PMC is open) |
| Zotero tests | `ZOTERO_API_KEY` + `PERSPICACITE_ZOTERO_LIBRARY_ID` |

Tests that require keys check for them at the start of the test function and skip with
a clear message if the key is not set.

### Audit tests (`tests/audit/`)

End-to-end harnesses that run against a live server and real databases. These are
used to validate that a code change has not broken the live pipeline for a set of
representative papers. Not run in CI by default.

```bash
# Run the full pipeline audit (requires a running server + Anthropic key)
uv run python tests/audit/run_audit.py
```

Results are written to `tests/audit/results/` as JSON and Markdown.

---

## Running the full suite

```bash
# Full suite with coverage report
uv run pytest --cov=src/perspicacite --cov-report=term-missing

# Coverage XML for CI upload
uv run pytest --cov=src/perspicacite --cov-report=xml:coverage.xml
```

---

## Test structure

```
tests/
  unit/
    test_config.py              # Config parsing + validation
    test_models.py              # Paper, PaperSource, Author models
    test_bibtex.py              # BibTeX parser edge cases
    test_chunking.py            # Chunking strategies
    test_screening.py           # BM25 + LLM screen logic
    test_snowball.py            # Citation-graph filter + dedup
    test_provenance.py          # ProvenanceRecord construction
    test_zotero.py              # Zotero client mocking
  integration/
    test_content_pipeline.py    # PMC, arXiv, Crossref, Unpaywall live fetchers
    test_rag_modes.py           # All 6 RAG modes with a live LLM
    test_mcp_live.py            # MCP tool calls against a running server
    test_kb_lifecycle.py        # Create → ingest → query → delete
  audit/
    run_audit.py                # Full pipeline audit harness
    results/                    # Audit output (JSON + Markdown, git-ignored)
```

---

## CI configuration

CI runs are defined in `.github/workflows/ci.yml`. The CI matrix:
- Runs `uv run pytest tests/unit/ -m "not live"` on every push and PR
- Uses Python 3.12
- Does not run integration or audit tests (no API keys in CI)

---

## Writing new tests

For unit tests:
- Use `pytest` fixtures in `tests/conftest.py` for shared mocks (config, AppState,
  Chroma collection, etc.)
- Mock external HTTP calls with `respx` or `httpx` test clients
- Do not use `asyncio.run` in tests — use `pytest-asyncio` with `@pytest.mark.asyncio`

For integration tests:
- Use `@pytest.mark.live` to gate on API key availability
- Check for the key at the top of the test: `if not os.getenv("OPENAI_API_KEY"): pytest.skip(...)`
- Keep the test focused: one external service call per test function

---

## Related topics

- [development/contributing.md](contributing.md) — setting up the dev environment
- [MANUAL_QA.md](MANUAL_QA.md) — manual QA checklist for UI features
