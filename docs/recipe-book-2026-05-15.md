# Perspicacité recipe book

Task-oriented "how to do X" pages for the CLI, MCP, and Web surfaces.
Each recipe is self-contained — copy-paste the commands and adapt the
KB names / DOIs to your case.

The reference README documents every flag exhaustively. This document
documents *intent* — "what should I run when I want to ...?"

> **Prereqs:** `pip install -e .`, `perspicacite serve` is reachable
> at `http://localhost:8000` for the recipes that need it, and a
> `.env` with at least one `*_API_KEY` is in place. See
> `config.example.yml` for the canonical config layout.

---

## Recipe 1 — Build a KB from a BibTeX file

Goal: hand a `.bib` file (e.g. exported from Zotero) to Perspicacité
and end up with a queryable knowledge base.

```bash
perspicacite create-kb my-kb \
    --description "Papers I care about for project X" \
    --bibtex path/to/library.bib
```

What happens:
1. Each `@article` in the BibTeX gets resolved to a DOI (via Crossref
   if missing).
2. PDFs are fetched (Unpaywall → publisher → user-configured
   `alternative_endpoint` fallback; the chain order is in `config.yml`).
3. Each PDF is parsed (text + figures/tables if Wave 4.1 multimodal is
   enabled).
4. Chunks are embedded and persisted to the `my-kb` Chroma collection.
5. Provenance event log (Wave 4.3) records each `paper_added` event.

To verify:

```bash
perspicacite list-kb
perspicacite query --kb my-kb "what does paper X say about Y?"
```

---

## Recipe 2 — Build a KB from a search query

Goal: "find me ~30 papers about exoplanet biosignatures and put them
in a KB called `astrobio`."

```bash
perspicacite search-to-kb astrobio \
    --query "exoplanet biosignature spectroscopy" \
    --max-papers 30 \
    --year-min 2020
```

The `--year-min` / `--year-max` filters (Wave 4.2) constrain the
search to a date range before the SciLEx call. The chain:

```
SciLEx (multi-DB search) → relevance screening → PDF fetch →
ingest_dois_into_kb → checkpoint (Wave 3.3 — resumable)
```

If the run is interrupted (network drops, rate limit), re-run the
same command — the checkpoint store skips already-ingested papers.

---

## Recipe 3 — Ingest local PDFs you already have

Goal: a folder full of PDFs on disk.

```bash
perspicacite ingest-local my-kb path/to/pdf-folder/
```

Each PDF is parsed, chunked, embedded, and added. No network calls
beyond the embedding API (or local sentence-transformers if
`knowledge_base.embedding_model` is set to a local model).

---

## Recipe 4 — Expand a KB by following citations

Goal: you have 20 seed papers; pull in their bibliographies (1 hop).

```bash
perspicacite expand-kb my-kb --max-new 50 --depth 1
```

The expander reads `references` from each seed paper, deduplicates
against papers already in the KB, fetches the new DOIs, ingests them,
and writes `paper_added` events to the KB log under the
`expand_kb_via_citations` source command.

---

## Recipe 5 — Generate a literature review

Goal: a polished synthesis report over a KB.

**MCP (Claude Desktop / Cursor):**

Use the Wave 5.2 prompt `literature_review`:

```
/literature_review topic="self-supervised representations for cryo-EM"
                  kb_name="cryoem"
                  max_papers=25
```

The prompt instructs the model to call `search_knowledge_base` then
`generate_report` with style "literature review". The model handles
the tool-orchestration.

**CLI:**

```bash
perspicacite query --kb cryoem \
    --style literature_review \
    --max-papers 25 \
    "self-supervised representations for cryo-EM"
```

Output: a multi-section report with citations, written to stdout (or
`--out report.md`).

---

## Recipe 6 — Compare two papers head-to-head

Goal: side-by-side comparison of two papers.

```
/compare_papers paper_a="10.1234/foo" paper_b="10.5678/bar"
```

The prompt fetches both via `get_paper_content`, builds a table
(research question / methods / dataset / findings / limitations /
reproducibility), and writes a 2-paragraph synthesis.

If both papers live in a KB, add `kb_name=...` so the model can pull
related context.

---

## Recipe 7 — Screen a KB for relevance to a topic

Goal: "which papers in `mykb` are about retrieval-augmented
generation?"

```
/screen_topic topic="retrieval-augmented generation"
              kb_name="mykb"
              threshold=0.6
```

The model calls `screen_papers` and reports above-threshold matches
ranked by score with a one-line rationale each.

CLI variant:

```bash
perspicacite screen-papers --kb mykb --topic "retrieval-augmented generation" \
                          --threshold 0.6
```

---

## Recipe 8 — Resume an interrupted ingest

Goal: a 100-DOI ingest crashed at paper 42. Re-run without re-fetching
papers 1–41.

```bash
# Same command as before — the checkpoint store handles the rest.
perspicacite search-to-kb my-kb --query "..." --max-papers 100
```

Wave 3.3 atomic checkpoint files survive SIGKILL mid-write. Status:

```bash
cat .perspicacite/checkpoints/<run-id>.json
```

To start fresh anyway: delete the checkpoint file before re-running.

---

## Recipe 9 — Export a KB for sharing

Goal: produce a BibTeX file + CSL-JSON + RIS that someone can import
into their own reference manager.

```bash
perspicacite export-kb my-kb --formats bibtex,csl,ris --out exports/
```

Wave 4.5 ships three formats simultaneously:

- `exports/my-kb.bib` (BibTeX with DOI, year, authors, title).
- `exports/my-kb.csl.json` (CSL JSON — drag into Zotero / Mendeley).
- `exports/my-kb.ris` (RIS — older but widely supported).

Add `--with-pdfs` to also copy the cached PDFs into a sibling folder.

---

## Recipe 10 — Browse a KB before querying

Goal: in Claude Desktop, you've connected the Perspicacité MCP and
you want to know what KBs exist before issuing a query.

The Wave 5.1 resources let the client browse:

- `perspicacite://kbs` — index of all KBs.
- `perspicacite://kb/{name}` — single KB metadata + summary URIs.
- `perspicacite://kb/{name}/papers` — paper IDs + titles.
- `perspicacite://kb/{name}/log` — append-only event history.

Claude Desktop surfaces these in its resource browser. You can
preview the KB before writing the prompt that uses it.

---

## Recipe 11 — Switch to local LLMs (no API keys)

Goal: run the synthesis stage on Ollama instead of Claude/OpenAI.

In `config.yml`:

```yaml
llm:
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "llama3.1:70b"
  providers_per_stage:
    synthesis: "ollama"            # or a chain:
    synthesis_heavy: ["ollama", "anthropic"]   # fallback to API if Ollama fails (Wave 3.2)
```

Ensure `ollama serve` is running. The fallback chain (Wave 3.2) lets
you keep API access as a safety net.

---

## Recipe 12 — Cap spend per run

Goal: never let a run exceed $5 in LLM costs.

In `config.yml`:

```yaml
llm:
  budget_usd_per_run: 5.0
```

Wave 2.4 tracks accumulated cost via the `BudgetTracker` contextvar.
When the cap is hit, the run aborts with a `BudgetExceededError` and
exits cleanly. Already-completed work (ingested chunks, written
checkpoints) is preserved.

To monitor mid-run:

```bash
# The budget tracker logs to stdout when warn_at thresholds (75%, 90%)
# are crossed. Tail the agent log.
tail -f .perspicacite/logs/agent.log | grep budget
```

---

## Recipe 13 — Use Claude Code (or Codex) instead of the API

Goal: route the synthesis stage through the local `claude` CLI
binary instead of the Anthropic API.

In `config.yml`:

```yaml
llm:
  providers:
    claude_cli:
      type: agent_cli
      binary: claude
      args: ["--print", "--output-format", "json"]
      usage_input_tokens_path: "usage.input_tokens"
      usage_output_tokens_path: "usage.output_tokens"
  providers_per_stage:
    synthesis: claude_cli
```

Wave 2.3 parses the token-usage fields out of the JSON output so the
budget tracker stays accurate even on the CLI path. Caveats live in
`docs/agent-cli-caveats.md`.

---

## Recipe 14 — Disambiguate authors via ORCID

Goal: turn `"Smith J."` into a canonical ORCID ID.

The Wave 4.4 resolver is exposed as a Python API:

```python
from pathlib import Path
from perspicacite.pipeline.orcid import AuthorResolver

resolver = AuthorResolver(cache_path=Path("data/orcid_cache.db"))
res = await resolver.resolve("Smith J.")
# AuthorResolution(orcid="0000-0001-...", display_name="John Smith",
#                  works_count=147, confidence=0.74)
```

Tune `kb.orcid_confidence_threshold` (default 0.20) to trade off
recall vs precision.

---

## Recipe 15 — Run the regression suite before a release

Goal: sanity-check that nothing has broken.

```bash
# Fast unit tests
pytest tests/unit/ -v

# E2E pipelines (deterministic mocks)
pytest tests/e2e/ -v

# Persistence + concurrency
pytest tests/integration/test_persistence_integrity.py -v

# Perf regression check vs stored baseline
pytest tests/integration/test_perf_baseline.py -m perf -v

# Provider matrix (live; needs API keys — slow)
pytest tests/integration/test_provider_matrix.py -v -m live
```

If perf drifts on a faster / slower machine, regenerate:

```bash
PERSPICACITE_UPDATE_PERF_BASELINE=1 pytest tests/integration/test_perf_baseline.py -m perf
git add tests/data/perf_baseline.json && git commit -m "perf: refresh baseline on <machine>"
```

---

## Cross-references

- Multi-paper-citation integration: `docs/versioned-kbs-2026-05-14.md`
- LLM cache + cost accounting: `docs/llm-cache-2026-05-14.md`, `docs/budget-caps-2026-05-14.md`
- Rate-limit handling + provider fallback: `docs/rate-limit-2026-05-14.md`, `docs/fallback-chain-2026-05-14.md`
- Checkpoint / resume: `docs/checkpoint-resume-2026-05-14.md`
- Multimodal (figure/table) ingest: `docs/multimodal-extraction-2026-05-14.md`
- ORCID disambiguation: `docs/orcid-disambiguation-2026-05-14.md`
- Architecture overview: `docs/architecture-2026-05-15.md`
