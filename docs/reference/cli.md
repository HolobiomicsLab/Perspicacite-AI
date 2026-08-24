# CLI Reference

All CLI commands follow the pattern:

```bash
perspicacite [-c config.yml] [-v] <subcommand> [options]
```

Global flags:
- `-c, --config PATH` — path to `config.yml` (default: `config.yml` in the current directory)
- `-v, --verbose` — enable verbose/debug logging

Structured JSON logs go to stderr; clean output (DOI lists, JSON results, progress
summaries) goes to stdout so you can pipe into `jq`, `tee`, etc.

---

## `serve`

Start the web server (UI + REST API + MCP server).

```bash
perspicacite -c config.yml serve [--host HOST] [--port PORT] [--reload] [--no-mcp] [--no-ui]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host HOST` | from config | Bind address (e.g., `0.0.0.0` or `127.0.0.1`) |
| `--port, -p PORT` | from config | Port number |
| `--reload` | false | Enable auto-reload (development only) |
| `--no-mcp` | false | Disable MCP server |
| `--no-ui` | false | Headless mode (REST API only, no web UI) |

---

## `create-kb`

Create a new knowledge base, optionally importing from BibTeX.

```bash
perspicacite -c config.yml create-kb NAME [--description TEXT] [--from-bibtex FILE]
```

| Flag | Description |
|------|-------------|
| `NAME` (positional) | KB name (alphanumeric + hyphens/underscores) |
| `--description, -d TEXT` | Human-readable description |
| `--from-bibtex, -b FILE` | BibTeX file to import on creation |
| `--session-db PATH` | Override default SQLite path |
| `--chroma-dir PATH` | Override default Chroma directory |

Without `--from-bibtex`, creates an empty KB. Papers can be added later with
`add-to-kb` or via the REST API.

---

## `add-to-kb`

Add papers to an existing KB.

```bash
perspicacite -c config.yml add-to-kb KB_NAME [--from-bibtex FILE]
```

---

## `list-kb`

List all knowledge bases with paper counts.

```bash
perspicacite list-kb [--json]
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON array |

---

## `query`

Ask a question against a KB using RAG.

```bash
perspicacite -c config.yml query QUESTION --kb KB_NAME [--mode MODE]
```

| Flag | Default | Description |
|------|---------|-------------|
| `QUESTION` (positional) | — | The research question |
| `--kb KB_NAME` | — | KB to query (required) |
| `--mode MODE` | `basic` | RAG mode: `basic`, `advanced`, `profound`, `contradiction` |

---

## `ingest-local`

Ingest local PDFs, Markdown, or code files into a KB.

```bash
perspicacite -c config.yml ingest-local --kb KB_NAME [--path PATH]
```

The path must be under a configured `local_docs.allowed_roots` entry in `config.yml`.

---

## `screen-papers`

Score candidate papers by relevance to a query. No server required.

```bash
perspicacite -c config.yml screen-papers \
  --input refs.bib \
  --candidates cand.bib \
  --output out.bib \
  [--threshold 0.3] \
  [--csv]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input FILE` | — | BibTeX of the reference set (defines the topic) |
| `--candidates FILE` | — | BibTeX of candidates to screen |
| `--output FILE` | — | Filtered BibTeX output |
| `--threshold FLOAT` | 0.3 | Minimum BM25 score to include |
| `--csv` | false | Also write a CSV score sheet |

---

## `pubmed-search`

Search PubMed and export results to BibTeX. No server required.

```bash
perspicacite -c config.yml pubmed-search \
  --query "microbiome" \
  --max-results 50 \
  --output hits.bib
```

---

## `build-capsule`

Build a capsule for a single paper (figures, references, code, SI).

```bash
perspicacite -c config.yml build-capsule --paper-id DOI_OR_ID --kb KB_NAME [--force]
```

---

## `build-capsules`

Build capsules for all papers in a KB (idempotent).

```bash
perspicacite -c config.yml build-capsules --kb KB_NAME [--force]
```

---

## `fetch-resources`

Mine and fetch external resources (GitHub, Zenodo, Crossref) for a paper.

```bash
perspicacite -c config.yml fetch-resources --paper-id DOI_OR_ID --kb KB_NAME
```

---

## `import-browser-cookies`

Export session cookies from a browser for institutional-access PDF downloads.

```bash
perspicacite import-browser-cookies \
  --browser BROWSER \
  --domain DOMAIN [--domain DOMAIN ...] \
  --output FILE
```

| Flag | Description |
|------|-------------|
| `--browser` | Browser name: `chrome`, `brave`, `firefox`, `edge`, `safari`, `opera`, `arc` |
| `--domain DOMAIN` | Domain to filter cookies for (repeatable). Required unless `--all-domains` |
| `--all-domains` | Export every cookie in the browser. See the warning below |
| `--output FILE` | Output path for Netscape `cookies.txt` |

At least one `--domain` is required. Without a filter the export would write
every logged-in session the browser holds — mail, bank, everything — as
plaintext, so that needs the explicit `--all-domains` flag.

Requires `uv pip install -e ".[cookies]"`.

---

## `check-cookies`

Check freshness of cookies in the configured cookies file.

```bash
perspicacite check-cookies
```

Exits non-zero if any configured domain has expired cookies.

---

## `search-to-kb`

Search academic databases and ingest results into a KB.

```bash
perspicacite -c config.yml search-to-kb \
  --query QUERY \
  --kb KB_NAME \
  [--max-results N] \
  [--min-year YEAR] \
  [--max-year YEAR] \
  [--min-citations N] \
  [--require-abstract] \
  [--screen bm25|llm] \
  [--screen-threshold FLOAT] \
  [--kb-aware] \
  [--rephrase N] \
  [--resolve-missing-dois] \
  [--resolve-doi-budget N] \
  [--resolve-doi-browser] \
  [--dry-run]
```

`--resolve-missing-dois` recovers a verified DOI from the title for hits that
arrive without one (Google Scholar above all), which would otherwise be dropped
as `no_doi`. Bounded by `--resolve-doi-budget` (default 25).

Requires SciLEx: `uv pip install -e ".[scilex]"`. See
[guides/search-to-kb.md](../guides/search-to-kb.md).

---

## `delete-kb`

Permanently delete a KB (metadata + Chroma collection).

```bash
perspicacite delete-kb KB_NAME
```

Cached PDFs under `data/papers/` are not deleted.

---

## `expand-kb`

Grow a KB by following forward/backward citation links.

```bash
perspicacite -c config.yml expand-kb \
  --kb KB_NAME \
  [--direction forward|backward|both] \
  [--max-per-seed N] \
  [--min-year YEAR] \
  [--min-citations N] \
  [--screen bm25|llm] \
  [--screen-threshold FLOAT] \
  [--dry-run]
```

See [guides/expand-via-citations.md](../guides/expand-via-citations.md).

---

## `enrich-cite-graph`

Update citation metadata (counts, references) for papers already in a KB.

```bash
perspicacite -c config.yml enrich-cite-graph --kb KB_NAME
```

---

## `export-kb`

Export a KB as BibTeX + PDF folder or Obsidian vault.

```bash
perspicacite -c config.yml export-kb \
  --kb KB_NAME \
  --output DIR \
  [--format bibtex|obsidian-vault] \
  [--with-pdfs] \
  [--with-supplementary]
```

---

## `version`

Print the installed version.

```bash
perspicacite version
```

---

## Related topics

- [getting-started.md](../getting-started.md) — first-run walkthrough
- [reference/config.md](config.md) — configuration reference
- [reference/mcp-tools.md](mcp-tools.md) — equivalent MCP tool signatures
