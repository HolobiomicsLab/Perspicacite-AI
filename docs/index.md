# Perspicacité — Documentation

Perspicacité is a local-first RAG framework for searching, understanding, and
organizing academic literature. It exposes a web UI, a REST API, a CLI, and an MCP
server — all backed by the same retrieval and reasoning engine.

## Where to start

- **New here?** Read [getting-started.md](getting-started.md) — install, configure,
  create your first KB, and ask your first question in under ten minutes.
- **Want to understand the design?** Read [VISION.md](VISION.md) — the problem this
  project addresses, the design philosophy, and the architecture overview.
- **Looking for a specific feature?** Browse the sections below.

---

## Documentation sections

### Concepts

Background reading for understanding how Perspicacité works internally.

| File | What it covers |
|------|---------------|
| [concepts/knowledge-bases.md](concepts/knowledge-bases.md) | What a KB is, how to create one, multi-KB routing |
| [concepts/rag-modes.md](concepts/rag-modes.md) | The six RAG modes and when to use each |
| [concepts/capsules.md](concepts/capsules.md) | Per-paper capsule structure: figures, references, code, SI |
| [concepts/provenance.md](concepts/provenance.md) | How the retrieval trace is built, where it lives, how to export it |
| [concepts/citation-graph.md](concepts/citation-graph.md) | Snowball walks, OpenAlex + Semantic Scholar fallback, arXiv handling |

### Guides

Step-by-step walkthroughs for common tasks.

| File | What it covers |
|------|---------------|
| [guides/ingest-bibtex.md](guides/ingest-bibtex.md) | Import a `.bib` file into a knowledge base |
| [guides/search-to-kb.md](guides/search-to-kb.md) | The search → screen → ingest workflow |
| [guides/expand-via-citations.md](guides/expand-via-citations.md) | Growing a KB by following citations |
| [guides/zotero-integration.md](guides/zotero-integration.md) | Pushing papers to Zotero, building KBs from collections |
| [guides/obsidian-export.md](guides/obsidian-export.md) | Exporting a KB as an Obsidian Markdown vault |
| [guides/institutional-pdf-access.md](guides/institutional-pdf-access.md) | Browser cookie flow for paywalled PDFs |

### Reference

Exact interface specifications: flags, endpoints, tool signatures, config keys.

| File | What it covers |
|------|---------------|
| [reference/cli.md](reference/cli.md) | All `perspicacite <subcommand>` calls and their flags |
| [reference/rest-api.md](reference/rest-api.md) | REST endpoints, request/response shapes |
| [reference/mcp-tools.md](reference/mcp-tools.md) | The 32 MCP tools |
| [reference/config.md](reference/config.md) | `config.yml` schema with all fields and defaults |
| [reference/paper-source-enum.md](reference/paper-source-enum.md) | All `PaperSource` values and when each is used |

### Development

For contributors and people extending the codebase.

| File | What it covers |
|------|---------------|
| [development/contributing.md](development/contributing.md) | Dev-loop setup, CLA workflow |
| [development/architecture.md](development/architecture.md) | Code tour of the layered source tree |
| [development/testing.md](development/testing.md) | How to run pytest, unit vs integration, API-key gates |
| [development/superpowers-workflow.md](development/superpowers-workflow.md) | How the brainstorm→spec→plan→subagent cycle works |

---

## Quick links

- [README](../README.md) — project overview and quick start
- [VISION.md](VISION.md) — framework vision and design philosophy
- [CONTRIBUTING.md](../CONTRIBUTING.md) — contributor agreement and CLA workflow
- [MANUAL_QA.md](development/MANUAL_QA.md) — manual QA checklist for UI features
- [ROADMAP.md](../ROADMAP.md) — high-level roadmap
