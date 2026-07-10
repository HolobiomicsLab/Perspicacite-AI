# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed
- The KB-metadata SQLite path now honours `database.path` (and `PERSPICACITE_DB_PATH`) in the
  web app, the MCP server and the CLI. It was hardcoded to a working-directory-relative
  `./data/perspicacite.db` while the vector store already honoured `database.chroma_path`, so a
  second instance listed the main hub's knowledge bases and wrote its vectors elsewhere. When
  `database.path` is not set, the legacy location is kept, so existing deployments are unaffected.
  Both resolved paths are now logged at startup.

### Changed
- **BREAKING:** the `indicia` and `adapters` optional extras are removed. They required the
  private, unpublished `indicium` stack, and `uv lock` resolves every extra whether or not it
  is requested, so their mere presence broke `uv sync` on every fresh clone. Maintainers now
  install the sibling checkouts into the synced environment with
  `uv pip install -e ../indicium ...`, and pass `--inexact` on subsequent syncs. See `CLAUDE.md`.
- **BREAKING:** `extract_claims_from_passages` MCP tool: `domain: str | None` renamed to
  `domains: list[str] | None`. Pass a single domain as `["metabolomics"]`. Multiple
  domain IDs are resolved and composed into a `CompositeAdapter` so all adapters'
  context, qualifier, enrichment, and SHACL shapes are applied together.
- **BREAKING:** `generate_report` MCP tool: same `domain` → `domains` rename and composition logic.

### Added
- `graph` optional extra (`rdflib`, `pyoxigraph`) — the RDF/SPARQL backend behind the claim
  graph. Both packages are on PyPI, so `uv sync --extra graph` works for everyone. `rdflib` was
  previously an undeclared dependency, reaching the environment only through `indicium`.
- `uv-resolve` CI job and `tests/unit/test_pyproject_packaging.py`, which fail if a local-path
  `[tool.uv.sources]` entry or a private distribution is reintroduced.
- `domains` multi-adapter support for both claim-extraction MCP tools (see Changed above).
- `claims_to_graph()` now serializes the `ontology_terms` dict from enriched claims as
  `asb:{slot}_ontology_term` RDF literals, enabling SHACL property-shape validation on
  ontology identifiers.

### Fixed
- `uv sync` failed on every fresh clone with `Distribution not found at: .../indicium`, because
  `[tool.uv.sources]` pointed at private sibling checkouts that only exist on maintainer machines
  (#29, reintroducing #5). The sources are removed and CI now exercises `uv lock` on a clean
  checkout; the previous CI installed with `pip`, which ignores `[tool.uv.sources]` entirely.
- `domain_adapter` is now correctly passed into `extract_claims()` (not applied as a manual
  post-processing loop), enabling LLM context enrichment and domain qualifier acceptance during
  extraction rather than only after.
- `claims_to_graph()` no longer serializes `None` ontology term values as the literal string
  `"None"`; `None`/falsy values are silently skipped.

---

## [2.0.0] - 2026-05-15

Initial public release of the Perspicacite 2.x series with the redesigned MCP server,
multi-repo knowledge-base support, and the `indicium` claim/evidence standard integration.
