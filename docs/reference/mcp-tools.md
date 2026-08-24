# MCP Tools Reference

Perspicacité exposes 27 tools via the Model Context Protocol at
`http://localhost:5468/mcp` (streamable-HTTP transport, port from `config.yml`).

To connect from Claude Code:

```json
{
  "mcpServers": {
    "perspicacite": {
      "type": "http",
      "url": "http://localhost:5468/mcp"
    }
  }
}
```

For other MCP clients (Mimosa-AI, SmolAgents, Cursor), use the same URL.

---

## Literature search and discovery

### `search_literature`

Search academic databases (requires SciLEx).

**Parameters:**
- `query` (str) — search query
- `max_results` (int, default 10) — maximum papers to return
- `min_year` (int, optional) — year lower bound
- `max_year` (int, optional) — year upper bound
- `article_type` (str, optional) — e.g., `"journal-article"`
- `databases` (list[str], optional) — which SciLEx databases to search

**Returns:** JSON array of paper objects.

### `get_paper_content`

Fetch the full text and sections for a paper by DOI.

**Parameters:**
- `doi` (str) — the DOI

**Returns:** `{doi, title, content_type, full_text, sections, references, ...}`

### `get_paper_references`

Extract cited references from a paper.

**Parameters:**
- `doi` (str)

**Returns:** list of reference objects with titles, DOIs, and years.

### `screen_papers`

Score a list of candidate papers for relevance to a query.

**Parameters:**
- `query` (str)
- `papers` (list) — candidate paper objects
- `method` (str, default `"bm25"`) — `"bm25"` or `"llm"`
- `threshold` (float, default 0.3) — minimum score

**Returns:** filtered + scored list.

---

## Knowledge base management

### `list_knowledge_bases`

List all KBs with paper counts, embedding model, and creation time.

**Parameters:** none.

### `create_knowledge_base`

Create a new knowledge base.

**Parameters:**
- `name` (str)
- `description` (str, optional)

### `delete_knowledge_base`

Permanently delete a KB (metadata + Chroma collection).

**Parameters:**
- `kb_name` (str)

### `add_papers_to_kb`

Add papers to a KB with auto-download and indexing.

**Parameters:**
- `kb_name` (str)
- `papers` (list) — paper objects with at least `title` and `doi`

### `add_dois_to_kb`

Bulk-add papers to a KB from a list of DOIs (max 200 per call).

**Parameters:**
- `kb_name` (str)
- `dois` (list[str])

### `search_knowledge_base`

Semantic + BM25 hybrid search within a KB.

**Parameters:**
- `kb_name` (str)
- `query` (str)
- `top_k` (int, default 10)

**Returns:** list of matching chunks with paper metadata and scores.

### `route_kbs`

Score all KBs for relevance to a query and return a ranked list. Use for
introspection or to pass `kb_names` to `generate_report` without running synthesis.

**Parameters:**
- `query` (str)
- `method` (str, default `"bm25"`) — `"bm25"` or `"llm"`
- `top_k` (int, default 3)

### `ingest_local_documents`

Ingest local PDFs or documents from a server-side path.

**Parameters:**
- `kb_name` (str)
- `path` (str) — must be under a `local_docs.allowed_roots` entry

---

## RAG and reporting

### `generate_report`

Synthesize a research report using RAG.

**Parameters:**
- `query` (str)
- `kb_name` (str, optional) — use `"auto"` for routing
- `kb_names` (list[str], optional) — multi-KB alternative
- `mode` (str, default `"basic"`) — RAG mode

**Returns:** `{answer, sources, mode, latency_ms, provenance}`

### `build_kb_from_search`

Search SciLEx, filter, screen, fetch PDFs, and ingest into a KB in one call.

**Parameters:**
- `query` (str)
- `kb_name` (str) — new or existing KB name
- `max_results` (int, default 10)
- `min_year` (int, optional)
- `min_citations` (int, optional)
- `screen_method` (str, optional) — `"bm25"` or `"llm"`
- `screen_threshold` (float, default 0.3)
- `resolve_missing_dois` (bool, default false) — recover a verified DOI from the
  title for hits that arrive without one (Google Scholar and other scraped
  providers). Without it those hits are dropped; check `filter_reasons.no_doi`
  in the report.
- `resolve_doi_budget` (int, default 25) — max title lookups per run

Counts for the lookup come back under `doi_backfill` in the report.

---

## Citation graph

### `expand_kb_via_citations`

Grow a KB by following citation links from its existing papers.

**Parameters:**
- `kb_name` (str)
- `direction` (str, default `"both"`) — `"forward"`, `"backward"`, or `"both"`
- `max_per_seed` (int, default 50)
- `min_year` (int, optional)
- `min_citations` (int, optional)
- `screen_method` (str, optional)
- `screen_threshold` (float, default 0.3)

### `enrich_kb_from_cite_graph_tool`

Update citation metadata for papers already in a KB without adding new papers.

**Parameters:**
- `kb_name` (str)

---

## Capsules

### `build_capsule`

Build a capsule for a single paper (figures, references, code, SI).

**Parameters:**
- `paper_id` (str) — DOI, PMID, or internal UUID
- `kb_name` (str)
- `force` (bool, default false) — rebuild if capsule already exists

### `build_capsules_for_kb`

Build capsules for all papers in a KB (idempotent).

**Parameters:**
- `kb_name` (str)
- `force` (bool, default false)

### `fetch_paper_resources`

Fetch external resources (GitHub, Zenodo, Crossref) for a paper.

**Parameters:**
- `paper_id` (str)
- `kb_name` (str)

### `fetch_supplementary`

Download Supplementary Information files for a paper.

**Parameters:**
- `paper_id` (str)
- `kb_name` (str)

---

## Export and integration

### `export_kb`

Export a KB as BibTeX or Obsidian vault.

**Parameters:**
- `kb_name` (str)
- `format` (str) — `"bibtex"` or `"obsidian-vault"`
- `with_pdfs` (bool, default false)

### `push_to_zotero`

Push papers to Zotero by DOI list, with optional PDF and SI attachment.

**Parameters:**
- `dois` (list[str])
- `attach_pdf` (bool, default false)
- `attach_supplementary` (bool, default false)

### `build_kbs_from_zotero`

Build one KB per Zotero top-level collection.

**Parameters:**
- `library_id` (str, optional) — override config default

---

## Example: JSON-RPC call from Python

```python
import httpx

# Initialize session
r = httpx.post("http://localhost:5468/mcp", json={
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "my-agent", "version": "1.0"}
    }
}, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
session_id = r.headers["mcp-session-id"]

# Call a tool
r = httpx.post("http://localhost:5468/mcp", json={
    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {
        "name": "search_knowledge_base",
        "arguments": {"kb_name": "my-kb", "query": "diamond magnetometry", "top_k": 5}
    }
}, headers={
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "Mcp-Session-Id": session_id
})
```

---

## Zotero read path (for ASB integration)

### `zotero_list_collections`

List all Zotero library collections as a nested tree.

**Parameters:**
- `library_id` (str, optional) — override configured library_id
- `include_subcollections` (bool, default true) — return nested tree

**Returns:** `{collections: [{id, name, parent_id, item_count, subcollections}], library_id, library_type}`

**Cache:** 1-hour in-memory cache keyed on (library_id, library_type).

**Errors:** `ZOTERO_NOT_CONFIGURED`, `ZOTERO_AUTH_FAILED`, `ZOTERO_RATE_LIMITED`, `LIBRARY_NOT_FOUND`

---

### `zotero_get_collection_items`

Return papers in a collection with metadata and per-paper license classification.

**Parameters:**
- `collection_id` (str) — Zotero collection key
- `library_id` (str, optional)
- `include_abstract` (bool, default true)
- `limit` (int, default 200, max 500)
- `cursor` (str, optional) — pagination token from previous call's `next_cursor`

**Returns:** `{collection_id, items: [{zotero_key, doi, title, authors, year, abstract, item_type, tags, license: {spdx, classification, policy, source}, has_attachments}], total, next_cursor}`

**License policy:** `classification=permissive` → `policy=verbatim` (text may be copied verbatim); `classification=closed|unknown` → `policy=reflavor` (must paraphrase).

**License resolution chain:** Crossref → OpenAlex → Zotero tags → heuristic (is_oa flag).

**Errors:** `ZOTERO_NOT_CONFIGURED`, `ZOTERO_AUTH_FAILED`, `ZOTERO_RATE_LIMITED`, `COLLECTION_NOT_FOUND`, `INVALID_CURSOR`

---

### `zotero_get_paper_resources`

Return ordered file access options for a single paper. Local paths come first (Perspicacité's PDF cache and capsule supplementary-file storage); remote URLs follow.

**Parameters:**
- `doi` (str, optional) — the paper's DOI
- `zotero_key` (str, optional) — Zotero item key; use when DOI is ambiguous
- `library_id` (str, optional)

Exactly one of `doi` or `zotero_key` must be provided.

**Returns:** `{doi, zotero_key, license: {...}, resources: [{role, filename, access: [{type, path|url, via?}]}], notes: [str]}`

`role` values: `fulltext_pdf`, `supplementary`, `note`. Access `type`: `local` (on-disk path) or `remote` (URL + `via` label: `publisher`, `doi_resolver`).

**Errors:** `ZOTERO_NOT_CONFIGURED`, `ZOTERO_AUTH_FAILED`, `ZOTERO_RATE_LIMITED`, `PAPER_NOT_FOUND`, `AMBIGUOUS_DOI`

---

### `zotero_ingest_collection_to_kb`

Ingest a Zotero collection into a Perspicacité KB. Returns immediately with a `job_id` when running under the full web server (poll `poll_url` for SSE completion events); runs inline in MCP-only mode.

**Parameters:**
- `collection_id` (str) — Zotero collection key
- `kb_name` (str, optional) — KB name; defaults to sanitized collection name
- `library_id` (str, optional)
- `force_reingest` (bool, default false) — re-embed already-indexed papers

**Returns (async mode):** `{job_id, kb_name, collection_id, item_count, status: "running", poll_url}`

**Returns (inline mode):** `{per_kb: [...]}`

After the job completes, use `search_knowledge_base` or `generate_report` with the returned `kb_name`.

**Errors:** `ZOTERO_NOT_CONFIGURED`, `ZOTERO_AUTH_FAILED`, `ZOTERO_RATE_LIMITED`, `COLLECTION_NOT_FOUND`

---

## Related topics

- [reference/rest-api.md](rest-api.md) — REST equivalents
- [reference/cli.md](cli.md) — CLI equivalents
- [concepts/rag-modes.md](../concepts/rag-modes.md) — mode selection for `generate_report`
- [concepts/citation-graph.md](../concepts/citation-graph.md) — how `expand_kb_via_citations` works
