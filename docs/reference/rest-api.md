# REST API Reference

The REST API is served by FastAPI at the same port as the web UI (default `:8000`).
All endpoints return JSON unless noted. Long-running operations return a `job_id`
and stream progress via Server-Sent Events (SSE).

Base URL: `http://localhost:8000`

---

## Authentication

The API is unauthenticated until you set a token. Set one via the
`PERSPICACITE_AUTH_TOKEN` environment variable (or `auth.token` in `config.yml`)
and every `/api/*` route except `GET /api/health` then requires it:

```
Authorization: Bearer <your-token>
```

Requests without it get `401` and a `WWW-Authenticate: Bearer` header. Setting
`auth.enabled: false` disables enforcement even when a token is present.

### Binding to a network interface

Because an unauthenticated API exposes `POST /api/llm/proxy` — which spends your
own LLM credits — and read/write access to every knowledge base, the server
**refuses to start** on a non-loopback host unless a token is set. Bind
`127.0.0.1` (the default), or set a token, or, if a reverse proxy in front
already authenticates callers, set `auth.allow_insecure_network_bind: true`.

---

## Health

### `GET /api/health`

Returns server status.

```json
{"status": "ok", "version": "2.0.0"}
```

---

## Chat

### `POST /api/chat`

Run a RAG query. Supports streaming SSE or non-streaming JSON.

**Request body:**

```json
{
  "query": "your research question",
  "kb_name": "my-kb",           // or "auto" for routing, or omit for web search only
  "kb_names": ["kb-1", "kb-2"], // multi-KB alternative to kb_name
  "mode": "basic",              // basic | advanced | profound | agentic | literature_survey | contradiction
  "stream": true                // true for SSE, false for blocking JSON
}
```

**Streaming response** (`stream: true`): server-sent events. Event types include
`thinking`, `kb_route` (when auto-routing), `chunk`, `answer`, `provenance`, `done`.

**Non-streaming response** (`stream: false`):

```json
{
  "answer": "...",
  "sources": [...],
  "mode": "basic",
  "latency_ms": 1234
}
```

---

## Conversations

### `GET /api/conversations`

List all conversations with message counts.

### `POST /api/conversations`

Create a conversation.

```json
{"title": "My research session"}
```

### `GET /api/conversations/{conv_id}`

Get a conversation with all messages.

### `DELETE /api/conversations/{conv_id}`

Delete a conversation and its messages.

### `DELETE /api/conversations`

Delete all conversations.

### `GET /api/conversations/search?q=QUERY`

Full-text search across conversation content (SQLite FTS5).

### `POST /api/conversations/{conv_id}/messages`

Add a message to a conversation (triggers RAG synthesis).

### `GET /api/conversations/{conv_id}/messages/{msg_id}/provenance`

Get the retrieval trace for a specific answer.

### `GET /api/conversations/{conv_id}/provenance`

Get all provenance records for a conversation.

### `GET /api/conversations/{conv_id}/export`

Export a conversation. Format controlled by `?format=` query parameter:
- `?format=markdown` — Markdown document with inline citations
- `?format=ro-crate` — RO-Crate 1.1 zip bundle with full provenance

---

## Knowledge Bases

### `GET /api/kb`

List all knowledge bases.

```json
[{"name": "my-kb", "paper_count": 42, "embedding_model": "text-embedding-3-small", ...}]
```

### `POST /api/kb`

Create a knowledge base.

```json
{"name": "my-kb", "description": "Diamond sensors papers"}
```

### `GET /api/kb/{name}`

Get KB metadata.

### `DELETE /api/kb/{name}`

Permanently delete a KB (metadata + Chroma collection).

### `GET /api/kb/{name}/stats`

KB statistics: paper count, chunk count, year histogram, content-type breakdown,
source-database breakdown.

### `POST /api/kb/{name}/papers`

Add papers to a KB from a list of paper objects.

### `POST /api/kb/{name}/bibtex`

Synchronous BibTeX import. Body: `{"bibtex": "<bibtex string>"}`.

### `POST /api/kb/{name}/bibtex/async`

Asynchronous BibTeX import. Returns `{"job_id": "..."}`. Poll progress via
`GET /api/jobs/{job_id}/events`.

### `POST /api/kb/{name}/dois`

Synchronous bulk DOI add. Body: `{"dois": ["10.1234/...", ...]}`.

### `POST /api/kb/{name}/dois/async`

Asynchronous bulk DOI add. Returns `{"job_id": "..."}`.

### `GET /api/kb/{name}/export`

Export a KB. Format controlled by `?format=` query parameter:
- `?format=obsidian-vault` — Obsidian Markdown vault zip
- `?format=bibtex` — BibTeX file (with optional PDFs in a folder)

### `GET /api/kb/{name}/chunks`

List or search chunks in a KB. Query parameters: `?query=TEXT&top_k=N`.

### `POST /api/kb/{name}/local-files`

Ingest local files (paths must be under `local_docs.allowed_roots`).

### `POST /api/kb/{name}/local-paths`

Ingest a directory tree from a local path.

### `POST /api/kb/{name}/build-capsules`

Trigger capsule building for all papers in a KB (async). Returns `{"job_id": "..."}`.

### `POST /api/kb/{name}/paper/{paper_id}/fetch-resources`

Fetch external resources (GitHub, Zenodo) for a single paper.

### `GET /api/kb/{name}/papers/{paper_id}/figures`

List figure metadata for a paper (from its capsule).

### `GET /api/kb/{name}/papers/{paper_id}/figure/{fig_id}`

Download a specific figure image.

---

## Paper

### `GET /api/paper?doi=DOI`

Fetch discovery metadata and content-type availability for a DOI. Hits OpenAlex
and Unpaywall without ingesting the paper.

---

## Jobs

### `GET /api/jobs/{job_id}`

Get the status of an async ingestion job.

```json
{
  "job_id": "...",
  "status": "running",  // pending | running | done | failed
  "progress": {"papers_processed": 5, "papers_total": 20}
}
```

### `GET /api/jobs/{job_id}/events`

SSE stream of job progress events. Each event is a JSON line. Emit until
`{"type": "done"}` or `{"type": "failed"}`.

---

## Zotero

### `GET /api/zotero/status`

Check Zotero integration status and connectivity.

### `POST /api/zotero/push`

Push papers to Zotero by DOI list.

```json
{
  "dois": ["10.1234/...", ...],
  "attach_pdf": true,
  "attach_supplementary": false
}
```

---

## Zotero ingest

### `GET /api/zotero-ingest/plan`

Preview which KBs would be created from Zotero collections.

### `POST /api/zotero-ingest/build-kbs/async`

Build one KB per Zotero collection. Body: `{"library_id": "..."}` (optional override).
Returns `{"job_id": "..."}`.

---

## Survey

### `GET /api/survey/{session_id}`

Get the status of a literature survey session.

### `POST /api/survey/{session_id}/select`

Select which paper clusters to include in the survey report.

### `POST /api/survey/{session_id}/generate`

Generate the survey report from selected clusters.

---

## Related topics

- [reference/mcp-tools.md](mcp-tools.md) — MCP equivalents for most of these endpoints
- [concepts/provenance.md](../concepts/provenance.md) — what the provenance endpoints return
- [reference/config.md](config.md) — server and auth configuration
