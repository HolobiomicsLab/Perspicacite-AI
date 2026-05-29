# Configuration Reference

Perspicacité is configured via a YAML file. Copy `config/example.yml` to `config.yml`
and edit from there. The file path is passed with the `-c` flag:

```bash
perspicacite -c config.yml serve
```

This document covers the most important configuration sections. The definitive source
of truth is the Pydantic model in `src/perspicacite/config/schema.py`.

---

## Top-level

```yaml
version: "2.0.0"        # must start with "2."
config_name: "default"  # arbitrary label for your config
```

---

## `server`

```yaml
server:
  host: "0.0.0.0"
  port: 5468
  reload: false         # enable uvicorn hot-reload (development only)
  cors_origins:
    - "http://localhost:3000"
    - "http://localhost:5173"
    - "http://localhost:5468"
```

---

## `mcp`

```yaml
mcp:
  enabled: true
  host: "0.0.0.0"
  port: 5001            # port for dedicated MCP server (when separate from web)
  transport: "streamable-http"  # streamable-http | sse | stdio
```

In the default configuration, the MCP server is merged into the web server at
`/mcp` on the same port. A separate `mcp.port` is only used when the MCP transport
is configured as a standalone process.

---

## `database`

```yaml
database:
  path: "~/.local/share/perspicacite/memory.db"   # SQLite
  chroma_path: "~/.local/share/perspicacite/chroma"
```

Both paths support `~` expansion.

---

## `knowledge_base`

```yaml
knowledge_base:
  embedding_model: "text-embedding-3-small"
  chunk_size: 1000
  chunk_overlap: 200
  chunking_method: "token"          # token | semantic | agentic
  default_top_k: 10
  similarity_threshold: 0.7
  use_two_pass: true
  markdown_heading_aware: true      # heading-aware chunking for .md files
  code_language_aware: true         # language-aware chunking for code files
  code_chunking: "auto"             # auto | ast | splitter
  contextual_retrieval_tier: "none" # none | abstract | summary | chunk
  contextual_retrieval_model: "claude-haiku-4-5"
  contextual_retrieval_provider: "anthropic"
  contextual_retrieval_max_chars: 400
```

---

## `llm`

```yaml
llm:
  default_provider: "deepseek"       # deepseek | openai | anthropic | ollama | agent_cli
  default_model: "deepseek-chat"

  # Per-stage model overrides (optional; all default to default_model)
  models:
    routing:    "claude-haiku-4-5"   # KB router (auto mode)
    screening:  "claude-haiku-4-5"   # paper relevance screen
    rephrase:   "claude-haiku-4-5"   # query rephrasing
    contextual: "claude-haiku-4-5"   # contextual retrieval prefix

  # Per-stage provider overrides (optional)
  providers_per_stage:
    screening: "ollama"

  # MCP sampling — use the connected client's credentials instead of direct API
  use_mcp_sampling: false

  # Agent CLI routing (for claude, codex, openclaw, hermes, etc.)
  providers:
    agent_cli:
      executable: "claude"
      # ... see config/providers/claude-code.yml
```

---

## `rag_modes`

```yaml
rag_modes:
  # KB auto-routing
  route_method: "bm25"       # bm25 | llm
  route_top_k: 3
  route_threshold: 0.1

  # Agentic mode
  max_iterations: 5

  # Profound mode
  profound_max_cycles: 3

  # Screening defaults (overridable per-call)
  screen_method: "bm25"
  screen_threshold: 0.3
```

---

## `scilex`

```yaml
scilex:
  enabled: true
  databases:
    - semantic_scholar
    - openalex
    - pubmed
    - arxiv
    - hal
    - dblp
  semantic_scholar_api_key: ""   # optional; higher rate limits with a key
```

---

## `pdf_download`

```yaml
pdf_download:
  unpaywall_email: "your@email.com"   # required for OA PDF discovery
  timeout: 30.0
  max_retries: 3
  cache_pdfs: true
  cache_dir: "data/papers"

  # Browser-cookie institutional access
  cookies_path: null           # path to Netscape cookies.txt
  cookie_domains: []           # domains to attach cookies to (empty = all)

  # Publisher API keys (all optional)
  elsevier_api_key: null
  springer_api_key: null
  wiley_tdm_token: null
  semantic_scholar_api_key: null
  # rsc_api_key, aaas_api_key also available
```

---

## `zotero`

```yaml
zotero:
  enabled: false
  api_key: ""              # or ZOTERO_API_KEY env var
  library_id: ""           # or PERSPICACITE_ZOTERO_LIBRARY_ID env var
  library_type: "user"     # user | group
  base_url: "https://api.zotero.org"   # or http://localhost:23119/api for desktop
```

---

## `cite_graph`

```yaml
cite_graph:
  min_year_offset: 7       # drop papers older than now - offset years
  min_citations: 1
  max_papers: 50
  venue_denylist: []
  w_citations: 0.30        # scoring weights for candidate ranking
  w_recency:   0.20
  w_oa:        0.20
  w_match:     0.30
```

---

## `capsule`

```yaml
capsule:
  build_on_add: false      # auto-build capsule when adding a paper
```

---

## `multimodal`

```yaml
multimodal:
  mode: "auto"             # auto | force | off
  show_code: false         # include AST code excerpts in RAG responses
```

---

## `local_docs`

```yaml
local_docs:
  allowed_roots:
    - "/Users/you/Documents/papers"
    - "/home/user/research"
```

Only paths under these roots can be ingested via `ingest-local` or the MCP
`ingest_local_documents` tool.

---

## `logging`

```yaml
logging:
  level: "INFO"     # DEBUG | INFO | WARNING | ERROR
  format: "json"    # json | text
```

---

## `auth`

```yaml
auth:
  enabled: true
  token: null   # set via PERSPICACITE_AUTH_TOKEN env var
```

---

## `ui`

```yaml
ui:
  theme: "system"          # light | dark | system
  citation_format: "nature"  # nature | apa | mla | ieee
```

---

## Environment variable overrides

All sensitive values can be set via environment variables, which take precedence over
`config.yml`:

| Variable | Config field |
|----------|-------------|
| `PERSPICACITE_AUTH_TOKEN` | `auth.token` |
| `ZOTERO_API_KEY` or `PERSPICACITE_ZOTERO_API_KEY` | `zotero.api_key` |
| `PERSPICACITE_ZOTERO_LIBRARY_ID` | `zotero.library_id` |
| `PERSPICACITE_ZOTERO_BASE_URL` | `zotero.base_url` |
| `DEEPSEEK_API_KEY` | used by LiteLLM for DeepSeek provider |
| `ANTHROPIC_API_KEY` | used by LiteLLM for Anthropic provider |
| `OPENAI_API_KEY` | used by LiteLLM for OpenAI provider |

---

## Related topics

- [getting-started.md](../getting-started.md) — minimal config to get started
- [guides/institutional-pdf-access.md](../guides/institutional-pdf-access.md) —
  `pdf_download.cookies_path` setup
- [guides/zotero-integration.md](../guides/zotero-integration.md) — `zotero.*` config
- [reference/paper-source-enum.md](paper-source-enum.md) — `PaperSource` values that
  appear in provenance and stats
