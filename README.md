# Perspicacité — AI-Powered Scientific Literature Research Assistant

**Perspicacité** (French for "insight") helps scientists, researchers, and students search, understand, and organize academic literature using AI grounded in real research papers.

## Features

- **Multi-database search** — Semantic Scholar, OpenAlex, PubMed, arXiv, HAL, DBLP, and more
- **Unified content pipeline** — Retrieves structured full text (PMC JATS XML, arXiv HTML), PDFs, or abstracts with quality-based priority routing
- **5 RAG modes** — From fast KB retrieval to multi-cycle agentic research and systematic literature surveys
- **Knowledge base management** — Import from BibTeX, add papers by DOI, semantic search within your collections
- **MCP server** — 8 tools exposed via Model Context Protocol for integration with AI agents (Mimosa, SmolAgents, etc.)
- **REST API** — Full JSON API for chat, KB management, conversations, and literature surveys
- **Local-first** — Data stays on your machine; only API calls go to LLM providers

## Quick Start

### Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install

```bash
git clone <repository-url>
cd perspicacite_v2
uv sync --dev
```

### Configure

```bash
cp .env.example .env
# Edit .env — add at least one LLM API key
```

| Provider | Get a key at |
|----------|-------------|
| **DeepSeek** (default) | [platform.deepseek.com](https://platform.deepseek.com/) |
| **OpenAI** | [platform.openai.com](https://platform.openai.com/) |
| **Anthropic** | [console.anthropic.com](https://console.anthropic.com/) |

### Run

```bash
uv run perspicacite -c config.yml serve
```

Open **http://localhost:8000** in your browser.

> The MCP server runs on the same port at `/mcp`. Use `--no-mcp` to disable it.

## How to Use Perspicacité

### Using the Web Interface

#### 1. Choose Your Knowledge Base (or Don't!)

In the left sidebar, you'll see a "Knowledge Base" section:

- **"No KB (web search only)"** — Searches the entire web for papers
- **Your own KBs** — Searches only papers you've added

**To create a new Knowledge Base:**
1. Click "+ Create new KB"
2. Enter a KB name and drag-and-drop a `.bib` file
3. Click "Create from BibTeX" to import papers

#### 2. Ask a Question

Type your research question in the chat box. Examples:
- "What are the effects of green tea extract on metabolism?"
- "How is feature-based molecular networking used in metabolomics?"
- "Compare transformer models to CNNs for medical imaging"

#### 3. Choose a Mode

Select a mode from the dropdown. See the RAG Modes table below for details.

#### 4. Review the Answer

Perspicacité will:
1. Show its "thinking process" (click to expand)
2. Search relevant papers
3. Filter and score them for relevance
4. Download full texts when possible
5. Generate an answer with citations

#### 5. Save Interesting Papers

At the bottom of each response, you'll see papers found during research. Click "Add to KB" to save them to your knowledge base.

### Building Your Knowledge Base

**Method 1: Import from BibTeX**
1. Export your references as BibTeX from Zotero, Mendeley, or EndNote
2. Click "+ Create new KB", drag your `.bib` file, enter a name
3. Click "Create from BibTeX"

**Method 2: Add Papers from Search Results**
When Perspicacité finds papers during research, click "Add to KB" on any paper.

**Method 3: Add via MCP or API**
```python
# Via MCP tool
add_papers_to_kb(kb_name="my-kb", papers=[{"title": "...", "doi": "..."}])

# Via REST API
POST /api/kb/my-kb/papers  [{"title": "...", "doi": "..."}]
```

### Tips for Best Results

**Writing good questions:**
- Be specific: "What are the antioxidant properties of green tea catechins?" over "Tell me about tea"
- Ask research-focused questions — Perspicacité summarizes literature, it doesn't write original content

**Managing KBs:**
- Keep KBs focused — create separate ones for different projects
- Start with 10-20 key papers, expand as needed
- Pay attention to relevance scores — high-scoring papers are most useful

**When to use each mode:**
- **Basic**: You have a well-curated KB and want quick answers
- **Advanced**: Your KB might need broader search
- **Profound**: Complex questions needing multiple perspectives
- **Agentic**: Questions requiring web search beyond your KB
- **Literature Survey**: Mapping a research field with AI-identified themes



## RAG Modes

| Mode | Description | Speed |
|------|-------------|-------|
| **Basic** | Single-query retrieval from your KB with hybrid vector+BM25 search | Fast |
| **Advanced** | Query expansion, WRRF fusion scoring, reranking | Medium |
| **Profound** | Multi-cycle research (up to 3 iterations) with planning and self-evaluation | Slower |
| **Agentic** | Intent-based agent with tool use (web search, PDF download), up to 5 iterations | Variable |
| **Literature Survey** | Systematic field mapping: broad search, theme clustering, AI recommendations, paper selection | Slowest |

## Content Retrieval Pipeline

Paper content is retrieved through a unified pipeline with quality-based priority:

```
1. Discovery — OpenAlex + Unpaywall → learn PMCID, arXiv ID, OA status, abstract
2. Structured full text — PMC JATS XML (sections + references) or arXiv HTML
3. PDF full text — OA PDF, arXiv PDF, Unpaywall, publisher APIs (Springer, Wiley, Elsevier, etc.)
4. Abstract only — from discovery metadata
5. Discard — papers with no retrievable content
```

Structured content (PMC, arXiv) provides sections and references. PDF content provides raw text via PyMuPDF. Papers behind paywalls with no OA version are served as abstracts.

## MCP Server

Perspicacité exposes an MCP server with 8 tools, accessible via:
- **MCP protocol** — native tool discovery and invocation
- **HTTP JSON-RPC** — `POST /mcp` with `{"method": "tools/call", "params": {"name": "...", "arguments": {...}}}`

### Tools

| Tool | Description |
|------|-------------|
| `search_literature` | Search academic databases with year range and article-type filters |
| `get_paper_content` | Fetch full text + sections by DOI through the unified pipeline |
| `get_paper_references` | Extract cited references from a paper |
| `create_knowledge_base` | Create a new KB |
| `add_papers_to_kb` | Add papers with auto-download and indexing |
| `search_knowledge_base` | Semantic search within a KB |
| `list_knowledge_bases` | List all KBs with stats |
| `generate_report` | Synthesize a research report using RAG |

### Example: JSON-RPC Call

```python
import httpx

# Initialize session
r = httpx.post("http://localhost:8000/mcp", json={
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
               "clientInfo": {"name": "my-agent", "version": "1.0"}}
}, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
session_id = r.headers["mcp-session-id"]

# Call a tool
r = httpx.post("http://localhost:8000/mcp", json={
    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
    "params": {"name": "search_literature", "arguments": {"query": "flash attention"}}
}, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": session_id})
```

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Chat endpoint (streaming SSE or non-streaming JSON) |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/conversations` | List conversations |
| `POST` | `/api/conversations` | Create conversation |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |
| `GET` | `/api/kb` | List knowledge bases |
| `POST` | `/api/kb` | Create KB |
| `GET` | `/api/kb/{name}` | Get KB details |
| `DELETE` | `/api/kb/{name}` | Delete KB |
| `POST` | `/api/kb/{name}/papers` | Add papers to KB |
| `POST` | `/api/kb/{name}/bibtex` | Import from BibTeX |
| `GET` | `/api/survey/{session_id}` | Get literature survey status |
| `POST` | `/api/survey/{session_id}/generate` | Generate survey report |

Non-streaming chat: pass `"stream": false` to `/api/chat` to get a JSON response instead of SSE.

## CLI Commands

```bash
# Start the server (web + MCP)
perspicacite -c config.yml serve [--host 0.0.0.0] [--port 8000] [--no-mcp] [--reload]

# Create a KB from BibTeX
perspicacite -c config.yml create-kb my-kb --from-bibtex papers.bib

# Show version
perspicacite version
```

## Configuration

Copy and edit `config.example.yml`. Key sections:

```yaml
llm:
  default_provider: "deepseek"   # deepseek, openai, anthropic
  default_model: "deepseek-chat"

knowledge_base:
  embedding_model: "text-embedding-3-small"
  chunk_size: 1000
  chunk_overlap: 200
  chunking_method: "token"       # token, semantic, agentic

pdf_download:
  unpaywall_email: "your@email.com"
  # Optional publisher API keys:
  # elsevier_api_key: "..."
  # springer_api_key: "..."
  # wiley_tdm_token: "..."

mcp:
  enabled: true
```

Academic database search APIs are configured under `scilex:` — enabled sources include Semantic Scholar, OpenAlex, PubMed, arXiv, HAL, DBLP by default.

## Knowledge Bases

**Create from BibTeX:**
- In the web UI: click "+ Create new KB", drag a `.bib` file, enter a name
- Via CLI: `perspicacite create-kb my-kb --from-bibtex refs.bib`
- Via MCP: `create_knowledge_base` then `add_papers_to_kb`

**Add papers during research:**
- Agentic mode finds and downloads papers — click "Add to KB" to save
- Literature Survey mode lets you select recommended papers and add in bulk

**Chunking strategies:**
- `token` — fixed-size token chunks (default)
- `semantic` — splits at semantic boundaries
- `agentic` — AI-driven chunking optimized for RAG

## Development

### Run Tests

```bash
# Unit tests
uv run pytest tests/unit/ -v

# Live MCP tests (requires running server)
uv run python tests/test_mcp_live.py --all --port 8000
uv run python tests/test_mcp_live.py --test search
uv run python tests/test_mcp_live.py --test kb
```

### Project Structure

```
src/perspicacite/
  cli.py                      # CLI commands (serve, create-kb, version)
  config/schema.py            # Configuration model
  mcp/server.py               # MCP server with 8 tools
  pipeline/
    download/                 # Content retrieval pipeline
      discovery.py            # OpenAlex + Unpaywall discovery
      unified.py              # Unified retrieve_paper_content()
      europepmc.py            # PMC JATS XML fetcher
      arxiv.py                # arXiv HTML + PDF
    parsers/pdf.py            # PyMuPDF-based parser
    bibtex_kb.py              # BibTeX → KB pipeline
  rag/
    engine.py                 # RAGEngine (routes to mode handlers)
    modes/                    # basic, advanced, profound, agentic, literature_survey
    tools/                    # Tool registry, KB search, LOTUS
  search/scilex_adapter.py    # Multi-database literature search
  retrieval/                  # ChromaDB vector store + hybrid search
web_app_full.py               # FastAPI web application
templates/index.html          # Single-page chat UI
```

## Privacy & Data

- **Your data stays local** — KBs are stored in ChromaDB and SQLite on your machine
- **API calls only** — Questions are sent to your configured LLM provider
- **No tracking** — No usage data collected

## Contributing

See `CONTRIBUTING.md` for guidelines.

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`.

## Acknowledgments

- **ChromaDB** for vector storage
- **OpenAlex** and **Semantic Scholar** for academic search
- **Unpaywall** for open access discovery
- **SciLEx** for literature exploration toolkit

## References

```bibtex
@inproceedings{pradi2025perspicacite,
  title     = {An AI Pipeline for Scientific Literacy and Discovery: a Demonstration of Perspicacit\'{e}-AI Integration with Knowledge Graphs},
  author    = {Pradi, Lucas and Jiang, Tao and Feraud, Matthieu and Bekbergenova, Madina and Taghzouti, Yousouf and Nothias, Louis-Felix},
  booktitle = {ISWC-C 2025},
  year      = {2025}
}
```

```bibtex
@softwareversion{scilex2026,
  title = {SciLEx, Science Literature Exploration Toolkit},
  author = {Ringwald, C\'{e}lian and Navet, Benjamin},
  url = {https://github.com/Wimmics/SciLEx},
  year = {2026}
}
```
