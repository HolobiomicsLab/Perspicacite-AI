# Perspicacité v2 — Implementation Guide

**Purpose**: Complete specification for building Perspicacité v2 from scratch.
**Audience**: A coding agent that will implement the entire project.
**Date**: 2026-03-16
**Approach**: Clean rewrite with selective extraction from v1.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Directory Structure](#3-directory-structure)
4. [Reference Packages](#4-reference-packages)
5. [Data Models](#4-data-models)
6. [Configuration System](#5-configuration-system)
7. [Implementation Phases](#6-implementation-phases)
   - [Phase 1: Foundation](#phase-1-foundation)
   - [Phase 2: LLM Layer](#phase-2-llm-layer)
   - [Phase 3: Vector Store & Retrieval](#phase-3-vector-store--retrieval)
   - [Phase 4: Document Pipeline](#phase-4-document-pipeline)
   - [Phase 5: RAG Engine](#phase-5-rag-engine)
   - [Phase 6: SciLEx Integration](#phase-6-scilex-integration)
   - [Phase 7: Memory & Sessions](#phase-7-memory--sessions)
   - [Phase 8: API Layer](#phase-8-api-layer)
   - [Phase 9: Minimal UI](#phase-9-minimal-ui)
   - [Phase 10: MCP Server](#phase-10-mcp-server)
8. [v1 Code Extraction Guide](#7-v1-code-extraction-guide)
9. [Testing Strategy](#8-testing-strategy)
10. [Deployment](#9-deployment)
11. [Appendix D: Error Handling Matrix](#appendix-d-error-handling-matrix)
12. [Appendix E: Streaming Protocol Specification](#appendix-e-streaming-protocol-specification)
13. [Appendix F: Security Model](#appendix-f-security-model)
14. [Appendix G: v1 Migration Guide](#appendix-g-v1-migration-guide)

---

## 1. Project Overview

### What is Perspicacité v2?

An AI-powered scientific literature research assistant. Users build knowledge bases from academic papers (BibTeX, PDFs, web search) and query them using RAG (Retrieval-Augmented Generation) with multiple modes of increasing depth.

### Core capabilities

1. **Knowledge Base Management**: Create, populate, and query vector-based knowledge bases from BibTeX files, PDFs, URLs, or academic API searches.
2. **Multi-Mode RAG**: Five retrieval modes from simple vector search to multi-cycle research with planning and reflection.
3. **Literature Search**: Integration with 11 academic APIs via SciLEx (Semantic Scholar, OpenAlex, PubMed, ArXiv, IEEE, Springer, Elsevier, HAL, DBLP, Istex, PubMed Central).
4. **Dynamic KB Building**: Agent can discover and add papers to a KB during a research session (Chroma enables real-time additions).
5. **Session Persistence**: Cross-session memory of past research, user preferences, and paper interactions.
6. **MCP Server**: Expose Perspicacité as an MCP tool for external AI agents (Mimosa-AI).

### v2.0 scope (this guide)

- Backend: Full async Python package with FastAPI
- Minimal working UI: React chat interface with KB selection
- SciLEx integration via adapter pattern
- RAG modes: Quick, Standard, Advanced, Deep, Citation
- Simple token-based auth
- MCP server
- Comprehensive test suite

### Out of scope for v2.0 (future versions)

- Full Agentic RAG mode with multi-agent architecture (v2.1)
- Chemical structure search / RDKit integration (v2.1)
- Memory-to-skill evolution pipeline (v2.2)
- Toolomics MCP client integration (v2.1)
- Multi-tenant enterprise deployment (v2.1)
- Full UI redesign (v2.1)

---

## 2. Technology Stack

### Backend

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Language** | Python ≥3.11 | async/await, modern typing |
| **Package manager** | `uv` | Fast, lockfile support, matches SciLEx |
| **Web framework** | FastAPI + Uvicorn | Async-native, OpenAPI docs, SSE support |
| **LLM routing** | LiteLLM | Multi-provider (OpenAI, Anthropic, DeepSeek, Gemini, Ollama) |
| **Vector store** | ChromaDB (primary) | Dynamic additions, metadata filtering, hybrid search |
| **Embeddings** | LiteLLM embeddings + SentenceTransformers | Multi-provider, local fallback |
| **BM25** | `rank_bm25` | Keyword search component of hybrid retrieval |
| **Reranking** | `sentence-transformers` CrossEncoder | Two-stage retrieval accuracy |
| **PDF parsing** | `pdfplumber` | Reliable text extraction (carried from v1) |
| **HTML parsing** | `beautifulsoup4` + `lxml` | Web content extraction (carried from v1) |
| **Session store** | SQLite via `aiosqlite` | Lightweight, zero-config persistence |
| **Caching** | SQLite (same DB) | Query cache, embedding cache |
| **MCP** | `fastmcp` | MCP server implementation |
| **Config** | PyYAML + Pydantic v2 | YAML files with type-safe validation |
| **Logging** | `structlog` | Structured JSON logging |
| **HTTP client** | `httpx` | Async HTTP for web search, PDF download |
| **Testing** | `pytest` + `pytest-asyncio` + `pytest-cov` | Async test support, coverage |
| **Linting** | `ruff` | Fast, comprehensive |
| **Typing** | `mypy` (strict) | Type safety |

### Frontend (Minimal)

| Component | Technology |
|-----------|-----------|
| **Framework** | React 18 + TypeScript |
| **Build** | Vite |
| **UI components** | shadcn/ui (carried from v1) |
| **State** | React Context + TanStack Query |
| **Streaming** | Native EventSource / fetch-event-source |
| **Styling** | Tailwind CSS |

### What we are NOT using (and why)

| Dropped | Reason |
|---------|--------|
| **LangChain** | Heavy dependency; we use Chroma and LiteLLM directly |
| **LangGraph** | Over-engineered for v2 scope; custom agent loop suffices |
| **FAISS** | Replaced by Chroma for dynamic KB support |
| **conda** | Replaced by `uv` (faster, lighter) |
| **Streamlit** | v1 remnant; using React |

---

## 3. Directory Structure

```
perspicacite_v2/
├── pyproject.toml
├── uv.lock
├── README.md
├── config.example.yml              # Example configuration
├── .env.example                    # Example environment variables
├── .github/
│   └── workflows/
│       └── ci.yml                  # Test + lint CI
│
├── src/
│   └── perspicacite/               # Main Python package
│       ├── __init__.py             # Package version, metadata
│       ├── __main__.py             # `python -m perspicacite` entry point
│       ├── cli.py                  # CLI commands (click or argparse)
│       │
│       ├── config/                 # Configuration system
│       │   ├── __init__.py
│       │   ├── schema.py           # Pydantic config models
│       │   ├── loader.py           # YAML + env + CLI layered loading
│       │   └── defaults.py         # Built-in default values
│       │
│       ├── models/                 # Core data models (Pydantic)
│       │   ├── __init__.py
│       │   ├── papers.py           # Paper, Author, PaperMetadata
│       │   ├── documents.py        # DocumentChunk, ChunkMetadata
│       │   ├── search.py           # SearchResult, SearchQuery, SearchFilters
│       │   ├── kb.py               # KnowledgeBase, KBInfo, KBStats
│       │   ├── rag.py              # RAGRequest, RAGResponse, SourceReference
│       │   ├── messages.py         # Message, Conversation, Session
│       │   └── api.py              # API request/response models
│       │
│       ├── llm/                    # LLM abstraction layer
│       │   ├── __init__.py
│       │   ├── client.py           # AsyncLLMClient (wraps LiteLLM)
│       │   ├── embeddings.py       # EmbeddingProvider protocol + implementations
│       │   ├── providers.py        # Provider config, model registry
│       │   └── tokens.py           # Token counting, context window management
│       │
│       ├── retrieval/              # Vector store + search
│       │   ├── __init__.py
│       │   ├── protocols.py        # VectorStore protocol definition
│       │   ├── chroma_store.py     # Chroma implementation
│       │   ├── bm25.py             # BM25 scoring
│       │   ├── hybrid.py           # Hybrid retrieval (vector + BM25)
│       │   └── reranker.py         # Cross-encoder reranking
│       │
│       ├── pipeline/               # Document processing pipeline
│       │   ├── __init__.py
│       │   ├── bibtex.py           # BibTeX parsing and processing
│       │   ├── chunking.py         # Text chunking (token/semantic/section-aware)
│       │   ├── curation.py         # LLM-based document curation
│       │   ├── citations.py        # Citation formatting and handling
│       │   ├── download.py         # PDF/content download with retries
│       │   ├── kb_builder.py       # KB creation orchestrator
│       │   └── parsers/
│       │       ├── __init__.py
│       │       ├── pdf.py          # PDF text extraction
│       │       ├── html.py         # HTML/web content extraction
│       │       ├── youtube.py      # YouTube transcript extraction
│       │       └── github.py       # GitHub README extraction
│       │
│       ├── rag/                    # RAG engine
│       │   ├── __init__.py
│       │   ├── protocols.py        # RAGMode protocol, Tool protocol
│       │   ├── engine.py           # Main RAG orchestrator
│       │   ├── tools.py            # Tool registry and implementations
│       │   ├── context.py          # Context window builder (7-layer)
│       │   └── modes/
│       │       ├── __init__.py
│       │       ├── quick.py        # QuickRAG - single-pass vector search
│       │       ├── standard.py     # StandardRAG - hybrid + reranking
│       │       ├── advanced.py     # AdvancedRAG - query expansion + multi-hop
│       │       ├── deep.py         # DeepRAG - research planning + iteration
│       │       └── citation.py     # CitationRAG - citation network analysis
│       │
│       ├── search/                 # Literature search (web)
│       │   ├── __init__.py
│       │   ├── protocols.py        # SearchProvider protocol
│       │   ├── scilex_adapter.py   # SciLEx integration adapter
│       │   ├── google_scholar.py   # Google Scholar (fallback/simple)
│       │   └── doi_resolver.py     # DOI → metadata resolution
│       │
│       ├── memory/                 # Session & memory management
│       │   ├── __init__.py
│       │   ├── session_store.py    # SQLite session persistence
│       │   ├── chat_history.py     # In-session chat history management
│       │   ├── research_memory.py  # Cross-session research recall
│       │   └── user_prefs.py       # User preferences storage
│       │
│       ├── api/                    # FastAPI application
│       │   ├── __init__.py
│       │   ├── app.py              # FastAPI app factory
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── chat.py         # POST /api/chat (streaming + non-streaming)
│       │   │   ├── knowledge_bases.py  # KB CRUD endpoints
│       │   │   ├── search.py       # Literature search endpoints
│       │   │   ├── sessions.py     # Session management endpoints
│       │   │   └── health.py       # Health check, info
│       │   ├── streaming.py        # SSE streaming utilities
│       │   ├── auth.py             # Simple token authentication
│       │   ├── middleware.py        # CORS, rate limiting, error handling
│       │   └── dependencies.py     # FastAPI dependency injection
│       │
│       ├── mcp/                    # MCP server
│       │   ├── __init__.py
│       │   └── server.py           # FastMCP server with tools/resources
│       │
│       └── logging.py              # Structured logging setup
│
├── website/                        # React frontend
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── config.ts
│       ├── types.ts                # TypeScript types matching API models
│       ├── api/
│       │   ├── client.ts           # API client (fetch wrapper)
│       │   └── streaming.ts        # SSE stream consumer
│       ├── context/
│       │   └── AuthContext.tsx
│       ├── pages/
│       │   ├── ChatPage.tsx        # Main chat interface
│       │   ├── KnowledgeBasesPage.tsx  # KB management
│       │   └── SettingsPage.tsx    # API keys, preferences
│       ├── components/
│       │   ├── ChatInput.tsx
│       │   ├── MessageList.tsx
│       │   ├── MessageItem.tsx
│       │   ├── SourceCard.tsx
│       │   ├── Sidebar.tsx
│       │   ├── KBSelector.tsx
│       │   ├── ModeSelector.tsx
│       │   └── ui/                 # shadcn/ui components
│       └── lib/
│           └── utils.ts
│
└── tests/
    ├── conftest.py                 # Shared fixtures, test config
    ├── fixtures/                   # Test data files
    │   ├── sample.bib
    │   ├── sample.pdf
    │   └── sample_papers.json
    ├── unit/
    │   ├── test_config.py
    │   ├── test_models.py
    │   ├── test_llm_client.py
    │   ├── test_embeddings.py
    │   ├── test_chroma_store.py
    │   ├── test_bm25.py
    │   ├── test_hybrid_retrieval.py
    │   ├── test_reranker.py
    │   ├── test_chunking.py
    │   ├── test_pdf_parser.py
    │   ├── test_html_parser.py
    │   ├── test_bibtex.py
    │   ├── test_kb_builder.py
    │   ├── test_rag_quick.py
    │   ├── test_rag_standard.py
    │   ├── test_rag_advanced.py
    │   ├── test_rag_deep.py
    │   ├── test_rag_citation.py
    │   ├── test_scilex_adapter.py
    │   ├── test_session_store.py
    │   ├── test_chat_history.py
    │   └── test_context_builder.py
    ├── integration/
    │   ├── test_api_chat.py
    │   ├── test_api_kb.py
    │   ├── test_api_search.py
    │   ├── test_api_streaming.py
    │   ├── test_kb_pipeline.py     # BibTeX → parse → chunk → embed → store → query
    │   └── test_mcp_server.py
    └── e2e/
        └── test_research_session.py  # Full session: create KB, query, follow-up
```

---

## 4. Reference Packages

Two existing codebases are available for reference and extraction during implementation.

### 4.1 Perspicacité v1 (Reference Implementation)

**Location**: `packages_to_use/Perspicacite-AI-release/`

**Purpose**: Source of proven algorithms, parsers, and logic to extract and adapt.

**Key modules to extract**:

| v1 Path | Lines | Purpose | v2 Destination |
|---------|-------|---------|----------------|
| `bibtex2kb/src/chunking.py` | ~662 | Text chunking algorithms | `pipeline/chunking.py` |
| `bibtex2kb/src/pdf_parser.py` | ~114 | PDF text extraction | `pipeline/parsers/pdf.py` |
| `bibtex2kb/src/html_parser.py` | ~327 | Web content extraction | `pipeline/parsers/html.py` |
| `bibtex2kb/src/youtube_parser.py` | ~243 | YouTube transcript extraction | `pipeline/parsers/youtube.py` |
| `bibtex2kb/src/github_parser.py` | ~56 | GitHub README extraction | `pipeline/parsers/github.py` |
| `bibtex2kb/src/citation_handler.py` | ~183 | Citation formatting | `pipeline/citations.py` |
| `bibtex2kb/src/url_handler.py` | ~117 | URL resolution, Unpaywall | `pipeline/download.py` |
| `bibtex2kb/src/cache_manager.py` | ~111 | Download caching | `pipeline/cache.py` |
| `bibtex2kb/src/document_curator.py` | ~330 | LLM text curation | `pipeline/curation.py` |
| `core/hybrid_retrieval.py` | ~137 | Vector + BM25 fusion | `retrieval/hybrid.py` |
| `core/llm_utils.py` | ~341 | Provider registry | `llm/providers.py` |
| `core/profonde.py` | ~1,226 | Deep RAG implementation | `rag/modes/deep.py` (reference) |
| `core/core.py` | ~928 | Standard/Advanced RAG | `rag/modes/` (reference) |

**Extraction notes**:
- Most v1 modules use synchronous I/O — wrap async wrappers where needed
- v1 uses LangChain for some abstractions — replace with direct implementations
- Some v1 modules have print() statements — replace with structlog

### 4.2 SciLEx (Literature Collection Engine)

**Location**: `packages_to_use/SciLEx/`

**Purpose**: Academic API collection (10+ sources) to integrate via adapter.

**Key modules**:

| Module | Purpose | Integration Point |
|--------|---------|-------------------|
| `scilex/crawlers/collectors/` | 10 API collectors (Semantic Scholar, OpenAlex, PubMed, etc.) | SciLExAdapter will call these |
| `scilex/aggregate_collect.py` | Deduplication and quality filtering | Wrap results through this |
| `scilex/quality_validation.py` | Paper quality scoring | Use for filtering results |
| `scilex/citations/` | Citation network analysis | CitationRAG mode |
| `scilex/export_to_bibtex.py` | Export to BibTeX format | KB export feature |

**Installation**:
```bash
# SciLEx is installed as an editable dependency
uv pip install -e packages_to_use/SciLEx
```

**Adapter approach**:
- Create `scilex_adapter.py` that imports from `scilex.crawlers.collectors`
- Wrap synchronous SciLEx calls in `asyncio.to_thread()`
- Map SciLEx output format to Perspicacité `Paper` models
- Graceful fallback to Google Scholar if SciLEx unavailable

### 4.3 Using Reference Packages During Development

**During implementation**:

1. **Read v1 source first** — Understand the algorithm/logic before rewriting
2. **Copy-paste adapted code** — Don't rewrite working logic unnecessarily
3. **Add type hints** — v1 has minimal typing; v2 requires full type coverage
4. **Convert to async** — v1 is sync; v2 uses async/await throughout
5. **Replace dependencies** — Remove LangChain, use httpx instead of requests

**Example extraction workflow**:

```python
# v1: bibtex2kb/src/chunking.py (sync, minimal types)
def chunk_text(text, chunk_size=1000):
    # ... algorithm ...
    return chunks

# v2: src/perspicacite/pipeline/chunking.py (async, typed)
async def chunk_text(
    text: str,
    config: ChunkConfig,
) -> list[DocumentChunk]:
    # Same algorithm, but:
    # - Async (for semantic chunking that calls embeddings)
    # - Full type hints
    # - Returns Pydantic models
    # - Uses structlog for logging
```

---

## 5. Data Models

All models use Pydantic v2 BaseModel. These are the canonical types used throughout the system.

### 5.1 Paper & Document Models (`models/papers.py`, `models/documents.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


class PaperSource(str, Enum):
    BIBTEX = "bibtex"
    SCILEX = "scilex"
    WEB_SEARCH = "web_search"
    USER_UPLOAD = "user_upload"
    CITATION_FOLLOW = "citation_follow"


class Author(BaseModel):
    name: str
    given: Optional[str] = None
    family: Optional[str] = None
    orcid: Optional[str] = None


class Paper(BaseModel):
    """Canonical paper representation used across the entire system."""
    id: str = Field(description="Unique ID: DOI, PMID, or generated UUID")
    title: str
    authors: list[Author] = []
    abstract: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    source: PaperSource = PaperSource.BIBTEX
    keywords: list[str] = []
    full_text: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    paper_id: str
    chunk_index: int
    section: Optional[str] = None
    page_number: Optional[int] = None
    source: PaperSource = PaperSource.BIBTEX
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None


class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None
```

### 5.2 Knowledge Base Models (`models/kb.py`)

```python
class ChunkConfig(BaseModel):
    method: Literal["token", "semantic", "section_aware"] = "token"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_sections: bool = True


class KnowledgeBase(BaseModel):
    name: str
    description: Optional[str] = None
    collection_name: str  # Chroma collection name
    embedding_model: str = "text-embedding-3-small"
    chunk_config: ChunkConfig = ChunkConfig()
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    paper_count: int = 0
    chunk_count: int = 0


class KBStats(BaseModel):
    name: str
    paper_count: int
    chunk_count: int
    embedding_model: str
    created_at: datetime
    size_mb: Optional[float] = None
```

### 5.3 Search & Retrieval Models (`models/search.py`)

```python
class SearchFilters(BaseModel):
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    authors: Optional[list[str]] = None
    journals: Optional[list[str]] = None
    sources: Optional[list[PaperSource]] = None
    has_full_text: Optional[bool] = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    retrieval_method: Literal["vector", "bm25", "hybrid"]


class SearchQuery(BaseModel):
    text: str
    kb_name: str = "default"
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    filters: Optional[SearchFilters] = None
    top_k: int = 10
    rerank: bool = True
```

### 5.4 RAG Models (`models/rag.py`)

```python
class RAGMode(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    ADVANCED = "advanced"
    DEEP = "deep"
    CITATION = "citation"


class SourceReference(BaseModel):
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    relevance_score: float = 0.0
    chunk_text: Optional[str] = None


class RAGRequest(BaseModel):
    query: str
    kb_name: str = "default"
    mode: RAGMode = RAGMode.STANDARD
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_iterations: Optional[int] = None
    use_web_search: bool = False
    filters: Optional[SearchFilters] = None
    conversation_id: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceReference] = []
    mode: RAGMode
    iterations: int = 1
    confidence: Optional[float] = None
    research_plan: Optional[list[str]] = None
    web_search_used: bool = False
    tokens_used: Optional[int] = None


class StreamEvent(BaseModel):
    """Structured SSE event — replaces v1's magic prefix protocol."""
    event: Literal[
        "status",       # Processing status updates
        "content",      # Answer text delta
        "source",       # Source reference
        "reasoning",    # Chain-of-thought (Deep mode)
        "plan",         # Research plan step
        "tool_call",    # Tool invocation
        "tool_result",  # Tool result
        "error",        # Error message
        "done",         # Stream complete
    ]
    data: str  # JSON-encoded payload
```

### 5.5 Message & Session Models (`models/messages.py`)

```python
class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: list[SourceReference] = []
    metadata: dict = Field(default_factory=dict)


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    kb_name: str = "default"
    messages: list[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "default"
    conversations: list[str] = []  # Conversation IDs
    preferences: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
```

### 5.6 API Models (`models/api.py`)

```python
class ChatRequest(BaseModel):
    messages: list[Message]
    kb_name: str = "default"
    mode: RAGMode = RAGMode.STANDARD
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    stream: bool = True
    use_web_search: bool = False
    conversation_id: Optional[str] = None
    max_iterations: Optional[int] = None


class ChatResponse(BaseModel):
    message: Message
    sources: list[SourceReference] = []
    conversation_id: str
    mode: RAGMode


class KBCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    source_type: Literal["bibtex", "papers", "empty"] = "empty"
    source_path: Optional[str] = None  # BibTeX file path
    papers: Optional[list[str]] = None  # DOIs or URLs
    embedding_model: str = "text-embedding-3-small"
    chunk_config: ChunkConfig = ChunkConfig()


class KBAddPapersRequest(BaseModel):
    papers: list[str]  # DOIs, URLs, or PDF paths
    auto_chunk: bool = True


class SearchRequest(BaseModel):
    query: str
    apis: list[str] = ["semantic_scholar", "openalex", "pubmed"]
    max_results: int = 20
    year_min: Optional[int] = None
    year_max: Optional[int] = None
```

---

## 6. Configuration System

### 6.1 Config File (`config.example.yml`)

```yaml
version: "2.0.0"

server:
  host: "0.0.0.0"
  port: 5468
  reload: false
  mcp:
    enabled: true
    transport: "stdio"

database:
  path: "~/.local/share/perspicacite/data.db"
  chroma_path: "~/.local/share/perspicacite/chroma"

knowledge_base:
  embedding_model: "text-embedding-3-small"
  chunk_size: 1000
  chunk_overlap: 200

llm:
  default_provider: "anthropic"
  default_model: "claude-sonnet-4-20250514"
  timeout: 120
  max_retries: 3

rag_modes:
  quick:
    max_iterations: 1
    tools: ["kb_search"]
  standard:
    max_iterations: 1
    tools: ["kb_search"]
    rerank: true
  advanced:
    max_iterations: 2
    tools: ["kb_search", "web_search"]
    rerank: true
    query_expansion: true
  deep:
    max_iterations: 5
    tools: ["kb_search", "web_search", "fetch_pdf", "citation_network"]
    rerank: true
    enable_planning: true
    enable_reflection: true
  citation:
    max_iterations: 3
    tools: ["kb_search", "web_search", "citation_network"]
    rerank: true
    build_citation_graph: true

scilex:
  enabled: true
  config_path: null  # Auto-discover or specify path to scilex.config.yml
  default_apis:
    - "semantic_scholar"
    - "openalex"
    - "pubmed"

web_search:
  providers: ["google_scholar", "semantic_scholar"]
  cache_ttl: 3600

logging:
  level: "INFO"
  format: "json"

auth:
  enabled: true
  token: null  # Set via env: PERSPICACITE_AUTH_TOKEN

ui:
  theme: "system"
  citation_format: "nature"
```

### 6.2 Environment Variables (`.env.example`)

```bash
# Required: At least one LLM provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
# DEEPSEEK_API_KEY=...
# GOOGLE_API_KEY=...

# Auth
PERSPICACITE_AUTH_TOKEN=my-secret-token

# Optional: SciLEx API keys (for higher rate limits)
# SCILEX_SEMANTIC_SCHOLAR_API_KEY=...
# SCILEX_IEEE_API_KEY=...
# SCILEX_SPRINGER_API_KEY=...
# SCILEX_ELSEVIER_API_KEY=...

# Optional: Override config
# PERSPICACITE_CONFIG_PATH=./config.yml
# PERSPICACITE_LOG_LEVEL=DEBUG
```

### 6.3 Pydantic Config Models (`config/schema.py`)

Implement a layered config loader:
1. Built-in defaults (`config/defaults.py`)
2. Config file (`config.yml`)
3. Environment variables (`PERSPICACITE_*`)
4. CLI arguments

Use Pydantic v2 `BaseSettings` with `SettingsConfigDict(env_prefix="PERSPICACITE_")` for automatic env var loading. The config loader reads YAML first, then overlays env vars.

---

## 7. Implementation Phases

Each phase produces a testable, working increment. Complete all tests for a phase before moving to the next.

---

### Phase 1: Foundation

**Goal**: Project scaffolding, data models, configuration, logging. Everything compiles and tests pass.

#### Step 1.1: Project Setup

Create `pyproject.toml`:

```toml
[project]
name = "perspicacite"
version = "2.0.0"
description = "AI-powered scientific literature research assistant"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "sse-starlette>=2.0.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "pyyaml>=6.0",
    "litellm>=1.50.0",
    "chromadb>=0.5.0",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=3.0.0",
    "httpx>=0.27.0",
    "aiosqlite>=0.20.0",
    "pdfplumber>=0.11.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "bibtexparser>=1.4.0",
    "structlog>=24.0.0",
    "tenacity>=9.0.0",
    "tiktoken>=0.7.0",
    "numpy>=1.26.0",
    "click>=8.1.0",
    "fastmcp>=0.4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "pytest-httpx>=0.30.0",
    "ruff>=0.7.0",
    "mypy>=1.12.0",
    "pre-commit>=3.8.0",
]
scilex = [
    "scilex",  # Installed from local path: pip install -e ../SciLEx
]

[project.scripts]
perspicacite = "perspicacite.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/perspicacite"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "unit: Unit tests (no external services)",
    "integration: Integration tests (may need services)",
    "live: Tests requiring API keys (deselect with -m 'not live')",
]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "TCH", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.coverage.run]
source = ["src/perspicacite"]
omit = ["*/tests/*"]

[tool.coverage.report]
fail_under = 70
```

#### Step 1.2: Data Models

Implement all models from Section 4. Each model file should:
- Use Pydantic v2 `BaseModel`
- Include `model_config = ConfigDict(frozen=False)` where mutation is needed
- Include proper `__repr__` for debugging
- Include factory methods where useful (e.g., `Paper.from_bibtex(entry)`, `Paper.from_scilex(record)`)

#### Step 1.3: Configuration System

Implement `config/schema.py`, `config/loader.py`, `config/defaults.py`:
- `load_config(path: Optional[str] = None) -> Config` — main entry point
- Search order: CLI path → `PERSPICACITE_CONFIG_PATH` → `./config.yml` → `~/.config/perspicacite/config.yml` → defaults
- Validate all fields with Pydantic
- Expand `~` in paths
- Mask secrets in logs

#### Step 1.4: Logging

Implement `logging.py` using `structlog`:
- JSON output for production, colored console for dev
- Include request_id, user_id context
- Configure log level from config

#### Step 1.5: CLI Skeleton

Implement `cli.py` with Click:
```
perspicacite serve        # Start FastAPI + optional MCP
perspicacite create-kb    # Create knowledge base from BibTeX
perspicacite list-kb      # List knowledge bases
perspicacite query        # CLI query against a KB
perspicacite version      # Print version
```

At this phase, commands can be stubs that print "not implemented yet".

#### Phase 1 Tests

- `test_config.py`: Loading from YAML, env override, defaults, validation errors
- `test_models.py`: Model creation, serialization, factory methods, edge cases

---

### Phase 2: LLM Layer

**Goal**: Async LLM client that works with multiple providers. Embedding generation.

#### Step 2.1: LLM Client (`llm/client.py`)

```python
from typing import Protocol, AsyncIterator


class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict],
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str: ...

    async def stream(
        self,
        messages: list[dict],
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncIterator[str]: ...


class AsyncLLMClient:
    """
    Wraps LiteLLM with:
    - Async completion and streaming
    - Automatic retries (tenacity) on transient errors
    - Provider-specific config (API keys from env)
    - Token counting before/after calls
    - Structured logging of all calls
    """
```

Key behaviors:
- Use `litellm.acompletion()` for async
- Use `litellm.acompletion(stream=True)` for streaming
- Retry on rate limit (429) and server errors (5xx) with exponential backoff, max 3 attempts
- Log: model, provider, input tokens, output tokens, latency, success/failure
- Handle provider-specific model name mapping (e.g., `anthropic/claude-sonnet-4-20250514`)

#### Step 2.2: Embedding Provider (`llm/embeddings.py`)

```python
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    
    @property
    def dimension(self) -> int: ...
    
    @property
    def model_name(self) -> str: ...


class LiteLLMEmbeddingProvider:
    """Uses litellm.aembedding() — supports OpenAI, Cohere, Voyage, etc."""

class SentenceTransformerEmbeddingProvider:
    """Local embeddings via sentence-transformers — no API key needed."""
```

Key behaviors:
- Batch embedding with configurable batch size (default 32)
- Caching: maintain an in-memory LRU cache of recent embeddings (optional)
- Fallback: if API embedding fails, fall back to local SentenceTransformer

#### Step 2.3: Token Management (`llm/tokens.py`)

- `count_tokens(text: str, model: str) -> int` using `tiktoken`
- `truncate_to_tokens(text: str, max_tokens: int, model: str) -> str`
- `estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float`

#### Step 2.4: Provider Registry (`llm/providers.py`)

```python
PROVIDERS = {
    "anthropic": {
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", ...],
        "env_key": "ANTHROPIC_API_KEY",
        "supports_streaming": True,
        "supports_tools": True,
    },
    "openai": { ... },
    "deepseek": { ... },
    "gemini": { ... },
    "ollama": { ... },
}


def get_available_providers() -> list[str]:
    """Return providers that have API keys configured."""

def get_models_for_provider(provider: str) -> list[str]:
    """Return available models for a provider."""

def validate_provider_config(provider: str, model: str) -> None:
    """Raise ConfigError if provider/model combination is invalid."""
```

#### Phase 2 Tests

- `test_llm_client.py`: Mock LiteLLM calls, test retry logic, streaming, error handling
- `test_embeddings.py`: Mock embedding calls, test batching, dimension validation
- Test with `pytest-httpx` for mocking HTTP calls

---

### Phase 3: Vector Store & Retrieval

**Goal**: Chroma-based vector store with hybrid search and reranking.

#### Step 3.1: VectorStore Protocol (`retrieval/protocols.py`)

```python
class VectorStore(Protocol):
    async def create_collection(self, name: str, embedding_dim: int) -> None: ...
    async def delete_collection(self, name: str) -> None: ...
    async def list_collections(self) -> list[str]: ...
    async def add_documents(
        self, collection: str, chunks: list[DocumentChunk]
    ) -> int: ...
    async def search(
        self, collection: str, query_embedding: list[float],
        top_k: int = 10, filters: Optional[dict] = None
    ) -> list[RetrievedChunk]: ...
    async def get_collection_stats(self, collection: str) -> dict: ...
    async def delete_documents(self, collection: str, ids: list[str]) -> int: ...
```

#### Step 3.2: Chroma Implementation (`retrieval/chroma_store.py`)

```python
class ChromaVectorStore:
    """
    Chroma-backed vector store.
    
    - Uses PersistentClient for disk-backed storage
    - Supports metadata filtering (year, source, section, etc.)
    - Handles embedding via configured EmbeddingProvider
    - Thread-safe for concurrent access
    """
    
    def __init__(self, persist_dir: str, embedding_provider: EmbeddingProvider):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_provider = embedding_provider
```

Key behaviors:
- `add_documents`: Embed texts in batches, add to Chroma with metadata
- `search`: Embed query, search Chroma, return `RetrievedChunk` list
- Metadata filter mapping: Convert `SearchFilters` → Chroma `where` clause
- Handle collection creation with proper embedding dimension

#### Step 3.3: BM25 Scoring (`retrieval/bm25.py`)

```python
class BM25Index:
    """
    BM25 index for keyword-based retrieval.
    
    Built alongside the vector index for hybrid search.
    Stored as a pickle file alongside Chroma data.
    """
    
    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.documents: list[DocumentChunk] = []
    
    async def build(self, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index from document chunks."""
    
    async def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Search by BM25 score."""
    
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

#### Step 3.4: Hybrid Retrieval (`retrieval/hybrid.py`)

```python
class HybridRetriever:
    """
    Combines vector similarity and BM25 for better recall.
    
    Uses Reciprocal Rank Fusion (RRF) to merge results:
    score = sum(1 / (k + rank_i)) for each retriever i
    """
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ): ...
    
    async def search(
        self,
        query: str,
        collection: str,
        top_k: int = 10,
        filters: Optional[SearchFilters] = None,
    ) -> list[RetrievedChunk]:
        """
        1. Vector search (top_k * 3 candidates)
        2. BM25 search (top_k * 3 candidates)
        3. RRF fusion
        4. Return top_k
        """
```

**Important**: Fix the division-by-zero bug from v1's `hybrid_retrieval.py` where normalization fails when all scores are equal.

#### Step 3.5: Cross-Encoder Reranker (`retrieval/reranker.py`)

```python
class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.
    
    Two-stage retrieval:
    1. Fast retrieval (vector/hybrid) → top_k * 3 candidates
    2. Cross-encoder scoring → top_k final results
    
    Cross-encoder is more accurate because it sees query + document together,
    unlike bi-encoders which encode them separately.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Score (query, chunk.text) pairs and re-sort."""
```

#### Phase 3 Tests

- `test_chroma_store.py`: CRUD operations, search, metadata filtering (use temp directory)
- `test_bm25.py`: Index building, search, edge cases (empty index, single doc)
- `test_hybrid_retrieval.py`: RRF fusion, weight balancing, deduplication
- `test_reranker.py`: Reranking changes order, handles empty input

---

### Phase 4: Document Pipeline

**Goal**: Process BibTeX files, PDFs, HTML, and other sources into document chunks ready for indexing.

#### Step 4.1: Parsers (`pipeline/parsers/`)

**Extract from v1** with these modifications:

| v1 File | v2 File | Changes |
|---------|---------|---------|
| `bibtex2kb/src/pdf_parser.py` (114 lines) | `pipeline/parsers/pdf.py` | Add type hints, async wrapper for I/O, add `extract_sections()` |
| `bibtex2kb/src/html_parser.py` (327 lines) | `pipeline/parsers/html.py` | Add type hints, fix `process_pdf(content=...)` bug, use `httpx` instead of `requests` |
| `bibtex2kb/src/youtube_parser.py` (243 lines) | `pipeline/parsers/youtube.py` | Add type hints, async wrapper |
| `bibtex2kb/src/github_parser.py` (56 lines) | `pipeline/parsers/github.py` | Add type hints, use `httpx` |

Each parser should implement:

```python
class ContentParser(Protocol):
    async def parse(self, source: str) -> ParsedContent: ...

class ParsedContent(BaseModel):
    text: str
    title: Optional[str] = None
    sections: dict[str, str] = {}  # section_name → text
    metadata: dict = {}
```

#### Step 4.2: Chunking (`pipeline/chunking.py`)

**Extract directly from v1** `bibtex2kb/src/chunking.py` (~662 lines). This module is well-structured and reusable.

Changes:
- Add type hints where missing
- Make tokenizer selection configurable via `ChunkConfig`
- Ensure `chunk_text()` is the main public API
- Add `async` wrapper for the semantic chunking mode (which calls embeddings)

Supported modes:
- `token`: Fixed-size token chunks with overlap
- `semantic`: Sentence embedding similarity-based splitting
- `section_aware`: Respect section boundaries + token chunking within sections

#### Step 4.3: BibTeX Processing (`pipeline/bibtex.py`)

**Decompose v1's monolithic `bibtex_processor.py`** into focused functions:

```python
async def parse_bibtex_file(path: str) -> list[dict]:
    """Parse .bib file into raw entry dicts."""

async def resolve_paper_content(
    entry: dict,
    http_client: httpx.AsyncClient,
) -> Paper:
    """
    Resolve a BibTeX entry to a Paper with full text.
    
    Resolution order:
    1. Local PDF (if 'file' field is a local path)
    2. PDF URL (if 'file' field is a URL)
    3. DOI → Unpaywall → OA PDF URL
    4. URL field → HTML parse
    5. Abstract only (fallback)
    """

async def process_bibtex_to_papers(
    bib_path: str,
    download_pdfs: bool = True,
    max_concurrent: int = 5,
) -> list[Paper]:
    """Full pipeline: parse BibTeX → resolve content → return Papers."""
```

Extract helper functions from v1:
- Unpaywall integration (`bibtex2kb/src/url_handler.py`)
- Citation formatting (`bibtex2kb/src/citation_handler.py`)
- Cache management (`bibtex2kb/src/cache_manager.py`)

#### Step 4.4: Document Curation (`pipeline/curation.py`)

**Refactor from v1** `bibtex2kb/src/document_curator.py`:

```python
async def curate_document(
    text: str,
    llm_client: AsyncLLMClient,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """
    LLM-based text curation: remove artifacts, clean formatting.
    
    Chunks long text, curates each chunk, reassembles.
    Optional — skipped if no LLM API key configured.
    """
```

Changes from v1:
- Fix `pkg_resources.resource_filename('bibtex2faiss', ...)` bug
- Use async LLM calls
- Remove LangChain JSONLoader dependency
- Accept config as parameter, not global

#### Step 4.5: Content Download (`pipeline/download.py`)

**Extract from v1** `bibtex2kb/src/url_handler.py`:

```python
async def download_pdf(
    url: str,
    http_client: httpx.AsyncClient,
    timeout: float = 30.0,
) -> Optional[bytes]:
    """Download PDF with retry logic."""

async def get_open_access_url(
    doi: str,
    http_client: httpx.AsyncClient,
) -> Optional[str]:
    """Query Unpaywall for OA PDF URL."""

def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF."""
```

Use `httpx.AsyncClient` instead of `requests`.

#### Step 4.6: KB Builder (`pipeline/kb_builder.py`)

```python
class KBBuilder:
    """
    Orchestrates knowledge base creation.
    
    Pipeline: Source → Parse → Curate → Chunk → Embed → Store
    """
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        llm_client: Optional[AsyncLLMClient] = None,
        chunk_config: ChunkConfig = ChunkConfig(),
    ): ...
    
    async def create_from_bibtex(
        self,
        name: str,
        bib_path: str,
        curate: bool = False,
    ) -> KnowledgeBase:
        """
        Create KB from BibTeX file.
        
        1. Parse BibTeX entries
        2. Download/resolve content for each entry
        3. Optionally curate with LLM
        4. Chunk all documents
        5. Embed and store in Chroma
        6. Build BM25 index
        7. Save KB metadata
        """
    
    async def add_papers(
        self,
        kb_name: str,
        papers: list[Paper],
    ) -> int:
        """
        Dynamically add papers to existing KB.
        Returns number of chunks added.
        """
    
    async def add_from_urls(
        self,
        kb_name: str,
        urls: list[str],
    ) -> int:
        """Download, parse, chunk, and add papers from URLs."""
```

#### Phase 4 Tests

- `test_pdf_parser.py`: Parse test PDF, handle corrupt PDF, empty PDF
- `test_html_parser.py`: Parse HTML page, extract main content
- `test_bibtex.py`: Parse .bib file, resolve entries, handle malformed BibTeX
- `test_chunking.py`: All three modes, overlap behavior, section preservation
- `test_kb_builder.py`: End-to-end KB creation with mocked parsers/embeddings

---

### Phase 5: RAG Engine

**Goal**: Multi-mode RAG system with tool calling for advanced modes.

#### Step 5.1: RAG Protocol (`rag/protocols.py`)

```python
class RAGModeHandler(Protocol):
    """Interface that all RAG mode implementations must satisfy."""
    
    @property
    def name(self) -> str: ...
    
    @property
    def description(self) -> str: ...
    
    async def execute(
        self,
        request: RAGRequest,
        retriever: HybridRetriever,
        llm: AsyncLLMClient,
        tools: "ToolRegistry",
        context: "ContextBuilder",
    ) -> RAGResponse: ...
    
    async def execute_stream(
        self,
        request: RAGRequest,
        retriever: HybridRetriever,
        llm: AsyncLLMClient,
        tools: "ToolRegistry",
        context: "ContextBuilder",
    ) -> AsyncIterator[StreamEvent]: ...


class Tool(Protocol):
    """Interface for tools available to RAG modes."""
    
    @property
    def name(self) -> str: ...
    
    @property
    def description(self) -> str: ...
    
    @property
    def parameters_schema(self) -> dict: ...
    
    async def execute(self, **kwargs) -> str: ...
```

#### Step 5.2: Tool Registry (`rag/tools.py`)

```python
class ToolRegistry:
    """
    Registry of tools available to RAG modes.
    Tools are registered at startup and resolved by name.
    """
    
    tools: dict[str, Tool]
    
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def list_tools(self, allowed: Optional[list[str]] = None) -> list[Tool]: ...


# Built-in tool implementations:

class KBSearchTool(Tool):
    """Search a knowledge base."""
    name = "kb_search"
    
class WebSearchTool(Tool):
    """Search academic web sources."""
    name = "web_search"

class FetchPDFTool(Tool):
    """Download and parse a PDF."""
    name = "fetch_pdf"

class CitationNetworkTool(Tool):
    """Get citations for a paper."""
    name = "citation_network"

class AddToKBTool(Tool):
    """Dynamically add papers to KB."""
    name = "add_to_kb"
```

#### Step 5.3: Context Builder (`rag/context.py`)

Implements the 7-layer context construction from the planning document:

```python
class ContextBuilder:
    """
    Constructs the LLM context window from multiple sources.
    
    Layers (in order):
    1. System prompt (always included, ~500 tokens)
    2. User preferences (always included, ~100 tokens)
    3. Chat history (recent turns verbatim, older summarized)
    4. Research memory (cross-session, if relevant)
    5. Retrieved documents (from KB search)
    6. Web search results (if KB insufficient)
    7. User query
    
    Total target: ≤ max_context_tokens (configurable, default 8000)
    """
    
    def __init__(
        self,
        llm_client: AsyncLLMClient,
        max_tokens: int = 8000,
    ): ...
    
    async def build(
        self,
        query: str,
        retrieved_docs: list[RetrievedChunk],
        chat_history: list[Message],
        user_prefs: Optional[dict] = None,
        research_memory: Optional[list] = None,
        web_results: Optional[list] = None,
    ) -> list[dict]:
        """
        Build messages list for LLM call.
        Prioritizes documents by relevance, truncates if needed.
        """
```

#### Step 5.4: RAG Mode Implementations

##### QuickRAG (`rag/modes/quick.py`)

```
Query → Embed → Vector search (top 5) → Generate answer
```

Single-pass, no reranking, no web search. Fastest mode.

##### StandardRAG (`rag/modes/standard.py`)

```
Query → Hybrid search (vector + BM25) → Rerank → Generate answer with sources
```

Default mode. Good for most queries.

##### AdvancedRAG (`rag/modes/advanced.py`)

```
Query → Generate expanded queries (3-5 variants)
     → Hybrid search each variant
     → Merge results (RRF)
     → Rerank
     → If insufficient: web search
     → Generate answer
```

Uses query expansion for better recall. Optional web search fallback.

##### DeepRAG (`rag/modes/deep.py`)

This is the most complex mode. Refactored from v1's `profonde.py`.

```
Query → Create research plan (list of sub-questions)
     → For each research step:
         → KB search
         → Evaluate quality (sufficient?)
         → If not: web search
         → If web found papers: optionally add to KB
         → Analyze documents
     → Create iteration summary
     → Should continue? (confidence check, max iterations)
     → If yes: adjust plan based on findings, loop
     → If no: generate final answer with all sources
```

Key classes:

```python
@dataclass
class ResearchStep:
    query: str
    purpose: str
    documents: list[RetrievedChunk]
    analysis: str
    confidence: float
    success: bool


@dataclass
class ResearchIteration:
    plan: list[str]
    steps: list[ResearchStep]
    summary: str
    missing_aspects: list[str]
    should_continue: bool


class DeepRAGMode:
    """
    Multi-cycle research with planning and reflection.
    
    Implements:
    - Research plan generation (LLM creates sub-questions)
    - Iterative retrieval (up to max_iterations cycles)
    - Quality assessment (LLM evaluates document sufficiency)
    - Plan adaptation (adjust plan based on findings)
    - Early exit (stop when confident enough)
    - Web search fallback (when KB is insufficient)
    - Dynamic KB building (add discovered papers)
    """
```

Improvements over v1 Profound:
- Fully async
- Clean streaming via `StreamEvent` (no magic prefixes)
- Tool-based architecture (tools are called by name, not hardcoded pipeline)
- Structured output (dataclasses for research state)
- No UI coupling in core logic

##### CitationRAG (`rag/modes/citation.py`)

```
Query → KB search for seed papers
     → Get citation network (forward + backward, depth 1-2)
     → Score papers by centrality + relevance
     → Identify seminal works
     → Generate literature review with citation graph
```

#### Step 5.5: RAG Engine (`rag/engine.py`)

```python
class RAGEngine:
    """
    Main entry point for RAG operations.
    Routes requests to the appropriate mode handler.
    """
    
    def __init__(
        self,
        llm_client: AsyncLLMClient,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        tool_registry: ToolRegistry,
        config: Config,
    ): ...
    
    async def query(self, request: RAGRequest) -> RAGResponse:
        """Execute a RAG query (non-streaming)."""
    
    async def query_stream(
        self, request: RAGRequest
    ) -> AsyncIterator[StreamEvent]:
        """Execute a RAG query with streaming."""
    
    def _get_mode_handler(self, mode: RAGMode) -> RAGModeHandler:
        """Return the handler for the given mode."""
```

#### Phase 5 Tests

- `test_rag_quick.py`: Basic query → answer with sources
- `test_rag_standard.py`: Hybrid search, reranking, source attribution
- `test_rag_advanced.py`: Query expansion, multi-query fusion
- `test_rag_deep.py`: Research planning, iteration, early exit, plan adjustment
- `test_rag_citation.py`: Citation network traversal, seminal paper identification
- `test_context_builder.py`: Layer assembly, token budgeting, truncation
- All tests should mock LLM calls and use a small in-memory Chroma DB

---

### Phase 6: SciLEx Integration

**Goal**: Use SciLEx as the literature collection engine via an adapter.

#### Step 6.1: Search Provider Protocol (`search/protocols.py`)

```python
class SearchProvider(Protocol):
    """Interface for literature search providers."""
    
    @property
    def name(self) -> str: ...
    
    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        apis: Optional[list[str]] = None,
    ) -> list[Paper]: ...
```

#### Step 6.2: SciLEx Adapter (`search/scilex_adapter.py`)

```python
class SciLExAdapter:
    """
    Adapter to use SciLEx as a search provider.
    
    Integration approach: Option B (library import with adapter).
    
    SciLEx is installed as a dependency (pip install -e ../SciLEx).
    This adapter:
    1. Prepares SciLEx config files in a temp directory
    2. Calls SciLEx collection functions programmatically
    3. Maps SciLEx dict/DataFrame output to Perspicacite Paper models
    4. Wraps sync calls in asyncio.to_thread()
    
    Fallback: If SciLEx is not installed, gracefully degrades
    to built-in Google Scholar search.
    """
    
    def __init__(self, api_config: Optional[dict] = None):
        self._scilex_available = self._check_scilex()
    
    def _check_scilex(self) -> bool:
        try:
            import scilex
            return True
        except ImportError:
            return False
    
    async def search(
        self,
        query: str,
        max_results: int = 20,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        apis: Optional[list[str]] = None,
    ) -> list[Paper]:
        if not self._scilex_available:
            return await self._fallback_search(query, max_results)
        
        return await asyncio.to_thread(
            self._scilex_search_sync,
            query, max_results, year_min, year_max, apis
        )
    
    def _scilex_search_sync(
        self,
        query: str,
        max_results: int,
        year_min: Optional[int],
        year_max: Optional[int],
        apis: Optional[list[str]],
    ) -> list[Paper]:
        """
        Sync SciLEx collection. Runs in a thread via asyncio.to_thread().
        
        Exact SciLEx calls used:
        """
        import tempfile, os, json
        from scilex.crawlers.collector_collection import CollectCollection
        from scilex.crawlers.aggregate import FORMAT_CONVERTERS
        from scilex.crawlers.aggregate_parallel import simple_deduplicate
        from scilex.config_defaults import DEFAULT_RATE_LIMITS
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Build SciLEx main_config dict (mirrors scilex.config.yml structure)
            main_config = {
                "KEYWORDS": {
                    "SEARCH_TERMS": {"group_1": query.split()},
                    "SEARCH_MODE": "single_group",
                },
                "COLLECTION": {
                    "output_path": tmpdir,
                    "collect_name": "perspicacite_search",
                    "MAX_ARTICLES_PER_QUERY": max_results,
                    "START_YEAR": year_min or 2000,
                    "END_YEAR": year_max or 2026,
                },
                "APIs": {api: True for api in (apis or self._default_apis)},
                "QUALITY_FILTERS": {"enabled": True},
            }
            
            # 2. Build api_config dict (mirrors api.config.yml structure)
            api_config = self._api_config or {}
            
            # 3. Run collection
            collector = CollectCollection(main_config, api_config)
            collector.create_collects_jobs()
            # SciLEx writes JSON files to tmpdir/perspicacite_search/{api}/{idx}/
            
            # 4. Load and aggregate results
            # Each API's JSON files contain lists of dicts
            all_records = []
            collect_dir = os.path.join(tmpdir, "perspicacite_search")
            for api_dir in os.listdir(collect_dir):
                api_path = os.path.join(collect_dir, api_dir)
                if not os.path.isdir(api_path):
                    continue
                converter = FORMAT_CONVERTERS.get(api_dir)
                if not converter:
                    continue
                for root, _, files in os.walk(api_path):
                    for f in files:
                        if f.endswith(".json"):
                            with open(os.path.join(root, f)) as fp:
                                data = json.load(fp)
                            results = data if isinstance(data, list) else data.get("results", [])
                            for record in results:
                                converted = converter(record)
                                if converted:
                                    all_records.append(converted)
            
            # 5. Deduplicate
            import pandas as pd
            if all_records:
                df = pd.DataFrame(all_records)
                df = simple_deduplicate(df)
                return [self._map_scilex_to_paper(row) for _, row in df.iterrows()]
            return []
    
    def _map_scilex_to_paper(self, row) -> Paper:
        """
        Map a SciLEx aggregated row (Zotero format) to Paper model.
        
        SciLEx Zotero-format columns → Paper fields:
          title           → title
          creators/authors → authors (semicolon-separated string)
          abstractNote    → abstract
          date/year       → year
          publicationTitle → journal
          DOI             → doi
          url             → url
          pdf_url         → pdf_url
          citation_count  → citation_count
          archive         → metadata["archive"] (which APIs found it)
          archiveID       → metadata["archive_id"]
          itemType        → metadata["item_type"]
        """
        from perspicacite.models.papers import Paper, Author, PaperSource
        
        authors = []
        raw_authors = row.get("authors") or row.get("creators") or ""
        if isinstance(raw_authors, str):
            for name in raw_authors.split(";"):
                name = name.strip()
                if name:
                    authors.append(Author(name=name))
        
        doi = row.get("DOI") or row.get("doi")
        paper_id = f"doi:{doi}" if doi else f"scilex:{hash(row.get('title', ''))}"
        
        return Paper(
            id=paper_id,
            title=str(row.get("title", "")),
            authors=authors,
            abstract=row.get("abstractNote") or row.get("abstract"),
            year=int(row["year"]) if row.get("year") else None,
            journal=row.get("publicationTitle"),
            doi=doi,
            url=row.get("url"),
            pdf_url=row.get("pdf_url"),
            citation_count=int(row["citation_count"]) if row.get("citation_count") else None,
            source=PaperSource.SCILEX,
            metadata={
                "archive": row.get("archive", ""),
                "item_type": row.get("itemType", ""),
            },
        )
```

#### Step 6.3: SciLEx Error Handling & Rate Limiting

```python
class SciLExAdapter:
    # ... (continued from above)
    
    # Rate limiting: SciLEx handles its own per-API rate limiting internally
    # (base.py: _rate_limit_wait, circuit breaker). No coordination needed.
    # However, we add a global timeout to prevent hanging:
    
    SEARCH_TIMEOUT: float = 120.0  # Max seconds for a SciLEx search
    
    async def search(self, query: str, **kwargs) -> list[Paper]:
        if not self._scilex_available:
            return await self._fallback_search(query, kwargs.get("max_results", 20))
        
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._scilex_search_sync, query, **kwargs),
                timeout=self.SEARCH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("SciLEx search timed out", query=query)
            return await self._fallback_search(query, kwargs.get("max_results", 20))
        except ImportError as e:
            logger.error("SciLEx import error", error=str(e))
            self._scilex_available = False
            return await self._fallback_search(query, kwargs.get("max_results", 20))
        except Exception as e:
            # SciLEx collector errors (API failures, parsing errors)
            # are mostly caught internally. This catches anything that escapes.
            logger.error("SciLEx search failed", error=str(e), query=query)
            return await self._fallback_search(query, kwargs.get("max_results", 20))
    
    async def _fallback_search(self, query: str, max_results: int) -> list[Paper]:
        """Fall back to Google Scholar when SciLEx is unavailable."""
        logger.info("Using Google Scholar fallback", query=query)
        return await self._google_scholar.search(query, max_results=max_results)
    
    async def search_single_api(
        self,
        query: str,
        api: str,
        max_results: int = 20,
    ) -> list[Paper]:
        """
        Search a single SciLEx API directly (bypasses aggregation).
        Useful for targeted searches when you know which API to use.
        
        Directly instantiates one collector:
        """
        if not self._scilex_available:
            return []
        
        return await asyncio.to_thread(
            self._single_api_search_sync, query, api, max_results
        )
    
    def _single_api_search_sync(
        self, query: str, api: str, max_results: int
    ) -> list[Paper]:
        """
        Direct single-API collection without full aggregation pipeline.
        """
        from scilex.crawlers.collectors.semantic_scholar import SemanticScholar_collector
        from scilex.crawlers.collectors.openalex import OpenAlex_collector
        from scilex.crawlers.collectors.pubmed import PubMed_collector
        from scilex.crawlers.collectors.arxiv import Arxiv_collector
        # ... other collectors
        
        COLLECTOR_MAP = {
            "semantic_scholar": SemanticScholar_collector,
            "openalex": OpenAlex_collector,
            "pubmed": PubMed_collector,
            "arxiv": Arxiv_collector,
            # Add others as needed
        }
        
        collector_cls = COLLECTOR_MAP.get(api)
        if not collector_cls:
            logger.warning(f"Unknown SciLEx API: {api}")
            return []
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            data_query = {
                "keywords": {"group_1": query.split()},
                "search_mode": "single_group",
                "start_year": 2000,
                "end_year": 2026,
                "max_articles_per_query": max_results,
            }
            api_key = (self._api_config or {}).get(api, {}).get("api_key")
            collector = collector_cls(data_query, tmpdir, api_key)
            collector.runCollect()
            
            # Read collected JSON and map to Papers
            # ... (similar to _scilex_search_sync but for single API)
```

#### Step 6.4: Google Scholar Fallback (`search/google_scholar.py`)

Simple fallback when SciLEx is not available. Extract and simplify from v1's `core/web_search.py`.

```python
class GoogleScholarSearcher:
    """
    Lightweight Google Scholar search via scholarly library.
    Used as fallback when SciLEx is not installed.
    """
    
    async def search(self, query: str, max_results: int = 10) -> list[Paper]:
        return await asyncio.to_thread(self._search_sync, query, max_results)
    
    def _search_sync(self, query: str, max_results: int) -> list[Paper]:
        try:
            from scholarly import scholarly
        except ImportError:
            logger.warning("scholarly not installed, cannot search Google Scholar")
            return []
        
        papers = []
        for result in scholarly.search_pubs(query):
            if len(papers) >= max_results:
                break
            bib = result.get("bib", {})
            papers.append(Paper(
                id=f"scholar:{hash(bib.get('title', ''))}",
                title=bib.get("title", ""),
                authors=[Author(name=a) for a in bib.get("author", [])],
                abstract=bib.get("abstract"),
                year=int(bib["pub_year"]) if bib.get("pub_year") else None,
                journal=bib.get("venue"),
                url=result.get("pub_url"),
                source=PaperSource.WEB_SEARCH,
            ))
            time.sleep(1)  # Rate limiting for scholarly
        return papers
```

#### Step 6.5: DOI Resolver (`search/doi_resolver.py`)

```python
async def resolve_doi(doi: str, http_client: httpx.AsyncClient) -> Paper:
    """Resolve DOI to Paper metadata via CrossRef API."""

async def resolve_dois_batch(
    dois: list[str], http_client: httpx.AsyncClient
) -> list[Paper]:
    """Batch DOI resolution."""
```

#### Phase 6 Tests

- `test_scilex_adapter.py`:
  - Mock SciLEx imports: `monkeypatch` the imports to simulate SciLEx being installed/unavailable
  - Test `_map_scilex_to_paper` with sample Zotero-format dicts covering all fields, missing fields, and edge cases
  - Test timeout handling: mock `_scilex_search_sync` to sleep longer than `SEARCH_TIMEOUT`
  - Test fallback: verify Google Scholar is called when SciLEx import fails
  - Test error recovery: verify adapter returns fallback results when SciLEx raises exceptions
- `test_google_scholar.py`: Mock `scholarly` library, test mapping, test rate limiting
- `test_doi_resolver.py`: Mock CrossRef API responses via `pytest-httpx`
- Tests with `@pytest.mark.live` marker for actual API calls

---

### Phase 7: Memory & Sessions

**Goal**: Persist conversations, research history, and user preferences across sessions.

#### Step 7.1: SQLite Schema

Use a single SQLite database (`data.db`) via `aiosqlite`:

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences TEXT DEFAULT '{}'  -- JSON
);

CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    title TEXT,
    kb_name TEXT DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',  -- JSON array of SourceReference
    metadata TEXT DEFAULT '{}',  -- JSON
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE research_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    findings_summary TEXT,
    papers_used TEXT DEFAULT '[]',  -- JSON array of paper IDs
    kbs_accessed TEXT DEFAULT '[]',  -- JSON array of KB names
    success_rating INTEGER CHECK (success_rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE kb_metadata (
    name TEXT PRIMARY KEY,
    description TEXT,
    collection_name TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_config TEXT NOT NULL,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    paper_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id);
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_research_memory_user ON research_memory(user_id);
```

#### Step 7.2: Session Store (`memory/session_store.py`)

```python
class SessionStore:
    """
    SQLite-backed session and conversation persistence.
    """
    
    async def init_db(self) -> None:
        """Create tables if not exist."""
    
    async def create_session(self, user_id: str) -> Session: ...
    async def get_session(self, session_id: str) -> Optional[Session]: ...
    
    async def create_conversation(
        self, session_id: str, kb_name: str
    ) -> Conversation: ...
    async def get_conversation(self, conv_id: str) -> Optional[Conversation]: ...
    async def list_conversations(self, session_id: str) -> list[Conversation]: ...
    
    async def add_message(self, conv_id: str, message: Message) -> None: ...
    async def get_messages(
        self, conv_id: str, limit: int = 50
    ) -> list[Message]: ...
    
    async def save_kb_metadata(self, kb: KnowledgeBase) -> None: ...
    async def get_kb_metadata(self, name: str) -> Optional[KnowledgeBase]: ...
    async def list_kbs(self) -> list[KnowledgeBase]: ...
```

#### Step 7.3: Chat History Manager (`memory/chat_history.py`)

```python
class ChatHistoryManager:
    """
    Manages conversation history for context window construction.
    
    - Recent messages: verbatim (last N turns)
    - Older messages: summarized via LLM
    - Relevant messages: semantic similarity retrieval (optional)
    """
    
    MAX_RECENT_TURNS: int = 10
    
    async def format_history(
        self,
        messages: list[Message],
        max_tokens: int = 2000,
    ) -> str:
        """Format chat history for inclusion in LLM context."""
    
    async def summarize_older(
        self,
        messages: list[Message],
        llm: AsyncLLMClient,
    ) -> str:
        """Summarize older messages to save tokens."""
```

#### Step 7.4: Research Memory (`memory/research_memory.py`)

```python
class ResearchMemory:
    """
    Cross-session memory of past research.
    
    Stores query + findings + papers used for each research session.
    Enables: "You researched this 3 days ago. Use those results?"
    """
    
    async def store(
        self,
        user_id: str,
        query: str,
        findings: str,
        papers_used: list[str],
        kbs_accessed: list[str],
    ) -> None: ...
    
    async def recall(
        self,
        user_id: str,
        query: str,
        max_results: int = 5,
    ) -> list[dict]:
        """
        Find past research similar to current query.
        Uses keyword matching (LIKE) for v2.
        Future: vector similarity on query embeddings.
        """
```

#### Phase 7 Tests

- `test_session_store.py`: CRUD for sessions, conversations, messages, KB metadata
- `test_chat_history.py`: Formatting, summarization, truncation
- Use temp SQLite databases for test isolation

---

### Phase 8: API Layer

**Goal**: FastAPI application exposing all functionality via REST + SSE.

#### Step 8.1: App Factory (`api/app.py`)

```python
from fastapi import FastAPI

def create_app(config: Config) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    1. Initialize services (LLM, vector store, SciLEx, etc.)
    2. Register routes
    3. Add middleware (CORS, auth, error handling)
    4. Set up lifespan events (startup/shutdown)
    """
```

Use FastAPI's lifespan context manager for startup/shutdown:
- Startup: Initialize Chroma client, LLM client, session store, tool registry
- Shutdown: Close HTTP clients, flush caches

#### Step 8.2: Dependencies (`api/dependencies.py`)

```python
from fastapi import Depends

async def get_config() -> Config: ...
async def get_llm_client() -> AsyncLLMClient: ...
async def get_vector_store() -> ChromaVectorStore: ...
async def get_rag_engine() -> RAGEngine: ...
async def get_session_store() -> SessionStore: ...
async def get_current_user(token: str = Header()) -> str: ...
```

Use FastAPI's dependency injection for clean service access.

#### Step 8.3: Authentication (`api/auth.py`)

Simple token-based auth:

```python
from fastapi import HTTPException, Header

async def verify_token(authorization: str = Header(None)) -> str:
    """
    Simple Bearer token auth.
    
    - If auth is disabled in config: allow all
    - If auth is enabled: require Authorization: Bearer <token>
    - Token is compared against PERSPICACITE_AUTH_TOKEN env var
    - Returns user_id (for now, derived from token or "default")
    """
```

#### Step 8.4: Chat Routes (`api/routes/chat.py`)

```python
@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    session_store: SessionStore = Depends(get_session_store),
    user_id: str = Depends(get_current_user),
) -> Union[ChatResponse, EventSourceResponse]:
    """
    Main chat endpoint.
    
    If stream=False: return ChatResponse
    If stream=True: return SSE stream of StreamEvent
    """
```

SSE streaming format (replaces v1's magic prefixes):

```
event: status
data: {"message": "Searching knowledge base..."}

event: content
data: {"delta": "Based on the literature, "}

event: content
data: {"delta": "metabolomics biomarkers for COVID-19 include..."}

event: source
data: {"title": "Wang et al.", "year": 2021, "doi": "10.1038/...", "score": 0.94}

event: reasoning
data: {"step": "Expanding search to PubMed for recent papers..."}

event: plan
data: {"step_index": 0, "total_steps": 3, "query": "...", "purpose": "..."}

event: tool_call
data: {"tool": "web_search", "args": {"query": "metabolomics COVID-19"}}

event: tool_result
data: {"tool": "web_search", "result": "Found 5 papers", "papers_count": 5}

event: error
data: {"message": "Web search timed out, continuing with KB results"}

event: done
data: {"conversation_id": "abc-123", "tokens_used": 4500, "mode": "deep", "iterations": 2}
```

See **Appendix E** for full streaming protocol specification including edge cases.

#### Step 8.5: Knowledge Base Routes (`api/routes/knowledge_bases.py`)

```python
@router.get("/api/kb")
async def list_knowledge_bases() -> list[KBStats]: ...

@router.post("/api/kb")
async def create_knowledge_base(request: KBCreateRequest) -> KnowledgeBase: ...

@router.get("/api/kb/{name}")
async def get_knowledge_base(name: str) -> KBStats: ...

@router.delete("/api/kb/{name}")
async def delete_knowledge_base(name: str) -> dict: ...

@router.post("/api/kb/{name}/papers")
async def add_papers(name: str, request: KBAddPapersRequest) -> dict: ...
```

#### Step 8.6: Search Routes (`api/routes/search.py`)

```python
@router.post("/api/search")
async def search_literature(request: SearchRequest) -> list[Paper]:
    """Search academic databases via SciLEx adapter."""

@router.get("/api/search/providers")
async def list_search_providers() -> list[dict]:
    """List available search APIs and their status."""
```

#### Step 8.7: Session Routes (`api/routes/sessions.py`)

```python
@router.get("/api/conversations")
async def list_conversations() -> list[Conversation]: ...

@router.get("/api/conversations/{id}")
async def get_conversation(id: str) -> Conversation: ...

@router.delete("/api/conversations/{id}")
async def delete_conversation(id: str) -> dict: ...

@router.get("/api/providers")
async def list_providers() -> dict:
    """List available LLM providers and models."""
```

#### Step 8.8: Health Routes (`api/routes/health.py`)

```python
@router.get("/health")
async def health() -> dict:
    """Health check with service status."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "services": {
            "llm": check_llm_available(),
            "chroma": check_chroma_available(),
            "scilex": check_scilex_available(),
        }
    }

@router.get("/api/info")
async def info() -> dict:
    """System info: available models, KBs, config."""
```

#### Step 8.9: Middleware (`api/middleware.py`)

```python
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request, call_next):
    request_id = str(uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
```

#### Step 8.10: Streaming Utilities (`api/streaming.py`)

```python
async def stream_rag_response(
    event_stream: AsyncIterator[StreamEvent],
) -> AsyncIterator[str]:
    """
    Convert RAG StreamEvents to SSE format.
    
    Each StreamEvent becomes:
    event: {event.event}
    data: {event.data}
    """
```

#### Phase 8 Tests

- `test_api_chat.py`: Chat endpoint (stream=False and stream=True), auth, error cases
- `test_api_kb.py`: KB CRUD, add papers
- `test_api_search.py`: Literature search endpoint
- `test_api_streaming.py`: SSE event format, connection handling
- Use `httpx.AsyncClient` with `app=app` for test client (no server needed)

---

### Phase 9: Minimal UI

**Goal**: React frontend with chat, KB selection, and settings.

#### Step 9.1: Project Setup

Use Vite + React + TypeScript. Carry over shadcn/ui from v1.

```bash
cd website
npm create vite@latest . -- --template react-ts
npm install @radix-ui/react-* tailwindcss @tanstack/react-query
npx shadcn@latest init
```

Configure Vite proxy to forward `/api` to FastAPI backend.

#### Step 9.2: TypeScript Types (`types.ts`)

Mirror the Pydantic models from Section 4. Keep in sync manually (or generate via OpenAPI schema).

#### Step 9.3: API Client (`api/client.ts`)

```typescript
class PerspicaciteClient {
  constructor(private baseUrl: string, private token?: string) {}
  
  async chat(request: ChatRequest): Promise<ChatResponse> { ... }
  chatStream(request: ChatRequest): EventSource { ... }
  async listKBs(): Promise<KBStats[]> { ... }
  async createKB(request: KBCreateRequest): Promise<KnowledgeBase> { ... }
  async searchLiterature(request: SearchRequest): Promise<Paper[]> { ... }
  async listProviders(): Promise<ProviderInfo[]> { ... }
  async listConversations(): Promise<Conversation[]> { ... }
}
```

#### Step 9.4: SSE Stream Consumer (`api/streaming.ts`)

```typescript
function consumeSSEStream(
  url: string,
  body: ChatRequest,
  handlers: {
    onStatus: (msg: string) => void;
    onContent: (delta: string) => void;
    onSource: (source: SourceReference) => void;
    onReasoning: (text: string) => void;
    onDone: (meta: DoneMeta) => void;
    onError: (error: string) => void;
  }
): AbortController { ... }
```

Uses named SSE events (not v1's magic prefix protocol).

#### Step 9.5: Core Component Specifications

##### Application State (`context/AppContext.tsx`)

```typescript
interface AppState {
  // Auth
  token: string | null;
  
  // Settings
  settings: {
    provider: string;        // "anthropic" | "openai" | "deepseek" | ...
    model: string;           // "claude-sonnet-4-20250514" | ...
    apiKeys: Record<string, string>;  // provider → key
    citationFormat: string;  // "nature" | "apa" | ...
    theme: "light" | "dark" | "system";
  };
  
  // Knowledge bases
  knowledgeBases: KBStats[];
  activeKB: string;  // KB name
  
  // Conversations
  conversations: Conversation[];
  activeConversationId: string | null;
  
  // RAG
  activeMode: RAGMode;
  useWebSearch: boolean;
}
```

Use React Context for auth/settings (rarely changes) and TanStack Query for server data (KBs, conversations, messages).

##### TanStack Query Hooks (`hooks/`)

```typescript
// hooks/useChat.ts
function useChat() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamContent, setStreamContent] = useState("");
  const [streamSources, setStreamSources] = useState<SourceReference[]>([]);
  const [streamStatus, setStreamStatus] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  
  const sendMessage = async (request: ChatRequest) => {
    setIsStreaming(true);
    setStreamContent("");
    setStreamSources([]);
    
    abortRef.current = consumeSSEStream("/api/chat", request, {
      onStatus: (msg) => setStreamStatus(msg),
      onContent: (delta) => setStreamContent(prev => prev + delta),
      onSource: (source) => setStreamSources(prev => [...prev, source]),
      onReasoning: (text) => { /* append to reasoning accordion */ },
      onError: (error) => { toast.error(error); setIsStreaming(false); },
      onDone: (meta) => {
        setIsStreaming(false);
        queryClient.invalidateQueries(["conversations"]);
      },
    });
  };
  
  const cancel = () => abortRef.current?.abort();
  
  return { sendMessage, cancel, isStreaming, streamContent, streamSources, streamStatus };
}

// hooks/useKnowledgeBases.ts
function useKnowledgeBases() {
  return useQuery({
    queryKey: ["knowledgeBases"],
    queryFn: () => client.listKBs(),
    staleTime: 30_000,
  });
}

function useCreateKB() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (req: KBCreateRequest) => client.createKB(req),
    onSuccess: () => queryClient.invalidateQueries(["knowledgeBases"]),
  });
}

// hooks/useConversations.ts
function useConversations() {
  return useQuery({
    queryKey: ["conversations"],
    queryFn: () => client.listConversations(),
  });
}

// hooks/useProviders.ts
function useProviders() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: () => client.listProviders(),
    staleTime: 300_000,  // Rarely changes
  });
}
```

##### Component Props & Structure

```typescript
// ChatPage.tsx — Main layout
interface ChatPageProps {}
// State: activeConversationId, messages from useConversation(id)
// Layout: flex row → [Sidebar (w-72)] [ChatArea (flex-1)]

// components/Sidebar.tsx
interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  // KB selector
  knowledgeBases: KBStats[];
  activeKB: string;
  onKBChange: (name: string) => void;
  // Mode selector
  activeMode: RAGMode;
  onModeChange: (mode: RAGMode) => void;
  useWebSearch: boolean;
  onWebSearchToggle: (enabled: boolean) => void;
}

// components/MessageList.tsx
interface MessageListProps {
  messages: Message[];
  isStreaming: boolean;
  streamContent: string;
  streamSources: SourceReference[];
  streamStatus: string;
}
// Renders a scrollable list. Auto-scrolls to bottom on new content.
// Streaming message shown as a partial Message at the bottom.

// components/MessageItem.tsx
interface MessageItemProps {
  message: Message;
  isUser: boolean;
}
// User messages: right-aligned, simple text
// Assistant messages: left-aligned, markdown rendered, source cards below
// Markdown: use react-markdown with remark-gfm, rehype-highlight

// components/SourceCard.tsx
interface SourceCardProps {
  source: SourceReference;
  compact?: boolean;  // Inline vs expanded view
}
// Displays: title, authors, year, journal, relevance score badge
// Click: opens DOI/URL in new tab
// Compact mode: single line for inline citations

// components/ChatInput.tsx
interface ChatInputProps {
  onSend: (text: string) => void;
  onCancel: () => void;
  isStreaming: boolean;
  disabled: boolean;
  activeMode: RAGMode;  // Shown as badge next to input
}
// Auto-growing textarea (min 1 row, max 6 rows)
// Enter to send, Shift+Enter for newline
// Cancel button appears during streaming

// components/ModeSelector.tsx
interface ModeSelectorProps {
  value: RAGMode;
  onChange: (mode: RAGMode) => void;
}
// Dropdown or segmented control with mode names + icons
// Each option shows: name, brief description, estimated speed
// Quick ⚡ | Standard 🔍 | Advanced 🧠 | Deep 🔬 | Citation 📑

// components/KBSelector.tsx
interface KBSelectorProps {
  knowledgeBases: KBStats[];
  value: string;
  onChange: (name: string) => void;
}
// Dropdown showing KB name, paper count, last updated
// "Manage KBs" link at bottom → navigates to KnowledgeBasesPage

// KnowledgeBasesPage.tsx
interface KnowledgeBasesPageProps {}
// Layout: grid of KB cards
// Each card: name, description, paper count, chunk count, size, created date
// Actions: Delete (with confirmation dialog)
// Create button → opens CreateKBDialog

// components/CreateKBDialog.tsx
interface CreateKBDialogProps {
  open: boolean;
  onClose: () => void;
  onCreated: (kb: KnowledgeBase) => void;
}
// Form fields:
//   - Name (text input, required)
//   - Description (text input, optional)
//   - Source type: tabs [BibTeX File | DOIs/URLs | Empty]
//   - BibTeX tab: file upload (drag-and-drop zone + click to browse)
//   - DOIs tab: textarea for pasting DOIs/URLs (one per line)
//   - Embedding model: select dropdown (default: text-embedding-3-small)
//   - Chunk size: number input (default: 1000)
// Submit: calls useCreateKB mutation, shows progress

// components/FileUpload.tsx
interface FileUploadProps {
  accept: string;  // ".bib,.bibtex"
  onFile: (file: File) => void;
  label: string;
}
// Drag-and-drop zone with dashed border
// Shows file name and size after selection
// Upload is handled by creating a FormData POST to /api/kb
// The API endpoint for BibTeX upload:
//   POST /api/kb/upload-bibtex
//   Content-Type: multipart/form-data
//   Body: { file: File, name: string, description?: string }

// SettingsPage.tsx
interface SettingsPageProps {}
// Sections:
//   1. API Keys: per-provider input with show/hide toggle, "Test" button
//   2. Default Provider/Model: two linked dropdowns
//   3. Preferences: citation format select, theme toggle
// Saves to localStorage + POST /api/settings (if auth enabled)
```

##### Error Handling Patterns (Frontend)

```typescript
// Global error boundary
class ErrorBoundary extends React.Component<{children: ReactNode}> {
  state = { hasError: false, error: null };
  static getDerivedStateFromError(error: Error) { return { hasError: true, error }; }
  render() {
    if (this.state.hasError) return <ErrorFallback error={this.state.error} />;
    return this.props.children;
  }
}

// API error handling in client.ts
class PerspicaciteClient {
  private async fetch<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(this.token ? { "Authorization": `Bearer ${this.token}` } : {}),
        ...options?.headers,
      },
    });
    
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new APIError(response.status, body.detail || response.statusText);
    }
    
    return response.json();
  }
}

class APIError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// Toast notifications for errors
// Use sonner (shadcn/ui toast) for non-blocking error messages:
//   - Network errors: "Connection lost. Retrying..."
//   - Auth errors: "Session expired. Please re-enter your token."
//   - Server errors: "Something went wrong. Please try again."
//   - Validation errors: Show specific field error messages
```

##### BibTeX File Upload Flow

```
User drags .bib file onto CreateKBDialog
  → FileUpload component reads file via FileReader
  → Shows file name + size
  → User clicks "Create"
  → Frontend sends: POST /api/kb (multipart/form-data)
      Body: { file: .bib, name: "my-kb", embedding_model: "text-embedding-3-small" }
  → Backend: saves file to temp dir, calls KBBuilder.create_from_bibtex()
  → Backend streams progress via SSE (optional) or returns after processing
  → For v2.0: simple blocking request with loading spinner
  → Frontend: shows success toast, navigates to chat with new KB selected
```

Add this API endpoint to Phase 8:

```python
@router.post("/api/kb/upload-bibtex")
async def upload_bibtex(
    file: UploadFile,
    name: str = Form(...),
    description: str = Form(None),
    embedding_model: str = Form("text-embedding-3-small"),
):
    """Create KB from uploaded BibTeX file."""
    # Save to temp file, process, return KBStats
```

#### Step 9.6: UI Design

- Use shadcn/ui for all components (Button, Input, Card, Dialog, Select, Textarea, DropdownMenu, Tabs, Badge, Tooltip, ScrollArea, Separator)
- Tailwind CSS for layout
- Support light/dark/system theme via `next-themes` pattern (CSS variables)
- Responsive: two-column layout (sidebar + main) on desktop, drawer sidebar on mobile
- Clean, minimal research-tool aesthetic: neutral grays, accent blue for actions, green for sources
- Fonts: Inter for UI, JetBrains Mono for code/citations
- Loading states: skeleton components for lists, spinner for actions
- Empty states: illustrated empty state for "no conversations" and "no knowledge bases"

#### Phase 9 Tests

Frontend tests are optional for v2.0. Focus on backend test coverage.
If desired later, use Vitest + React Testing Library for component tests.

---

### Phase 10: MCP Server

**Goal**: Expose Perspicacité as an MCP server for external AI agents.

#### Step 10.1: Implementation (`mcp/server.py`)

```python
from fastmcp import FastMCP

mcp = FastMCP("perspicacite")


@mcp.tool
async def research_literature(
    query: str,
    mode: str = "deep",
    kb_name: str = "default",
    max_iterations: int = 3,
    use_web_search: bool = True,
) -> str:
    """
    Research a scientific question using Perspicacité's RAG system.
    
    Returns: Research answer with citations.
    """


@mcp.tool
async def search_knowledge_base(
    query: str,
    kb_name: str = "default",
    top_k: int = 5,
) -> str:
    """Quick search in a specific knowledge base."""


@mcp.tool
async def list_knowledge_bases() -> str:
    """List available knowledge bases."""


@mcp.tool
async def add_papers_to_kb(
    kb_name: str,
    papers: list[str],  # DOIs or URLs
) -> str:
    """Add papers to a knowledge base."""


@mcp.resource("perspicacite://info")
async def get_info() -> str:
    """Perspicacité capabilities and status."""


@mcp.resource("perspicacite://kb/{name}/stats")
async def get_kb_stats(name: str) -> str:
    """Knowledge base statistics."""
```

#### Step 10.2: Entry Point Integration

In `__main__.py`:

```python
@cli.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 5468,
    no_mcp: bool = False,
    no_ui: bool = False,
):
    """Start Perspicacité server."""
    # Start FastAPI (REST + optional static files)
    # Optionally start MCP server alongside
```

The MCP server should share the same service instances (LLM client, vector store, etc.) as the FastAPI app.

#### Phase 10 Tests

- `test_mcp_server.py`: Test tool calls, resource access, error handling

---

## 8. v1 Code Extraction Guide

### Files to extract (copy + modify)

| v1 Source | v2 Destination | Modifications |
|-----------|---------------|---------------|
| `bibtex2kb/src/chunking.py` (662 lines) | `pipeline/chunking.py` | Add type hints, make tokenizer configurable |
| `bibtex2kb/src/pdf_parser.py` (114 lines) | `pipeline/parsers/pdf.py` | Add type hints, async I/O wrapper |
| `bibtex2kb/src/html_parser.py` (327 lines) | `pipeline/parsers/html.py` | Fix `process_pdf(content=...)` bug, use `httpx` |
| `bibtex2kb/src/youtube_parser.py` (243 lines) | `pipeline/parsers/youtube.py` | Add type hints, async wrapper |
| `bibtex2kb/src/github_parser.py` (56 lines) | `pipeline/parsers/github.py` | Add type hints, use `httpx` |
| `bibtex2kb/src/citation_handler.py` (183 lines) | `pipeline/citations.py` | Add type hints |
| `bibtex2kb/src/url_handler.py` (117 lines) | `pipeline/download.py` | Use `httpx`, add async |
| `bibtex2kb/src/cache_manager.py` (111 lines) | `pipeline/cache.py` | Add type hints |
| `core/llm_utils.py` (341 lines) | `llm/providers.py` | Extract provider registry; `call_llm` → async via LiteLLM |
| `core/hybrid_retrieval.py` (137 lines) | `retrieval/hybrid.py` | Fix div-by-zero, add async, use RRF instead of weighted sum |

### Files to rewrite (use as reference only)

| v1 Source | v2 Destination | What to keep |
|-----------|---------------|-------------|
| `core/core.py` (928 lines) | `rag/modes/quick.py`, `standard.py`, `advanced.py` | Document quality assessment logic, query expansion prompts |
| `core/profonde.py` (1,226 lines) | `rag/modes/deep.py` | Research plan prompt templates, iteration logic structure |
| `core/llm_wrapper.py` (285 lines) | `llm/client.py` | Retry logic pattern (use tenacity) |
| `core/web_search*.py` (4 files) | Replaced by SciLEx adapter | None (SciLEx covers these APIs better) |
| `bibtex2kb/src/bibtex_processor.py` (566 lines) | `pipeline/bibtex.py` | Unpaywall integration, source routing logic |
| `bibtex2kb/src/document_curator.py` (330 lines) | `pipeline/curation.py` | Curation prompt template |
| `bibtex2kb/src/faiss_database_creator.py` (387 lines) | `pipeline/kb_builder.py` | Embedding model selection pattern |
| `fastAPI/app.py` (277 lines) | `api/app.py` | Endpoint structure reference |
| `fastAPI/call.py` (391 lines) | `rag/engine.py` | Mode routing logic reference |
| `mcp_server/server.py` (319 lines) | `mcp/server.py` | Tool definitions, resource structure |

### Files to ignore (not needed in v2)

- `evaluation/` — Separate concern; build evaluation later
- `rr_benchmarking/` — Research artifact, not production code
- `legacy/` — Old code
- `bibtex_libraries/` — Superseded by `pipeline/`
- `bibtex2kb/src/bibtex_processor_bak.py` — Backup file
- `bibtex2kb/src/pubmed_explorer.py`, `arxiv_explorer.py`, `ref_explorer.py` — Superseded by SciLEx
- `tests/old_tests/` — Outdated
- `slurm_scrips/` — HPC-specific

---

## 9. Testing Strategy

### 9.1 Test Pyramid

```
           ┌──────────┐
           │   E2E    │  2-3 tests (full research sessions)
          ┌┴──────────┴┐
          │ Integration │  15-20 tests (API endpoints, pipeline)
         ┌┴────────────┴┐
         │    Unit       │  100+ tests (every module)
         └──────────────┘
```

### 9.2 Fixtures (`conftest.py`)

```python
import pytest
import tempfile
from perspicacite.config.schema import Config
from perspicacite.llm.client import AsyncLLMClient


@pytest.fixture
def config():
    """Test configuration with defaults."""
    return Config()


@pytest.fixture
def temp_dir():
    """Temporary directory for Chroma, SQLite, etc."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_llm_client():
    """Mock LLM that returns canned responses."""
    class MockLLM:
        async def complete(self, messages, **kwargs) -> str:
            return "This is a mock LLM response."
        
        async def stream(self, messages, **kwargs):
            for word in "This is a mock LLM response.".split():
                yield word + " "
    
    return MockLLM()


@pytest.fixture
def sample_papers():
    """List of sample Paper objects for testing."""
    return [
        Paper(id="doi:10.1234/test1", title="Test Paper 1", ...),
        Paper(id="doi:10.1234/test2", title="Test Paper 2", ...),
    ]


@pytest.fixture
def sample_chunks(sample_papers):
    """Pre-chunked documents for retrieval testing."""
    ...


@pytest.fixture
async def chroma_store(temp_dir, embedding_provider):
    """Chroma vector store with test data."""
    store = ChromaVectorStore(persist_dir=temp_dir, ...)
    # Add sample data
    return store


@pytest.fixture
async def session_store(temp_dir):
    """SQLite session store for testing."""
    store = SessionStore(db_path=f"{temp_dir}/test.db")
    await store.init_db()
    return store
```

### 9.3 Mocking Strategy

- **LLM calls**: Always mock in unit tests. Use `MockLLM` fixture.
- **Embeddings**: Use a fixed-dimension random embedding provider for tests.
- **HTTP**: Use `pytest-httpx` for mocking external API calls.
- **Chroma**: Use real Chroma with temp directory (it's fast enough).
- **SQLite**: Use real SQLite with temp file (in-memory or temp dir).
- **SciLEx**: Mock at the import level; test adapter mapping separately.

### 9.4 Coverage Targets

| Module | Target |
|--------|--------|
| `models/` | 95% |
| `config/` | 90% |
| `llm/` | 80% |
| `retrieval/` | 85% |
| `pipeline/` | 80% |
| `rag/` | 75% |
| `search/` | 75% |
| `memory/` | 85% |
| `api/` | 80% |
| `mcp/` | 70% |
| **Overall** | **≥70%** |

### 9.5 CI Configuration (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --dev
      - run: uv run ruff check src/ tests/
      - run: uv run pytest -m "not live" --cov --cov-report=xml
      - run: uv run mypy src/perspicacite/
```

---

## 10. Deployment

### 10.1 Local Development

```bash
# Clone and setup
git clone <repo>
cd perspicacite_v2
uv sync --dev

# Configure
cp config.example.yml config.yml
cp .env.example .env
# Edit .env with API keys

# Install SciLEx (optional)
uv pip install -e packages_to_use/SciLEx

# Run
uv run perspicacite serve --reload

# Frontend (separate terminal)
cd website && npm install && npm run dev
```

### 10.2 Docker

```dockerfile
FROM python:3.11-slim AS backend
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev
COPY src/ src/
COPY config.example.yml config.yml

FROM node:20-slim AS frontend
WORKDIR /app/website
COPY website/package.json website/package-lock.json ./
RUN npm ci
COPY website/ .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY --from=backend /app /app
COPY --from=frontend /app/website/dist /app/website/dist
EXPOSE 5468
CMD ["uv", "run", "perspicacite", "serve", "--host", "0.0.0.0", "--port", "5468"]
```

### 10.3 Entry Point (`__main__.py`)

```python
"""
python -m perspicacite
  → starts FastAPI server + optional MCP server
  → serves React UI from website/dist if available

Flags:
  --no-mcp    Disable MCP server
  --no-ui     Headless mode (API only)
  --reload    Dev mode with auto-reload
  --port      Server port (default: 5468)
  --host      Server host (default: 0.0.0.0)
"""
```

---

## Appendix A: Key Prompts

These are the LLM prompt templates used internally. Store them in a `prompts/` directory or as module-level constants.

### System Prompt

```
You are Perspicacité, a scientific literature research assistant.

Your role is to help researchers find, analyze, and synthesize scientific papers.
Provide evidence-based answers with citations.

Guidelines:
- Always cite sources as [Author et al., Year]
- If information is insufficient, say so clearly
- Highlight conflicting findings from different sources
- Use technical language appropriate for the field
- When uncertain, indicate confidence level
```

### Research Plan Generation (Deep mode)

```
Given the following research question, create a step-by-step research plan.
Each step should be a specific sub-question to investigate.

Research question: {query}
Available knowledge base: {kb_name} ({paper_count} papers)

Create 3-5 research steps. For each step, provide:
1. The specific query to search
2. The purpose of this step
3. What aspect of the main question it addresses

Return as JSON: [{"query": "...", "purpose": "...", "aspect": "..."}]
```

### Document Quality Assessment

```
Evaluate these retrieved documents for answering the question.

Question: {query}
Documents:
{formatted_documents}

For each document, rate relevance (0-10) and explain why.
Identify any missing aspects not covered by these documents.

Return as JSON: {
  "assessments": [{"doc_id": "...", "relevance": 8, "reason": "..."}],
  "missing_aspects": ["...", "..."],
  "overall_sufficiency": "sufficient" | "partial" | "insufficient"
}
```

---

## Appendix B: Phase Dependency Graph

```
Phase 1 (Foundation) ──────────┐
                                ├──► Phase 2 (LLM) ─────┐
                                │                         ├──► Phase 5 (RAG Engine)
Phase 3 (Vector Store) ────────┤                         │
                                ├──► Phase 4 (Pipeline) ──┘
                                │                              │
Phase 6 (SciLEx) ──────────────┤                              ├──► Phase 8 (API)
                                │                              │        │
Phase 7 (Memory) ──────────────┘                              │        ├──► Phase 9 (UI)
                                                               │        │
                                                               └────────┴──► Phase 10 (MCP)
```

Phases 1-4 can be built in sequence. Phases 6 and 7 can be built in parallel once Phase 1 is done. Phase 5 requires 2, 3, and 4. Phases 8-10 require Phase 5.

---

## Appendix C: Definition of Done (per Phase)

A phase is complete when:

1. All specified modules are implemented
2. All type hints are present and pass `mypy --strict`
3. All unit tests pass
4. Code passes `ruff check` with no errors
5. Coverage meets or exceeds the target for that module
6. Each public function/class has a docstring
7. No `print()` statements (use `structlog` logger)
8. No hardcoded API keys, paths, or secrets
9. All async functions use `async/await` (no blocking calls in async context)

---

## Appendix D: Error Handling Matrix

Every service can fail. This matrix defines what happens when each one does.

### Service Failure Behaviors

| Service | Failure Mode | Detection | Recovery | User Impact |
|---------|-------------|-----------|----------|-------------|
| **LLM Provider** | API timeout | `asyncio.TimeoutError` after 120s | Retry 3x with exponential backoff (2s, 4s, 8s). If all fail, return error. | "LLM provider is not responding. Please try again." |
| **LLM Provider** | Rate limit (429) | `litellm.RateLimitError` | Retry with `Retry-After` header value or 30s default. Max 3 retries. | Transparent retry; user sees slight delay. |
| **LLM Provider** | Auth error (401) | `litellm.AuthenticationError` | No retry. Log error. | "Invalid API key for {provider}. Please check settings." |
| **LLM Provider** | Model not found | `litellm.NotFoundError` | No retry. | "Model {model} is not available. Please select a different model." |
| **LLM Provider** | Context length exceeded | `litellm.ContextWindowExceededError` | Truncate context (remove oldest history, reduce doc count), retry once. | Transparent; slightly less context used. |
| **Chroma** | Collection not found | `chromadb.errors.InvalidCollectionException` | Return empty results. | "Knowledge base '{name}' not found." |
| **Chroma** | Disk full | `OSError` | Log error, return error. | "Storage is full. Please delete unused knowledge bases." |
| **Chroma** | Corrupt index | Various | Delete and rebuild collection from metadata. | "Knowledge base is being rebuilt. Please try again shortly." |
| **Embedding API** | Timeout/error | `httpx.TimeoutException` | Retry 3x. If all fail, try local SentenceTransformer fallback. | Transparent fallback; may be slower. |
| **SciLEx search** | Import error | `ImportError` | Switch to Google Scholar fallback. | Transparent; fewer APIs searched. |
| **SciLEx search** | API timeout | `asyncio.TimeoutError` (120s) | Use Google Scholar fallback. | "Academic search is slow. Using limited results." |
| **SciLEx search** | Individual API failure | Caught internally by SciLEx circuit breaker | SciLEx skips failed API, continues with others. | Transparent; results from fewer sources. |
| **PDF download** | 403/404 | `httpx.HTTPStatusError` | Try Unpaywall for alternative URL. If no alternative, skip PDF. | "Could not access PDF for {title}. Using abstract only." |
| **PDF download** | Timeout | `httpx.TimeoutException` | Retry once with doubled timeout. If fail, skip. | Same as above. |
| **PDF parsing** | Corrupt/encrypted PDF | `pdfplumber` exception | Skip document, log warning. | "Could not parse PDF for {title}. Using abstract only." |
| **SQLite** | Lock timeout | `aiosqlite.OperationalError` | Retry 3x with 100ms delay. | Transparent retry. |
| **SQLite** | Corrupt DB | `aiosqlite.DatabaseError` | Backup corrupt file, recreate schema. Sessions/history lost. | "Session data was reset due to an error." |
| **CrossRef API** | Timeout | `httpx.TimeoutException` | Skip citation resolution for this paper. | Citations may be incomplete. |
| **Frontend ↔ Backend** | Network error | `fetch` rejects | Show toast: "Connection lost". Auto-retry on reconnect. | "Connection lost. Reconnecting..." |
| **Frontend ↔ Backend** | 401 Unauthorized | Response status 401 | Clear token, show login/token entry. | "Session expired. Please enter your token." |

### Error Response Format (API)

All API errors return consistent JSON:

```python
class ErrorResponse(BaseModel):
    error: str           # Machine-readable error code
    detail: str          # Human-readable message
    status_code: int     # HTTP status code
    request_id: str      # For debugging/support

# Examples:
# 400: {"error": "invalid_request", "detail": "KB name is required", ...}
# 401: {"error": "unauthorized", "detail": "Invalid or missing auth token", ...}
# 404: {"error": "not_found", "detail": "Knowledge base 'xyz' not found", ...}
# 422: {"error": "validation_error", "detail": "Invalid RAG mode: 'turbo'", ...}
# 429: {"error": "rate_limited", "detail": "Too many requests. Retry after 30s", ...}
# 500: {"error": "internal_error", "detail": "An unexpected error occurred", ...}
# 503: {"error": "service_unavailable", "detail": "LLM provider is not responding", ...}
```

### Error Handling in RAG Modes

During a multi-step RAG execution (especially Deep mode), partial failures should not abort the entire operation:

```python
class DeepRAGMode:
    async def execute(self, ...):
        for step in research_plan:
            try:
                results = await self._execute_step(step)
            except ToolExecutionError as e:
                # Log the failure, emit a status event, continue with next step
                yield StreamEvent(event="error", data=json.dumps({
                    "message": f"Step '{step.purpose}' failed: {e}. Continuing..."
                }))
                step.success = False
                continue
            
            # Even if some steps fail, generate answer from what we have
        
        if all(not s.success for s in research_plan):
            yield StreamEvent(event="error", data=json.dumps({
                "message": "Could not find relevant information. Please rephrase your query."
            }))
            return
        
        # Generate answer from successful steps
        yield from self._generate_answer(successful_steps)
```

---

## Appendix E: Streaming Protocol Specification

### SSE Event Types (Complete)

| Event | Payload | When Emitted | Frequency |
|-------|---------|-------------|-----------|
| `status` | `{"message": str}` | Processing status updates | Multiple per request |
| `content` | `{"delta": str}` | Answer text chunks | Many (per token) |
| `source` | `SourceReference` (JSON) | When a source is cited | 1 per source |
| `reasoning` | `{"step": str}` | Chain-of-thought (Deep/Citation modes) | 1 per reasoning step |
| `plan` | `{"step_index": int, "total_steps": int, "query": str, "purpose": str}` | Research plan steps (Deep mode) | 1 per plan step |
| `tool_call` | `{"tool": str, "args": dict}` | Tool invocation | 1 per tool call |
| `tool_result` | `{"tool": str, "result": str, ...}` | Tool result | 1 per tool call |
| `error` | `{"message": str, "recoverable": bool}` | Non-fatal errors during processing | 0+ per request |
| `done` | `{"conversation_id": str, "tokens_used": int, "mode": str, "iterations": int}` | Stream complete | Exactly 1, always last |

### Client Disconnect Handling

```python
# api/streaming.py

async def stream_rag_response(
    event_stream: AsyncIterator[StreamEvent],
    request: Request,  # FastAPI request object
) -> AsyncIterator[str]:
    """
    Convert RAG StreamEvents to SSE format.
    Handles client disconnection gracefully.
    """
    try:
        async for event in event_stream:
            # Check if client is still connected
            if await request.is_disconnected():
                logger.info("Client disconnected mid-stream")
                break
            
            yield f"event: {event.event}\ndata: {event.data}\n\n"
    except asyncio.CancelledError:
        # Client closed connection; this is normal
        logger.info("Stream cancelled by client")
    except Exception as e:
        # Unexpected error during streaming
        logger.error("Stream error", error=str(e))
        error_event = StreamEvent(
            event="error",
            data=json.dumps({"message": "Stream interrupted", "recoverable": False})
        )
        yield f"event: {error_event.event}\ndata: {error_event.data}\n\n"
    finally:
        # Always emit done event if we haven't already
        # (The RAG engine should emit this, but as a safety net)
        pass
```

### Partial Response Recovery

When an LLM stream fails mid-generation:

```python
async def _stream_llm_with_recovery(
    self,
    messages: list[dict],
    llm: AsyncLLMClient,
) -> AsyncIterator[str]:
    """
    Stream LLM response with partial failure recovery.
    
    If stream breaks mid-way:
    1. Collect what we have so far
    2. Attempt non-streaming completion for the rest
    3. If that also fails, return what we have + error notice
    """
    collected = []
    try:
        async for chunk in llm.stream(messages):
            collected.append(chunk)
            yield chunk
    except Exception as e:
        logger.warning("LLM stream interrupted", error=str(e), collected_length=len(collected))
        
        if not collected:
            raise  # Nothing to recover; let caller handle
        
        # Try non-streaming completion with partial context
        try:
            partial = "".join(collected)
            messages_with_partial = messages + [
                {"role": "assistant", "content": partial},
                {"role": "user", "content": "(Please continue your response from where you left off.)"},
            ]
            remainder = await llm.complete(messages_with_partial)
            yield remainder
        except Exception:
            # Give up; return what we have with a notice
            yield "\n\n*[Response was interrupted. The above is a partial answer.]*"
```

### Backpressure Handling

SSE is a push protocol — the server pushes events and the client buffers them. For Perspicacité, backpressure is not a significant concern because:

1. **Content events** are rate-limited by LLM token generation speed (~50-100 tokens/sec)
2. **Source events** are discrete (5-10 per request)
3. **Total stream duration** is seconds to minutes, not hours

If needed, `sse-starlette` handles buffering internally. The main protection is the `request.is_disconnected()` check to stop wasting resources on abandoned streams.

### SSE Connection Lifecycle

```
Client                                  Server
  |                                        |
  |--- POST /api/chat (stream=true) ------>|
  |                                        |--- Validate request
  |                                        |--- Start RAG engine
  |<------ event: status ------------------|
  |<------ event: plan (Deep mode) --------|
  |<------ event: tool_call ---------------|
  |<------ event: tool_result -------------|
  |<------ event: status ------------------|
  |<------ event: content (many) ----------|
  |<------ event: source (per source) -----|
  |<------ event: content (more) ----------|
  |<------ event: done --------------------|
  |                                        |--- Save to session store
  |--- Connection closed ----------------->|
  
  Error case:
  |<------ event: error (recoverable) -----|  (stream continues)
  |<------ event: content (partial) -------|
  |<------ event: done --------------------|
  
  Client disconnect:
  |--- Connection reset ------------------>|
  |                                        |--- detect is_disconnected()
  |                                        |--- cancel RAG engine
  |                                        |--- cleanup resources
```

### Frontend SSE Consumer (Detailed)

```typescript
// api/streaming.ts

import { fetchEventSource } from "@microsoft/fetch-event-source";

export function consumeSSEStream(
  url: string,
  body: ChatRequest,
  token: string | null,
  handlers: StreamHandlers,
): AbortController {
  const ctrl = new AbortController();
  
  fetchEventSource(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { "Authorization": `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    signal: ctrl.signal,
    
    onopen: async (response) => {
      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }
    },
    
    onmessage: (ev) => {
      const data = JSON.parse(ev.data);
      switch (ev.event) {
        case "status":    handlers.onStatus(data.message); break;
        case "content":   handlers.onContent(data.delta); break;
        case "source":    handlers.onSource(data as SourceReference); break;
        case "reasoning": handlers.onReasoning(data.step); break;
        case "plan":      handlers.onPlan?.(data); break;
        case "tool_call": handlers.onToolCall?.(data); break;
        case "tool_result": handlers.onToolResult?.(data); break;
        case "error":     handlers.onError(data.message); break;
        case "done":      handlers.onDone(data); break;
      }
    },
    
    onerror: (err) => {
      // fetchEventSource retries by default; we want to stop on auth errors
      if (err instanceof Error && err.message.includes("401")) {
        handlers.onError("Authentication failed");
        ctrl.abort();
        throw err;  // Stop retrying
      }
      // For other errors, let it retry once then give up
      handlers.onError("Connection interrupted. Retrying...");
    },
    
    onclose: () => {
      // Server closed the connection without a done event
      // This happens on server restart or crash
      handlers.onError("Connection closed unexpectedly");
    },
  });
  
  return ctrl;
}

interface StreamHandlers {
  onStatus: (message: string) => void;
  onContent: (delta: string) => void;
  onSource: (source: SourceReference) => void;
  onReasoning: (text: string) => void;
  onPlan?: (plan: PlanStep) => void;
  onToolCall?: (call: ToolCall) => void;
  onToolResult?: (result: ToolResult) => void;
  onError: (message: string) => void;
  onDone: (meta: DoneMeta) => void;
}
```

---

## Appendix F: Security Model

### Token-Based Authentication (v2.0)

v2.0 uses a simple shared-secret token model. This is appropriate for single-user or small-team deployments.

#### Token Flow

```
1. Admin sets token in .env:
   PERSPICACITE_AUTH_TOKEN=my-secret-token-here

2. Frontend stores token in localStorage:
   localStorage.setItem("perspicacite_token", token)

3. Every API request includes:
   Authorization: Bearer my-secret-token-here

4. Backend validates:
   - If auth.enabled is false in config → skip check, allow all
   - If auth.enabled is true → compare token against env var
   - Invalid/missing token → 401 Unauthorized
```

#### Implementation (`api/auth.py`)

```python
from fastapi import HTTPException, Header, Depends
from perspicacite.config.schema import Config


class AuthManager:
    def __init__(self, config: Config):
        self.enabled = config.auth.enabled
        self.token = config.auth.token or os.environ.get("PERSPICACITE_AUTH_TOKEN")
    
    async def verify(self, authorization: str = Header(None)) -> str:
        """
        Verify auth token and return user_id.
        
        Returns:
            "default" for valid token (single-user mode)
        
        Raises:
            HTTPException(401) if token is invalid
        """
        if not self.enabled:
            return "default"
        
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header required")
        
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization format")
        
        token = authorization[7:]  # Strip "Bearer "
        
        if not self.token:
            raise HTTPException(
                status_code=500,
                detail="Auth is enabled but no token configured. Set PERSPICACITE_AUTH_TOKEN."
            )
        
        if not hmac.compare_digest(token, self.token):
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return "default"  # Single-user mode; future: decode JWT for multi-user


# FastAPI dependency
def get_auth_manager(config: Config = Depends(get_config)) -> AuthManager:
    return AuthManager(config)

async def get_current_user(
    auth: AuthManager = Depends(get_auth_manager),
    authorization: str = Header(None),
) -> str:
    return await auth.verify(authorization)
```

#### Session Expiration

For v2.0, tokens don't expire (they're static secrets). The frontend should:
- Persist token in `localStorage`
- On 401 response: clear token, show token entry UI
- No automatic token refresh needed

#### CORS Configuration

```python
# api/middleware.py

def configure_cors(app: FastAPI, config: Config):
    """
    CORS settings based on environment.
    """
    if config.server.reload:
        # Development: permissive
        origins = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5468"]
    else:
        # Production: restrict to same-origin + configured origins
        origins = config.server.cors_origins or []
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
```

#### Security Checklist

- [ ] Auth token compared with `hmac.compare_digest` (timing-safe)
- [ ] API keys never logged (masked in structlog output)
- [ ] API keys never returned in API responses
- [ ] Config file permissions: `chmod 600 config.yml`
- [ ] `.env` in `.gitignore`
- [ ] No secrets in Docker image layers (use runtime env vars)
- [ ] CORS restricted in production
- [ ] Rate limiting on auth endpoints (prevent brute force)
- [ ] File upload size limit (10MB for BibTeX files)
- [ ] Path traversal protection on KB names (alphanumeric + hyphen + underscore only)
- [ ] SQL injection: using parameterized queries via aiosqlite (never f-string SQL)

---

## Appendix G: v1 Migration Guide

### FAISS → Chroma Data Migration

For users with existing v1 knowledge bases (FAISS indexes), provide a migration script.

#### Migration Script (`scripts/migrate_v1_kb.py`)

```python
"""
Migrate a Perspicacité v1 FAISS knowledge base to v2 Chroma format.

Usage:
    python -m perspicacite.scripts.migrate_v1_kb \
        --v1-path /path/to/knowledge_bases/my_kb \
        --v2-name my_kb \
        --embedding-model text-embedding-3-small

What it does:
    1. Loads FAISS index + docstore from v1 path
    2. Extracts all documents with metadata
    3. Re-chunks if needed (v1 uses word-based, v2 uses token-based)
    4. Re-embeds with v2 embedding model (required — v1 embeddings may differ)
    5. Stores in Chroma collection
    6. Builds BM25 index
    7. Saves KB metadata to SQLite
"""
import asyncio
import click
from pathlib import Path


@click.command()
@click.option("--v1-path", required=True, type=click.Path(exists=True))
@click.option("--v2-name", required=True, type=str)
@click.option("--embedding-model", default="text-embedding-3-small")
@click.option("--rechunk", is_flag=True, default=False, help="Re-chunk documents with v2 settings")
def migrate(v1_path: str, v2_name: str, embedding_model: str, rechunk: bool):
    asyncio.run(_migrate(v1_path, v2_name, embedding_model, rechunk))


async def _migrate(v1_path: str, v2_name: str, embedding_model: str, rechunk: bool):
    """
    Migration steps:
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    
    v1_dir = Path(v1_path)
    
    # Step 1: Load v1 FAISS index
    # v1 stores: index.faiss, index.pkl in the KB directory
    print(f"Loading v1 FAISS index from {v1_dir}...")
    
    # We need the original embedding model to load FAISS
    # v1 default was OpenAI text-embedding-ada-002
    v1_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    try:
        v1_db = FAISS.load_local(
            str(v1_dir), v1_embeddings, allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Make sure OPENAI_API_KEY is set (needed for v1 embedding model).")
        return
    
    # Step 2: Extract all documents
    docstore = v1_db.docstore
    doc_ids = list(v1_db.index_to_docstore_id.values())
    
    documents = []
    for doc_id in doc_ids:
        doc = docstore.search(doc_id)
        if doc:
            documents.append({
                "text": doc.page_content,
                "metadata": doc.metadata or {},
            })
    
    print(f"Extracted {len(documents)} document chunks from v1.")
    
    # Step 3: Optionally re-chunk
    if rechunk:
        from perspicacite.pipeline.chunking import chunk_text
        
        # Group chunks back by paper (using metadata.source or similar)
        # Then re-chunk with v2 settings
        print("Re-chunking with v2 token-based chunking...")
        # ... implementation ...
    
    # Step 4: Create v2 KB
    from perspicacite.config.loader import load_config
    from perspicacite.llm.embeddings import LiteLLMEmbeddingProvider
    from perspicacite.retrieval.chroma_store import ChromaVectorStore
    from perspicacite.models.documents import DocumentChunk, ChunkMetadata
    
    config = load_config()
    embedding_provider = LiteLLMEmbeddingProvider(model=embedding_model)
    vector_store = ChromaVectorStore(
        persist_dir=config.database.chroma_path,
        embedding_provider=embedding_provider,
    )
    
    # Step 5: Create collection and add documents
    await vector_store.create_collection(v2_name, embedding_provider.dimension)
    
    chunks = []
    for i, doc in enumerate(documents):
        meta = doc["metadata"]
        chunks.append(DocumentChunk(
            id=f"migrated_{i}",
            text=doc["text"],
            metadata=ChunkMetadata(
                paper_id=meta.get("source", f"unknown_{i}"),
                chunk_index=i,
                section=meta.get("section"),
                title=meta.get("title"),
                authors=meta.get("authors"),
                year=int(meta["year"]) if meta.get("year") else None,
                doi=meta.get("doi"),
                url=meta.get("url"),
            ),
        ))
    
    added = await vector_store.add_documents(v2_name, chunks)
    print(f"Migrated {added} chunks to Chroma collection '{v2_name}'.")
    
    # Step 6: Save KB metadata
    from perspicacite.memory.session_store import SessionStore
    from perspicacite.models.kb import KnowledgeBase, ChunkConfig
    
    store = SessionStore(db_path=config.database.path)
    await store.init_db()
    await store.save_kb_metadata(KnowledgeBase(
        name=v2_name,
        description=f"Migrated from v1 FAISS: {v1_dir.name}",
        collection_name=v2_name,
        embedding_model=embedding_model,
        chunk_config=ChunkConfig(),
        paper_count=len(set(c.metadata.paper_id for c in chunks)),
        chunk_count=len(chunks),
    ))
    
    print(f"Migration complete. KB '{v2_name}' is ready to use.")


if __name__ == "__main__":
    migrate()
```

#### What Migrates vs. What Doesn't

| Data | Migrates? | Notes |
|------|-----------|-------|
| Document text chunks | Yes | Extracted from FAISS docstore |
| Chunk metadata (title, authors, DOI, URL) | Yes | From LangChain Document.metadata |
| Embeddings | **No** — re-embedded | v2 may use different embedding model |
| BM25 index | **No** — rebuilt | Rebuilt automatically from migrated chunks |
| Conversations / chat history | **No** | v1 has no persistence |
| User preferences | **No** | v1 has no persistence |
| API keys | Manual | User re-enters in v2 config |
| BibTeX source files | Manual | User can re-process with v2 pipeline for richer metadata |

#### Migration Recommendations for Users

1. **Simple migration**: Run the script to get existing KBs working in v2 quickly
2. **Best quality**: Re-process original BibTeX files with v2's improved pipeline (better chunking, metadata, curation). This gives:
   - Token-based chunking (vs v1's word-based)
   - Section-aware splitting
   - Richer metadata in Chroma
   - BM25 + hybrid search capabilities
3. **Hybrid approach**: Migrate for immediate use, then re-process source BibTeX when convenient

---

**End of Implementation Guide**
