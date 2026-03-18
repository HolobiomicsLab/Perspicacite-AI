# Code Quality Assessment: Perspicacité v2

**Review Date**: 2026-03-18
**Model**: MiniMax-M2.7
**Assessment Level**: Partial (reviewed ~60% of implementation based on spec completeness)

---

## Overall Status: **Partially Implemented (~40-50% complete)**

The planning documentation is comprehensive, but the implementation has significant gaps between the spec and the actual code.

---

## Strengths

1. **Architecture is sound** — The layered design (config → models → LLM → retrieval → pipeline → RAG → API → MCP) follows good software engineering principles. Clean separation of concerns.

2. **Pydantic v2 models are well-designed** — `papers.py`, `schema.py`, `config/` are properly typed with validators, good use of enums and field constraints.

3. **LLM client (`llm/client.py`)** — Clean async implementation with retry logic (tenacity), proper lazy loading, structured logging, streaming support, and fallback mechanism.

4. **Session store (`memory/session_store.py`)** — Well-structured SQLite schema with proper async usage, good use of `aiosqlite`.

5. **Configuration system** — Pydantic-based config with layered loading (YAML + env + CLI), good validators.

6. **Chroma store** — Proper async implementation with metadata conversion helpers, filter handling.

7. **pyproject.toml** — Modern setup with proper tool configs (ruff, mypy, pytest), appropriate dependencies.

---

## Critical Issues

### 1. API Routes are Stubbed Out

`src/perspicacite/api/routes/chat.py:14-43` — The `/chat` endpoint is a **placeholder** returning hardcoded test responses. It doesn't actually use the RAG engine:

```python
async def event_generator():
    # Placeholder - would use actual RAG engine
    yield {"event": "status", "data": '{"message": "Processing..."}'}
    yield {"event": "content", "data": '{"delta": "This is a "}'}
```

This is the **main entry point** and it's not connected to anything.

### 2. CLI `create_kb`, `list_kb`, `query` are Not Implemented

`src/perspicacite/cli.py:153-237` — Commands exist but just print `(not implemented yet)` / `(RAG not implemented yet)`.

### 3. MCP Server Not Implemented

`src/perspicacite/mcp/server.py` exists but no actual MCP tools/resources are defined — it's an empty shell.

### 4. Several RAG Modes are Incomplete

- `rag/modes/standard.py` — Hybrid search mentioned in comment but only vector search used
- `rag/modes/deep.py` — `_adjust_plan()` at line 168 just returns the same plan (no actual adjustment logic)
- `rag/modes/advanced.py`, `rag/modes/citation.py` — Not reviewed but likely similar stubbing

### 5. Missing Key Components

- `pipeline/bibtex.py` — Not in file list, BibTeX parsing not implemented
- `pipeline/curation.py` — Not in file list, LLM curation not implemented
- BM25 reranking in hybrid retrieval is referenced but not implemented
- `rag/tools.py` — Tool registry exists but actual tool implementations are minimal

### 6. Inconsistent Typing

The codebase has good Pydantic models for data transfer, but internal RAG mode implementations use `Any` liberally:

```python
async def execute(
    self,
    request: RAGRequest,
    llm: Any,              # Should be LLMClient protocol
    vector_store: Any,     # Should be VectorStore protocol
    embedding_provider: Any,  # Should be EmbeddingProvider protocol
    tools: Any,
) -> RAGResponse:
```

---

## Moderate Issues

### 7. No Tests for Core RAG Modes

Tests directory only has:
- `test_config.py`
- `test_models.py`
- `test_llm_client.py`
- `test_embeddings.py`
- `test_chroma_store.py`
- `test_scilex_adapter.py`
- `test_download.py`
- `test_deep_research.py` (integration)

But **no tests for RAG modes** exist — `test_rag_quick.py`, `test_rag_standard.py`, etc. are listed in the spec (IMPLEMENTATION_GUIDE.md lines 289-297) but don't exist in the `tests/unit/` directory.

### 8. Duplicated Files / Confusion

- `rag/chunking.py` and `pipeline/chunking.py` — potential duplication, unclear which is used
- `rag/dynamic_kb.py` and `rag/simple_embeddings.py` — not reviewed but exist as separate files from main spec
- `rag/assessment.py`, `rag/dynamic_kb.py`, `rag/modes/agentic.py`, `rag/modes/deep_research.py` — not referenced in planning doc, unclear purpose or relationship to planned modes

### 9. Hardcoded Values / Magic Numbers

- `retrieval/chroma_store.py:36` — `top_k=20` then sliced to 10 in standard mode
- `rag/modes/deep.py:57` — `top_k=5`
- Various hardcoded limits without corresponding config entries

### 10. Error Handling is Generic

Most error handling just re-raises with a log message. No custom exceptions, no error recovery strategies beyond the tenacity retry decorator on LLM calls.

### 11. No Input Validation on API

`models/api.py` (ChatRequest) not reviewed in detail, but the API routes don't validate:
- KB existence before querying
- Mode validity against available/implemented modes
- User permissions or rate limits

---

## Minor Issues

- `chromadb` import has backward-compat try/except for `IncludeEnum` (`retrieval/chroma_store.py:7-11`) — would be cleaner to just use the new API
- `_filters_to_where()` in `chroma_store.py` uses string-based filter construction, could use a more type-safe approach
- `deep.py:143-158` — Very naive plan parsing (`response.split("\n")`), fragile
- No graceful degradation if Chroma vector store is unavailable
- Root-level Python files (`full_chatbot.py`, `minimal_chatbot.py`, `web_app_full.py`) appear to be scratch/exploration files rather than proper integration points

---

## Summary Table

| Area | Status | Quality | Notes |
|------|--------|---------|-------|
| Config/Schema | ✅ Complete | Good | Well-structured Pydantic models |
| Pydantic Models | ✅ Complete | Good | Proper validation, frozen configs |
| LLM Client | ✅ Complete | Good | Async, retries, streaming, fallbacks |
| Chroma Store | ✅ Complete | Good | Async, proper metadata conversion |
| Session Store | ✅ Complete | Good | SQLite with proper schema |
| API Routes | ❌ Stubbed | Poor | Chat endpoint is placeholder |
| CLI | ❌ Stubbed | Poor | Commands print "not implemented" |
| RAG Modes | ⚠️ Partial | Mixed | Core works, features missing |
| MCP Server | ❌ Empty | Poor | Shell only, no tools defined |
| Pipeline (Bibtex, Chunking, Curation) | ⚠️ Unknown | Unknown | Files may not exist |
| Tests | ⚠️ Minimal | Needs Work | No RAG mode tests |
| Documentation | ✅ Excellent | — | Comprehensive planning docs |

---

## Recommendations (Priority Order)

1. **Fix the API chat endpoint** — Connect `api/routes/chat.py` to the RAG engine immediately. This is blocking all frontend development.

2. **Implement at least one complete RAG mode end-to-end** — Standard mode with hybrid search (vector + BM25) and reranking.

3. **Add integration tests for the KB pipeline** — BibTeX → parse → chunk → embed → store → query. This validates the entire system.

4. **Clean up root-level Python files** — Remove or relocate `full_chatbot.py`, `minimal_chatbot.py`, `web_app_full.py` to proper locations.

5. **Add protocol definitions** — Create `VectorStore`, `EmbeddingProvider`, `LLMClient` protocols rather than using `Any` type. Enables better type checking and testing.

6. **Create custom exception hierarchy** — Domain-specific exceptions (e.g., `KnowledgeBaseNotFoundError`, `RAGModeNotSupportedError`) instead of generic `Exception`.

7. **Decide fate of extra files** — `rag/assessment.py`, `rag/dynamic_kb.py`, `rag/modes/agentic.py`, `rag/modes/deep_research.py`, `rag/chunking.py` — either integrate or remove.

8. **Add configuration for magic numbers** — `top_k`, `rerank_top_n`, iteration counts should come from config, not be hardcoded.

---

## Files Reviewed

### Core Implementation (reviewed)
- `src/perspicacite/models/papers.py`
- `src/perspicacite/llm/client.py`
- `src/perspicacite/rag/engine.py`
- `src/perspicacite/retrieval/chroma_store.py`
- `src/perspicacite/api/routes/chat.py`
- `src/perspicacite/rag/modes/standard.py`
- `src/perspicacite/rag/modes/deep.py`
- `src/perspicacite/rag/modes/base.py`
- `src/perspicacite/config/schema.py`
- `src/perspicacite/api/app.py`
- `src/perspicacite/memory/session_store.py`
- `src/perspicacite/cli.py`

### Configuration (reviewed)
- `pyproject.toml`
- `config.example.yml` (mentioned in spec)

### Documentation (reviewed)
- `IMPLEMENTATION_GUIDE.md`
- `Perspicacite_v2_Planning.md`

### Not Reviewed (incomplete list)
- `src/perspicacite/pipeline/bibtex.py`
- `src/perspicacite/pipeline/curation.py`
- `src/perspicacite/rag/tools.py`
- `src/perspicacite/search/scilex_adapter.py`
- `src/perspicacite/mcp/server.py`
- `src/perspicacite/rag/modes/quick.py`
- `src/perspicacite/rag/modes/advanced.py`
- `src/perspicacite/rag/modes/citation.py`
