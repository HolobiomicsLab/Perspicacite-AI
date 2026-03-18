# Dynamic KB Building for Agentic RAG - Implementation Summary

## Overview

This implementation adds a **DeepResearch** mode that builds a dynamic knowledge base from scratch for each research query, following the workflow:

```
User Query → SciLEx Search → PDF Download → Relevance Assessment → Dynamic KB → Answer
```

## Files Created/Modified

### 1. PDF Downloader (`src/perspicacite/pipeline/download.py`)
**Purpose**: Download and parse PDFs from URLs

**Key Classes**:
- `DownloadResult`: Result of PDF download with content and metadata
- `PaperParser`: Parse PDF content using pymupdf
- `Unpaywall`: Query Unpaywall API for Open Access PDF URLs
- `PDFDownloader`: Main downloader with retry logic and progress tracking

**Features**:
- Async HTTP with httpx
- Retry with exponential backoff (tenacity)
- PDF magic bytes verification
- Progress callbacks
- Unpaywall integration for OA papers

### 2. Paper Assessment (`src/perspicacite/rag/assessment.py`)
**Purpose**: LLM-based relevance assessment

**Key Classes**:
- `RelevanceAssessment`: Structured assessment result
- `PaperAssessor`: Assess paper relevance to query
- `QueryRefiner`: Refine search queries based on failed results

**Features**:
- JSON-structured LLM responses
- Parallel batch assessment
- Relevance scoring (0.0-1.0)
- Confidence scoring
- Query refinement for iterative search

### 3. Dynamic Knowledge Base (`src/perspicacite/rag/dynamic_kb.py`)
**Purpose**: Session-scoped vector collections

**Key Classes**:
- `KnowledgeBaseConfig`: Configuration for chunking, retrieval
- `DynamicKnowledgeBase`: Session-scoped KB with auto-cleanup
- `DynamicKBFactory`: Factory for creating KB instances

**Features**:
- Automatic collection naming (session UUID)
- Document chunking with overlap
- Embedding generation
- Semantic search with score filtering
- Async context manager for cleanup

### 4. DeepResearch Mode (`src/perspicacite/rag/modes/deep_research.py`)
**Purpose**: End-to-end research workflow

**Key Classes**:
- `DeepResearchResult`: Complete result with metadata
- `DeepResearchMode`: Main workflow orchestrator

**Workflow**:
1. Search SciLEx for papers
2. Download PDFs for papers without full text
3. Assess relevance of each paper
4. Build dynamic KB from relevant papers
5. Search KB and generate answer
6. Query refinement if insufficient results

### 5. Demo Script (`demo_dynamic_kb.py`)
**Purpose**: Demonstrate full workflow with simulated services

**Usage**:
```bash
python demo_dynamic_kb.py
```

### 6. Integration Tests (`tests/integration/test_deep_research.py`)
**Purpose**: Test the complete workflow with mocks

**Tests**:
- `test_deep_research_full_workflow`: End-to-end test
- `test_deep_research_no_papers`: No results handling
- `test_deep_research_query_refinement`: Iterative search

## Key Design Decisions

### 1. Session-Scoped Collections
- Each research query gets its own vector collection
- Named with UUID prefix for isolation
- Auto-created and cleaned up with context manager

### 2. Separation of Concerns
- `PDFDownloader`: Only downloads/parse PDFs
- `PaperAssessor`: Only assesses relevance
- `DynamicKnowledgeBase`: Only manages vector storage
- `DeepResearchMode`: Orchestrates the workflow

### 3. Async Throughout
- All I/O operations are async
- Parallel PDF downloads
- Parallel paper assessments
- Efficient resource usage

### 4. Graceful Degradation
- If PDF download fails, continue with abstract
- If assessment fails, mark as not relevant
- If no papers found, return helpful message

## Integration Points

### Required Services:
```python
# To use DeepResearchMode:
deep_research = DeepResearchMode(
    scilex_searcher=scilex_client,      # SciLex API client
    pdf_downloader=PDFDownloader(),      # PDF download
    paper_assessor=PaperAssessor(llm),   # Relevance assessment
    query_refiner=QueryRefiner(llm),     # Query improvement
    kb_factory=DynamicKBFactory(         # KB creation
        vector_store=chroma_store,
        embedding_service=embedding_svc,
    ),
    llm_client=llm,                      # Answer generation
)
```

### Usage:
```python
result = await deep_research.execute(
    query="How are transformers used in medical imaging?"
)

print(result.answer)
print(f"Found {result.papers_relevant} relevant papers")
```

## Demo Output

```
🧠 Dynamic KB Building for Agentic RAG - Demo
======================================================================

📦 Step 0: Initialize Services
  ✅ All services initialized

🎯 Step 1: User Query
  Query: How are transformers used in medical imaging?

🔍 Step 2: SciLEx Search
  🔍 Searching SciLEx for: 'How are transformers used in medical imaging?'
  Found 2 papers
    - Vision Transformers for Medical Image Analysis (2024)
    - Self-Attention Mechanisms in Radiology (2023)

📥 Step 3: PDF Download
  📥 Downloading PDF from: https://example.com/paper1.pdf
  📥 Downloading PDF from: https://example.com/paper2.pdf
  Downloaded 2 PDFs

⚖️  Step 4: Relevance Assessment
  Vision Transformers for Medical Image Analysis... - ✅ Relevant (score: 0.85)
  Self-Attention Mechanisms in Radiology... - ✅ Relevant (score: 0.85)

  2/2 papers marked as relevant

🧠 Step 5: Dynamic KB → Answer
  📚 Creating collection: session_226f7f32
  Retrieved 1 relevant passages from KB

======================================================================
📤 Generated Answer:
======================================================================

Vision Transformers (ViTs) have shown remarkable success in medical imaging tasks.
Key findings from the literature:
...
```

## Next Steps

1. **Install Dependencies**: `pip install chromadb tenacity httpx pymupdf`
2. **Configure API Keys**: Add SciLEx, Unpaywall, and LLM credentials
3. **Integration**: Connect DeepResearchMode to RAG engine
4. **Testing**: Run integration tests with real services
5. **UI**: Add DeepResearch option to user interface
