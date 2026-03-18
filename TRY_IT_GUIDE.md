# Try It Guide - Dynamic KB & Agentic RAG

This guide shows you how to run the demos and integrate the new features.

## Prerequisites

```bash
# Python 3.11+
python --version

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Using uv (Recommended)

```bash
# Create virtual environment and install all dependencies
uv sync --dev

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# Verify installation
python -c "import perspicacite; print('✅ Installed successfully')"
```

### Alternative: Using pip

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

---

## Quick Start - Run the Demos

### 1. Dynamic KB Building Demo

This demo shows the 5-step workflow with simulated services:

```bash
cd /mnt/d/new_repos/perspicacite_v2
python demo_dynamic_kb.py
```

**Expected Output:**
```
🧠 Dynamic KB Building for Agentic RAG - Demo
==================================================
🎯 Step 1: User Query
   Query: How are transformers used in medical imaging?

🔍 Step 2: SciLEx Search
   Found 2 papers
     - Vision Transformers for Medical Image Analysis (2024)
     - Self-Attention Mechanisms in Radiology (2023)

📥 Step 3: PDF Download
   Downloaded 2 PDFs

⚖️  Step 4: Relevance Assessment
   Vision Transformers... - ✅ Relevant (score: 0.85)
   Self-Attention Mechanisms... - ✅ Relevant (score: 0.85)

🧠 Step 5: Dynamic KB → Answer
   📚 Creating collection: session_226f7f32
   Retrieved 1 relevant passages from KB

📤 Generated Answer:
[Comprehensive answer with citations]
```

### 2. Agentic RAG Demo

This demo shows true agentic capabilities:

```bash
python demo_agentic_rag.py
```

**Expected Output:**
```
🤖 Agentic RAG Demo - True Agent-Based Research
==================================================
🎯 Research Question:
   'How are transformers used in medical imaging?'

🔧 Agentic RAG Capabilities:
   ✓ Document quality assessment
   ✓ Early exit when question answered
   ✓ Dynamic plan adjustment
   ✓ Web search fallback
   ✓ Tool selection and execution
   ✓ Self-evaluation

🚀 Starting Agentic Research...
==================================================

🔄 Research Cycle 1/3
--------------------------------------------------
   📋 Plan created with 3 steps

   Step 1: Investigate transformer architectures...
   🔍 Searching KB...
   ✅ Success (confidence: 0.85)
   📊 Key findings: 3

   🎯 Early exit triggered! (confidence: 0.88)

📊 Research Statistics:
   Iterations: 1
   Total Steps: 1
   LLM Calls: 6

📤 Generated Answer:
[Structured answer with findings]
```

---

## Integration Guide

### Step 1: Install with uv

```bash
# Navigate to project
cd /mnt/d/new_repos/perspicacite_v2

# Create venv and install all dependencies (including dev)
uv sync --dev

# Activate virtual environment
source .venv/bin/activate

# Verify
python -c "import perspicacite; print('✅ Ready to use')"
```

### Optional: Additional Dependencies

```bash
# For SciLEx integration (if package is available)
uv pip install scilex

# For specific embeddings provider (choose one)
uv pip install voyageai
uv pip install openai
uv pip install anthropic
```

### Step 2: Configure Environment

Create `.env` file:

```env
# LLM Configuration
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional

# SciLEx Configuration
SCILEX_API_KEY=your_key_here
SCILEX_API_URL=https://api.scilex.ai

# Unpaywall (for PDF downloads)
UNPAYWALL_EMAIL=your_email@example.com

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db

# Optional: SerpAPI for web search
SERPAPI_KEY=your_key_here
```

### Step 3: Use DeepResearch Mode

```python
import asyncio
from perspicacite.pipeline.download import PDFDownloader
from perspicacite.rag.assessment import PaperAssessor, QueryRefiner
from perspicacite.rag.dynamic_kb import DynamicKBFactory, KnowledgeBaseConfig
from perspicacite.rag.modes.deep_research import DeepResearchMode

async def main():
    # Initialize services
    pdf_downloader = PDFDownloader()
    
    paper_assessor = PaperAssessor(llm_client=your_llm)
    query_refiner = QueryRefiner(llm_client=your_llm)
    
    kb_factory = DynamicKBFactory(
        vector_store=your_vector_store,
        embedding_service=your_embedding_service,
    )
    
    # Create DeepResearch mode
    deep_research = DeepResearchMode(
        scilex_searcher=your_scilex_client,
        pdf_downloader=pdf_downloader,
        paper_assessor=paper_assessor,
        query_refiner=query_refiner,
        kb_factory=kb_factory,
        llm_client=your_llm,
        max_search_iterations=3,
        min_relevant_papers=3,
    )
    
    # Execute research
    result = await deep_research.execute(
        query="How are transformers used in medical imaging?",
        search_params={"topn": 10, "source": "hybrid"}
    )
    
    print(f"Answer: {result.answer}")
    print(f"Papers found: {result.papers_found}")
    print(f"Papers relevant: {result.papers_relevant}")
    print(f"Iterations: {result.iterations}")

asyncio.run(main())
```

### Step 4: Use Agentic RAG Mode

```python
import asyncio
from perspicacite.rag.modes.agentic import AgenticRAGMode
from perspicacite.models.rag import RAGRequest, RAGMode

async def main():
    # Create Agentic RAG mode
    config = your_config  # Config object with rag_modes.agentic settings
    mode = AgenticRAGMode(config)
    
    # Create request
    request = RAGRequest(
        query="How are transformers used in medical imaging?",
        mode=RAGMode.AGENTIC,
        kb_name="your_kb",
        max_iterations=3,
    )
    
    # Execute with your services
    response = await mode.execute(
        request=request,
        llm=your_llm_client,
        vector_store=your_vector_store,
        embedding_provider=your_embedding_provider,
        tools=your_tool_registry,
    )
    
    print(f"Answer: {response.answer}")
    print(f"Iterations: {response.iterations}")
    print(f"Sources: {len(response.sources)}")
    for source in response.sources:
        print(f"  - {source.title} ({source.year})")

asyncio.run(main())
```

---

## Testing

### Run Tests with uv

```bash
# Make sure venv is activated
source .venv/bin/activate

# Run all tests
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run specific test file
uv run pytest tests/unit/test_pdf_download.py -v
uv run pytest tests/integration/test_deep_research.py -v

# Run with coverage
uv run pytest tests/ --cov=src/perspicacite --cov-report=html
```

### Manual Testing

```python
# Test PDF downloader
import asyncio
from perspicacite.pipeline.download import PDFDownloader

async def test_pdf():
    downloader = PDFDownloader()
    
    # Test Unpaywall lookup
    url = await downloader.get_open_access_url("10.1038/s41586-021-03819-2")
    print(f"OA URL: {url}")
    
    # Test download
    result = await downloader.download_and_parse(url)
    print(f"Content length: {len(result.content)}")

asyncio.run(test_pdf())
```

---

## Configuration Options

### DeepResearch Config

```python
# In your config file
rag_modes:
  deep_research:
    max_search_iterations: 3        # Max search-query-refine cycles
    min_relevant_papers: 3          # Stop when we have this many
    relevance_threshold: 0.6        # Min score to be "relevant"
    max_papers_to_download: 10      # Limit PDF downloads
    chunk_size: 1000                # For KB chunking
    chunk_overlap: 200
    top_k: 5                        # Retrieve top K from KB
```

### Agentic RAG Config

```python
# In your config file
rag_modes:
  agentic:
    max_iterations: 3               # Max research cycles
    early_exit_confidence: 0.85     # Threshold for early exit
    max_consecutive_failures: 2     # Trigger plan review
    quality_threshold: 0.7          # Min document quality score
```

---

## Troubleshooting

### Virtual Environment Issues

```bash
# Check if venv exists
ls -la .venv

# If not, create it with uv
uv sync --dev

# Activate venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Verify Python is from venv
which python  # Should show .venv/bin/python
```

### Import Errors

```bash
# Make sure venv is activated
source .venv/bin/activate

# Reinstall in editable mode
uv pip install -e .

# Verify
python -c "import perspicacite; print('✅ OK')"
```

### Missing Dependencies

```bash
# If you get "No module named 'chromadb'"
uv pip install chromadb

# If you get "No module named 'structlog'"
uv pip install structlog

# Or reinstall all dependencies
uv sync --dev
```

### Demo Import Issues

The demos use `importlib` to bypass the dependency chain. If you see import errors:

```bash
# Make sure you're running from project root with venv activated
cd /mnt/d/new_repos/perspicacite_v2
source .venv/bin/activate
python demo_dynamic_kb.py
```

---

## Next Steps

1. **Try the demos** - Run `demo_dynamic_kb.py` and `demo_agentic_rag.py`
2. **Set up real services** - Configure LLM, SciLEx, and vector store
3. **Run integration tests** - Verify with real (or mocked) services
4. **Integrate into your app** - Use DeepResearch or AgenticRAG modes
5. **Customize** - Adjust thresholds and prompts for your use case

---

## uv Quick Reference

```bash
# Create venv and install from pyproject.toml
uv sync

# Install with dev dependencies
uv sync --dev

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Run command in venv without activating
uv run python script.py

# Run pytest
uv run pytest

# Update lock file after editing pyproject.toml
uv lock

# Show installed packages
uv pip list

# Check for outdated packages
uv pip list --outdated
```

---

## Example: Full Workflow

```python
"""
Complete example showing DeepResearch + AgenticRAG
"""
import asyncio
from perspicacite.pipeline.download import PDFDownloader
from perspicacite.rag.assessment import PaperAssessor, QueryRefiner
from perspicacite.rag.dynamic_kb import DynamicKBFactory, KnowledgeBaseConfig
from perspicacite.rag.modes.deep_research import DeepResearchMode
from perspicacite.rag.modes.agentic import AgenticRAGMode
from perspicacite.models.rag import RAGRequest, RAGMode

async def research_with_dynamic_kb(query: str):
    """Use DeepResearch mode for comprehensive research."""
    
    # Initialize services (replace with your actual services)
    pdf_downloader = PDFDownloader()
    paper_assessor = PaperAssessor(llm_client=your_llm)
    query_refiner = QueryRefiner(llm_client=your_llm)
    
    kb_factory = DynamicKBFactory(
        vector_store=your_vector_store,
        embedding_service=your_embedding_service,
        default_config=KnowledgeBaseConfig(
            chunk_size=1000,
            chunk_overlap=200,
            top_k=5,
        )
    )
    
    # Create DeepResearch mode
    deep_research = DeepResearchMode(
        scilex_searcher=your_scilex_client,
        pdf_downloader=pdf_downloader,
        paper_assessor=paper_assessor,
        query_refiner=query_refiner,
        kb_factory=kb_factory,
        llm_client=your_llm,
        max_search_iterations=3,
        min_relevant_papers=3,
    )
    
    # Execute
    result = await deep_research.execute(query)
    
    return {
        "answer": result.answer,
        "papers_found": result.papers_found,
        "papers_relevant": result.papers_relevant,
        "search_history": result.search_history,
    }

async def research_with_agentic_rag(query: str):
    """Use AgenticRAG mode for agent-based research."""
    
    # Create mode
    mode = AgenticRAGMode(your_config)
    
    # Create request
    request = RAGRequest(
        query=query,
        mode=RAGMode.AGENTIC,
        kb_name="your_kb",
        max_iterations=3,
    )
    
    # Execute
    response = await mode.execute(
        request=request,
        llm=your_llm,
        vector_store=your_vector_store,
        embedding_provider=your_embedding_provider,
        tools=your_tool_registry,
    )
    
    return {
        "answer": response.answer,
        "iterations": response.iterations,
        "sources": response.sources,
    }

async def main():
    query = "How are transformers used in medical imaging?"
    
    print("=" * 60)
    print("Option 1: DeepResearch (Dynamic KB Building)")
    print("=" * 60)
    result1 = await research_with_dynamic_kb(query)
    print(f"Papers found: {result1['papers_found']}")
    print(f"Papers relevant: {result1['papers_relevant']}")
    print(f"Answer: {result1['answer'][:500]}...")
    
    print("\n" + "=" * 60)
    print("Option 2: AgenticRAG (Agent-Based Research)")
    print("=" * 60)
    result2 = await research_with_agentic_rag(query)
    print(f"Iterations: {result2['iterations']}")
    print(f"Sources: {len(result2['sources'])}")
    print(f"Answer: {result2['answer'][:500]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Support

If you encounter issues:

1. Check the demo scripts - they show working examples
2. Review `AGENTIC_RAG_ASSESSMENT.md` for feature details
3. Check `DYNAMIC_KB_IMPLEMENTATION.md` for architecture
4. Run with `DEBUG=1` for verbose logging
