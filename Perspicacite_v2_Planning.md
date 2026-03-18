# Perspicacité v2 - Planning Document

**Date**: 2026-03-12  
**Status**: Initial Planning Phase  
**Purpose**: Living document to track v2 architecture decisions, features, and open questions.

---

## 1. Package Overview

### 1.1 Perspicacité-AI (Current v1)

**Purpose**: AI-powered research assistant for *using* existing literature

| Component | Purpose | Lines of Code (approx) |
|-----------|---------|----------------------|
| `bibtex2kb` | BibTeX → FAISS Knowledge Base | ~80K |
| `core/` | RAG engine (Basic/Advanced/Profound) | ~40K |
| `website/` | React UI | ~30 files |
| `fastAPI/` | REST API backend | ~500 |
| `mcp_server/` | MCP server for Claude Desktop | ~320 |

**Key Features**:
- BibTeX processing with PDF/HTML/GitHub/YouTube parsers
- Document curation with LLM
- FAISS vector database creation
- Three RAG modes: Basic, Advanced, Profound
- Multi-provider LLM support (OpenAI, Anthropic, DeepSeek, Gemini, etc.)
- Web search integration (Google Scholar, PubMed, OpenAlex, Semantic Scholar)

**Current Limitations**:
- FAISS-only vector store support
- Monolithic core modules (`core.py` ~1,400 lines, `profonde.py` ~2,400 lines)
- Stateless design (no session persistence)
- No async/await in core retrieval logic
- Docker + conda deployment (heavyweight)

---

### 1.2 SciLEx

**Purpose**: Systematic literature review toolkit for *collecting* papers

| Component | Purpose |
|-----------|---------|
| `crawlers/` | 10 academic API collectors |
| `aggregate_collect.py` | Deduplication, 5-phase quality filtering |
| `citations/` | Citation network analysis (CrossRef, OpenCitations) |
| `enrich_with_hf.py` | HuggingFace metadata enrichment |
| `export_to_bibtex.py` / `push_to_zotero.py` | Export capabilities |

---

### 1.3 Toolomics (Existing MCP Infrastructure)

**Purpose**: Suite of MCP tools for scientific research (Holobiomics Lab)

**Location**: `/home/tjiang/repos/Mimosa_project/toolomics`

| MCP Server | Purpose | Relevance to Perspicacité v2 |
|------------|---------|------------------------------|
| `graph_rag/` | Knowledge graph-based RAG (Microsoft GraphRAG) | **High** - Graph RAG for v2 Agentic mode |
| `sibils/` | Scientific literature search (PubMed, PMC) + text mining | **High** - Literature search with entity extraction |
| `browser/` | Web browser automation (SearXNG) | **High** - Web search capabilities |
| `pdf/` | PDF processing and extraction | **Medium** - Alternative PDF parser |
| `Rscript/` | R script execution | **Medium** - Statistical analysis |
| `cheminformatics/` | Chemistry tools (RDKit) | **Low-Medium** - Domain-specific |
| `decimer/` | Chemical structure recognition | **Low** - Specialized chemistry |
| `python_editor/` | Python code execution | **High** - Code execution for Agentic mode |
| `shell/` | Shell command execution | **Medium** - System operations |
| `html/` | HTML processing | **Low** - Alternative HTML parser |
| `txt_editor/` | Text file editing | **Low** - File operations |

**Architecture**:
- Multi-instance deployment (isolated workspaces)
- Docker support for complex dependencies
- Centralized workspace directory
- Automatic port assignment
- FastMCP-based implementation

**Key Insight**: Toolomics provides the **tool infrastructure** needed for true "Agentic" RAG mode. Instead of building from scratch, Perspicacité v2 could:
1. Use Toolomics as the tool layer
2. Integrate with SciLEx for literature collection
3. Focus on the orchestration layer (Agentic RAG controller)

---

**Supported APIs**:
| API | Key Required | Best For |
|-----|-------------|----------|
| SemanticScholar | Optional | CS/AI papers, citation networks |
| OpenAlex | Optional | Broad coverage, ORCID data |
| IEEE | Yes | Engineering, CS conferences |
| Arxiv | No | Preprints, physics, CS |
| Springer | Yes | Journals, books |
| Elsevier | Yes | Medical, life sciences |
| PubMed | Optional | 35M biomedical papers |
| HAL | No | French research, theses |
| DBLP | No | CS bibliography, 95%+ DOI |
| Istex | No | French institutional access |

**Key Capabilities**:
- Multi-API parallel collection
- Smart deduplication (DOI, URL, fuzzy title matching)
- 5-phase quality filtering with time-aware citation thresholds
- Citation network extraction
- HuggingFace enrichment (ML models, datasets, GitHub stats)
- Export to Zotero or BibTeX with PDF links

**Tech Stack**: Python ≥3.10, `uv` package manager, modern Python practices

---

## 2. Integration Strategy: Perspicacité + SciLEx

### 2.1 Why Integrate?

SciLEx was identified as the collection engine for Perspicacité v2 because:
- Robust 10-API collection infrastructure (no need to rebuild)
- Battle-tested deduplication and quality filtering
- Citation network analysis capabilities
- HuggingFace enrichment for AI/ML papers
- Maintained by a colleague (collaborative development)

### 2.2 Integration Options

#### Option A: Library Import (Recommended)

```python
# Perspicacité v2 - new web search module
from scilex.crawlers.collectors import (
    SemanticScholarCollector,
    OpenAlexCollector,
    PubMedCollector,
)
from scilex.aggregate_collect import aggregate_results, deduplicate_papers
from scilex.quality_validation import validate_paper_quality

class SciLExSearchProvider:
    """Adapter to use SciLEx as Perspicacité's web search backend"""
    
    def search(self, query: str, config: SearchConfig) -> List[Document]:
        # 1. Run SciLEx collection for single query
        # 2. Deduplicate & quality filter
        # 3. Download PDFs
        # 4. Convert to Perspicacité Document format
        pass
```

**Pros**:
- Shared memory, fast execution
- Clean Python API
- Full access to SciLEx internals

**Cons**:
- SciLEx becomes a hard dependency
- Version coupling

---

#### Option B: CLI Wrapper

```python
import subprocess

def scilex_search(query: str, output_dir: str) -> List[Document]:
    # Run scilex-collect with temporary config
    subprocess.run([
        "scilex-collect", 
        "--query", query,
        "--output", output_dir
    ])
    # Parse results and convert to Documents
```

**Pros**:
- Loose coupling
- SciLEx can evolve independently
- Language-agnostic

**Cons**:
- Slower (process spawn overhead)
- Complex error handling
- File I/O overhead

---

#### Option C: Shared Interface (Most Flexible)

```python
from abc import ABC, abstractmethod

class LiteratureSearcher(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> SearchResults:
        pass
    
    @abstractmethod
    def get_pdf(self, paper: PaperMetadata) -> Optional[bytes]:
        pass

# SciLEx implementation
class SciLExSearcher(LiteratureSearcher):
    pass

# Perspicacité legacy implementation  
class PerspicaciteSearcher(LiteratureSearcher):
    pass
```

**Pros**:
- Swappable implementations
- Easy testing with mocks
- Clean separation of concerns

**Cons**:
- More upfront design work
- Interface maintenance overhead

---

### 2.3 Key Integration Points

| Feature | SciLEx Provides | Perspicacité Uses |
|---------|----------------|-------------------|
| **API Coverage** | 10 academic APIs | Broader literature search |
| **Deduplication** | DOI + fuzzy title matching | Clean result sets |
| **Quality Filter** | 5-phase validation | Higher quality documents |
| **Citation Networks** | CrossRef + OpenCitations | Identify seminal papers |
| **HuggingFace** | Model/dataset metadata | Enrich AI/ML papers |
| **Export** | BibTeX with PDF URLs | Input to KB pipeline |

---

### 2.4 Open Questions - Integration

1. **Granularity**: 
   - Should Perspicacité call SciLEx per-query (on-demand)?
   - Or batch collect and cache results?

2. **Real-time vs Batch**:
   - Current Perspicacité: Real-time web search during Profound mode
   - SciLEx: Designed for batch collection
   - Do we need a hybrid approach?

3. **Metadata Flow**:
   - SciLEx collects rich metadata (citation counts, quality scores)
   - Should Perspicacité use this for ranking?
   - Store in FAISS metadata or sidecar JSON?

4. **Configuration**:
   - Should users configure SciLEx separately or unified config?
   - Share API keys or separate management?

5. **Output Handling**:
   - SciLEx exports BibTeX → Perspicacité processes it
   - Or direct in-memory transfer (skip file I/O)?

---

## 3. Architecture for v2

### 3.1 Current State (v1)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────▶│   FastAPI       │────▶│  Core Modules   │
│   (Vite/TS)     │◄────│   (app.py)      │◄────│  (core/profonde)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                          ┌─────────────────┐
│  MCP Server     │                          │  FAISS /        │
│  (Claude Desktop)│                         │  Web Searchers  │
└─────────────────┘                          └─────────────────┘
```

### 3.2 Proposed v2 Architecture: Standalone + Optional MCP Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Perspicacité v2 Architecture                         │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         User Interfaces                                │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────┐  │  │
│  │  │   React Web UI  │  │   MCP Server    │  │   CLI Interface       │  │  │
│  │  │  (Primary Mode) │  │ (For Mimosa-AI) │  │  (Optional)           │  │  │
│  │  └─────────────────┘  └─────────────────┘  └───────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         API Layer (FastAPI)                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  │
│  │  │  REST API   │  │   MCP       │  │  WebSocket  │                   │  │
│  │  │  (Standard) │  │  (Optional) │  │  (Streaming)│                   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Core RAG Engine                                  │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  RAG Modes: Quick → Standard → Advanced → Deep → Citation      │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                         │
│        ┌───────────────────────────┼───────────────────────────┐             │
│        ▼                           ▼                           ▼             │
│  ┌───────────────┐   ┌─────────────────────┐   ┌─────────────────────────┐  │
│  │  Knowledge    │   │  Literature Search  │   │  Optional: Toolomics    │  │
│  │  Base Layer   │   │  (SciLEx-Powered)   │   │  (MCP Client)           │  │
│  │               │   │                     │   │                         │  │
│  │ • FAISS       │   │ ┌─────────────────┐ │   │ • GraphRAG              │  │
│  │ • Chroma      │   │ │ SciLEx Adapter  │ │   │ • SIBiLS                │  │
│  │ • Pinecone    │   │ │  10 API Collect │ │   │ • Browser               │  │
│  │ • Weaviate    │   │ │  Deduplication  │ │   │ • Python Editor         │  │
│  │ • Graph (KG)  │   │ │  Quality Filter │ │   │ • etc.                  │  │
│  └───────────────┘   │ └─────────────────┘ │   └─────────────────────────┘  │
│                      └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```
        │                                 │ │  │ Quality Filter│  │ │
        │                                 │ │  ├───────────────┤  │ │
        │                                 │ │  │ Citation Net  │  │ │
        │                                 │ │  └───────────────┘  │ │
        │                                 │ └─────────────────────┘ │
        │                                 └─────────────────────────┘
        │                                              │
        │    ┌─────────────────────────────────────────┘
        │    ▼
        │  ┌─────────────────────────────────────────────────────────┐
        │  │              Toolomics MCP Layer                         │
        │  │  (External Tool Execution - /home/tjiang/repos/...)      │
        │  │                                                          │
        │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐    │
        │  │  │  GraphRAG   │ │   SIBiLS    │ │     Browser     │    │
        │  │  │  (KG-RAG)   │ │(PubMed/Text │ │ (Web Search)    │    │
        │  │  └─────────────┘ │  Mining)    │ └─────────────────┘    │
        │  │  ┌─────────────┤ └─────────────┘ ┌─────────────────┐    │
        │  │  │   Python    │ ┌─────────────┐ │   PDF Parser    │    │
        │  │  │   Editor    │ │  Cheminf.   │ │                 │    │
        │  │  │ (Code Exec) │ │  (RDKit)    │ └─────────────────┘    │
        │  │  └─────────────┘ └─────────────┘                        │
        │  └─────────────────────────────────────────────────────────┘
        │                              │
        └─────────────────────┬────────┘
                              ▼
                  ┌─────────────────────────┐
                  │    Document Pipeline    │
                  │  (Parse → Curate → Chunk│
                  │   → Embed → Store)      │
                  └─────────────────────────┘
```

#### Key Integration Points

1. **SciLEx Adapter**: Abstracts SciLEx functionality into the Literature Search layer
2. **MCP Server (Optional)**: Exposes Perspicacité capabilities to Mimosa-AI
3. **Toolomics Client (Optional)**: MCP client for enhanced tool capabilities
4. **Temp Cache**: Short-term storage for on-demand web search results
5. **KB Update**: Batch collection results can be merged into persistent KB
6. **Citation Networks**: SciLEx's citation analysis feeds into RAG ranking

#### Three Usage Patterns

```
Pattern A: Standalone Web UI (Primary)
──────────────────────────────────────
User → React UI → FastAPI → Core RAG → Answer
                 │
                 ├── QuickRAG: Simple vector search
                 ├── StandardRAG: Hybrid search
                 ├── AdvancedRAG: Query expansion
                 ├── DeepRAG: Multi-cycle research
                 └── CitationRAG: Network analysis

Pattern B: Batch Collection → KB Building
──────────────────────────────────────────
Research Topic → SciLEx Collect → Aggregate → 
→ Export BibTeX → Process with bibtex2kb → 
→ Build FAISS KB → Query via Pattern A

Pattern C: Mimosa-AI Integration (Optional)
───────────────────────────────────────────
User Goal → Mimosa-AI Planner → Discovers Perspicacité MCP
            │
            ├── Calls perspicacite_research(query)
            ├── Calls perspicacite_kb_search(query)
            └── Combines with other Toolomics tools
                (cheminformatics, docking, etc.)
                
Pattern D: Enhanced with Toolomics (Optional)
──────────────────────────────────────────────
User Query → DeepRAG + Toolomics
├── Use SciLEx for literature collection
├── Use Toolomics GraphRAG for KG queries
├── Use Toolomics SIBiLS for PubMed mining
└── Synthesize → Verify → Deliver Answer
```

### 3.3 Key Architectural Improvements

| Area | Current (v1) | Proposed (v2) |
|------|-------------|---------------|
| **Vector Store** | FAISS only | Multi-backend (FAISS, Chroma, Pinecone, Weaviate, Graph) |
| **Search** | Vector only | Hybrid (Vector + BM25 + Metadata filters) |
| **Chemical Search** | None | Fingerprint + Canonical SMILES (RDKit) |
| **Enterprise Scale** | Single-user | Multi-tenant, millions of docs support |
| **Query Efficiency** | O(n) brute force | Hierarchical + Caching + Pre-filtering |
| **Tool Layer** | Web search only | Toolomics MCP integration (optional) |
| **Async** | Blocking I/O | Full async/await support |
| **Session** | Stateless | Persistent sessions with Redis/PostgreSQL |
| **Modularity** | Monolithic | Plugin-based RAG modes and searchers |
| **Collection** | Built-in crawlers | SciLEx integration (10 APIs) |
| **Deployment** | Docker + conda | Docker + `uv` (faster, lighter) |

---

### 3.4 Perspicacité v2 Positioning: Standalone + Integration

**Primary Role**: Standalone literature chatbot for researchers
**Secondary Role**: Knowledge provider to Mimosa-AI (via MCP/A2A)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Perspicacité v2                                 │   │
│  │                    (Standalone Literature Chatbot)                   │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │   │
│  │   │  QuickRAG   │  │ StandardRAG │  │  DeepRAG    │  │Citation  │  │   │
│  │   │             │  │             │  │ (Profound)  │  │  RAG     │  │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘  │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                    React UI + FastAPI                        │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                     ┌──────────────┴──────────────┐                         │
│                     │                             │                         │
│                     ▼                             ▼                         │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │   SciLEx Integration            │  │  Optional: Toolomics (MCP)      │   │
│  │   (10 API collectors)           │  │  (Enhanced capabilities)        │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ MCP Server Mode (optional)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Mimosa-AI                                     │   │
│  │              (Omni-Assistant for Science)                            │   │
│  │                                                                      │   │
│  │   Discovers Perspicacité as MCP tool:                               │   │
│  │   - "perspicacite_research(query)" → DeepRAG result                  │   │
│  │   - "perspicacite_kb_search(query)" → QuickRAG result                │   │
│  │   - "perspicacite_citation_network(paper)" → Citation analysis       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**

1. **Standalone First**: Perspicacité works independently as a literature research tool
2. **Seamless Integration**: MCP server starts by default alongside React UI
3. **Clean Separation**: Core RAG logic is separate from MCP interface
4. **Bidirectional Value**: Users can use Perspicacité directly OR through Mimosa

---

### 3.5 MCP Server Mode for Mimosa Integration

**Default Behavior (Option C)**: Both UI and MCP server start simultaneously

```bash
# Default: Starts React UI + MCP server
python -m perspicacite

# UI only (no MCP)
python -m perspicacite --no-mcp-server

# MCP only (headless, for Mimosa-only deployments)
python -m perspicacite --no-ui
```

When running as an MCP server, Perspicacité exposes these tools:

```python
# Perspicacité MCP Server (optional mode)
@mcp.tool
def research_literature(
    query: str,
    mode: str = "deep",  # quick, standard, deep, citation
    max_cycles: int = 3,
    use_web_search: bool = True
) -> str:
    """
    Research a scientific question using Perspicacité's RAG system.
    
    Use this tool when you need:
    - Comprehensive literature review
    - Evidence-based answers from research papers
    - Citation-backed claims
    
    Returns: Research answer with citations
    """
    pass

@mcp.tool
def search_knowledge_base(
    query: str,
    knowledge_base: str = "default",
    top_k: int = 5
) -> List[Document]:
    """
    Quick search in a specific knowledge base.
    
    Use this tool when you need:
    - Fast lookup in a known KB
    - Specific paper retrieval
    """
    pass

@mcp.tool
def analyze_citation_network(
    paper_doi: str,
    depth: int = 2
) -> Dict:
    """
    Analyze citation network for a paper.
    
    Use this tool when you need:
    - Find seminal papers
    - Understand research lineage
    - Discover related work
    """
    pass

@mcp.tool
def get_kb_list() -> List[str]:
    """List available knowledge bases."""
    pass
```

**Integration Benefits:**

| For Perspicacité Users | For Mimosa Users |
|------------------------|------------------|
| Standalone web UI | Access to Perspicacité via natural language |
| Direct control over RAG modes | Automatic tool discovery |
| Full feature set | Integrated into broader workflows |
| No dependencies | Combined with other Toolomics tools |

---

### 3.6 Relationship to Toolomics

**Toolomics** = General scientific tools (chemistry, biology, etc.)
**Perspicacité** = Specialized literature/knowledge tools

```
Mimosa-AI Orchestrator
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────────────┐        ┌─────────────────────────┐
│      Toolomics          │        │    Perspicacité v2      │
│   (General Science)     │        │  (Literature/Knowledge) │
│                         │        │                         │
│ • Cheminformatics       │        │ • Literature search     │
│ • Docking               │        │ • Citation networks     │
│ • Mass spec analysis    │        │ • Knowledge bases       │
│ • GraphRAG (general)    │        │ • SciLEx collection     │
│ • Python/R execution    │        │ • PDF/HTML parsers      │
└─────────────────────────┘        └─────────────────────────┘
```

**No Overlap**: Perspicacité does NOT duplicate Toolomics tools - it complements them.
- Toolomics executes experiments and analyzes data
- Perspicacité provides the knowledge foundation (literature, prior work)

---

### 3.7 Skills vs Memory: A Unified Knowledge Model

**Analogy (Validated Understanding)**:

| Concept | Analogy | Characteristics |
|---------|---------|-----------------|
| **Skills** | Knowledge from a course | General, reusable, structured, curated |
| **Memory** | Lab activity notes / Project notes | Specific, contextual, accumulated daily |
| **Learned Skills** | Lab protocols that became standard procedures | Memory that proved valuable → converted to skills |

**Key Insight**: Skills and memory exist on a **capability continuum**, not as binary categories.

```
Static Skills ◄────────────────────────────────────► Dynamic Memory
     │                                                  │
     │         Learned Skills (Procedural Memory)        │
     │                    ▲                              │
     │    Memory ────────┘                               │
     │    proves valuable                                │
     │    over time                                      │
     │                                                   │
     └──► Copied from Toolomics                          └──► Session history
          (citation-management)                               (user preferences)
```

**For Perspicacité v2**:

| Type | Example | Storage | Source |
|------|---------|---------|--------|
| **Static Skills** | "How to conduct a systematic literature review" | SKILL.md files | Copied from Toolomics |
| **Episodic Memory** | "User prefers Nature citation format" | Vector DB | Runtime accumulation |
| **Procedural Memory** | "For metabolomics queries, search PubMed first" | JSON/YAML rules | Learned from experience |

**Memory → Skill Conversion Pipeline** (v2.1+):

```python
class KnowledgeEvolution:
    """
    Converts valuable memory into reusable skills
    """
    
    def monitor_memory(self, memory: EpisodicMemory):
        """Track memory usage and success rates"""
        # Example: User always asks for PubMed first for bio queries
        # Success rate: 95% vs 60% for general search
        
    def propose_skill(self, memory_pattern: Pattern) -> Skill:
        """Convert proven memory pattern to skill"""
        if memory_pattern.success_rate > 0.9 and memory_pattern.frequency > 10:
            return Skill(
                name=f"{memory_pattern.domain}_optimization",
                description=memory_pattern.description,
                rules=memory_pattern.extract_rules(),
                source="learned_from_memory"
            )
    
    def review_with_human(self, proposed_skill: Skill) -> bool:
        """Human-in-the-loop validation"""
        # Show proposed skill to user/admin for approval
        # Prevents overfitting to specific user habits
```

**Example Evolution**:

```
Week 1-4 (Memory accumulation):
- Session 1: User asks about "metabolomics LC-MS" → Use general search
- Session 5: User asks about "metabolomics workflow" → Use general search  
- Session 12: User asks about "metabolite identification" → Use general search
- Pattern detected: User works in metabolomics

Week 5 (Procedural memory formation):
- Learned: "User research area = metabolomics"
- Applied: Prioritize PubMed, suggest metabolomics KBs

Week 10+ (Skill conversion candidate):
- Pattern: 95% success with PubMed-first for metabolomics
- Proposal: Create "metabolomics_research" skill
- Validation: Admin approves → New SKILL.md created
```

**v2 Scope**: Static skills only (from Toolomics)
**v2.1 Scope**: Add memory layer
**v2.2 Scope**: Memory → Skill conversion (self-evolving)

---

### 3.8 Enterprise-Scale Architecture

**Scenario**: Big international company with millions of internal documents

#### Query Efficiency Strategies

**1. Hierarchical Search (Avoid O(n) scanning)**
```python
class HierarchicalRetriever:
    """
    Two-stage retrieval for enterprise scale
    """
    
    async def search(self, query: str, filters: dict):
        # Stage 1: Metadata filtering (fast, no embedding cost)
        # e.g., department, date range, doc type
        candidate_pool = self.metadata_index.filter(
            department=filters.get("department"),
            date_range=filters.get("date_range"),
            doc_type=filters.get("doc_type")
        )
        # Reduces 10M docs → 100K docs
        
        # Stage 2: Vector search on reduced pool
        results = self.vector_db.search(
            query=query,
            filter_ids=candidate_pool,  # Only search within filtered set
            k=100
        )
        
        # Stage 3: BM25 re-ranking
        return self.bm25_rerank(query, results)
```

**2. Query Result Caching**
```python
class QueryCache:
    """
    Cache frequent queries (enterprise users often search similar things)
    """
    
    def get(self, query: str, user_context: dict):
        # Hash query + user department/role
        cache_key = self.hash(query + user_context["department"])
        
        # Check Redis cache
        if cached := self.redis.get(cache_key):
            return json.loads(cached)
        
        # Not cached → perform search
        results = self.retriever.search(query)
        
        # Cache for 1 hour (configurable)
        self.redis.setex(cache_key, 3600, json.dumps(results))
        return results
```

**3. Pre-computed Clusters**
```python
# Offline: Cluster documents by topic
clusters = cluster_documents(kmeans_on_embeddings, n_clusters=1000)
# e.g., cluster_42 = "metabolomics LC-MS pharmaceuticals"

# Online: Route query to relevant clusters
async def cluster_routed_search(query: str):
    query_embedding = embed(query)
    
    # Find top-5 relevant clusters
    relevant_clusters = find_nearest_clusters(query_embedding, k=5)
    # Only search within 5% of total documents
    
    results = []
    for cluster in relevant_clusters:
        results.extend(
            vector_search_in_cluster(query, cluster.id, k=20)
        )
    
    return merge_and_deduplicate(results)
```

---

### 3.9 Chemical Structure Search

**Use Case**: Finding papers containing specific molecules, substructure search

**Current Gap**: Vector search on text can't match "glucose" to "C6H12O6" or similar structures

**Solution: Multi-Modal Embeddings**

```python
class ChemicalRetriever:
    """
    Retrieve by chemical structure using molecular fingerprints
    """
    
    def __init__(self):
        self.rdkit_available = self._check_rdkit()
        self.text_db = Chroma(collection="papers_text")
        self.chem_db = Chroma(collection="papers_chemicals")
    
    def index_paper(self, paper: Document):
        """Extract and index chemical structures"""
        # Extract chemicals from paper
        chemicals = self.extract_chemicals(paper.text)
        
        for chem in chemicals:
            if self.rdkit_available:
                # Generate Morgan fingerprint
                mol = Chem.MolFromSmiles(chem.canonical_smiles)
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048
                )
                
                # Store in chemical index
                self.chem_db.add(
                    ids=[f"{paper.id}_{chem.name}"],
                    embeddings=[fingerprint.ToList()],
                    metadatas=[{
                        "paper_id": paper.id,
                        "chemical_name": chem.name,
                        "smiles": chem.canonical_smiles,
                        "inchi": chem.inchi_key
                    }]
                )
    
    async def search_by_structure(self, query_smiles: str, similarity: float = 0.8):
        """
        Search papers by chemical structure
        
        Args:
            query_smiles: SMILES string of query molecule
            similarity: Tanimoto similarity threshold (0-1)
        """
        query_mol = Chem.MolFromSmiles(query_smiles)
        query_fp = AllChem.GetMorganFingerprintAsBitVect(
            query_mol, radius=2, nBits=2048
        )
        
        # Find similar chemicals by fingerprint
        similar_chemicals = self.chem_db.similarity_search_by_vector(
            embedding=query_fp.ToList(),
            k=50,
            score_threshold=similarity  # Tanimoto similarity
        )
        
        # Get unique papers
        paper_ids = set(chem.metadata["paper_id"] for chem in similar_chemicals)
        
        return self.get_papers_by_ids(paper_ids)
    
    async def substructure_search(self, substructure_smarts: str):
        """
        Find papers containing substructure
        
        Example: "c1ccccc1" (benzene ring) finds all aromatic compounds
        """
        substructure = Chem.MolFromSmarts(substructure_smarts)
        
        # This requires scanning (expensive), so:
        # 1. Pre-filter by fingerprint similarity
        candidates = self.search_by_structure(substructure_smarts, similarity=0.6)
        
        # 2. Verify with RDKit substructure match
        results = []
        for paper in candidates:
            for chem in paper.chemicals:
                mol = Chem.MolFromSmiles(chem.smiles)
                if mol.HasSubstructMatch(substructure):
                    results.append(paper)
                    break
        
        return results
```

**Query Examples:**
```python
# 1. Find papers about similar molecules
results = await retriever.search_by_structure(
    query_smiles="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    similarity=0.85
)

# 2. Find papers with specific functional group
results = await retriever.substructure_search(
    substructure_smarts="[NX3](=O)(=O)"  # Nitro group
)

# 3. Hybrid: Text + Chemical
results = await retriever.hybrid_search(
    text_query="COVID-19 drug repurposing",
    chemical_query="C1=CC=NC=C1"  # Pyridine ring
)
```

---

### 3.10 Essential Tool Set (v2 Core)

**Design Philosophy**: Minimal essential tools that work reliably. Toolomics integration comes later (v2.1+).

**Key Feature**: Dynamic KB building - agent can add papers on-the-fly (Chroma advantage over static FAISS)

#### Core Tool Registry

```python
class PerspicaciteTools:
    """
    Essential tools for literature research.
    Each tool is a function the agent can call.
    """
    
    # ============================================================
    # CATEGORY 1: WEB SEARCH (Literature Discovery)
    # ============================================================
    
    async def web_search(
        query: str,
        sources: List[str] = ["google_scholar", "pubmed", "semantic_scholar"],
        max_results: int = 10,
        date_range: Optional[Tuple[str, str]] = None
    ) -> SearchResults:
        """
        Search academic web sources for papers.
        
        Use when:
        - KB doesn't have relevant papers
        - Need recent publications (last 6 months)
        - Expanding search beyond local KB
        
        Returns: List of papers with title, authors, abstract, PDF URL
        """
        pass
    
    async def fetch_pdf(self, url: str, download: bool = True) -> PDFDocument:
        """
        Download and parse PDF from URL.
        
        Use when:
        - Web search found relevant paper
        - Need full text for analysis
        
        Returns: Parsed text, metadata, extracted sections
        """
        pass
    
    # ============================================================
    # CATEGORY 2: SCILEX INTEGRATION (Collection)
    # ============================================================
    
    async def scilex_collect(
        query: str,
        apis: List[str] = ["semantic_scholar", "openalex", "pubmed"],
        max_papers: int = 100,
        quality_threshold: float = 0.7
    ) -> CollectionJob:
        """
        Batch collect papers using SciLEx.
        
        Use when:
        - Building a new knowledge base
        - Comprehensive literature survey needed
        - Multiple API coverage required
        
        Returns: Collection job ID, status tracking
        """
        pass
    
    async def scilex_status(self, job_id: str) -> CollectionStatus:
        """Check progress of SciLEx collection job."""
        pass
    
    async def scilex_to_kb(
        self,
        job_id: str,
        kb_name: str,
        deduplicate: bool = True
    ) -> KnowledgeBase:
        """
        Convert SciLEx results to KB.
        
        Pipeline: SciLEx output → BibTeX → Process → FAISS
        """
        pass
    
    # ============================================================
    # CATEGORY 3: KNOWLEDGE BASE (Local RAG)
    # ============================================================
    
    async def kb_search(
        query: str,
        kb_name: str = "default",
        mode: str = "hybrid",  # vector, bm25, hybrid
        filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[RetrievedDocument]:
        """
        Search local knowledge base.
        
        Use when:
        - User has a specific KB selected
        - Need fast, offline-capable search
        - Working with curated paper collection
        
        Returns: Chunks with relevance scores, metadata
        """
        pass
    
    async def kb_list(self) -> List[KnowledgeBaseInfo]:
        """List available knowledge bases."""
        pass
    
    async def kb_create(
        self,
        name: str,
        source: Union[str, List[str]],  # BibTeX file or paper list
        embedding_model: str = "openai"
    ) -> KnowledgeBase:
        """Create new KB from papers."""
        pass
    
    async def kb_add_papers(
        self,
        kb_name: str,
        papers: List[Union[str, Paper]],  # URLs, DOIs, or Paper objects
        auto_chunk: bool = True
    ) -> AddResult:
        """
        Dynamically add papers to existing KB.
        
        CRITICAL: This enables agent to build KB during research!
        
        Use when:
        - Web search found relevant papers not in KB
        - User provides PDF URLs
        - Agent discovers papers via citations
        - Incrementally building research collection
        
        Flow:
        1. Download/fetch papers
        2. Parse PDFs
        3. Chunk documents
        4. Generate embeddings
        5. Add to Chroma collection
        
        Returns: Success count, failed papers, duplicates detected
        """
        pass
    
    # ============================================================
    # CATEGORY 4: PDF PROCESSING
    # ============================================================
    
    async def pdf_extract(
        pdf_path: str,
        extract_tables: bool = True,
        extract_images: bool = False
    ) -> ExtractedContent:
        """
        Extract text and structure from PDF.
        
        Use when:
        - PDF not in KB yet
        - Need specific sections (methods, results)
        - Extracting tables/figures
        
        Returns: Full text, sections, tables, metadata
        """
        pass
    
    async def pdf_chunk(
        self,
        pdf_content: ExtractedContent,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[DocumentChunk]:
        """Smart chunking preserving sections."""
        pass
    
    # ============================================================
    # CATEGORY 5: CITATION & METADATA
    # ============================================================
    
    async def get_citation(
        paper_id: str,
        style: str = "bibtex"  # apa, mla, nature, etc.
    ) -> str:
        """Get formatted citation for paper."""
        pass
    
    async def get_citation_network(
        paper_id: str,
        direction: str = "both",  # forward, backward, both
        depth: int = 1
    ) -> CitationNetwork:
        """
        Get papers citing or cited by given paper.
        
        Use for:
        - Finding seminal works
        - Discovering related research
        """
        pass
    
    async def resolve_doi(self, doi: str) -> PaperMetadata:
        """Get paper metadata from DOI."""
        pass
    
    # ============================================================
    # CATEGORY 6: MEMORY (User-Specific)
    # ============================================================
    
    async def recall_research(
        query: str,
        user_id: str,
        recency_days: Optional[int] = None
    ) -> List[ResearchMemory]:
        """
        Recall user's past research on similar topics.
        
        Use when:
        - User asks follow-up question
        - Building on prior research
        - Avoiding duplicate searches
        """
        pass
    
    async def store_research(
        user_id: str,
        query: str,
        findings: str,
        papers_used: List[str],
        kb_accessed: List[str]
    ) -> None:
        """Store research session for future recall."""
        pass
    
    # ============================================================
    # CATEGORY 7: ANALYSIS (v2.1+)
    # ============================================================
    
    async def summarize_papers(
        paper_ids: List[str],
        focus: Optional[str] = None
    ) -> Summary:
        """Multi-paper synthesis."""
        pass
    
    async def compare_papers(
        paper_ids: List[str],
        aspects: List[str] = ["methodology", "results"]
    ) -> Comparison:
        """Side-by-side paper comparison."""
        pass
```

#### Tool Selection Flow

```
User Query
    │
    ├─► Contains "recent" or "latest"? ──► web_search (for recency)
    │
    ├─► Specific KB mentioned? ──► kb_search
    │
    ├─► Building new collection? ──► scilex_collect
    │
    ├─► Citation format question? ──► get_citation
    │
    ├─► "Related to my previous work"? ──► recall_research
    │
    ├─► Web search found good papers? ──► kb_add_papers (dynamic KB building!)
    │
    └─► Default ──► kb_search + web_search (if insufficient)
```

#### Dynamic KB Building: Chroma vs FAISS

**The Key Difference:**

| Feature | FAISS (v1) | Chroma (v2) |
|---------|-----------|-------------|
| **Add documents** | Rebuild entire index | ✅ Add incrementally |
| **Runtime updates** | ❌ Offline batch only | ✅ Real-time additions |
| **Agent autonomy** | ❌ Cannot self-expand KB | ✅ Can add papers during research |

**Why This Matters:**

```python
# v1 (FAISS): Static KB
# 1. Build KB offline from BibTeX
# 2. Agent can only search existing KB
# 3. If paper not found → "Sorry, not in KB"

# v2 (Chroma): Dynamic KB
# Agent workflow:
async def research_with_dynamic_kb(query: str):
    # 1. Search existing KB
    results = await kb_search(query)
    
    if len(results) < 3:
        # 2. Not enough? Search web
        web_results = await web_search(query)
        
        # 3. Found good papers? ADD TO KB!
        await kb_add_papers(
            kb_name="current_session",
            papers=[r.url for r in web_results[:3]]
        )
        
        # 4. Now re-search with expanded KB
        results = await kb_search(query)
    
    return results
```

**Use Cases for Dynamic KB Building:**

1. **Incremental Research Session**
   ```
   User: "Start researching metabolomics biomarkers"
   Agent: Creates empty "session_2026_03_12" KB
   
   [Turn 3] Agent finds 3 papers via web search
   Agent: Adds them to session KB
   
   [Turn 5] User asks follow-up
   Agent: Searches now-expanded KB
   ```

2. **Citation Following**
   ```
   User: "Tell me about Smith et al. 2021"
   Agent: Finds paper in KB
   Agent: "This paper cites Jones et al. 2020. Add it?"
   User: "Yes"
   Agent: kb_add_papers(["doi:10.xxx/jones2020"])
   ```

3. **User-Provided Papers**
   ```
   User: "Here's a PDF link: https://..."
   Agent: Downloads, parses, adds to KB
   Agent: "Added to your research collection"
   ```

**Implementation with Chroma:**

```python
async def kb_add_papers(self, kb_name: str, papers: List[str]):
    """
    Dynamically add papers to KB during research
    """
    collection = self.chroma.get_collection(kb_name)
    
    for paper_url in papers:
        # 1. Download PDF
        pdf = await self.fetch_pdf(paper_url)
        
        # 2. Parse and extract text
        text = await self.pdf_extract(pdf)
        
        # 3. Smart chunking (preserve sections)
        chunks = self.chunk_with_metadata(
            text,
            chunk_size=1000,
            preserve_sections=True
        )
        
        # 4. Add to Chroma (auto-embeds)
        collection.add(
            documents=[c.text for c in chunks],
            metadatas=[{
                "paper_id": paper_url,
                "chunk_index": c.index,
                "section": c.section,  # "abstract", "methods", etc.
                "added_at": datetime.now().isoformat(),
                "added_by": "agent_web_search"
            } for c in chunks],
            ids=[f"{paper_url}_{i}" for i in range(len(chunks))]
        )
    
    return {"added": len(papers), "chunks": total_chunks}
```

**Advantages Over v1:**

1. **No rebuild needed** - Add papers in real-time
2. **Session-based KBs** - Each research session can have growing KB
3. **Agent autonomy** - Agent can self-improve its knowledge
4. **Collaborative** - Multiple users can add to shared KB
5. **Persistent** - Added papers stay in KB for future queries

**This is why Chroma > FAISS for Perspicacité v2.**

#### Search Efficiency Considerations

**The Trade-off**: Dynamic capability vs raw performance

| Metric | FAISS | Chroma | Notes |
|--------|-------|--------|-------|
| **Query latency (1M docs)** | ~5-10ms | ~20-50ms | FAISS is C++ optimized |
| **Index build time** | Fast (batched) | Slower (incremental) | Chroma optimizes per-add |
| **Concurrent queries** | Excellent | Good | Both handle typical loads |
| **Memory footprint** | Lower | Higher | Chroma stores metadata |
| **Dynamic updates** | ❌ Rebuild required | ✅ Real-time | Chroma's key advantage |

#### Query Accuracy (Retrieval Quality)

**The key question**: Does Chroma find the *right* documents, or just any documents?

**Short answer**: Chroma enables **better accuracy** than pure FAISS through hybrid capabilities.

**1. Pure Vector Search (Both Similar)**

```
Query: "metabolomics biomarkers for COVID-19"
        ↓
[Embedding model] → vector([0.12, -0.34, 0.89, ...])
        ↓
┌─────────────────┐     ┌─────────────────┐
│  FAISS Index    │     │  Chroma (FAISS  │
│  (Flat L2/IP)   │     │   backend)      │
│                 │     │                 │
│  Cosine Sim:    │ ≈   │  Cosine Sim:    │
│  Doc A: 0.91 ✓  │     │  Doc A: 0.91 ✓  │
│  Doc B: 0.87 ✓  │     │  Doc B: 0.87 ✓  │
│  Doc C: 0.82    │     │  Doc C: 0.82    │
└─────────────────┘     └─────────────────┘
```

*Same embedding model = same vector similarity results*

**2. Where Chroma Wins: Hybrid Search**

```python
# Chroma can do what FAISS alone cannot:

# A) Metadata Pre-filtering (narrow search space)
results = collection.query(
    query_embeddings=[query_emb],
    where={
        "$and": [
            {"year": {"$gte": 2020}},           # Recent papers
            {"journal": {"$in": ["Nature", "Science"]}},  # High-impact
            {"has_full_text": True}              # Not just abstracts
        ]
    },
    n_results=10
)
# Result: Search only high-quality candidates → better precision

# B) Keyword + Vector (Hybrid)
# FAISS: Vector only
# Chroma: Can combine with BM25/full-text

results = collection.query(
    query_embeddings=[query_emb],
    query_texts=["COVID-19 metabolomics"],  # Also do keyword match
    n_results=10
)
# Result: Captures exact keyword matches + semantic similarity

# C) Re-ranking with Cross-encoder
# Retrieve 50 candidates with Chroma, then re-rank top 10
```

**3. Accuracy Comparison Table**

| Capability | FAISS Only | Chroma | Impact on Accuracy |
|------------|-----------|--------|-------------------|
| **Pure vector search** | ✅ Exact same | ✅ Same (uses FAISS backend) | Identical |
| **Metadata filtering** | ❌ None | ✅ Pre-filter | **↑ Higher precision** |
| **Keyword (BM25)** | ❌ None | ✅ Hybrid search | **↑ Better recall** |
| **Multi-vector (colbert)** | ⚠️ Manual | ✅ Built-in support | **↑ Better ranking** |
| **Re-ranking** | ⚠️ External | ✅ Integrated | **↑ Better top-k** |

**4. Real-World Example**

```
Query: "What are the best methods for LC-MS metabolomics?"

FAISS (vector only):
┌────────────────────────────────────────┐
│ 1. "LC-MS methods for proteomics"      │ ← Wrong modality (0.88 sim)
│ 2. "Metabolomics using GC-MS"          │ ← Wrong technique (0.85 sim)
│ 3. "Best practices for LC-MS"          │ ← Too vague (0.82 sim)
│ 4. "LC-MS metabolomics review 2023"    │ ← Correct! (0.81 sim)
└────────────────────────────────────────┘
Problem: Semantic similarity ≠ relevance

Chroma (hybrid + metadata filter):
┌────────────────────────────────────────┐
│ Filter: journal in ["Metabolomics",    │
│         "Analytical Chemistry"]        │
│         AND has "LC-MS" in text        │
│                                        │
│ 1. "LC-MS metabolomics review 2023"    │ ← Exact match
│ 2. "LC-MS protocols for metabolomics" │ ← Method-focused
│ 3. "Comparative LC-MS methods"         │ ← Relevant comparison
└────────────────────────────────────────┘
Result: Better precision through filtering
```

**5. v2 Hybrid Search Implementation**

```python
class HybridRetriever:
    """
    Best of both worlds: Chroma's flexibility + accurate ranking
    """
    
    async def search(
        self,
        query: str,
        kb_name: str,
        filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[RetrievedDocument]:
        """
        Two-stage retrieval for optimal accuracy
        """
        collection = self.chroma.get_collection(kb_name)
        
        # Stage 1: Pre-filter with metadata (if filters provided)
        # This eliminates irrelevant candidates early
        candidate_filter = self.build_filter(filters) if filters else None
        
        # Stage 2: Retrieve with hybrid (vector + BM25)
        candidates = collection.query(
            query_texts=[query],
            query_embeddings=[await self.embed(query)],
            where=candidate_filter,
            n_results=min(top_k * 3, 100),  # Over-fetch for re-ranking
            include=["documents", "metadatas", "distances"]
        )
        
        # Stage 3: Re-rank with cross-encoder (more accurate than bi-encoder)
        reranked = await self.cross_encoder_rerank(
            query=query,
            documents=candidates,
            top_k=top_k
        )
        
        return reranked
    
    async def cross_encoder_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        """
        Cross-encoder: query + doc together → more accurate relevance score
        Slower but much more accurate than cosine similarity alone
        """
        pairs = [(query, doc.text) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by cross-encoder score
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc for doc, _ in scored_docs[:top_k]]
```

**6. Accuracy Metrics (Expected)**

| Method | Recall@10 | Precision@10 | NDCG@10 |
|--------|-----------|--------------|---------|
| FAISS (vector only) | 0.72 | 0.65 | 0.68 |
| Chroma (vector only) | 0.72 | 0.65 | 0.68 |
| **Chroma (hybrid)** | **0.81** | **0.74** | **0.77** |
| **+ Metadata filter** | **0.78** | **0.82** | **0.80** |
| **+ Cross-encoder** | **0.85** | **0.88** | **0.87** |

**7. When to Use Each**

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| **General semantic search** | Either | Same embedding = same results |
| **Filtered search** (date, journal, author) | **Chroma** | Metadata filtering eliminates false positives |
| **Exact keyword needed** (chemical names, gene IDs) | **Chroma** | BM25 captures exact matches vector misses |
| **Maximum accuracy** | **Chroma + rerank** | Multi-stage retrieval + precise ranking |
| **Speed critical** (>10M docs, >100 QPS) | FAISS | Raw speed with quantized indices |

**8. Key Insight for v2**

```
Chroma doesn't hurt accuracy—it ENABLES better accuracy through:

1. Metadata filtering      → Eliminate irrelevant candidates
2. Hybrid (vector+BM25)    → Catch exact keyword matches
3. Flexible re-ranking     → Better final ranking
4. Multi-vector (ColBERT)  → Token-level matching

For a research assistant, accuracy matters more than raw speed.
A 50ms query that returns the RIGHT paper beats a 5ms query 
that returns the WRONG paper.
```

**Optimization Strategies for v2:**

```python
class OptimizedChromaKB:
    """
    Chroma with performance optimizations for large-scale use
    """
    
    def __init__(self):
        self.collection = None
        self.query_cache = {}  # LRU cache for repeated queries
        self.batch_buffer = []  # Buffer for batch additions
        
    async def search(
        self,
        query: str,
        kb_name: str,
        top_k: int = 10,
        use_cache: bool = True
    ) -> List[RetrievedDocument]:
        """
        Optimized search with caching
        """
        # 1. Check cache (exact match or embedding similarity)
        cache_key = hashlib.md5(f"{kb_name}:{query}:{top_k}".encode()).hexdigest()
        if use_cache and cache_key in self.query_cache:
            cached = self.query_cache[cache_key]
            if time.time() - cached["timestamp"] < 300:  # 5 min TTL
                return cached["results"]
        
        # 2. Generate embedding (with caching)
        query_embedding = await self.get_embedding(query)
        
        # 3. Search Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 4. Cache results
        if use_cache:
            self.query_cache[cache_key] = {
                "results": results,
                "timestamp": time.time()
            }
        
        return results
    
    async def add_papers_optimized(
        self,
        papers: List[Paper],
        batch_size: int = 100
    ):
        """
        Batch additions for efficiency (don't add one-by-one)
        """
        # Buffer papers for batch insertion
        self.batch_buffer.extend(papers)
        
        if len(self.batch_buffer) >= batch_size:
            await self._flush_batch()
    
    async def _flush_batch(self):
        """Process buffered papers in batch"""
        if not self.batch_buffer:
            return
        
        # 1. Download all PDFs in parallel
        pdfs = await asyncio.gather(*[
            self.fetch_pdf(p.url) for p in self.batch_buffer
        ])
        
        # 2. Extract text in parallel
        texts = await asyncio.gather(*[
            self.pdf_extract(pdf) for pdf in pdfs if pdf
        ])
        
        # 3. Generate all embeddings in single batch (GPU-efficient)
        all_chunks = []
        for text in texts:
            all_chunks.extend(self.chunk_document(text))
        
        embeddings = await self.embedder.embed_batch(
            [c.text for c in all_chunks],
            batch_size=32  # Optimal for GPU
        )
        
        # 4. Single Chroma add operation (much faster than individual)
        self.collection.add(
            documents=[c.text for c in all_chunks],
            metadatas=[c.metadata for c in all_chunks],
            embeddings=embeddings,
            ids=[c.id for c in all_chunks]
        )
        
        # Clear buffer
        self.batch_buffer = []
```

**When to Use FAISS vs Chroma:**

| Scenario | Recommendation |
|----------|---------------|
| **Production KB (millions of docs, rare updates)** | FAISS (export periodically) |
| **Research sessions (dynamic, growing)** | Chroma (default for v2) |
| **Hybrid approach** | Chroma for active sessions → export to FAISS for archive |

**v2 Recommendation:**

```python
# Use Chroma for active/dynamic KBs (default)
kb = ChromaKnowledgeBase()

# For large static collections, export to FAISS
if kb.size > 1_000_000:
    # Periodically export to FAISS for read-only speed
    faiss_index = kb.export_to_faiss()
```

**Performance Benchmarks (Expected):**

| Collection Size | Chroma Query | FAISS Query | Chroma Add | FAISS Rebuild |
|----------------|--------------|-------------|------------|---------------|
| 10K docs | 15ms | 3ms | 50ms | N/A |
| 100K docs | 25ms | 5ms | 80ms | 30s |
| 1M docs | 50ms | 10ms | 150ms | 5min |
| 10M docs | 150ms | 20ms | 300ms | 1hr |

**For Enterprise Scale:**
- Use **HNSW index** in Chroma (approximate search, sub-linear time)
- Implement **two-tier architecture**: Hot KB (Chroma) + Archive (FAISS)
- Add **Redis caching** for frequent queries
- Use **metadata pre-filtering** to reduce search space

#### Chemical Embeddings & Multi-Modal Search

**The Problem**: Standard text embeddings miss chemical relationships

```
Query: "Papers about aspirin and similar anti-inflammatory drugs"

Text embedding issue:
- "Aspirin" → vector([...])
- "Acetylsalicylic acid" → different vector (same molecule, different name!)
- Structure similarity is LOST in text space

Chemical embedding solution:
- SMILES: "CC(=O)Oc1ccccc1C(=O)O" → Morgan fingerprint → vector
- Similar structure → Similar fingerprint → Similar embedding
```

#### Technical Integration: Chemical Embeddings with Vector DBs

**1. The Core Challenge: Binary vs Dense Embeddings**

| Property | Text Embeddings | Chemical Fingerprints |
|----------|----------------|----------------------|
| **Source** | Sentence-BERT, OpenAI | RDKit Morgan/ECFP |
| **Values** | Float32 [-1, 1] | Binary {0, 1} |
| **Dimensions** | 384-1536 (dense) | 1024-4096 (sparse) |
| **Metric** | Cosine / Dot product | Tanimoto / Dice |
| **Index type** | HNSW (float) | Binary index (Hamming) |

**FAISS and Chroma both work with float32 vectors only.** Binary fingerprints must be converted.

**2. Three Integration Strategies**

**Strategy A: Binary-to-Float Conversion (Recommended for v2)**

```python
"""
Convert Morgan fingerprint (binary) to float embedding for standard vector DBs
"""
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class ChemicalEmbedder:
    def __init__(self, n_bits: int = 2048, projection_dim: int = 256):
        self.n_bits = n_bits
        self.projection_dim = projection_dim
        # Learned projection: binary 2048-dim → dense 256-dim
        self.projection = self._load_or_create_projection()
    
    def morgan_fingerprint(self, smiles: str) -> np.ndarray:
        """Generate binary Morgan fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.n_bits)
        return np.array(fp)
    
    def fingerprint_to_embedding(self, fp: np.ndarray) -> np.ndarray:
        """
        Convert binary fingerprint to dense float embedding
        
        Option 1: Direct cast (works but inefficient)
        - Input: 2048 binary bits
        - Output: 2048 float32 (sparse, 0.0 or 1.0)
        - Problem: High dimensionality, wastes space
        
        Option 2: PCA projection (better)
        - Input: 2048 binary bits
        - Output: 256 float32 (dense)
        - Trained on large chemical database
        
        Option 3: Autoencoder (best)
        - Input: 2048 binary bits
        - Output: 128-256 float32 (dense, meaningful)
        - Preserves chemical similarity structure
        """
        # Option 2: PCA projection
        if self.projection is not None:
            return fp.astype(np.float32) @ self.projection  # (2048,) @ (2048, 256) = (256,)
        
        # Fallback: direct cast (sparse but compatible)
        return fp.astype(np.float32)
    
    def embed_smiles(self, smiles: str) -> np.ndarray:
        """Full pipeline: SMILES → fingerprint → embedding"""
        fp = self.morgan_fingerprint(smiles)
        if fp is None:
            return None
        return self.fingerprint_to_embedding(fp)
```

**Strategy B: Use Pre-trained Chemical Embeddings (Alternative)**

```python
"""
Instead of fingerprints, use learned chemical embeddings (already dense)
"""
from transformers import AutoModel, AutoTokenizer

class LearnedChemicalEmbedder:
    """
    Use ChemBERTa or MoLFormer - already produces dense float embeddings
    """
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = 768  # ChemBERTa dimension
    
    def embed_smiles(self, smiles: str) -> np.ndarray:
        """SMILES → ChemBERTa embedding"""
        inputs = self.tokenizer(smiles, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        
        # Mean pooling over token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embedding[0]  # (768,) float32

"""
Comparison:
┌─────────────────┬──────────────────┬──────────────────┐
│ Aspect          │ Morgan + PCA     │ ChemBERTa        │
├─────────────────┼──────────────────┼──────────────────┤
│ Dimension       │ 256 (configurable│ 768 (fixed)      │
│ Interpretable   │ Yes (structure)  │ No (black box)   │
│ Tanimoto approx │ Yes (preserved)  │ No               │
│ Compute         │ Fast (RDKit)     │ Slow (NN forward)│
│ No GPU needed   │ Yes              │ Prefer GPU       │
│ Pre-training    │ None needed      │ Required         │
└─────────────────┴──────────────────┴──────────────────┘
"""
```

**Strategy C: Separate Binary Index (FAISS Only)**

```python
"""
FAISS supports binary indices directly - use for exact Tanimoto search
"""
import faiss

class FAISSChemicalIndex:
    """
    FAISS binary index for exact chemical similarity
    """
    def __init__(self, n_bits: int = 2048):
        self.n_bits = n_bits
        # Binary index using Hamming distance (≈ Tanimoto for bit vectors)
        self.index = faiss.IndexBinaryFlat(n_bits)
        self.paper_ids = []  # Map index position → paper_id
    
    def add_chemicals(self, chemicals: List[Dict]):
        """
        Add chemicals to binary index
        
        Args:
            chemicals: [{"smiles": "...", "paper_id": "...", "fp": np.array}, ...]
        """
        # Stack fingerprints into (n_chemicals, n_bits/8) uint8 array
        fp_matrix = np.vstack([c["fp"] for c in chemicals]).astype(np.uint8)
        
        # Pack bits into bytes (FAISS binary format requirement)
        fp_bytes = np.packbits(fp_matrix, axis=1)
        
        self.index.add(fp_bytes)
        self.paper_ids.extend([c["paper_id"] for c in chemicals])
    
    def search(self, query_smiles: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search by structure using Hamming distance
        """
        # Generate query fingerprint
        mol = Chem.MolFromSmiles(query_smiles)
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.n_bits)
        query_fp = np.array(query_fp).astype(np.uint8).reshape(1, -1)
        query_bytes = np.packbits(query_fp, axis=1)
        
        # Search (returns Hamming distances)
        distances, indices = self.index.search(query_bytes, k)
        
        # Convert Hamming distance to Tanimoto similarity
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            # Hamming distance = number of differing bits
            # Tanimoto = 1 - (Hamming / total_bits)
            tanimoto = 1 - (dist / self.n_bits)
            results.append((self.paper_ids[idx], tanimoto))
        
        return results

"""
Limitations of Strategy C:
- FAISS binary index is separate from float index
- Cannot do unified text+chemical search easily
- Must maintain two indices manually
- Chroma doesn't expose FAISS binary indices directly
"""
```

**3. Chroma Integration (Recommended for v2)**

```python
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions

class ChromaChemicalKB:
    """
    Chroma-based KB with chemical embeddings
    """
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = PersistentClient(path=db_path)
        
        # Text collection: standard sentence embeddings
        self.text_collection = self.client.get_or_create_collection(
            name="papers_text",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        # Chemical collection: custom embedding function
        self.chem_embedder = ChemicalEmbedder(n_bits=2048, projection_dim=256)
        self.chem_collection = self.client.get_or_create_collection(
            name="papers_chemicals",
            # Chroma will use our custom embedder
            embedding_function=self._chemical_embedding_function()
        )
    
    def _chemical_embedding_function(self):
        """Create Chroma-compatible embedding function"""
        class ChemicalEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, embedder):
                self.embedder = embedder
            
            def __call__(self, smiles_list: List[str]) -> List[List[float]]:
                """Chroma calls this to embed documents/queries"""
                embeddings = []
                for smiles in smiles_list:
                    emb = self.embedder.embed_smiles(smiles)
                    if emb is None:
                        emb = np.zeros(self.embedder.projection_dim)
                    embeddings.append(emb.tolist())
                return embeddings
        
        return ChemicalEmbeddingFunction(self.chem_embedder)
    
    async def add_paper(self, paper: Paper):
        """Add paper with both text and chemical indices"""
        paper_id = paper.doi or paper.pmid
        
        # 1. Index text chunks
        chunks = self.chunk_text(paper.full_text)
        for i, chunk in enumerate(chunks):
            self.text_collection.add(
                ids=[f"{paper_id}_chunk_{i}"],
                documents=[chunk.text],
                metadatas=[{
                    "paper_id": paper_id,
                    "chunk_index": i,
                    "section": chunk.section,
                    "has_chemicals": bool(paper.chemicals)
                }]
            )
        
        # 2. Index chemicals (if any)
        if paper.chemicals:
            for chem in paper.chemicals:
                # Generate embedding
                fp = self.chem_embedder.morgan_fingerprint(chem.smiles)
                emb = self.chem_embedder.fingerprint_to_embedding(fp)
                
                self.chem_collection.add(
                    ids=[f"{paper_id}_chem_{chem.inchi_key}"],
                    documents=[chem.canonical_smiles],  # Chroma stores this
                    embeddings=[emb.tolist()],  # Our custom embedding
                    metadatas=[{
                        "paper_id": paper_id,
                        "smiles": chem.smiles,
                        "inchi_key": chem.inchi_key,
                        "name": chem.name,
                        "fingerprint_bits": fp.tolist(),  # For Tanimoto re-ranking
                    }]
                )
    
    async def search_by_structure(self, query_smiles: str, k: int = 10):
        """
        Search papers by chemical structure similarity
        """
        # Chroma handles embedding via our custom function
        results = self.chem_collection.query(
            query_texts=[query_smiles],  # Chroma calls embedder
            n_results=k * 5,  # Over-fetch for re-ranking
            include=["metadatas", "distances", "embeddings"]
        )
        
        # Re-rank with exact Tanimoto (more accurate than cosine on projected vectors)
        query_fp = self.chem_embedder.morgan_fingerprint(query_smiles)
        
        reranked = []
        for idx, metadata in enumerate(results["metadatas"][0]):
            stored_fp = np.array(metadata["fingerprint_bits"])
            tanimoto = self._tanimoto(query_fp, stored_fp)
            
            reranked.append({
                "paper_id": metadata["paper_id"],
                "chemical_name": metadata["name"],
                "tanimoto": tanimoto,
                "smiles": metadata["smiles"]
            })
        
        # Sort by Tanimoto, return top-k
        reranked.sort(key=lambda x: x["tanimoto"], reverse=True)
        return reranked[:k]
    
    async def hybrid_search(
        self,
        text_query: str,
        chemical_query: Optional[str] = None,
        text_weight: float = 0.6,
        chem_weight: float = 0.4,
        k: int = 10
    ):
        """
        Hybrid search: text relevance + chemical similarity
        """
        # 1. Text search
        text_results = self.text_collection.query(
            query_texts=[text_query],
            n_results=k * 3
        )
        text_scores = {}
        for meta, dist in zip(text_results["metadatas"][0], text_results["distances"][0]):
            # Convert distance to similarity score
            text_scores[meta["paper_id"]] = 1 - dist  # Assuming normalized
        
        # 2. Chemical search (if provided)
        chem_scores = {}
        if chemical_query:
            chem_results = await self.search_by_structure(chemical_query, k=k*3)
            for r in chem_results:
                chem_scores[r["paper_id"]] = r["tanimoto"]
        
        # 3. Fuse scores
        all_papers = set(text_scores.keys()) | set(chem_scores.keys())
        fused = []
        for pid in all_papers:
            t_score = text_scores.get(pid, 0)
            c_score = chem_scores.get(pid, 0)
            final = text_weight * t_score + chem_weight * c_score
            fused.append((pid, final, t_score, c_score))
        
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]
    
    def _tanimoto(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Tanimoto similarity: intersection / union"""
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        return float(intersection) / float(union) if union > 0 else 0.0
```

**4. FAISS Integration (Alternative)**

```python
import faiss
import numpy as np

class FAISSChemicalKB:
    """
    FAISS-based implementation (for comparison)
    """
    def __init__(self, text_dim: int = 384, chem_dim: int = 256):
        self.text_dim = text_dim
        self.chem_dim = chem_dim
        
        # Two separate indices (FAISS limitation)
        self.text_index = faiss.IndexFlatIP(text_dim)  # Inner product = cosine for normalized
        self.chem_index = faiss.IndexFlatIP(chem_dim)
        
        # Mappings
        self.text_id_map = {}  # faiss_idx → paper_id
        self.chem_id_map = {}
    
    def add_paper(self, paper: Paper, text_emb: np.ndarray, chem_embs: List[np.ndarray]):
        """
        Add paper to both indices
        
        Note: FAISS requires managing indices separately - more complex than Chroma
        """
        # Add text
        text_idx = self.text_index.ntotal
        self.text_index.add(text_emb.reshape(1, -1))
        self.text_id_map[text_idx] = paper.id
        
        # Add chemicals
        for chem_emb in chem_embs:
            chem_idx = self.chem_index.ntotal
            self.chem_index.add(chem_emb.reshape(1, -1))
            self.chem_id_map[chem_idx] = (paper.id, chem.inchi_key)
    
    def search_chemical(self, query_emb: np.ndarray, k: int = 10):
        """Search chemical index"""
        distances, indices = self.chem_index.search(query_emb.reshape(1, -1), k)
        # Map back to paper IDs
        return [self.chem_id_map[i] for i in indices[0]]

"""
Chroma vs FAISS for Chemical Search:
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│ Aspect              │ Chroma                  │ FAISS                   │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│ Multi-collection    │ Native support          │ Manual management       │
│ Metadata filtering  │ Built-in                │ External (pre-filter)   │
│ Embedding function  │ Pluggable               │ Manual                  │
│ Persistence         │ Automatic               │ Manual save/load        │
│ Binary index        │ Not exposed             │ Direct access           │
│ Complexity          │ Lower                   │ Higher                  │
│ Flexibility         │ Higher (multi-modal)    │ Lower (single metric)   │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
"""
```

**5. Summary: Recommended v2 Architecture**

```python
"""
Recommended approach for Perspicacité v2:
"""

class PerspicaciteKnowledgeBase:
    def __init__(self, enable_chemistry: bool = True):
        # Chroma as primary backend
        self.client = PersistentClient(path="./kb")
        
        # Text collection: standard embeddings
        self.text = self.client.get_or_create_collection("text")
        
        # Chemical collection: custom fingerprint→dense projection
        self.chemical = None
        if enable_chemistry:
            self.chem_embedder = ChemicalEmbedder(projection_dim=256)
            self.chemical = self.client.get_or_create_collection(
                "chemical",
                embedding_function=self.chem_embedder.to_chroma_function()
            )
    
    async def search(self, query: Query) -> Results:
        """
        Unified search interface
        """
        if query.has_chemical_component() and self.chemical:
            # Hybrid: text + structure
            return await self.hybrid_search(query)
        else:
            # Text only
            return await self.text_search(query)

"""
Key Design Decisions:
1. Use Strategy A (fingerprint + PCA projection) for v2
   - Fast (RDKit, no NN inference)
   - Interpretable (structure-based)
   - Compatible with any vector DB

2. Chroma over FAISS for chemical search
   - Better multi-collection support
   - Built-in metadata handling
   - Pluggable embedding functions

3. Re-rank with exact Tanimoto
   - Vector search gives candidates (fast, approximate)
   - Tanimoto re-ranking gives accuracy (exact)
   - Two-stage approach balances speed and precision
"""
```

---

## 4. Memory Schema Design

### 4.1 What to Store

| Type | Example | Purpose |
|------|---------|---------|
| **Query** | "Recent metabolomics LC-MS papers" | Recall what was asked |
| **Findings** | Summary of 5 papers on topic | Avoid re-synthesizing |
| **Papers Used** | ["pmid:12345", "doi:10.xxx"] | Track sources |
| **KB Accessed** | ["metabolomics_kb", "covid_kb"] | Which collections |
| **User Preferences** | "Prefers Nature format" | Personalization |
| **Success Rating** | User thumbs up/down | Quality feedback |

### 4.2 Schema (PostgreSQL + Vector Store)

```sql
-- Research Sessions (Episodic Memory)
CREATE TABLE research_sessions (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    query_embedding VECTOR(1536),  -- For semantic recall
    findings_summary TEXT,
    papers_used TEXT[],  -- Array of DOIs/PMIDs
    kbs_accessed TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    success_rating INTEGER CHECK (success_rating BETWEEN 1 AND 5),
    metadata JSONB  -- Flexible: duration, tokens used, etc.
);

-- Create index for semantic search
CREATE INDEX ON research_sessions 
USING ivfflat (query_embedding vector_cosine_ops);

-- User Preferences (Semantic Memory)
CREATE TABLE user_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    citation_style VARCHAR(50) DEFAULT 'nature',
    preferred_databases TEXT[],  -- ['pubmed', 'scholar']
    research_domains TEXT[],  -- ['metabolomics', 'drug_discovery']
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Paper Interactions (For recommendations)
CREATE TABLE paper_interactions (
    id UUID PRIMARY KEY,
    user_id VARCHAR(255),
    paper_id VARCHAR(255),
    interaction_type VARCHAR(50),  -- 'viewed', 'cited', 'saved'
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### 4.3 Recall Mechanisms

```python
class MemoryManager:
    """
    Multi-modal memory retrieval
    """
    
    async def recall(
        self,
        user_id: str,
        query: str,
        strategy: str = "hybrid"
    ) -> List[ResearchSession]:
        """
        Recall relevant past research
        
        Strategies:
        - "semantic": Vector similarity on query
        - "keyword": Full-text search
        - "hybrid": Both + re-ranking
        - "recent": Time-based (last 7 days)
        """
        
        if strategy == "semantic":
            # Vector similarity
            query_embedding = self.embed(query)
            return await self.db.query(
                """
                SELECT * FROM research_sessions
                WHERE user_id = $1
                ORDER BY query_embedding <=> $2
                LIMIT 5
                """,
                user_id, query_embedding
            )
        
        elif strategy == "keyword":
            # Text search
            return await self.db.query(
                """
                SELECT * FROM research_sessions
                WHERE user_id = $1
                AND (
                    query ILIKE $2
                    OR findings_summary ILIKE $2
                )
                ORDER BY created_at DESC
                LIMIT 5
                """,
                user_id, f"%{query}%"
            )
        
        elif strategy == "hybrid":
            # Combine both
            semantic_results = await self.recall(user_id, query, "semantic")
            keyword_results = await self.recall(user_id, query, "keyword")
            return self.reciprocal_rank_fusion(semantic_results, keyword_results)
```

### 4.4 Memory Usage in Agent Flow

```python
async def research_with_memory(user_id: str, query: str):
    # 1. Check if similar research exists
    past_work = await memory.recall(user_id, query, strategy="hybrid")
    
    if past_work and past_work[0].similarity > 0.9:
        # Very similar query → suggest using prior results
        return {
            "type": "prior_work_available",
            "message": "You researched this 2 days ago. Use those results?",
            "prior_session": past_work[0]
        }
    
    # 2. Check for relevant papers from past sessions
    relevant_papers = set()
    for session in past_work[:3]:  # Top 3 similar sessions
        relevant_papers.update(session.papers_used)
    
    # 3. Boost these papers in search
    kb_results = await kb.search(
        query,
        boost_ids=list(relevant_papers)  # Higher ranking for known papers
    )
    
    # 4. Store new session
    await memory.store(user_id, query, findings, papers_used, kbs_accessed)
```

### 4.5 Complete Information Flow to LLM

**Scenario**: User asks "How does metabolomics help predict COVID-19 severity?"

#### Step-by-Step Context Construction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: SYSTEM PROMPT (Always included)                                   │
│  ~500 tokens                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  You are Perspicacité, a scientific literature research assistant.          │
│                                                                             │
│  Your role is to help researchers find, analyze, and synthesize            │
│  scientific papers. Provide evidence-based answers with citations.          │
│                                                                             │
│  Guidelines:                                                                │
│  - Always cite sources using [Author et al., Year](URL "Full Citation")    │
│  - If information is insufficient, say so clearly                          │
│  - Highlight conflicting findings from different sources                    │
│  - Use technical language appropriate for the field                         │
│                                                                             │
│  Available tools:                                                           │
│  - kb_search: Search local knowledge bases                                  │
│  - web_search: Search academic databases (PubMed, Scholar, etc.)           │
│  - fetch_pdf: Download and parse PDF papers                                 │
│  - get_citation_network: Find related papers via citations                  │
│  - recall_research: Access user's research history                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: USER PREFERENCES / PROFILE (Lightweight, ~100 tokens)            │
├─────────────────────────────────────────────────────────────────────────────┤
│  User Profile:                                                              │
│  - Research domain: Metabolomics, Systems Biology                           │
│  - Citation style: Nature                                                   │
│  - Preferred databases: PubMed, Metabolights                                │
│  - Recent focus: COVID-19, cytokine storm, LC-MS methods                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: SESSION CHAT HISTORY (Current conversation, ~500-2000 tokens)    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Current session started: 2026-03-12 14:30                                  │
│  Knowledge base: "metabolomics_v2"                                          │
│                                                                             │
│  [Turn 1]                                                                   │
│  User: "What are the main metabolomics approaches for disease biomarkers?" │
│  Assistant: "The main approaches are... [targeted vs untargeted summary]"   │
│                                                                             │
│  [Turn 2]                                                                   │
│  User: "Focus on LC-MS methods"                                             │
│  Assistant: "LC-MS metabolomics typically involves... [technical details]"  │
│                                                                             │
│  [Turn 3]                                                                   │
│  User: "Now how does this apply to COVID-19?"                               │
│  Assistant: "LC-MS metabolomics has been extensively applied to COVID-19..."│
│                                                                             │
│  [Older messages summarized]                                                │
│  [Summary: User is building knowledge from general → specific, focusing    │
│   on LC-MS metabolomics for infectious disease biomarkers]                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: RETRIEVED MEMORY (Cross-session, if relevant, ~200-500 tokens)   │
├─────────────────────────────────────────────────────────────────────────────┤
│  [From recall_research("metabolomics COVID-19") - Past sessions]            │
│                                                                             │
│  Similar past research (3 days ago, different session):                     │
│  Query: "Metabolomics biomarkers COVID severity"                           │
│  Key findings:                                                              │
│  - Wu et al. 2021 identified 5 metabolites associated with severity        │
│  - Strong correlation between kynurenine pathway and ICU admission         │
│  - Papers saved: ["pmid:33412345", "doi:10.1038/s41586-021-..."]          │
│  Rating: ⭐⭐⭐⭐⭐ (User found helpful)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: RETRIEVED DOCUMENTS (KB Search Results, ~2000-4000 tokens)       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Retrieved from knowledge base "metabolomics_v2" (Top 5 most relevant):    │
│                                                                             │
│  [Document 1] Relevance: 0.94                                               │
│  Source: Wang et al., Nature Metabolism, 2021                               │
│  URL: https://doi.org/10.1038/s42255-021-00442-2                           │
│  Content: "Metabolomic profiling reveals distinct signatures of             │
│  SARS-CoV-2 infection severity. We identified 33 metabolites               │
│  significantly altered in severe COVID-19, including..."                    │
│                                                                             │
│  [Document 2] Relevance: 0.91                                               │
│  Source: Shen et al., Cell Metabolism, 2020                                 │
│  URL: https://doi.org/10.1016/j.cmet.2020.10.007                           │
│  Content: "Serum metabolomics reveals kynurenine pathway activation        │
│  as a biomarker for COVID-19 severity..."                                   │
│                                                                             │
│  [Document 3] Relevance: 0.88                                               │
│  Source: (additional papers...)                                             │
│  ...                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 5: WEB SEARCH RESULTS (If KB insufficient, ~1500-2500 tokens)       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Web search results (PubMed, Semantic Scholar):                             │
│                                                                             │
│  [Paper 1] Title: "Metabolomic signatures predict COVID-19 severity"      │
│  Authors: Thompson et al.                                                   │
│  Year: 2023                                                                 │
│  Abstract: "We performed untargeted metabolomics on 500 COVID patients..." │
│  PDF Available: https://...                                                 │
│                                                                             │
│  [Paper 2] Title: "Kynurenine pathway in severe COVID-19"                  │
│  ...                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 6: SKILL INSTRUCTIONS (If activated, ~1000-2000 tokens)             │
├─────────────────────────────────────────────────────────────────────────────┤
│  [Skill: literature-review loaded via tool call]                            │
│                                                                             │
│  # Literature Review Methodology                                            │
│                                                                             │
│  When conducting a systematic review:                                       │
│  1. Define PICO: Population, Intervention, Comparison, Outcome             │
│  2. Search strategy: Use MeSH terms + free text                             │
│  3. Screening: Title/abstract → Full text                                   │
│  4. Data extraction: Standardized forms                                     │
│  5. Quality assessment: PRISMA guidelines                                   │
│                                                                             │
│  For this query about COVID severity prediction:                            │
│  - Focus on prognostic biomarkers                                           │
│  - Consider longitudinal studies                                            │
│  - Note effect sizes and confidence intervals                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 7: CURRENT USER QUERY                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  User: "How does metabolomics help predict COVID-19 severity?"             │
│                                                                             │
│  Context: Researching for grant proposal on biomarkers.                    │
│  Need specific metabolite names and predictive accuracy metrics.           │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Total Context Size Estimate

| Layer | Tokens (typical) | Notes |
|-------|-----------------|-------|
| System Prompt | 500 | Fixed |
| User Profile | 100 | Fixed |
| Chat History | 500-2000 | Last N turns (older summarized) |
| Memory | 0-500 | Cross-session, only if relevant |
| KB Results | 2000-4000 | Top 5-10 documents |
| Web Results | 0-2500 | Only if KB insufficient |
| Skill | 0-2000 | Only if methodology needed |
| User Query | 50-200 | Variable |
| **Total** | **~3,500 - 12,000** | Depends on query complexity |

#### Chat History Management

```python
class ChatHistoryManager:
    """
    Manage conversation history within a session
    """
    
    MAX_HISTORY_TURNS = 10  # Keep last 10 exchanges verbatim
    SUMMARIZE_THRESHOLD = 20  # Summarize older turns
    
    def format_history(self, messages: List[Message]) -> str:
        """
        Format chat history for context window
        
        Strategy:
        1. Recent messages: Include verbatim (full detail)
        2. Older messages: Summarize to save tokens
        3. System messages: Always include fully
        """
        if len(messages) <= self.MAX_HISTORY_TURNS:
            return self.format_verbatim(messages)
        
        # Split into recent and old
        recent = messages[-self.MAX_HISTORY_TURNS:]
        older = messages[:-self.MAX_HISTORY_TURNS]
        
        # Summarize older messages
        older_summary = self.summarize_messages(older)
        
        return f"""
[Earlier conversation summary]
{older_summary}

[Recent messages]
{self.format_verbatim(recent)}
"""
    
    def summarize_messages(self, messages: List[Message]) -> str:
        """Use LLM to create concise summary of older conversation"""
        prompt = f"""
        Summarize this conversation history in 2-3 sentences.
        Focus on: topic, key decisions, what was learned.
        
        Conversation:
        {self.format_verbatim(messages)}
        
        Summary:
        """
        return self.llm.complete(prompt, max_tokens=100)
    
    def get_relevant_history(self, query: str, messages: List[Message]) -> List[Message]:
        """
        Selectively retrieve relevant parts of history
        (Alternative to simple truncation)
        """
        # Embed query and all messages
        query_emb = self.embed(query)
        message_embs = [self.embed(m.content) for m in messages]
        
        # Find most relevant messages
        similarities = [cosine_sim(query_emb, m_emb) for m_emb in message_embs]
        
        # Return top-k most relevant + recent context
        relevant_indices = np.argsort(similarities)[-5:]
        recent_indices = range(len(messages) - 3, len(messages))
        
        selected = set(relevant_indices) | set(recent_indices)
        return [messages[i] for i in sorted(selected)]
```

#### Chat History vs Memory: Key Difference

| Aspect | Chat History | Memory |
|--------|--------------|--------|
| **Scope** | Current session only | Cross-session persistence |
| **Content** | Full conversation turns | Condensed findings, preferences |
| **Storage** | In-memory / Redis | PostgreSQL + Vector DB |
| **Lifetime** | Hours (session duration) | Months/years |
| **Trigger** | Always included (recent) | Retrieved if relevant |
| **Format** | Verbatim or summarized | Structured (JSON/metadata) |

**Example:**
```
Current Session (Chat History):
- 10 minutes ago: "What is metabolomics?"
- 5 minutes ago: "Focus on LC-MS"
- Now: "Apply to COVID-19"

Cross-Session (Memory):
- 3 days ago: Researched "metabolomics biomarkers"
- Last week: Saved 5 papers on "LC-MS methods"
- Preference: "Uses Nature citation format"
```

#### Optimization Strategies

```python
class ContextManager:
    """
    Manage context window efficiently
    """
    
    MAX_CONTEXT_TOKENS = 8000  # Leave room for response
    
    def build_context(self, query: str, user_id: str) -> str:
        parts = []
        token_count = 0
        
        # 1. System prompt (always included)
        parts.append(self.system_prompt)
        token_count += 500
        
        # 2. User profile (always included, small)
        parts.append(self.get_user_profile(user_id))
        token_count += 100
        
        # 3. Retrieved memory (if relevant)
        memory = self.recall_memory(user_id, query)
        if memory and memory.relevance > 0.8:
            parts.append(self.format_memory(memory))
            token_count += min(len(memory), 500)
        
        # 4. Documents (most important - prioritize by relevance)
        docs = self.retrieve_documents(query)
        remaining = self.MAX_CONTEXT_TOKENS - token_count - 1000  # Reserve for response
        
        formatted_docs = []
        for doc in docs:
            doc_text = self.format_document(doc)
            if token_count + len(doc_text) > remaining:
                break
            formatted_docs.append(doc_text)
            token_count += len(doc_text)
        
        parts.append("\n\n".join(formatted_docs))
        
        # 5. User query
        parts.append(f"\n\nUser Query: {query}")
        
        return "\n\n".join(parts)
    
    def prioritize_content(self, docs: List[Document], query: str) -> List[Document]:
        """
        Prioritize which documents to include based on:
        1. Relevance score
        2. Recency (for fast-moving fields)
        3. Citation count (influential papers)
        4. Diversity (avoid same author/cluster)
        """
        scored = []
        for doc in docs:
            score = (
                doc.relevance * 0.5 +
                self.recency_score(doc.year) * 0.2 +
                self.citation_score(doc.citations) * 0.2 +
                self.diversity_bonus(doc, scored) * 0.1
            )
            scored.append((score, doc))
        
        return [d for _, d in sorted(scored, reverse=True)]
```

#### Key Insights

1. **Not everything is loaded upfront** - Documents are retrieved on-demand based on query
2. **Skills are lazy-loaded** - Only loaded if LLM calls `load_skill` tool
3. **Memory is selective** - Only if similarity > 0.8
4. **Web search is fallback** - Only if KB results insufficient
5. **Context is ranked** - Most relevant documents first, truncate if needed
6. **Chat history is managed** - Recent turns verbatim, older summarized

This keeps the context manageable while maximizing information quality.

#### Example Flow (Multi-Turn Conversation)

**Turn 1: Initial Question**
```
User: "What are metabolomics approaches for biomarkers?"
    ↓
1. System prompt (500)
2. User profile (100)
3. Chat history: [empty - first message] (0)
4. Memory: None relevant (0)
5. KB search → 5 papers (3000)
6. Add user query (100)
    ↓
Total: ~3,700 tokens
```

**Turn 3: Follow-up Question**
```
User: "How does this apply to COVID-19?"
    ↓
1. System prompt (500)
2. User profile (100)
3. Chat history:
   - Turn 1 Q&A (400)
   - Turn 2 Q&A (400)
   Total: ~800
4. Memory: None relevant (0)
5. KB search → 5 papers (3000)
6. Add user query (50)
    ↓
Total: ~4,450 tokens
```

**Turn 10: Long Conversation**
```
User: "What about the kynurenine pathway specifically?"
    ↓
1. System prompt (500)
2. User profile (100)
3. Chat history:
   - Turns 1-7: Summarized (200)
   - Turns 8-9: Verbatim (600)
   Total: ~800
4. Memory: Found similar research 3 days ago (400)
5. KB search → 5 papers + boost prior papers (3000)
6. Add user query (50)
    ↓
Total: ~4,850 tokens
```

---

## 5. RAG Modes Analysis: Profound vs Agentic

### 4.1 Expert Analysis: Is Profound RAG Already Agentic?

After analyzing Perspicacité's Profound mode against expert definitions of Agentic RAG, here are the findings:

#### 4.1.1 Current Profound Mode Capabilities (v1)

Perspicacité's **Profound mode** already implements many "agentic" characteristics:

| Feature | Profound Mode Implementation | Agentic RAG Criteria |
|---------|------------------------------|---------------------|
| **Planning** | `_create_research_plan()` - LLM generates research steps with queries | ✅ Explicit planner |
| **Iteration** | Up to 5 research cycles with adaptive strategy | ✅ Multi-loop iterative |
| **Reflection** | `_analyze_documents()` - evaluates doc relevance & sufficiency | ✅ Self-reflection |
| **Plan Adjustment** | `_review_and_adjust_plan()` - modifies plan based on findings | ✅ Dynamic adaptation |
| **Early Exit** | `_is_question_answered()` - confidence-based termination | ✅ Stop condition evaluation |
| **Error Handling** | Detects unanswerable/false premise questions | ✅ Error recovery |
| **Quality Assessment** | Document quality scoring with missing aspect identification | ✅ Evidence evaluation |

**Architecture Flow (Profound Mode)**:
```
User Question → Create Plan → For Each Step:
    ├── Stage 1: Basic RAG
    ├── Assess Quality (sufficient?)
    ├── Stage 2: Advanced RAG (if needed)
    ├── Assess Quality (sufficient?)
    └── Stage 3: Web Search (if enabled & needed)
    └── Analyze Documents
├── Create Iteration Summary
├── Should Continue? (max cycles, early exit)
├── Review & Adjust Plan (if consecutive failures)
└── Generate Final Answer (with refinement)
```

---

#### 4.1.2 What Experts Say Defines "True" Agentic RAG

According to recent research and expert analyses (2024-2025):

**Core Agentic RAG Components** (from TechRxiv, XCube Labs, Kanerika):

| Component | Traditional RAG | Iterative RAG | Agentic RAG |
|-----------|----------------|---------------|-------------|
| **Workflow** | Query → Retrieve → Generate | Query → Retrieve → Reflect → Iterate | Plan → Retrieve → Reflect → Act → Iterate |
| **Planning** | None | Implicit via iteration | Explicit planner/controller agent |
| **Tool Use** | Static retriever | Multiple retrievers | APIs, calculators, code, databases, KG |
| **Memory** | Stateless | Conversation history | Episodic/long-term memory |
| **Multi-Agent** | No | No | Specialized agents (Planner, QA, Verifier) |
| **Self-Correction** | No | Limited | Multi-round correction with backtracking |
| **Reasoning** | Single-step | Multi-hop reasoning | Iterative planning + reflection |

**Key Differentiators of Agentic RAG**:

1. **Tool Integration Beyond Retrieval**
   - Agentic: APIs, calculators, code execution, SQL queries, knowledge graphs
   - Profound: Limited to KB + Web Search (Google Scholar, PubMed, etc.)

2. **Multi-Agent Architecture**
   - Agentic: Specialized agents (Router, Planner, Verifier, Memory)
   - Profound: Single orchestrator (ProfondeChain)

3. **Episodic Memory**
   - Agentic: Maintains reasoning paths, prior queries, intermediate conclusions
   - Profound: Within-session iteration history only

4. **Autonomous Tool Selection**
   - Agentic: Dynamically decides which tools to use
   - Profound: Fixed 3-stage pipeline (Basic → Advanced → Web)

---

#### 4.1.3 The Verdict: Profound is "Agentic-Lite"

**Consensus**: Perspicacité's Profound mode is essentially an **Iterative/Advanced RAG** system with some agentic characteristics, but not "full" Agentic RAG.

```
Traditional RAG ──────► Iterative RAG ──────► Profound RAG ──────► Agentic RAG
(Basic/Advanced)       (Multi-hop)           (Agentic-Lite)        (Full Autonomy)
     │                      │                      │                    │
     │                      │                      │                    │
  Simple              Query refinement      Research planning    Multi-agent
  retrieval           & re-retrieval        & dynamic strategy   tool orchestration
```

**What Profound lacks to be "fully agentic"**:

| Missing Component | Impact | v2 Opportunity |
|------------------|--------|----------------|
| **Diverse Tools** | Only KB + Web search | Add calculators, code exec, SQL, APIs |
| **Multi-Agent** | Single orchestrator | Split into Planner, Researcher, Verifier |
| **Persistent Memory** | Session-only context | Cross-session learning, user preferences |
| **Dynamic Tool Selection** | Fixed 3-stage pipeline | LLM decides which tools to use when |
| **Knowledge Graph** | Flat document storage | Graph RAG with entity relationships |
| **Self-Consistency** | Single reasoning path | Multiple reasoning paths + voting |

---

### 4.2 RAG Modes Taxonomy for v2

Based on expert analysis, here's the refined v2 RAG mode structure:

| Mode | Category | Use Case | Key Characteristics |
|------|----------|----------|-------------------|
| **Quick** ⚡ | Traditional | Fast lookup, FAQ | Single-pass, vector search only |
| **Standard** 🔍 | Traditional | General research | Hybrid search + reranking |
| **Advanced** 🧠 | Iterative | Complex queries | Query expansion + SW-RRF + multi-hop |
| **Deep (Profound v2)** 🔬 | Agentic-Lite | Scientific research | Research planning + iterative + web search |
| **Agentic** 🤖 | Full Agentic | Open-ended tasks | Multi-agent + diverse tools + KG + memory |
| **Citation** 📑 | Specialized | Literature reviews | Citation network analysis + bibliography |

---

### 4.3 Deep (Profound v2) - Refined Architecture

Rename "Profound" to "Deep" with these improvements:

```python
class DeepRAG:
    """
    Agentic-Lite research mode for scientific literature.
    
    Improvements over v1 Profound:
    - Integration with SciLEx for unified search
    - Citation network awareness
    - Better memory of intermediate conclusions
    - Optional tool use (calculator, code) via MCP
    """
    
    capabilities = [
        "research_planning",      # Multi-step plan generation
        "iterative_retrieval",    # Up to N cycles with evaluation
        "quality_assessment",     # Document relevance scoring
        "plan_adaptation",        # Dynamic plan adjustment
        "early_exit",            # Confidence-based termination
        "citation_tracking",     # Track paper relationships
        "limitation_detection",  # Detect unanswerable questions
    ]
```

---

### 4.4 Full Agentic RAG (v2.1 or v3) with Toolomics

Reserve "Agentic" mode for true multi-agent architecture using Toolomics as the tool layer:

```python
class AgenticRAG:
    """
    Full agentic research system with specialized agents.
    Toolomics provides the actual tool execution (MCP servers).
    """
    
    agents: List[Agent] = [
        RouterAgent(),      # Routes to appropriate specialist
        PlannerAgent(),     # Creates research plans
        ResearcherAgent(),  # Performs retrievals (KB, Web, APIs)
        VerifierAgent(),    # Self-consistency checking
        MemoryAgent(),      # Long-term memory management
    ]
    
    # Tools are provided by Toolomics MCP servers
    toolomics_tools: List[MCPTool] = [
        # From Toolomics (/home/tjiang/repos/Mimosa_project/toolomics)
        "graphrag_query",     # GraphRAG MCP - Knowledge graph queries
        "sibils_search",      # SIBiLS MCP - PubMed/text mining
        "sibils_annotations", # SIBiLS MCP - Entity extraction
        "browser_search",     # Browser MCP - Web search via SearXNG
        "browser_download",   # Browser MCP - File download
        "python_execute",     # Python Editor MCP - Code execution
        "pdf_extract",        # PDF MCP - Text extraction
        "shell_command",      # Shell MCP - System commands
        "rscript_execute",    # Rscript MCP - Statistical analysis
    ]
    
    # Internal tools (SciLEx integration)
    internal_tools: List[Tool] = [
        "scilex_search",      # SciLEx 10-API collection
        "scilex_aggregate",   # Deduplication & filtering
        "kb_vector_search",   # FAISS/Chroma/Pinecone
        "citation_analysis",  # CrossRef/OpenCitations
    ]
```

**Agentic Workflow with Toolomics:**

```
User Query: "Analyze the relationship between metabolite X and disease Y"

Step 1: Planner Agent creates research plan
├── Sub-task A: Search for metabolite X in literature
├── Sub-task B: Search for disease Y associations
├── Sub-task C: Find papers mentioning both
└── Sub-task D: Analyze statistical relationships

Step 2: Router Agent delegates to tools
├── Call Toolomics/SIBiLS: search_paper("metabolite X")
├── Call Toolomics/SIBiLS: search_paper("disease Y")
├── Call Toolomics/GraphRAG: query_kg("X → related_to → Y")
├── Call SciLEx: batch_collect("metabolite X disease Y")
└── Call Toolomics/Python: analyze_citations()

Step 3: Researcher Agent synthesizes findings
├── Integrate results from all tools
├── Identify gaps in information
└── Request additional searches if needed

Step 4: Verifier Agent checks quality
├── Cross-reference findings
├── Verify citations exist
└── Check for contradictions

Step 5: Memory Agent updates state
├── Store research path
├── Cache intermediate results
└── Update user preferences
```

---

### 4.5 Summary: Profound vs Agentic

| Aspect | Profound (v1) | Deep (v2) | Agentic (v2.1+) |
|--------|--------------|-----------|-----------------|
| **Architecture** | Single chain | Single chain + SciLEx | Multi-agent system |
| **Planning** | ✅ Research plans | ✅ Enhanced planning | ✅ Specialized planner agent |
| **Iteration** | ✅ Up to 5 cycles | ✅ Configurable cycles | ✅ Dynamic iteration |
| **Reflection** | ✅ Doc analysis | ✅ Better assessment | ✅ Multi-agent verification |
| **Tools** | KB + 4 web searchers | + SciLEx 10 APIs | + Calc, Code, SQL, KG |
| **Memory** | Session-only | Cross-session | Episodic + semantic |
| **Citation Networks** | ❌ | ✅ Via SciLEx | ✅ Full graph analysis |
| **Self-Correction** | Plan adjustment | Better backtracking | Multi-round correction |
| **Multi-Agent** | ❌ | ❌ | ✅ Specialized agents |

**Recommendation for v2**:
1. Rename "Profound" → "Deep" (clearer naming)
2. Integrate SciLEx as the web search backend
3. Add citation network awareness
4. Keep "Agentic" as a future v2.1+ mode with full multi-agent architecture

---

## 5. New Capabilities for v2

### 5.1 Multi-Modal Support
- [ ] Process figures, tables, charts from PDFs
- [ ] Support for video content (YouTube lectures, conference talks)
- [ ] Image understanding for scientific diagrams

### 5.2 Citation Network Analysis
- [ ] Build citation graphs from retrieved papers
- [ ] Identify seminal papers (high centrality)
- [ ] Suggest related work through co-citation analysis
- [ ] Visual exploration of paper relationships

### 5.3 Research Session Management
```python
@dataclass
class ResearchSession:
    session_id: str
    knowledge_bases: List[str]
    conversation_history: List[Message]
    retrieved_documents: DocumentGraph  # Not just flat list
    hypotheses: List[Hypothesis]        # Track evolving understanding
    confidence_over_time: List[float]   # Research progress metric
```

### 5.4 Collaborative Features
- [ ] Share research sessions between users
- [ ] Annotations on documents
- [ ] Peer review mode (challenge/suggest additions)

### 5.5 Evaluation & Benchmarking
- [ ] Built-in evaluation framework for RAG performance
- [ ] A/B testing different retrieval strategies
- [ ] Automatic benchmark dataset creation from user feedback

### 5.6 LLM-Independent Embeddings
- [ ] Support for local embedding models (SentenceTransformers, BGE, etc.)
- [ ] Multi-modal embeddings (CLIP for images)
- [ ] Custom fine-tuned embeddings for specific domains

### 5.7 Agent Swarm Research Mode (v2.2+)

**Status:** Planned  
**Skill Reference:** `/home/tjiang/.config/agents/skills/agent-swarm/SKILL.md`

#### Vision
Implement true multi-agent research using Kimi Code CLI's agent-swarm patterns. Multiple specialized agents collaborate in parallel to conduct comprehensive literature research.

#### Architecture

```
User Query
    ↓
[Research Coordinator Agent]
    ↓
    ├── [Literature Search Agent] → SciLEx APIs (PubMed, Scholar, etc.)
    ├── [Document Analysis Agent] → PDF parsing & chunking
    ├── [Citation Network Agent] → CrossRef/OpenCitations
    ├── [Synthesis Agent] → Answer generation & summarization
    └── [Verifier Agent] → Quality check & fact validation
    ↓
[Integration & Verification]
    ↓
Final Answer with Citations
```

#### Swarm Patterns to Implement

**1. Coordinator-Worker Pattern**
- Coordinator analyzes query and delegates subtasks
- Workers research in parallel across different sources
- Results aggregated and synthesized

**2. Maker-Checker Pattern (for quality)**
- Maker agent generates answer draft
- Checker agent verifies completeness & accuracy
- Iterate until quality threshold met (max 3 iterations)

**3. Fan-out/Fan-in (for large KBs)**
- Search multiple knowledge bases in parallel
- Merge and re-rank results from all sources

#### Files to Create
```
src/perspicacite/rag/agents/
├── __init__.py
├── coordinator.py          # Main orchestrator (CoordinatorAgent)
├── researcher.py           # Literature search agent (ResearcherAgent)
├── analyst.py              # Document analysis agent (AnalystAgent)
├── synthesizer.py          # Answer synthesis agent (SynthesizerAgent)
├── verifier.py             # Quality check agent (VerifierAgent)
└── swarm_engine.py         # Swarm orchestration logic
```

#### Integration Points

| Swarm Component | Perspicacité Integration | Pattern |
|-----------------|-------------------------|---------|
| Research Coordinator | New `CoordinatorAgent` class | Coordinator-Worker |
| Literature Searcher | SciLEx adapter + Web search tools | Fan-out/Fan-in |
| Document Analyst | PDF parser + Chunking pipeline | Worker |
| Citation Analyst | CitationNetworkTool | Worker |
| Synthesizer | LLM client with context building | Maker |
| Verifier | New verification layer | Checker |

#### New RAG Mode: "Swarm"

```python
class SwarmRAGMode(BaseRAGMode):
    """
    Multi-agent research mode using agent-swarm patterns.
    
    Agents:
    - Coordinator: Decomposes query, assigns subtasks
    - Researcher: Parallel searches across multiple sources
    - Analyst: Extracts key information from documents
    - Synthesizer: Combines findings into coherent answer
    - Verifier: Checks accuracy and completeness
    """
```

#### Configuration

```yaml
rag_modes:
  swarm:
    enabled: true
    max_agents: 5
    patterns:
      - "coordinator-worker"
      - "maker-checker"
    agents:
      coordinator:
        model: "claude-3-opus"  # High reasoning capability
      researcher:
        max_parallel_searches: 3
      verifier:
        max_iterations: 3
        quality_threshold: 0.9
```

#### Benefits
- **Parallel research** - Multiple sources searched simultaneously
- **Quality assurance** - Verification layer catches errors
- **Scalability** - Easy to add new specialized agents
- **Transparency** - Each agent's contribution visible in stream events

---

## 6. Technology Stack Recommendations

| Component | Current (v1) | Proposed (v2) |
|-----------|-------------|---------------|
| **Backend** | FastAPI | Keep FastAPI, add asyncpg |
| **Vector DB** | FAISS | Multi-backend abstraction |
| **Frontend** | React + Vite | Keep, add TanStack Query |
| **Session Store** | None | Redis or PostgreSQL |
| **Task Queue** | None | Celery or RQ |
| **Deployment** | Docker + conda | Docker + `uv` |
| **Package Mgmt** | pip | `uv` with lockfiles |

---

## 7. BibTeX Workflow Considerations

### 7.1 Current Gap: PDF URL Handling

**Problem**: Perspicacité v1 only handles local PDF paths

```python
# bibtex_processor.py line 155 - Only checks local files!
if chunk.lower().endswith('.pdf') and os.path.exists(chunk):
```

SciLEx exports `file = {https://...pdf}` URLs, but Perspicacité ignores them.

**v2 Fix Needed**:
- Detect URL vs local path in `file` field
- Download from URLs when `pdf_url` is provided
- Fall back to Unpaywall only when needed

### 7.2 Metadata Enrichment

SciLEx exports fields Perspicacité v1 doesn't use:

| SciLEx Field | Contains | v2 Opportunity |
|--------------|----------|----------------|
| `archiveprefix` | Source API | Prioritize/weight sources |
| `note` | HuggingFace URL | Trigger HF model parsing |
| `howpublished` | GitHub repo URL | Trigger GitHub parser |
| `keywords` | HF tags (TASK:NER;PTM:BERT) | Enrich embeddings |
| `copyright` | License info | Filter OA content |

---

## 8. Open Questions - General

### 8.1 Scope & Approach
- [ ] Is this a refactor of existing code or a clean rewrite?
- [ ] Can v2 drop backward compatibility with v1 KBs?
- [ ] Target timeline for v2 release?
- [ ] Team size: solo or collaborative?

### 8.2 Priorities
- [ ] Which new capabilities are must-have vs nice-to-have?
- [x] **Clarified**: Agentic RAG with Toolomics → v2.1+ (Deep mode for v2)
- [x] **Clarified**: v2 focuses on: SciLEx integration + Deep mode + MCP server
- [ ] Should v2 focus on backend improvements first, or UI refresh?

### 8.3 Integration Decisions
- [ ] Which SciLEx integration option (A, B, or C)?
- [ ] Real-time or batch collection?
- [ ] Unified configuration or separate?

### 8.4 Deployment & Distribution
- [ ] Keep Docker as primary deployment method?
- [ ] Need cloud-native features (K8s, serverless)?
- [ ] Package for PyPI distribution?

---

## 9. Next Steps / Action Items

### Critical (P0)
- [ ] **URGENT**: Restore v1 Profound capabilities to v2 DeepRAG (see `RAG_CRITICAL_ASSESSMENT.md`)
  - [ ] Port `_analyze_documents()` - Document quality assessment
  - [ ] Port `_is_question_answered()` - Early exit logic
  - [ ] Port `_review_and_adjust_plan()` - Plan adaptation
  - [ ] Port `_process_documents()` - Three-stage retrieval with web search
  - [ ] Implement tool orchestration (currently defined but unused)
- [ ] Write comprehensive tests for DeepRAG agentic behavior

### High Priority (P1)
- [ ] Decide on integration approach with SciLEx
- [ ] Define RAG mode requirements
- [ ] Create proof-of-concept for new architecture
- [ ] Evaluate vector store abstraction libraries
- [ ] Design session management schema
- [ ] Define migration path from v1 KBs

### Future (P2)
- [ ] Implement Agent Swarm Research Mode (v2.2+)
- [ ] Integrate Toolomics MCP tools for Agentic RAG
- [ ] Design multi-agent orchestration architecture

---

## 10. Meeting Log

### 2026-03-12 - Initial Discussion
- Reviewed both packages (Perspicacité-AI and SciLEx)
- Identified SciLEx as the collection engine for v2
- Discussed three integration options (Library, CLI, Interface)
- Outlined architecture improvements for v2
- Proposed new RAG modes including Agentic RAG
- Documented open questions for future discussion

### 2026-03-12 - Profound vs Agentic RAG Analysis
- Deep dive into Profound mode implementation (core/profonde.py ~2,400 lines)
- Research on expert definitions of Agentic RAG (TechRxiv, XCube Labs, Kanerika, Gartner)
- **Key Finding**: Profound mode is "Agentic-Lite" - has planning, iteration, reflection, but lacks:
  - Diverse tool integration (only KB + Web search)
  - Multi-agent architecture (single orchestrator)
  - Episodic memory (session-only)
  - Dynamic tool selection (fixed 3-stage pipeline)
- **Conclusion**: 
  - v2: Rename "Profound" → "Deep", integrate SciLEx, add citation awareness
  - v2.1+: True "Agentic" mode with multi-agent architecture
- Updated RAG modes taxonomy: Quick → Standard → Advanced → Deep → Agentic → Citation

### 2026-03-12 - Toolomics Discovery
- Explored `/home/tjiang/repos/Mimosa_project/toolomics`
- **Major Finding**: Existing MCP infrastructure with tools needed for Agentic RAG:
  - `graph_rag/` - Microsoft GraphRAG (knowledge graph RAG)
  - `sibils/` - PubMed/PMC search + text mining (chemicals, diseases, genes)
  - `browser/` - Web browser automation via SearXNG
  - `python_editor/` - Code execution
  - Plus: cheminformatics, PDF processing, Rscript, etc.
- **Implication for v2**: 
  - Toolomics provides the **tool layer** for Agentic RAG
  - Perspicacité v2 should focus on **orchestration** (Agentic RAG controller)
  - Integration pattern: Perspicacité → calls Toolomics MCP servers as tools
  - SciLEx provides literature collection, Toolomics provides tool execution
- **Architecture shift**: Perspicacité becomes the "brain", Toolomics provides the "hands"

### 2026-03-12 - Positioning Clarification
- **Clarified**: Perspicacité is FIRST a standalone literature chatbot
- **Secondary role**: Knowledge provider to Mimosa-AI via MCP protocol
- **Key distinction**:
  - Perspicacité = Specialized literature/knowledge tool
  - Mimosa-AI = Omni-assistant for science (uses Perspicacité as one of many tools)
  - Toolomics = General scientific tools (chemistry, biology, etc.)
- **No overlap**: Perspicacité does NOT duplicate Toolomics - it complements it
- **Integration pattern**: 
  - Standalone: React UI + FastAPI
  - MCP Mode: Exposes `research_literature()`, `search_kb()`, `analyze_citations()` tools
  - Mimosa discovers and uses Perspicacité MCP tools when literature research needed

### 2026-03-12 - MCP Server Mode Decision
- **Decision**: Option C - Both UI and MCP server start simultaneously by default
- **Default behavior**: `python -m perspicacite` starts React UI + MCP server
- **Override flags**:
  - `--no-mcp-server`: UI only (no Mimosa integration)
  - `--no-ui`: MCP only (headless mode for Mimosa-only usage)
- **Rationale**: Maximum flexibility - works standalone AND integrates with Mimosa seamlessly

### 2026-03-12 - Enterprise Scale & Chemical Search Requirements
- **Enterprise use case**: Big international company with millions of internal documents
- **Query efficiency critical**: Need hierarchical search, caching, pre-filtering (not O(n) scan)
- **Current search confirmed**: Hybrid (Vector + BM25) - keep and optimize
- **Chemical search added**: 
  - Molecular fingerprints (Morgan/RDKit)
  - Canonical SMILES matching
  - Substructure search
  - Multi-modal embeddings (text + chemical)
- **Architecture implications**: 
  - Two-stage retrieval (metadata filter → vector search)
  - Separate chemical index alongside text index
  - Cluster-based routing for scalability
  - Redis caching for frequent queries

### 2026-03-12 - Skills vs Memory Understanding
- **Insight**: Skills vs Memory distinction clarified through analogy:
  - **Skills** = Course knowledge (general, reusable, structured)
  - **Memory** = Lab notes / Project notes (specific, contextual, daily accumulation)
  - **Learned Skills** = Lab protocols that became standard (valuable memory → skills)
- **Key realization**: They exist on a **capability continuum**, not binary categories
- **Self-evolving concept**: Memory can be promoted to skills when proven valuable
- **v2 plan**: Start with static skills (Toolomics), add memory layer in v2.1, add conversion pipeline in v2.2
- **Documented**: Section 3.7 added with knowledge evolution pipeline

### 2026-03-12 - Dynamic KB Building with Chroma
- **Major finding**: Chroma enables real-time KB expansion (FAISS is static)
- **New tool added**: `kb_add_papers()` - allows agent to add papers during research
- **Use cases documented**:
  - Incremental research sessions (KB grows as agent finds papers)
  - Citation following (auto-add cited papers)
  - User-provided papers (PDF URLs)
- **Key advantage**: Agent can self-improve knowledge base dynamically
- **Contrast with v1**: v1 FAISS requires offline batch rebuild; v2 Chroma adds incrementally
- **Decision**: Chroma preferred over FAISS for v2 (dynamic > static)

### 2026-03-12 - Search Efficiency Analysis
- **Trade-off acknowledged**: Chroma is 2-5x slower than FAISS for queries
- **Benchmarks documented**: FAISS ~5-10ms vs Chroma ~20-50ms (1M docs)
- **Optimization strategies added**:
  - Query caching (5-min TTL for repeated searches)
  - Batch additions (buffer papers, flush in batches)
  - Parallel PDF processing and embedding generation
  - HNSW index for approximate search at scale
- **Hybrid recommendation**: Chroma for active/dynamic KBs, FAISS for archived/cold storage
- **Enterprise strategy**: Two-tier (Hot: Chroma + Archive: FAISS) with Redis caching
- **Key insight**: Dynamic capability worth the performance cost for interactive research sessions

### 2026-03-17 - CRITICAL: RAG Implementation Assessment
- **CRITICAL FINDING**: v2 DeepRAG is a significant regression from v1 Profound
- **Assessment Document**: `RAG_CRITICAL_ASSESSMENT.md` created
- **Key Issues Identified**:
  - Tools are defined but NEVER used (web search non-functional)
  - No document quality assessment (core agentic feature missing)
  - No early exit logic (only max_iterations)
  - Plan adaptation is a stub (`return current_plan`)
  - No reflection or iteration memory
  - ~14% of v1 capabilities migrated (225 lines vs 1,634 lines)
- **Root Cause**: Architecture modernized but logic not ported from v1
- **Reference Implementation**: `packages_to_use/Perspicacite-AI-release/core/profonde.py`
- **Action Required**: Port v1 logic to v2 async architecture IMMEDIATELY
- **Priority**: P0 - This is the core differentiator of the product

### 2026-03-17 - Agent Swarm Feature Added to Roadmap
- **New Capability**: Agent Swarm Research Mode (v2.2+)
- **Skill Reference**: `/home/tjiang/.config/agents/skills/agent-swarm/SKILL.md`
- **Patterns to Implement**:
  - Coordinator-Worker: Dynamic task distribution
  - Maker-Checker: Quality verification loop
  - Fan-out/Fan-in: Parallel source searching
- **New RAG Mode**: "Swarm" with multi-agent orchestration
- **Files to Create**: `src/perspicacite/rag/agents/` module
- **Agents**: Coordinator, Researcher, Analyst, Synthesizer, Verifier
- **Benefits**: Parallel research, quality assurance, scalability
- **Status**: Planned for v2.2+ (after DeepRAG is fixed)

### 2026-03-12 - Query Accuracy Analysis
- **Clarification**: Pure vector search accuracy is IDENTICAL (same embeddings)
- **Chroma advantages for accuracy**:
  - Metadata pre-filtering (eliminate false positives)
  - Hybrid search (vector + BM25 for exact keyword matches)
  - Integrated re-ranking (cross-encoder for better final ranking)
- **Expected metrics improvement**:
  - FAISS (vector only): Recall@10=0.72, Precision@10=0.65
  - Chroma hybrid + filter: Recall@10=0.78, Precision@10=0.82
  - + Cross-encoder rerank: Recall@10=0.85, Precision@10=0.88
- **Two-stage retrieval designed**:
  1. Pre-filter with metadata (narrow candidate pool)
  2. Hybrid retrieve (over-fetch candidates)
  3. Cross-encoder rerank (precise final ranking)
- **Key insight**: For research accuracy > speed. 50ms with RIGHT result > 5ms with WRONG result

### 2026-03-12 - Chemical Embeddings Technical Analysis
- **Core problem**: Binary fingerprints (2048 bits) vs float embeddings (384-1536 dims) are incompatible
  - FAISS and Chroma only support float32 vectors
  - Morgan fingerprints must be converted for storage
- **Three integration strategies documented**:
  1. **Binary-to-float conversion** (recommended): PCA projection 2048-dim → 256-dim dense
  2. **Learned embeddings**: ChemBERTa/MoLFormer (already dense, 768-dim)
  3. **Separate binary index**: FAISS IndexBinaryFlat (Hamming distance, exact Tanimoto)
- **Chroma integration** (recommended for v2):
  - Custom `EmbeddingFunction` for chemical SMILES
  - Two collections: `papers_text` (sentence embeddings) + `papers_chemicals` (fingerprint projections)
  - Re-ranking: Vector search (fast, approximate) → Tanimoto re-ranking (exact)
- **FAISS comparison**:
  - Chroma: Better multi-collection support, built-in metadata, pluggable embedders
  - FAISS: Direct binary index access, but manual multi-index management
- **Technical implementation**:
  - Morgan fingerprint (RDKit) → PCA projection → float32 embedding
  - Tanimoto coefficient calculated post-search for verification
  - Optional RDKit dependency with graceful degradation
- **Performance characteristics**:
  - Strategy A (fingerprint+PCA): Fast, no GPU needed, interpretable
  - Strategy B (ChemBERTa): Slow (NN inference), needs GPU, black-box
  - Strategy C (binary index): Exact Tanimoto, but separate from text index

---

## 6. Configuration Schema Design

### 6.1 Design Principles

1. **Layered Configuration**: Defaults → Config File → Environment Variables → CLI Args
2. **Secrets Separation**: API keys in environment variables, not files
3. **Profile Support**: dev/staging/prod profiles for different environments
4. **Validation**: Pydantic models for type safety and validation
5. **Hot-Reload**: Config changes without restart (where possible)

### 6.2 Recommended Schema (YAML)

```yaml
# config.yml - Main configuration file
# Located at: ~/.config/perspicacite/config.yml (user) or ./config.yml (project)

# ============================================================
# VERSION & METADATA
# ============================================================
version: "2.0.0"  # Config schema version
config_name: "default"  # Profile name

# ============================================================
# SERVER SETTINGS
# ============================================================
server:
  # FastAPI backend
  host: "0.0.0.0"
  port: 5468
  reload: false  # Auto-reload on code changes (dev only)
  
  # MCP Server (optional)
  mcp:
    enabled: true
    host: "0.0.0.0"
    port: 5500
    transport: "stdio"  # "stdio" or "sse"
  
  # CORS settings
  cors:
    origins: ["http://localhost:3000", "http://localhost:5468"]
    credentials: true

# ============================================================
# DATABASE SETTINGS
# ============================================================
database:
  # SQLite for memory (sessions, preferences)
  sqlite:
    path: "~/.local/share/perspicacite/memory.db"
    # Connection pool settings
    pool_size: 5
    max_overflow: 10
  
  # Chroma for knowledge bases
  chroma:
    persist_directory: "~/.local/share/perspicacite/chroma"
    # Chroma server (optional - for external Chroma)
    host: null  # null = embedded mode
    port: null
    
  # Redis (optional - for caching at scale)
  redis:
    enabled: false
    host: "localhost"
    port: 6379
    db: 0
    password: null  # Set via env: PERSPICACITE_REDIS_PASSWORD

# ============================================================
# KNOWLEDGE BASE DEFAULTS
# ============================================================
knowledge_base:
  # Default embedding model
  embedding_model: "text-embedding-3-small"  # OpenAI
  # Alternatives:
  # embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  # embedding_model: " voyage-3"
  
  # Chunking settings
  chunk_size: 1000
  chunk_overlap: 200
  preserve_sections: true
  
  # Search defaults
  default_top_k: 10
  similarity_threshold: 0.7
  
  # Chemical search (if RDKit available)
  chemistry:
    enabled: true
    fingerprint_bits: 2048
    projection_dim: 256
    similarity_metric: "tanimoto"

# ============================================================
# LLM PROVIDER SETTINGS
# ============================================================
llm:
  # Default provider and model
  default_provider: "anthropic"
  default_model: "claude-3-5-sonnet-20241022"
  
  # Provider configurations
  providers:
    anthropic:
      # API key from env: ANTHROPIC_API_KEY
      base_url: "https://api.anthropic.com"
      timeout: 60
      max_retries: 3
      
    openai:
      # API key from env: OPENAI_API_KEY
      base_url: "https://api.openai.com/v1"
      timeout: 60
      max_retries: 3
      
    deepseek:
      # API key from env: DEEPSEEK_API_KEY
      base_url: "https://api.deepseek.com"
      timeout: 60
      max_retries: 3
      
    gemini:
      # API key from env: GOOGLE_API_KEY
      base_url: "https://generativelanguage.googleapis.com"
      timeout: 60
      max_retries: 3
  
  # Context management
  context:
    max_tokens: 8000  # Leave room for response
    chat_history_turns: 10
    summarize_threshold: 20

# ============================================================
# RAG MODE SETTINGS
# ============================================================
rag_modes:
  quick:
    max_iterations: 1
    tools: ["kb_search"]
    max_tokens: 2000
    
  standard:
    max_iterations: 1
    tools: ["kb_search", "web_search"]
    max_tokens: 4000
    
  advanced:
    max_iterations: 2
    tools: ["kb_search", "web_search", "get_citation_network"]
    max_tokens: 6000
    
  deep:
    max_iterations: 5
    tools: ["kb_search", "web_search", "get_citation_network", "fetch_pdf"]
    max_tokens: 8000
    enable_planning: true
    enable_reflection: true
    
  citation:
    max_iterations: 3
    tools: ["kb_search", "web_search", "get_citation_network"]
    max_tokens: 6000
    build_citation_graph: true
    min_citation_depth: 2

# ============================================================
# SCILEX INTEGRATION (10 Academic APIs)
# ============================================================
scilex:
  # API keys come from environment variables
  # Format: SCILEX_<API_NAME>_API_KEY
  
  apis:
    semantic_scholar:
      enabled: true
      rate_limit: 100  # requests per minute
      
    openalex:
      enabled: true
      rate_limit: 100
      
    pubmed:
      enabled: true
      rate_limit: 10  # NCBI requires lower rate
      api_key: null  # Optional: SCILEX_PUBMed_API_KEY for higher limits
      
    ieee:
      enabled: false  # Requires API key
      rate_limit: 100
      
    springer:
      enabled: false  # Requires API key
      rate_limit: 100
      
    elsevier:
      enabled: false  # Requires API key
      rate_limit: 100
      
    arxiv:
      enabled: true
      rate_limit: 100
      
    hal:
      enabled: true
      rate_limit: 100
      
    dblp:
      enabled: true
      rate_limit: 100
      
    istex:
      enabled: true
      rate_limit: 100
  
  # Collection settings
  collection:
    default_max_papers: 100
    quality_threshold: 0.7
    deduplicate: true
    download_pdfs: true

# ============================================================
# PDF PROCESSING
# ============================================================
pdf:
  # Parser selection
  parser: "pymupdf"  # "pymupdf", "pdfplumber", "grobid"
  
  # Extraction settings
  extract_tables: true
  extract_images: false  # Expensive
  extract_equations: false
  
  # Chunking
  respect_section_boundaries: true
  min_chunk_size: 100
  max_chunk_size: 2000

# ============================================================
# WEB SEARCH
# ============================================================
web_search:
  # Search providers (priority order)
  providers:
    - "google_scholar"
    - "semantic_scholar"
    - "pubmed"
    - "openalex"
  
  # Browser settings (for JavaScript-heavy sites)
  browser:
    enabled: false  # Use browser automation
    headless: true
    timeout: 30
  
  # Caching
  cache_ttl: 3600  # 1 hour

# ============================================================
# LOGGING
# ============================================================
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "json"  # "json" (prod) or "text" (dev)
  
  # Output destinations
  outputs:
    - type: "console"
    - type: "file"
      path: "~/.local/share/perspicacite/logs/perspicacite.log"
      rotation: "1 day"
      retention: "30 days"
  
  # Sensitive data masking
  mask_secrets: true

# ============================================================
# USER INTERFACE
# ============================================================
ui:
  # React UI settings
  theme: "system"  # "light", "dark", "system"
  
  # Default view
  default_view: "chat"  # "chat", "search", "library"
  
  # Citation display
  citation_format: "nature"  # "nature", "apa", "mla", "ieee"
  
  # Feature flags
  features:
    enable_citation_network: true
    enable_pdf_preview: true
    enable_export: true
    enable_memory: true

# ============================================================
# SECURITY
# ============================================================
security:
  # API key encryption (at rest)
  encryption:
    enabled: false  # Enable for multi-user deployments
    key_derivation: "pbkdf2"
  
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 60
  
  # Allowed hosts (for production)
  allowed_hosts: ["localhost", "127.0.0.1"]
```

### 6.3 Environment Variables

```bash
# Required: At least one LLM provider
ANTHROPIC_API_KEY="sk-ant-..."
OPENAI_API_KEY="sk-..."
# or DEEPSEEK_API_KEY, GOOGLE_API_KEY

# Optional: SciLEx API keys (for higher rate limits)
SCILEX_IEEE_API_KEY="..."
SCILEX_SPRINGER_API_KEY="..."
SCILEX_ELSEVIER_API_KEY="..."
SCILEX_PUBMED_API_KEY="..."  # NCBI API key

# Optional: External services
PERSPICACITE_REDIS_PASSWORD="..."
PERSPICACITE_ENCRYPTION_KEY="..."

# Development
PERSPICACITE_CONFIG_PATH="./config.dev.yml"
PERSPICACITE_LOG_LEVEL="DEBUG"
```

### 6.4 Pydantic Validation Model

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Literal
from pathlib import Path

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(5468, ge=1024, le=65535)
    reload: bool = False
    
class MCPConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(5500, ge=1024, le=65535)
    transport: Literal["stdio", "sse"] = "stdio"

class DatabaseConfig(BaseModel):
    sqlite_path: Path = Field(default=Path("~/.local/share/perspicacite/memory.db"))
    chroma_path: Path = Field(default=Path("~/.local/share/perspicacite/chroma"))

class LLMProviderConfig(BaseModel):
    base_url: str
    timeout: int = 60
    max_retries: int = 3
    # API key loaded from environment, not stored here

class Config(BaseModel):
    version: str = "2.0.0"
    server: ServerConfig = ServerConfig()
    mcp: MCPConfig = MCPConfig()
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    
    @validator('version')
    def validate_version(cls, v):
        if not v.startswith("2."):
            raise ValueError("Config version must be 2.x")
        return v

# Usage
config = Config.from_yaml("~/.config/perspicacite/config.yml")
config = Config.from_env()  # Override with env vars
```

### 6.5 Profile Support

```bash
# Different profiles for different environments

# Development
perspicacite --profile dev
# Loads: ~/.config/perspicacite/config.dev.yml

# Production
perspicacite --profile prod
# Loads: ~/.config/perspicacite/config.prod.yml

# Or via environment
export PERSPICACITE_PROFILE=prod
perspicacite
```

### 6.6 Configuration Loading Order

```
1. Built-in defaults (code)
   ↓
2. System config (/etc/perspicacite/config.yml) [Linux only]
   ↓
3. User config (~/.config/perspicacite/config.yml)
   ↓
4. Project config (./config.yml) [if exists]
   ↓
5. Profile-specific config (config.{profile}.yml)
   ↓
6. Environment variables (PERSPICACITE_*)
   ↓
7. CLI arguments (--port, --host, etc.)

Later sources override earlier sources.
```

### 6.7 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YAML over JSON** | Comments support, human-readable, standard for configs |
| **Secrets in env vars** | Security - never commit API keys to git |
| **XDG directories** | Follow Linux standard (`~/.config`, `~/.local/share`) |
| **Pydantic validation** | Type safety, auto-completion, clear error messages |
| **Profile support** | Easy switching between dev/prod/test environments |
| **Layered loading** | Flexibility - override only what you need |
| **Hot-reload optional** | Most settings require restart (simpler), some can reload |

---

**Document Owner**: tjiang  
**Last Updated**: 2026-03-12  
**Next Review**: TBD
