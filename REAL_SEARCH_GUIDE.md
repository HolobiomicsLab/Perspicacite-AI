# Real Academic Search Guide

This guide shows you how to perform real academic research using actual APIs.

## Quick Start

### Step 1: Get API Keys

You'll need at least one of these:

| Service | Purpose | Free Tier | Get Key |
|---------|---------|-----------|---------|
| **Anthropic Claude** | LLM | $5 free credits | [console.anthropic.com](https://console.anthropic.com/) |
| **OpenAI** | LLM | $5 free credits | [platform.openai.com](https://platform.openai.com/) |
| **Semantic Scholar** | Paper search | Unlimited (no key) | Free, no signup |

Optional but recommended:
| Service | Purpose | Free Tier | Get Key |
|---------|---------|-----------|---------|
| **SerpAPI** | Google Scholar | 100 searches/month | [serpapi.com](https://serpapi.com/) |
| **Unpaywall** | OA PDFs | Unlimited (needs email) | Just use your email |

### Step 2: Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit with your keys
nano .env  # or use your favorite editor
```

Your `.env` file should look like:

```env
# Required - pick one or both
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
# OR
OPENAI_API_KEY=sk-your-actual-key-here

# Optional - for better results
UNPAYWALL_EMAIL=your-email@example.com
SERPAPI_KEY=your-serpapi-key
```

### Step 3: Run Real Search

```bash
# Check setup
python real_search.py --setup

# Run a search
python real_search.py "How are transformers used in medical imaging?"

# Use Agentic RAG mode
python real_search.py "Your question" --mode agentic

# Run both and compare
python real_search.py "Your question" --mode both
```

## How It Works

### Real Services Used

1. **LLM (Claude/GPT)**: 
   - Creates research plans
   - Assesses document quality
   - Generates answers
   - Costs: ~$0.01-0.05 per search

2. **Semantic Scholar API** (Free):
   - Searches 200M+ papers
   - Returns titles, abstracts, citations
   - No API key needed!

3. **Unpaywall** (Free with email):
   - Finds Open Access PDFs
   - Resolves DOIs to PDF URLs

4. **ChromaDB** (Local):
   - Stores embeddings locally
   - Free, runs on your machine

### Cost Estimates

| Component | Cost per Search |
|-----------|----------------|
| LLM (Claude) | $0.01-0.05 |
| Semantic Scholar | Free |
| Unpaywall | Free |
| ChromaDB | Free (local) |
| **Total** | **~$0.01-0.05** |

## Example Usage

### Basic Search

```bash
$ python real_search.py "CRISPR applications in cancer therapy"

🧠 REAL ACADEMIC SEARCH
======================================================================
🔑 API Key Status:
   Anthropic (Claude): ✅
   OpenAI (GPT):       ❌
   SciLEx:             ❌ (will use fallback)

🤖 LLM configured: anthropic / claude-3-5-sonnet-20241022

🔬 DYNAMIC KB WORKFLOW (REAL)
======================================================================
🎯 Research Query: CRISPR applications in cancer therapy

📡 Step 1: Searching academic databases...
   📚 Using Semantic Scholar API (SciLEx not configured)
   Found 5 papers:
   1. CRISPR-Cas systems in cancer therapy: a comprehensive review... (2024)
   2. Gene editing for cancer immunotherapy... (2023)
   3. CRISPR screens identify therapeutic targets... (2024)
   ...

📥 Step 2: Attempting to fetch full texts...
   ✅ Downloaded via Unpaywall: CRISPR-Cas systems in cancer...
   📄 Using abstract: Gene editing for cancer immunotherapy...
   Retrieved text for 2/5 papers

⚖️  Step 3: Filtering by relevance...
   Selected 3 most relevant papers

🏗️  Step 4: Building dynamic knowledge base...
      📚 Created collection: session_a1b2c3d4
      💾 Added 6 documents

🔍 Step 5: Searching KB and generating answer...
   Retrieved 5 relevant passages

🧹 Step 6: Cleaning up...
      🗑️  Deleted collection: session_a1b4c3d4

📊 RESULTS
======================================================================
Stats: {'llm_calls': 2, 'papers_found': 5, 'papers_downloaded': 1, 'papers_relevant': 3}

Sources:
  - CRISPR-Cas systems in cancer therapy: a comprehensive review...
  - Gene editing for cancer immunotherapy: current status...
  - CRISPR screens identify therapeutic targets in triple-negative...

📤 ANSWER
======================================================================
CRISPR-Cas systems have emerged as powerful tools for cancer therapy...
[Full synthesized answer with citations]
```

### Agentic Mode

```bash
$ python real_search.py "What are the latest advances in protein folding?" --mode agentic

🤖 AGENTIC RAG WORKFLOW (REAL)
======================================================================
🎯 Research Query: What are the latest advances in protein folding?

🔄 Research Cycle 1/2
--------------------------------------------------
   📋 Creating research plan with LLM...
   Created 3 research steps
      1. Investigate AlphaFold and recent structure prediction methods...
      2. Explore applications of predicted structures in drug discovery...
      3. Review limitations and accuracy improvements...

   🔍 Step 1: Investigate AlphaFold and recent structure prediction...
      Found 3 papers

   🔍 Step 2: Explore applications of predicted structures...
      Found 2 papers

   🔍 Step 3: Review limitations and accuracy improvements...
      Found 4 papers

   📊 Evaluating if question is answered...
      ✅ Early exit! (confidence: 0.88)

📝 Generating final answer...

📤 ANSWER
======================================================================
## Recent Advances in Protein Folding

### AlphaFold and Structure Prediction
Recent advances in protein structure prediction have been dominated by...
[Comprehensive answer]
```

## Troubleshooting

### "No API key found"

```bash
# Check your .env file
cat .env | grep -E "ANTHROPIC|OPENAI"

# Should show your key (not empty)
# If empty, edit the file:
nano .env
```

### "Rate limit exceeded"

The free tiers have rate limits:
- Anthropic: ~5 requests/minute on free tier
- OpenAI: ~3 requests/minute on free tier

Solution: Wait a minute between searches, or upgrade to paid tier.

### "No papers found"

Try:
1. Use more general keywords
2. Check your internet connection
3. Try different phrasing

Example:
```bash
# Too specific - might fail
python real_search.py "Transformer architecture with 12 layers for COVID-19 CT"

# Better - more general
python real_search.py "Deep learning for COVID-19 medical imaging"
```

### "PDF extraction failed"

Some PDFs are:
- Scanned images (not text)
- Behind paywalls
- Corrupted

The script automatically falls back to using abstracts.

## Advanced Usage

### Using with Your Own Data

To use real ChromaDB with your own papers:

```python
from real_search import RealVectorStore, RealLLMClient

async def search_my_papers(query: str):
    # Use existing ChromaDB with your papers
    vector_store = RealVectorStore(persist_dir="./my_papers_db")
    llm = RealLLMClient()
    
    # Search your collection
    results = await vector_store.search("my_collection", query)
    
    # Generate answer
    context = "\n".join([r["text"] for r in results])
    answer = await llm.complete([{
        "role": "user",
        "content": f"Answer based on:\n{context}\n\nQuestion: {query}"
    }])
    
    return answer
```

### Customizing Search

Edit `real_search.py` to customize:

```python
# Change number of papers
search_result = await searcher.search(query, topn=10)  # Default: 5

# Change chunk size
chunk_size = 500  # Default: 1000

# Change early exit threshold
early_exit_confidence = 0.90  # Default: 0.85
```

### Adding More Sources

To add arXiv, PubMed, etc.:

```python
async def search_arxiv(self, query: str, topn: int = 5) -> dict:
    """Search arXiv for papers."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": topn,
    }
    # ... implementation
```

Then add to the `RealPaperSearcher` class.

## Privacy & Data

- **LLM calls**: Sent to Anthropic/OpenAI (see their privacy policies)
- **Search queries**: Sent to Semantic Scholar (anonymous)
- **PDFs**: Downloaded from publishers (respect rate limits)
- **Embeddings**: Stored locally on your machine only

## Next Steps

1. **Get API keys** and configure `.env`
2. **Try a search**: `python real_search.py "your question"`
3. **Read the code**: `real_search.py` is well-commented
4. **Customize**: Modify for your specific use case
5. **Scale**: Consider caching, rate limiting for production

## Support

- API Issues: Check status at [status.anthropic.com](https://status.anthropic.com/) or [status.openai.com](https://status.openai.com/)
- Rate Limits: Upgrade to paid tier for higher limits
- Feature Requests: Modify the code - it's open source!

## Comparison: Mock vs Real

| Feature | Mock (`try_it.py`) | Real (`real_search.py`) |
|---------|-------------------|-------------------------|
| Papers | Simulated | Real from Semantic Scholar |
| PDFs | Fake text | Actual downloads |
| LLM | Fake responses | Claude/GPT |
| Vector DB | Simulated | Real ChromaDB |
| Cost | Free | ~$0.01-0.05/search |
| Speed | Instant | 10-30 seconds |
| Use Case | Testing/Demo | Actual research |
