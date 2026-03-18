# Quick Start Guide

Get started with Dynamic KB & Agentic RAG in 30 seconds!

## Prerequisites

```bash
# Make sure you have the virtual environment activated
source .venv/bin/activate

# Or run with uv
uv run python try_it.py
```

## Usage

### 1. List Example Queries

```bash
python try_it.py --list
```

Output:
```
Example queries you can try:

  1. How are transformers used in medical imaging?
  2. What are the benefits of self-attention in radiology?
  3. Compare Vision Transformers to CNNs for pathology
  ...
```

### 2. Run a Quick Demo

```bash
# Run Agentic RAG workflow
python try_it.py --query "How are transformers used in medical imaging?" --mode agentic

# Run Dynamic KB workflow
python try_it.py --query "How are transformers used in medical imaging?" --mode dynamic_kb

# Run both workflows and compare
python try_it.py --query "How are transformers used in medical imaging?" --mode both
```

### 3. Interactive Mode

```bash
# Start interactive demo
python try_it.py
```

Then follow the prompts:
- Enter `1`, `2`, or `3` to use an example query
- Enter `c` to type your own question
- Enter `q` to quit

## Example Output

### Agentic RAG

```
🤖 AGENTIC RAG WORKFLOW
======================================================================
🎯 Research Query: How are transformers used in medical imaging?

🔄 Research Cycle 1/3
--------------------------------------------------
   📋 Creating research plan...
   Created 3 research steps
      1. Investigate transformer architectures...
      2. Compare performance with CNNs...
      3. Review medical applications...

   🔍 Step 1: Investigate transformer architectures...
      Tool: KB Search
      Assessing document quality...
      Quality: 0.85
      Analyzing documents...

      🎯 Checking for early exit...
      ✅ Early exit triggered! (confidence: 0.88)

📊 AGENTIC RAG STATS
======================================================================
  Iterations: 1
  Total steps: 1
  LLM calls: 6
  Early exit: Yes

📤 ANSWER
======================================================================
[Generated answer with findings]
```

### Dynamic KB

```
🔬 DYNAMIC KB WORKFLOW
======================================================================
🎯 Research Query: How are transformers used in medical imaging?

📡 Step 1: Searching SciLEx...
   Found 3 papers:
   1. Vision Transformers for Medical Image Analysis (2024)
   2. Self-Attention Mechanisms Improve Radiology Diagnosis (2023)
   3. Computational Efficiency of Transformers (2024)

📥 Step 2: Downloading PDFs...
   ✅ Downloaded 2 PDFs

⚖️  Step 3: Assessing relevance...
   ✅ 2/3 papers relevant

🏗️  Step 4: Building dynamic knowledge base...
      📚 Created collection: session_abc123
      💾 Added 4 documents

🔍 Step 5: Retrieving from KB...
   Retrieved 2 relevant passages

🧹 Step 6: Cleaning up...
      🗑️  Cleaned up collection

📊 DYNAMIC KB STATS
======================================================================
  Papers found: 3
  Papers relevant: 2
  LLM calls: 1
```

## Try Your Own Questions

The script uses simulated services, so you can ask any research question:

```bash
python try_it.py --query "What is the role of attention mechanisms in NLP?"
python try_it.py --query "Compare RNNs and Transformers for time series"
python try_it.py --query "Explain contrastive learning in computer vision"
```

## Next Steps

1. **Run the demos** to see both workflows in action
2. **Read the implementation** in `src/perspicacite/rag/`
3. **Check the flowcharts** in `WORKFLOW_ASCII.txt`
4. **Integrate into your project** using the real services

## Troubleshooting

```bash
# If you get import errors, make sure venv is activated
source .venv/bin/activate

# If dependencies are missing
uv sync --dev

# For help
python try_it.py --help
```

## What's Happening?

The script simulates the entire workflow:

1. **Simulated SciLEx Search** - Returns mock academic papers
2. **Simulated PDF Download** - Pretends to fetch full texts
3. **Simulated LLM** - Returns structured JSON responses
4. **Simulated Vector Store** - Pretends to store/retrieve embeddings

In production, you would replace these with real services (OpenAI, SciLEx API, ChromaDB, etc.)

## Modify the Script

Open `try_it.py` and:

1. **Replace `SimulatedLLM`** with real LLM client (OpenAI, Anthropic, etc.)
2. **Replace `SimulatedSciLexSearcher`** with actual SciLEx API
3. **Replace `SimulatedVectorStore`** with real ChromaDB instance
4. **Keep the workflow logic** - that's the core value!

See `TRY_IT_GUIDE.md` for full integration instructions.
