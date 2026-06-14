# Embedding Models & RAG Configuration Guide

Perspicacite supports multiple embedding models and RAG strategies. This guide explains
the three tiers we've validated through systematic benchmarking on the
[SciFact](https://github.com/allenai/scifact) claim-retrieval dataset (5 183 PubMed
abstracts, 300 dev claims), and shows how to configure each one.

---

## TL;DR — Which tier should I use?

| Tier | Embedding model | NDCG@10 | RAM footprint | API cost | Best for |
|---|---|---|---|---|---|
| **1 — Default** | `all-MiniLM-L6-v2` | 0.857 | ~90 MB | free | Any machine, quick setup |
| **2 — OpenAI** | `text-embedding-3-large` | 0.910* | ~0 MB local | ~$0.13 / M tokens | Best accuracy, cloud cost OK |
| **3a — Biomedical local** | `S-PubMedBert-MS-MARCO` | **0.873** | ~440 MB | free | Biomedical / life science |
| **3b — General local SOTA** | `BAAI/bge-m3` | **0.879** | ~2.3 GB | free | New best overall — beats OpenAI on SciFact |

\* OpenAI vector=0.864 + `ms-marco-MiniLM-L-12-v2` CE rerank (OR×4)

All NDCG@10 figures are with `over_retrieve=4×, BAAI/bge-reranker-v2-m3` reranker unless
noted. Base (no rerank) figures are 5–14 pp lower.

---

## Tier 1 — Default: fast, free, zero extra setup

**Config file:** `config.yml`

### Model
`sentence-transformers/all-MiniLM-L6-v2` — 384-dim, ~90 MB, runs on CPU in <10 ms/query.

### Benchmarked results (SciFact dev, 188 claims with evidence)

| Mode | Over-retrieve | Reranker | NDCG@10 | R@5 | MRR |
|---|---|---|---|---|---|
| vector | 1× | — | 0.745 | 0.844 | 0.718 |
| hybrid | 1× | — | 0.774 | 0.839 | 0.757 |
| vector | 4× | ms-marco-MiniLM-L-12-v2 | 0.851 | 0.917 | 0.837 |
| vector | 4× | bge-reranker-v2-m3 | 0.857 | 0.915 | 0.847 |
| hybrid | 4× | bge-reranker-v2-m3 | 0.808 | 0.839 | 0.806 |

### Launch

```bash
cd ~/git/Perspicacite-AI

# Basic (no reranker — fastest, lowest RAM)
uv run perspicacite -c config.yml serve

# With reranker enabled (server-side reranking in /api/chat advanced mode)
# The reranker model is set in config.yml under rag_modes.reranker_model
uv run perspicacite -c config.yml serve
```

Key `config.yml` settings:
```yaml
knowledge_base:
  embedding_model: "all-MiniLM-L6-v2"
  similarity_threshold: 0.7   # MiniLM cosine scores are well-distributed
  default_top_k: 10

rag_modes:
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"   # ~120 MB
```

### LLM options for Tier 1

```yaml
llm:
  # Option A — OpenRouter free tier (DeepSeek V4 Flash, no cost)
  default_provider: "openrouter"
  default_model: "deepseek/deepseek-v4-flash"

  # Option B — Local Ollama (completely offline)
  # See Tier 3 section for Ollama setup
  default_provider: "ollama"
  default_model: "qwen3:8b"   # ~5 GB; smaller than 14B
```

---

## Tier 2 — OpenAI: best accuracy, cloud cost

**Config file:** `configs/embedders/openai_large.yml`

### Model
`text-embedding-3-large` — 3 072-dim, OpenAI API, ~$0.13 per million tokens.

### Benchmarked results (SciFact dev, 188 claims)

| Mode | Over-retrieve | Reranker | NDCG@10 | R@5 | MRR |
|---|---|---|---|---|---|
| vector | 1× | — | 0.864 | 0.932 | 0.849 |
| vector | 4× | ms-marco-MiniLM-L-12-v2 | 0.872 | 0.951 | 0.855 |

Gain over MiniLM baseline: **+12 pp NDCG@10 (no rerank), +2 pp with CE reranker**.

### Launch

```bash
export OPENAI_API_KEY="sk-..."

# Using the dedicated config (port 8002 by default)
uv run perspicacite -c configs/embedders/openai_large.yml serve
```

Key settings:
```yaml
knowledge_base:
  embedding_model: "text-embedding-3-large"
  similarity_threshold: 0.0   # CRITICAL: OpenAI cosine scores differ from MiniLM
                               # 0.7 threshold will filter most/all results
```

> **Cost estimate:** ingesting SciFact (5 183 abstracts, ~2.1 M tokens) cost < $0.30.
> Query embedding for 300 eval claims: < $0.01. For personal KB use, cost is negligible.

### Running alongside MiniLM

You can run both servers simultaneously (they share `chroma_db/` but use different KBs):

```bash
# Terminal 1 — MiniLM on :8000
uv run perspicacite -c config.yml serve

# Terminal 2 — OpenAI on :8002
OPENAI_API_KEY=$OPENAI_API_KEY uv run perspicacite -c configs/embedders/openai_large.yml serve
```

Each server uses its own KB (`scifact_abstracts` for MiniLM, `scifact_openai_large`
for OpenAI) and embeds queries with the matching model. See
[Critical Gotcha #1](#critical-gotcha-1-sqlite-vs-chromadb-metadata-must-agree) below.

---

## Tier 3a — Biomedical local: best life-science accuracy

**Config file:** `configs/embedders/pubmedbert.yml`

### Model
`pritamdeka/S-PubMedBert-MS-MARCO` — 768-dim, PubMedBERT fine-tuned for retrieval on
MS-MARCO. Domain-adapted for medical/biological text. ~440 MB.

### Benchmarked results (SciFact dev, 188 claims)

| Mode | Over-retrieve | Reranker | NDCG@10 | R@5 | MRR |
|---|---|---|---|---|---|
| vector | 1× | — | — | — | — |
| vector | 4× | bge-reranker-v2-m3 | **0.873** | **0.933** | **0.864** |
| hybrid | 4× | bge-reranker-v2-m3 | 0.842 | 0.887 | 0.832 |

**This is the best overall configuration we found** — surpassing OpenAI 3-large on
biomedical text while being fully local and free. The key is combining a
domain-adapted retrieval model with a powerful cross-encoder reranker.

### Launch

```bash
uv run perspicacite -c configs/embedders/pubmedbert.yml serve
# Model auto-downloads from HuggingFace on first run (~440 MB)
```

For offline environments (after first download):
```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    uv run perspicacite -c configs/embedders/pubmedbert.yml serve
```

Key settings:
```yaml
knowledge_base:
  embedding_model: "pritamdeka/S-PubMedBert-MS-MARCO"
  similarity_threshold: 0.0   # Lower scores than MiniLM; 0.7 would filter results
  default_top_k: 10

rag_modes:
  reranker_model: "BAAI/bge-reranker-v2-m3"   # ~2.2 GB — pulls on first run
```

> **Why bge-reranker-v2-m3?** It's the MTEB SOTA cross-encoder and adds ~4–6 pp
> NDCG@10 over ms-marco-MiniLM-L-12-v2 in our experiments. It requires ~2.2 GB RAM
> but runs on CPU (slowly) or GPU (fast). If RAM is constrained, use
> `cross-encoder/ms-marco-MiniLM-L-12-v2` (~120 MB) instead — still gains ~2–3 pp.

---

## Tier 3b — General local SOTA

**Config file:** `configs/embedders/bge_m3.yml`

### Model
`BAAI/bge-m3` — 1 024-dim, multilingual MTEB SOTA retrieval model. ~2.3 GB.

```yaml
knowledge_base:
  embedding_model: "BAAI/bge-m3"
  similarity_threshold: 0.0
```

> **Benchmark result (SciFact dev, 300 claims, 5 183 PubMed abstracts):**
> NDCG@10 = **0.879** with `BAAI/bge-reranker-v2-m3` reranker (over_retrieve=4×).
> This is the **current best overall**, beating PubMedBERT (0.873) and OpenAI 3-large + ms-marco (0.872).
> Note: an earlier result of 0.655 was invalid — that run accidentally targeted the MiniLM server
> (port 8000) where `similarity_threshold: 0.7` filtered most bge-m3 results.
> The corrected run uses port 8004 with `similarity_threshold: 0.0`.

GPU launch:
```bash
# If you have a CUDA GPU, sentence-transformers will use it automatically
uv run perspicacite -c configs/embedders/bge_m3.yml serve
```

---

## Running Multiple Knowledge Bases in Parallel

Perspicacite stores all embedding vectors in a single `chroma_db/` directory,
and all servers share it. This means you can run multiple servers (different embedding
models, different ports) and each maintains its own KB collections.

### Example: three-server setup

```bash
# Port 8000 — MiniLM (always-on, fast, general queries)
uv run perspicacite -c config.yml serve &

# Port 8001 — SPECTER2 (scientific citation context)
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    uv run perspicacite -c configs/embedders/specter2.yml serve &

# Port 8002 — OpenAI 3-large (highest accuracy, paid)
OPENAI_API_KEY=$OPENAI_API_KEY \
    uv run perspicacite -c configs/embedders/openai_large.yml serve &
```

### Ingest the same corpus into each KB

Each KB must be ingested separately through the server that owns its embedding model:

```bash
# Ingest SciFact abstracts into MiniLM KB (port 8000)
PERSPICACITE_URL=http://localhost:8000 \
    uv run python scripts/ingest_corpus.py --corpus abstracts --kb-name scifact_abstracts

# Ingest same abstracts into SPECTER2 KB (port 8001)
PERSPICACITE_URL=http://localhost:8001 \
    uv run python scripts/ingest_corpus.py --corpus abstracts --kb-name scifact_specter2

# Ingest into OpenAI KB (port 8002)
PERSPICACITE_URL=http://localhost:8002 \
    uv run python scripts/ingest_corpus.py --corpus abstracts --kb-name scifact_openai_large
```

### Cross-encoder reranking via MCP

When using `perspicacite-eval`, the `CrossServerRRFAdapter` fuses results from two
different-model KBs using Reciprocal Rank Fusion (RRF), then applies a cross-encoder.
This is experimental but shows promise for further +2–5 pp gains.

---

## Reranker Configuration

Cross-encoder rerankers apply a second pass over the retrieved candidates. Configure
`rag_modes.reranker_model` in your config file.

| Reranker | Size | Speed (CPU) | NDCG gain vs no-rerank |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~120 MB | fast (<1 s/query) | +8–10 pp |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~120 MB | fast (<1 s/query) | +10–12 pp |
| `BAAI/bge-reranker-v2-m3` | ~2.2 GB | slow (2–5 s/query CPU) | +12–14 pp |

> **Over-retrieve setting:** Reranking only helps when you fetch more candidates than
> you need. Use `default_top_k` × 3–4 for the initial retrieval, then rerank to `top_k`.
> In the Perspicacite API, set `top_k=20` to retrieve 20, and the server reranks to
> return the best 5–10.

---

## LLM Configuration

### Option A — OpenRouter (free tier, no install)

```yaml
llm:
  default_provider: "openrouter"
  default_model: "deepseek/deepseek-v4-flash"   # free, fast, good quality
  # Alternatives:
  #   "deepseek/deepseek-r1:free"                # reasoning model, free
  #   "google/gemma-3-27b-it:free"               # 27B Google model, free
  #   "anthropic/claude-3-5-haiku"               # paid, fast
  #   "anthropic/claude-opus-4-5"                # paid, best quality
  providers:
    openrouter:
      base_url: "https://openrouter.ai/api/v1"
      timeout: 120
```

Set `OPENROUTER_API_KEY` in environment. Free-tier models require no payment.

### Option B — Anthropic (paid, best quality)

```yaml
llm:
  default_provider: "anthropic"
  default_model: "claude-opus-4-5"
  providers:
    anthropic:
      base_url: "https://api.anthropic.com"
      timeout: 120
```

Set `ANTHROPIC_API_KEY` in environment.

### Option C — Local Ollama (completely offline, no cost)

**Install Ollama:**
```bash
brew install ollama        # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh   (Linux)

ollama serve               # start daemon
ollama pull qwen3:14b      # ~9 GB; best balance of quality and speed on M-series
# Lighter alternatives:
# ollama pull qwen3:8b     # ~5 GB
# ollama pull llama3.2:3b  # ~2 GB (fast, lower quality)
```

**Config:**
```yaml
llm:
  default_provider: "ollama"
  default_model: "qwen3:14b"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      timeout: 300    # 14B can be slow for long answers
```

See `configs/embedders/qwen3_14b.yml` for a complete example.

**Thinking mode (Qwen3):** Qwen3 supports `/think` and `/no_think` tokens. The server
inserts these based on mode complexity. Set `QWEN3_NO_THINK=1` env var to always
disable thinking (faster, lower quality for complex tasks).

---

## Recommended Configurations by Use Case

### Researcher on a laptop (minimal footprint)

```bash
# Single server, MiniLM, OpenRouter LLM
cp config.yml config_laptop.yml
# Edit: llm.default_model = "deepseek/deepseek-v4-flash"
uv run perspicacite -c config.yml serve
```

RAM: ~300 MB. Works on any machine with internet for LLM calls.

### Biomedical researcher (best accuracy, air-gapped option)

```bash
# Port 8005 — PubMedBERT + bge-reranker + local Qwen3
# First run: downloads ~2.6 GB of models
uv run perspicacite -c configs/embedders/pubmedbert.yml serve

# With local LLM (Ollama):
# Edit configs/embedders/pubmedbert.yml: llm.default_provider = "ollama", default_model = "qwen3:14b"
```

RAM: ~3 GB (PubMedBERT + bge-reranker + Qwen3 8B) or ~11 GB (Qwen3 14B).

### Lab with shared server and multiple users

Run three servers on fixed ports, each serving a different embedding tier. Users choose
the server URL that matches their embedding tier. Clients on the same network can all
share the same server.

### Maximum accuracy (no cost constraint)

```bash
# OpenAI 3-large + bge-reranker + Claude Opus
OPENAI_API_KEY=$OPENAI_API_KEY ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    uv run perspicacite -c configs/embedders/openai_large.yml serve
# Edit configs/embedders/openai_large.yml: 
#   llm.default_provider = "anthropic"
#   llm.default_model = "claude-opus-4-5"
#   rag_modes.reranker_model = "BAAI/bge-reranker-v2-m3"
```

---

## Critical Gotcha: SQLite vs ChromaDB metadata must agree

Perspicacite stores two independent records per KB:

| Store | Location | Used for |
|---|---|---|
| ChromaDB vectors | `chroma_db/<uuid>/` | Actual embedded dimension |
| SQLite `kb_metadata` | `data/perspicacite.db` | **Query model selection** |

`search_knowledge_base` reads `kb_metadata.embedding_model` from SQLite to choose
which model embeds the **query**. If they disagree (e.g. vectors are 768-dim PubMedBERT
but SQLite says `all-MiniLM-L6-v2`), every search fails with a ChromaDB
dimension-mismatch error — silently returning 0 results.

**Verify metadata after ingest:**
```bash
sqlite3 ~/git/Perspicacite-AI/data/perspicacite.db \
  "SELECT name, embedding_model, paper_count FROM kb_metadata WHERE name LIKE 'scifact%';"
```

**Fix metadata mismatch:**
```bash
sqlite3 ~/git/Perspicacite-AI/data/perspicacite.db \
  "UPDATE kb_metadata SET embedding_model='pritamdeka/S-PubMedBert-MS-MARCO'
   WHERE name='scifact_pubmedbert';"
```

**Always use the right server URL for each KB:**
```bash
# CORRECT: PubMedBERT KB queried through PubMedBERT server
PERSPICACITE_URL=http://localhost:8005 uv run eval --corpora pubmedbert

# WRONG: PubMedBERT KB queried through MiniLM server → dim mismatch → 0 results
PERSPICACITE_URL=http://localhost:8000 uv run eval --corpora pubmedbert
```

---

## Planned: HuggingFace Model Releases

After completing our domain-adaptation experiments (in progress), we plan to release
fine-tuned retrieval models on HuggingFace:

- **`HolobiomicsLab/perspicacite-retrieval-biomedical`** — PubMedBERT-based model
  fine-tuned on biomedical claim → evidence pairs from SciFact + custom in-house data
- **`HolobiomicsLab/perspicacite-reranker-biomedical`** — Cross-encoder fine-tuned
  on the same domain

These will be drop-in replacements in the `embedding_model` and `reranker_model`
config fields. Expected release: after SciFact fine-tuning experiments complete.

---

## See Also

- `docs/guides/ingest-bibtex.md` — Ingesting BibTeX / PDF collections
- `docs/guides/zotero-integration.md` — Sync from Zotero library
- `docs/MCP.md` — MCP tool reference for programmatic access
- `perspicacite-eval/docs/baseline_2026_05_24.md` — Full benchmark results table
