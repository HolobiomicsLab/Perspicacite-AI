# Embedding Model & Reranker Selection Guide

This document records the empirical basis for the default embedding model and reranker
choices in each Perspicacité configuration file.

Full benchmark data: [perspicacite-eval/docs/retrieval_benchmark_2026_05_26.md](https://github.com/HolobiomicsLab/perspicacite-eval/blob/main/docs/retrieval_benchmark_2026_05_26.md)

---

## Quick Reference

| Config file | Embedding | Dims | Reranker | NDCG@10 (SciFact) | Use when |
|---|---|---|---|---|---|
| `config.yml` | all-MiniLM-L6-v2 | 384 | ms-marco-MiniLM-L-12-v2 | **0.851** | Dev, resource-constrained, fast setup |
| `configs/embedders/bge_m3.yml` | BAAI/bge-m3 | 1024 | bge-reranker-v2-m3 | **0.879** | Production biomedical (recommended) |
| `configs/embedders/openai_large.yml` | text-embedding-3-large | 3072 | ms-marco-MiniLM-L-12-v2 | **0.872** | Cross-domain, best generalisation |

---

## Evaluation Protocol

All scores are from a controlled benchmark campaign (2026-05-26) using:

- **Dataset**: SciFact dev set — 5,183 PubMed abstracts, 188 non-NEI claims (primary),
  300 all-claim BEIR variant (extended), BEIR/NFCorpus 323 nutrition queries (cross-domain)
- **Metric**: NDCG@10, Recall@5, MRR on the gold `cited_doc_ids`
- **Retrieval**: vector mode, over-retrieve 4× (fetch 20 docs), cross-encoder rerank, evaluate at k=5
- **System**: Perspicacité v2 + perspicacite-eval harness (open source)

---

## Model Comparison — SciFact Biomedical (188 claims)

| Embedding | Dims | Reranker | NDCG@10 | R@5 | MRR |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | ms-marco-L-12 | 0.851 | 0.917 | 0.837 |
| all-MiniLM-L6-v2 | 384 | bge-reranker-v2-m3 | 0.857 | 0.915 | 0.847 |
| **BAAI/bge-m3** | **1024** | **bge-reranker-v2-m3** | **0.879** | **0.936** | **0.870** |
| text-embedding-3-large (OpenAI) | 3072 | ms-marco-L-12 | 0.872 | 0.951 | 0.855 |
| text-embedding-3-large (OpenAI) | 3072 | bge-reranker-v2-m3 | 0.851 ⚠️ | 0.915 | 0.837 |
| NeuML/pubmedbert-base-embeddings | 768 | bge-reranker-v2-m3 | 0.875 | 0.932 | 0.865 |
| CrossServerRRF (PubMedBERT+OpenAI) | 768+3072 | bge-reranker-v2-m3 | 0.881 | 0.942 | 0.868 |

---

## Cross-Domain Comparison — BEIR/NFCorpus (323 nutrition queries)

| Embedding | NDCG@10 | Notes |
|---|---|---|
| all-MiniLM-L6-v2 | 0.242 | Poor generalisation |
| BAAI/bge-m3 | 0.249 | Biomedical gains don't transfer (+0.7 pp over MiniLM) |
| **text-embedding-3-large** | **0.327** | **Best generaliser (+7.8 pp over bge-m3)** |

---

## The Reranker Pairing Rule

**This is the most important practical finding of the benchmark.**

```
Weak / domain-specific embedding  →  use bge-reranker-v2-m3  (aggressive, helps weak pools)
Strong / general embedding         →  use ms-marco-MiniLM-L-12-v2  (conservative, preserves quality)
```

**Evidence:** bge-reranker-v2-m3 applied to OpenAI text-embedding-3-large scores
**0.851 NDCG** — worse than ms-marco (0.872) AND worse than no reranker (0.864).
The reranker actively demotes correct documents. The top-20 from OpenAI 3-large are
already uniformly excellent; aggressive reranking mistakes subtle embedding signal for noise.

Applied to weaker models (MiniLM, BGE-M3, PubMedBERT), bge-reranker correctly rescues
relevant documents from ranks 6–20, delivering gains of +6 to +11 pp NDCG.

---

## Retrieval k — How Many Papers to Fetch

The internal RAG pipeline in `basic` mode applies a 3× over-retrieve multiplier:

```
default_top_k=10  →  retrieves 30 docs internally
                   →  reranks with cross-encoder
                   →  passes top-5 to LLM synthesis
```

This was validated against our eval baseline (4× over-retrieve: fetch 20, evaluate at k=5).
The 3× internal multiplier performs similarly; going to 4× gives negligible gain (<0.5 pp).

**Recommendation:** keep `default_top_k: 10` for all configurations.

For MCP `search_knowledge_base` direct calls (bypasses RAG pipeline),
pass `top_k=20` for optimal retrieval before client-side reranking.

---

## HyDE — Do Not Use

Hypothetical Document Embeddings were evaluated on all three tiers:

| Model | Vector NDCG | HyDE NDCG | Delta |
|---|---|---|---|
| MiniLM | 0.857 | 0.834 | **−2.3 pp** |
| OpenAI | 0.872 | 0.810 | **−6.2 pp** |

HyDE consistently degrades retrieval quality. The stronger the embedding, the larger the
loss. HyDE is **not recommended** for any current Perspicacité configuration.

---

## BGE-M3 — Important Configuration Note

`similarity_threshold` **must be 0.0** for BGE-M3 (and OpenAI). The default threshold
of 0.7 (tuned for MiniLM's cosine distribution) filters out virtually all BGE-M3 results
because BGE-M3's dot-product cosine scores occupy a lower range. Setting threshold=0.7
with BGE-M3 produces empty results and NDCG=0.000.

---

## References

- Benchmark harness: [perspicacite-eval](https://github.com/HolobiomicsLab/perspicacite-eval)
- Full results: `perspicacite-eval/docs/retrieval_benchmark_2026_05_26.md`
- Result JSON files: `perspicacite-eval/results/run_20260526_*.json`
- BEIR benchmark: Thakur et al. (2021), [arxiv:2104.08663](https://arxiv.org/abs/2104.08663)
- BGE-M3: Chen et al. (2024), [arxiv:2309.07597](https://arxiv.org/abs/2309.07597)
- bge-reranker-v2-m3: BAAI, [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3)
