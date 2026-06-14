# Configuration presets

The canonical, fully-documented template is **[`../config.example.yml`](../config.example.yml)**.
Copy it to `config.yml` (git-ignored) and edit:

```bash
cp config.example.yml config.yml
```

`config.yml` in the repo root is the default the CLI loads when you don't pass `-c`.
The files here are ready-made starting points for specific LLM providers or embedding
backends — copy one over `config.yml`, or point at it directly:

```bash
uv run perspicacite -c configs/embedders/openai_large.yml serve
```

## `llm/` — LLM provider presets

Swap the chat/synthesis backend. Each sets `llm.*` for one provider; embedding defaults
to the open local `all-MiniLM-L6-v2`.

| Preset | Backend |
|--------|---------|
| [`llm/claude_code.yml`](llm/claude_code.yml) | Claude Code subscription (CLI auth) |
| [`llm/codex.yml`](llm/codex.yml) | OpenAI Codex CLI subscription |
| [`llm/hermes.yml`](llm/hermes.yml) | Hermes Agent (Nous Research) |
| [`llm/ollama.yml`](llm/ollama.yml) | Local-only / zero cloud cost (Ollama) |
| [`llm/openclaw.yml`](llm/openclaw.yml) | OpenClaw agent |
| [`llm/openrouter-free.yml`](llm/openrouter-free.yml) | OpenRouter free tier |

## `embedders/` — embedding / retrieval presets

Swap the KB embedding model (and matching reranker). See
[`../docs/embedding-models.md`](../docs/embedding-models.md) for the benchmark table.
A KB must be **rebuilt** when you change its embedding model.

| Preset | Embedding model | Notes |
|--------|-----------------|-------|
| [`embedders/bge_m3.yml`](embedders/bge_m3.yml) | `BAAI/bge-m3` | Production biomedical (recommended) |
| [`embedders/openai_large.yml`](embedders/openai_large.yml) | `text-embedding-3-large` | Cross-domain, best generalisation |
| [`embedders/specter2.yml`](embedders/specter2.yml) | `allenai/specter2_base` | Scientific-paper embeddings |
| [`embedders/pubmedbert.yml`](embedders/pubmedbert.yml) | `pritamdeka/S-PubMedBert-MS-MARCO` | Biomedical |
| [`embedders/neuml_pubmedbert.yml`](embedders/neuml_pubmedbert.yml) | `NeuML/pubmedbert-base-embeddings` | Biomedical (NeuML) |
| [`embedders/biomedbert.yml`](embedders/biomedbert.yml) | `microsoft/BiomedNLP-BiomedBERT-…` | Biomedical (Microsoft) |
| [`embedders/bge_en_icl.yml`](embedders/bge_en_icl.yml) | `BAAI/bge-en-icl` | In-context-learning embeddings |
| [`embedders/gte_qwen2_7b.yml`](embedders/gte_qwen2_7b.yml) | `Alibaba-NLP/gte-Qwen2-7B-instruct` | Large instruct embedder |
| [`embedders/stella_1_5b.yml`](embedders/stella_1_5b.yml) | `dunzhang/stella_en_1.5B_v5` | Compact high-quality embedder |
| [`embedders/qwen3_14b.yml`](embedders/qwen3_14b.yml) | `text-embedding-3-large` | Qwen3-14B chat + OpenAI embeddings |
| [`embedders/code_kb.yml`](embedders/code_kb.yml) | `mistralai/codestral-embed-2505` | Code knowledge bases |

Every preset here is parse-validated against the config schema by
`tests/integration/test_config_audit.py`.
