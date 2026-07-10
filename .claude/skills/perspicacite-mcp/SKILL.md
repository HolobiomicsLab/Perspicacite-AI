---
name: perspicacite-mcp
description: Use the Perspicacité literature-research MCP server intelligently — query shaping, database selection, tool/mode choice, and reading the result envelope.
---

# Goal

Drive the Perspicacité MCP server well: translate and shape queries, pick the
right database fan-out, choose the correct tool and report mode, and read the
`{success, ...}` response envelope. This file is a stable snapshot; for the
authoritative, current tool index call `get_usage_guide` (see Live reference).

# 1. Translate non-English queries

The pipeline is tuned for English. Translate any non-English user query to
English before searching, but keep the original wording for display and for the
citation/quote you show back to the user.

# 2. Query shaping

- Pass the user's text through largely **verbatim**. Do not pre-rephrase it.
- For literature search, leave `optimize_query` on (default) so the server runs
  its own one-shot LLM rewrite; set `optimize_query=False` only when you need an
  exact verbatim query. The response surfaces `fallback_reason` if the rewrite
  failed and the verbatim query was used.
- **Author searches:** set `optimize_query=False`, or better, resolve the author
  to an ORCID/OpenAlex id and filter by it. The rewrite is tuned for topical
  recall and may drop a bare surname it does not recognise as a scientific term
  (e.g. "Libis" → dropped), turning an author search into a topic search.
- For passage search (`search_by_passage`), pass the **raw** sentence/paragraph
  text — that tool matches on the literal input, so do not rewrite it.

# 3. Database selection

`suggest_databases` runs a fast deterministic heuristic (no LLM, ~25ms) —
always worth calling when the domain is unfamiliar. Pass its `recommended`
list directly as `databases=[...]` to `search_literature` or `generate_report`.
Known databases: `semantic_scholar`, `openalex`, `pubmed`, `europepmc`,
`arxiv`, `pubchem`, `crossref`, `inspire`.

# 4. Tool-choice decision table

| You want | Call |
|----------|------|
| A synthesized, cited answer | `generate_report` |
| A raw list of candidate papers | `search_literature` |
| Passages similar to a sentence/paragraph you hold | `search_by_passage` |
| Keyword passages from a KB | `get_relevant_passages(adaptive=True)` |
| Structured numeric parameters from passages | `extract_parameters_from_passages` |
| Documented failure modes / limitations from passages | `extract_failure_modes_from_passages` |
| Ambiguous or multi-step research | `generate_report(mode="agentic")` |
| Which of my KBs is most relevant to a query | `route_kbs` |
| Grow a KB by following citation edges | `expand_kb_via_citations` |
| Search a skill KB with EDAM IRI pre-filter (L2 skill routing) | `search_skill_kb` |
| Extract typed claims (Bucur SuperPattern) from passages | `extract_claims_from_passages` |
| Build then query a reasoning graph of claims across a KB | `build_claim_graph` → `query_claim_graph` (`claim_graph_export` → JSON-LD; `claim_graph_status` to poll) |

The claim tools (`extract_claims_from_passages`, the `*_claim_graph` family, `export_astra`)
require the private `indicium` package, which is unpublished and therefore maintainer-only;
they return a clear error when it isn't installed. The claim graph's RDF backend is public
(`uv sync --extra graph`).

Use `adaptive=True` on `get_relevant_passages` for terse or jargon-heavy
queries: the server runs the optimizer once and retries if the first pass
returns nothing. The response `attempts` array and `refined_query` field show
whether the retry fired.

**Extraction tools (`extract_*_from_passages`) — critical:**  
Pass the `passages` list returned by `get_relevant_passages` **as-is** — do
not serialise it to a string, truncate it, or restructure it. The tool
validates the list structure on arrival; a stringified or restructured payload
causes an immediate error (~100ms). The correct call pattern is:

```
result = get_relevant_passages(query=..., kb_name=..., k=8)
extract_failure_modes_from_passages(
    passages=result["passages"],   # exact list, not str(result["passages"])
    context="<domain description>"
)
```

# 5. KB management pitfalls

**Paywalled or inaccessible DOIs silently produce empty KBs.**  
`add_dois_to_kb` gracefully skips papers it cannot retrieve (no OA PDF, no
PMC XML, no arXiv). The response shows `added` / `skipped` / `errored`
counts. Always check that `added >= 1` before calling `search_by_passage`,
`get_relevant_passages`, or `expand_kb_via_citations` on that KB — operating
on an empty KB returns zero results without an error, which can look like a
retrieval failure when it is really an ingestion failure.

**Abstract-only papers produce one chunk, not paragraph-level chunks.**  
When full text cannot be fetched, Perspicacité indexes the paper as a single
metadata chunk: `Title / Authors / Year / DOI / Abstract`. This chunk IS
searchable and will appear in `search_by_passage` results — but a paper with
full text produces many more chunks, so full-text papers dominate ranked
results. Expect lower recall from abstract-only papers on specific passage
queries.

**Multi-KB retrieval uses `kb_names` (list), not `kb_name` (string).**  
`search_by_passage`, `get_relevant_passages`, and `generate_report` accept
either `kb_name="single_kb"` or `kb_names=["kb_a", "kb_b"]` — never both.
When passing `kb_names`, all KBs must share the same embedding model; the
server checks compatibility and errors if they differ. Each result hit carries
`kb_name` identifying its origin KB.

**KB routing before blind multi-KB search.**  
When you hold several KBs and are unsure which is most relevant, call
`route_kbs(query=..., kb_names=[...])` first. It returns a ranked list so
you can target `search_by_passage` at the winner rather than fan-out across
all KBs.

**Full compatible-routing workflow (do this for multi-KB setups):**

1. `list_knowledge_bases` → each KB carries `embedding_model` and a
   `retrieval_hint` (`embedding_strength`, `recommended_reranker`,
   `recommended_hybrid`).
2. `route_kbs(query)` → rank KBs by relevance to the query.
3. **Group the chosen KBs by identical `embedding_model`** before passing them
   as `kb_names`. Mixed-embedding groups are rejected by the server (you'll get
   an embedding-compat error). If the top hits span two embeddings, query each
   compatible group separately and merge yourself.
4. Set `use_hybrid` / `use_reranker` for the call from the winning KB's
   `retrieval_hint` (see §6.1).

# 6. Mode and screening (`generate_report`)

**Default policy — prefer `basic` (standard RAG) or `advanced`.** These two cover
the large majority of queries at low latency. Use `advanced` as the standing
default (best single-claim retrieval); drop to `basic` for a fast answer from a
well-targeted KB. **Escalate to `deep_research`, `agentic`, `literature_survey`,
or `contradiction` only when the task explicitly asks for it** (e.g. "survey the
field", "do these papers agree/refute X?", open-ended multi-step research) **or
genuinely requires multi-paper synthesis or agreement analysis** — those modes
cost ~3–10× the latency, so do not reach for them by default.

- `basic` — quick single-pass retrieval + synthesis, no rerank. Lowest latency
  (~12s). Good for a fast factual answer from a well-targeted KB.
- `advanced` — query expansion + WRRF fusion + screening (default). **Best
  single-claim retrieval mode** in eval (78% stress-bench hit-rate, 93% on hard
  paraphrases, ~34s p50, zero errors). Use this unless you have a specific
  reason not to.
- `deep_research` — deep multi-cycle research with planning + reflection
  (`profound` is a deprecated alias). Highest quality for multi-paper synthesis
  but ~90–120s and benefits from a warm LLM cache. Reserve for genuine surveys.
- `contradiction` — surfaces agreement / disagreement / open questions across
  papers. Use for "does the literature support/refute X?" questions.
- `agentic` — multi-step, intent-driven orchestration with tool use. For
  ambiguous, multi-step tasks where the path isn't known up front.
- `literature_survey` — broad survey with theme clustering + recommendations.
  Broad-scope, **not retrieval-tuned** — don't use it to find a specific paper.

**Mode quick-pick:** specific factual answer → `basic` or `advanced`; best
recall/quality single claim → `advanced`; agree/refute survey → `contradiction`;
deep multi-paper synthesis (latency OK) → `deep_research`; open-ended/tooling →
`agentic`.

When precision matters, set `screen_method` (`bm25` | `rerank` | `llm`) and a
`screen_threshold` in [0, 1]. Use `max_papers_to_download` to cap full-text
fetches, `recency_weight` (>0) to bias toward recent work, and `kb_names` to
fan across several knowledge bases (they must share an embedding model).

# 6.1 Embedding-aware retrieval tuning (advanced)

The optimal retrieval recipe depends on the **strength of the KB's embedding
model** — read it from `list_knowledge_bases` → `embedding_model` /
`retrieval_hint`. Per-request knobs let you adapt without reconfiguring the
server:

| KB embedding | `use_hybrid` | `use_reranker` | Why |
|--------------|--------------|----------------|-----|
| Weak/local (`all-MiniLM-L6-v2`, base BGE, SPECTER2, PubMedBERT) | leave default (on) | leave default (on) | BM25 hybrid (+~3pp) and cross-encoder rerank (+~10pp on SciFact) add signal the bi-encoder lacks. |
| Strong instruction-tuned (`Qwen3-Embedding`, `codestral-embed`, `text-embedding-3-large`, `gemini-embedding`) | **`False`** | **`False`** | The embedder already produces a near-perfect top-k; BM25 blending and a general cross-encoder *demote* correct hits (measured: −1 to −5pp). |

Rule of thumb: **`retrieval_hint.recommended_hybrid` → `use_hybrid`** and
**`retrieval_hint.recommended_reranker` → `use_reranker`**. Both default to
`None` (= the server's configured behaviour), so omit them unless you're
overriding.

Finer control: `bm25_weight` / `vector_weight` set the hybrid blend explicitly
(e.g. `vector_weight=1.0, bm25_weight=0.0` = pure vector). For a strong embedder,
a *light* BM25 touch (≤0.25) is at best neutral; the balanced 0.5 default hurts.

**KB-compatibility constraint:** the embedding model is fixed at ingest time —
you cannot embed a query with a different model than the KB was built with
(dimension mismatch → silent zero results). So choose the *embedding per KB*,
and per query choose only the *KB*, the *mode*, and these *retrieval knobs*.

# 7. Reading results

- Every tool returns a JSON string with a `{success: true/false, ...}`
  envelope. Check `success` before using the payload; on failure read `error`.
- Some responses append `META: {...}` telemetry tails and embed fields such as
  `usage` (tokens/cost), `attempts` (passage-retrieval tries), and
  `query_rephrasings` / `refined_query`. Use these to see whether a retry fired
  or the query was rewritten.
- `add_dois_to_kb` response carries `added`, `skipped`, and `errored` counts.
  Treat `added == 0` as a signal to investigate access before proceeding.

# 8. Perspicacité-deeper-work principle

Perspicacité KB is used **after** the L1/L2 router has identified a relevant
skill — for deeper work *within* that skill's domain. Not as primary
search-from-cold-start.

The four-layer routing architecture (ASB-Skills release design §6):

| Layer | Mechanism | When it fires |
|-------|-----------|---------------|
| **L0** | Plugin README + router SKILL.md description | Session start |
| **L1** | `_router/SKILL.md` (auto-generated per collection), loaded by default; calls `search_skill_kb` | First domain mention |
| **L2** | `search_skill_kb(query, edam_topics?)` — EDAM-pre-filtered → embed-ranked | Router invokes it |
| **L3** | Load specific SKILL.md body via Read tool | After top-k decision |
| **L4** | Indicium claim verification | On-demand, accuracy-critical |

**When to use Perspicacité KB (correct):**
- The router has identified a relevant skill domain (L1 has fired)
- You need parameter ranges, failure modes, or related papers for *that* skill's domain
- You need claim evidence against the original literature

**When NOT to use Perspicacité KB as primary search:**
- Cold-start: user asks a question with no prior domain routing
- Use `search_literature` or `generate_report` from the live database instead

**`search_skill_kb` vs `search_knowledge_base`:**
Use `search_skill_kb` (not `search_knowledge_base`) when searching a KB that was
ingested from an ASB-Skill collection (`source_format=asb-skill-collection-v1`).
It adds an EDAM IRI pre-filter that cuts the candidate set ~10× before embedding
ranking — the precision multiplier that makes large skill libraries scale. Pass
`edam_topics` from the collection's known topics for best results.

# 9. Troubleshooting / failure diagnosis

| Symptom | Most likely cause | What to check / do |
|---------|-------------------|--------------------|
| Zero results, no error | Empty KB (ingestion failed) **or** embedding mismatch | Re-check `add_dois_to_kb` `added>=1`; confirm you queried the KB with its own embedding (single-KB path or same-embedding `kb_names`). |
| Zero results on multi-KB | Mixed embeddings in `kb_names` | `list_knowledge_bases` → group by `embedding_model`; query compatible groups separately. |
| Author search returns topic papers | `optimize_query` dropped the surname | Set `optimize_query=False` or filter by ORCID/OpenAlex id (§2). |
| Strong-embedder KB recall feels off after a rerank | Cross-encoder demoting good hits | Set `use_reranker=False` (and `use_hybrid=False`) for that call (§6.1). |
| `extract_*_from_passages` errors instantly | `passages` was stringified/restructured | Pass `result["passages"]` verbatim (§4). |
| `agentic` / `deep_research` return ERR for every claim | `llm_cache.db` is a 0-byte file | Recreate the `llm_cache` table (see Perspicacité CLAUDE.md Gotcha 5). |
| Very high latency | `deep_research` cold cache, or large `max_papers_to_download` | Prefer `advanced`; cap `max_papers_to_download`; set `max_total_seconds`. |
| Low recall from some papers | They're abstract-only (one chunk) | Expected; full-text papers dominate ranked results (§5). |

To localise a retrieval miss: try the same query in `basic` (pure vector) vs
`advanced` (hybrid+expansion). If `basic` finds it but `advanced` doesn't, a
high BM25 weight is likely demoting it — set `use_hybrid=False` or lower
`bm25_weight`.

# 10. Live reference

Treat this file as a snapshot. For the authoritative current capabilities,
decision rules, full tool index, and knob defaults, call `get_usage_guide` —
prefer it over trusting this snapshot when planning a multi-step task.
