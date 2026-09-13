# Passage retrieval contract

`search_by_passage` and `get_relevant_passages` return distinct document chunks.
Several passages from one article can survive a query. Normal paper/RAG search
keeps its existing paper-level deduplication.

Every passage carries its actual `chunk_id`, `kb_name`, `collection_name`, and
`content_sha256` (SHA-256 of the exact returned UTF-8 text). The same chunk ID in
two collections represents two results. Repeated IDs inside one collection are
deduplicated by highest score; conflicting text for one identity is an error.
Results sort by descending score, then collection and chunk ID, with the global
`k` applied after filtering. Missing bibliographic or license metadata stays
unknown. Retrieval does not establish scientific support or licensing permission.

## Failures and empty results

An existing, successfully searched collection can return no matches. A missing
or inaccessible requested collection is an error, including when other KBs
returned useful passages. The response contains `success: false`, `ok: false`,
`error`, and `collection_errors`, whose entries identify `kb_name`,
`collection_name`, and the error. Partial passages are not emitted as a complete
result. Adaptive query rewriting only runs after a successful empty query.

The Chroma store's ordinary `search` retains its historical empty result when a
collection cannot be opened. Its `search_strict` capability enables
`require_collection=True` for passage retrieval without a separate preflight.
Other store implementations must raise on failure, or expose `search_strict`
if their ordinary search suppresses collection errors.

## Embedding compatibility

Passage mode requires a nonempty, unambiguous embedding model in every selected
KB's metadata, matching the configured query provider. Legacy composite model
strings (`primary|fallback` or typed `default+type:model`) are refused. No provider
is automatically selected, downloaded or reconfigured to make the query work.
If the provider reports a different `last_used_model` after the query, the vector
is refused before searching. A query must produce exactly one finite, nonzero
vector of the provider's declared dimension.

Cached and Typed embedding wrappers currently lack a reliable query producer
identity and are explicitly unsupported in passage mode, including a Typed
wrapper with no routes. Use an explicitly configured unwrapped provider after
checking compatibility. This is a local configured/branch identity check, not
remote model-serving attestation or proof of how historical KB vectors were
ingested. Metadata alone cannot certify an old collection for benchmarking.

## Python callers

`DynamicKnowledgeBase.search_chunks` and `MultiKBRetriever.search_chunks` accept
`query`, `top_k`, `min_score`, and `filters`. They use the explicit
`search(..., result_unit="chunk")` path. The default result unit stays `paper`.
For a single KB, set `collection_name`, `kb_name`, and
`expected_embedding_model` from its persisted metadata before searching the
initialized collection. Multi-KB mode uses each supplied metadata record.
`min_score=0` is an explicit floor; `None` uses the configured default.

Wrap a chunk-capable retriever in `PassageRetriever` before calling the shared
`search_passages` normalizer. The adapter never falls back to paper search.
The legacy normalizer alone still accepts normalized dictionaries and can form
a fallback chunk ID; the real strict retrieval path requires an actual ID.

The MCP tools scope by KB. `get_relevant_passages.paper_doi` remains a reserved,
unenforced argument; do not treat it as evidence of a DOI-scoped search.
