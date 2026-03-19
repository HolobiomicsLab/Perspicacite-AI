# Perspicacité v2 — Known Problems / Bug Register

This file tracks user-reported issues found during interactive testing.

## 1) KB-first retrieval returns 0 results even when KB should contain relevant papers

- **Report**: AgenticRAG appears to skip/ineffectively use the Knowledge Base (KB) for retrieval.
- **Evidence**: In `logs/web_app_20260319_102057.log`, a chat is sent with KB `test2` selected:
  - Orchestrator logs show KB injection and execution:
    - `KB: test2` (L14)
    - `Injected kb_search step for KB 'test2'` (L29)
    - `KB_SEARCH: Searching collection 'kb_test2'` (L37)
    - `KB_SEARCH: Found 0 results` (L38)
  - Then the system proceeds to OpenAlex and finds the expected foundational FBMN paper(s) (L45–L67).

- **Expected**: `kb_search` should return relevant documents for the query “Summarize feature-based molecular networking (FBMN)” when KB `test2` has curated FBMN papers.
- **Actual**: `kb_search` returns 0 results and the agent falls back to OpenAlex.

- **Notes / likely causes to investigate next**
  - **KB content may not actually be in the expected collection**: UI/back-end uses the Chroma collection name `kb_{kb_name}` (e.g. `kb_test2`). If papers were added under a different KB name/collection, retrieval will be empty.
  - **Papers might be saved to SQLite metadata but not embedded into Chroma**: verify the `/api/kb/{name}/papers` path successfully adds documents into the vector store and persists to disk.
  - **Embedding mismatch / empty vectors**: if embeddings fail silently or are not computed, similarity search could return nothing.
  - **Query formulation mismatch**: KB contains “FBMN” but query uses “feature-based molecular networking”; consider adding lightweight alias expansion at KB-search time (e.g., include both `"FBMN"` and `"feature-based molecular networking"`), or store acronym expansions in metadata.

- **Repro (minimal)**
  - Create/select KB `test2`, add at least one FBMN paper via the “papers found” curation UI.
  - Refresh, re-select KB `test2`.
  - Ask: “Summarize feature-based molecular networking (FBMN)”.
  - Observe server log: KB search returns 0.

## 2) Need a way to delete KB (with confirmation)

- **Report**: There should be an explicit UI action to delete a KB, and it should require confirmation.
- **Current state (as implemented)**:
  - Backend supports deletion (`DELETE /api/kb/{name}`), but the UI does not expose a delete control.
  - Deletion is a destructive action and should require an explicit confirmation step in the UI (e.g., modal dialog that requires typing the KB name).

- **Expected**
  - User can delete a KB from the UI.
  - UI requires confirmation before issuing the delete request.
  - After deletion: KB dropdown refreshes, selection resets, and the user sees a success/failure toast.

## 3) Show the exact keywords used for each search

- **Report**: It’s currently hard to tell what exact keywords/queries are being sent to retrieval tools (KB search and OpenAlex).
- **Expected**
  - UI displays the **exact query string** used per tool invocation (e.g., “KB search query: …”, “OpenAlex query: …”).
  - Preferably show this both:
    - inline in the agent “actions” stream, and/or
    - in a small “debug/details” expandable area for each action.

## 4) Ability to collapse agent actions

- **Report**: The action-by-action trace is useful but can be noisy; it should be collapsible.
- **Expected**
  - A “collapse/expand” control for the agent’s action trace (globally and/or per step).
  - Default behavior could be “collapsed after completion” with a one-line summary (e.g., “Searched KB, searched OpenAlex, generated answer”).
  - When expanding an action/step, show **more detail**, such as:
    - the **exact query** used (KB query / OpenAlex query)
    - **counts** (e.g., KB hits, papers retrieved)
    - brief **tool output preview** (first N lines / first N papers)
    - optional **timing** (start/end or duration)
    - identifiers useful for debugging (e.g., KB collection name `kb_{name}`, step id)

