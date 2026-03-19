# Perspicacite v2 — Known Problems / Bug Register

This file tracks user-reported issues found during interactive testing.

---

## FIXED Issues

### 1) KB-first retrieval returns 0 results FIXED
- **Root cause**: `unified_agentic.py` was using `kb_name` directly as the Chroma collection name, but collections are named `kb_{kb_name}`.
- **Fix**: Changed line 882 in `src/perspicacite/rag/unified_agentic.py` from `collection=kb_name` to `collection=f"kb_{kb_name}"`.

### 2) Need a way to delete KB (with confirmation) FIXED
- **Fix**: Added delete button and confirmation modal in `web_app_full.py`:
  - Delete button appears in KB panel when a KB is selected
  - Modal requires typing the KB name to confirm deletion
  - After deletion: KB dropdown refreshes and shows success toast

### 3) Show the exact keywords used for each search FIXED
- **Fix**: Two-part fix in `web_app_full.py` and `src/perspicacite/rag/agentic/orchestrator.py`:
  - Backend now includes `query` field in `tool_call` events (line 215 in orchestrator.py)
  - Frontend `addThinkingStep()` now displays the query in a highlighted style for each tool invocation

### 4) Ability to collapse agent actions FIXED
- **Fix**: Added collapsible thinking panel in `web_app_full.py`:
  - "Collapse all / Expand all" button in thinking panel header
  - Individual steps are collapsible via chevron icon
  - Collapsed steps show one-line summary, expanded steps show full details and query

---

## Open Issues

(None currently - all reported issues have been addressed)

---

## Recently Fixed

### 5) Need deduplication mechanism for saving papers to KB FIXED
- **Fix**: Added `paper_exists()` method to `ChromaVectorStore` that queries Chroma by `paper_id` metadata
- Modified `add_papers_to_kb` endpoint to:
  - Check if paper exists before adding (by DOI or generated title hash)
  - Skip duplicates and return list of skipped papers
  - Only update `paper_count` and `chunk_count` for genuinely new papers
  - Return `skipped_duplicates` count and list in the API response

### 6) Final report should have academic-style References section FIXED
- **Fix**: Added `_format_references_section()` and `_extract_papers_from_results()` methods in `orchestrator.py`
- References are appended after the main answer with `## References` heading
- Format: numbered list with markdown links: `[Author et al., Year](https://doi.org/... "Full citation")`
- DOI links are clickable; papers without DOI still show full citation
- Inspired by Perspicacite Profonde citation style

---

## Historical Issues (for reference)

### Original Issue 1) KB-first retrieval returns 0 results

- **Report**: AgenticRAG appears to skip/ineffectively use the Knowledge Base (KB) for retrieval.
- **Evidence**: In `logs/web_app_20260319_102057.log`, a chat is sent with KB `test2` selected:
  - Orchestrator logs show KB injection and execution:
    - `KB: test2` (L14)
    - `Injected kb_search step for KB 'test2'` (L29)
    - `KB_SEARCH: Searching collection 'kb_test2'` (L37)
    - `KB_SEARCH: Found 0 results` (L38)
  - Then the system proceeds to OpenAlex and finds the expected foundational FBMN paper(s) (L45-L67).

- **Expected**: `kb_search` should return relevant documents for the query "Summarize feature-based molecular networking (FBMN)" when KB `test2` has curated FBMN papers.
- **Actual**: `kb_search` returns 0 results and the agent falls back to OpenAlex.

- **Notes / likely causes to investigate next**
  - **KB content may not actually be in the expected collection**: UI/back-end uses the Chroma collection name `kb_{kb_name}` (e.g. `kb_test2`). If papers were added under a different KB name/collection, retrieval will be empty.
  - **Papers might be saved to SQLite metadata but not embedded into Chroma**: verify the `/api/kb/{name}/papers` path successfully adds documents into the vector store and persists to disk.
  - **Embedding mismatch / empty vectors**: if embeddings fail silently or are not computed, similarity search could return nothing.
  - **Query formulation mismatch**: KB contains "FBMN" but query uses "feature-based molecular networking"; consider adding lightweight alias expansion at KB-search time (e.g., include both `"FBMN"` and `"feature-based molecular networking"`), or store acronym expansions in metadata.

- **Repro (minimal)**
  - Create/select KB `test2`, add at least one FBMN paper via the "papers found" curation UI.
  - Refresh, re-select KB `test2`.
  - Ask: "Summarize feature-based molecular networking (FBMN)".
  - Observe server log: KB search returns 0.

### Original Issue 2) Need a way to delete KB (with confirmation)

- **Report**: There should be an explicit UI action to delete a KB, and it should require confirmation.
- **Current state (as implemented)**:
  - Backend supports deletion (`DELETE /api/kb/{name}`), but the UI does not expose a delete control.
  - Deletion is a destructive action and should require an explicit confirmation step in the UI (e.g., modal dialog that requires typing the KB name).

- **Expected**
  - User can delete a KB from the UI.
  - UI requires confirmation before issuing the delete request.
  - After deletion: KB dropdown refreshes, selection resets, and the user sees a success/failure toast.

### Original Issue 3) Show the exact keywords used for each search

- **Report**: It's currently hard to tell what exact keywords/queries are being sent to retrieval tools (KB search and OpenAlex).
- **Expected**
  - UI displays the **exact query string** used per tool invocation (e.g., "KB search query: ...", "OpenAlex query: ...").
  - Preferably show this both:
    - inline in the agent "actions" stream, and/or
    - in a small "debug/details" expandable area for each action.

### Original Issue 4) Ability to collapse agent actions

- **Report**: The action-by-action trace is useful but can be noisy; it should be collapsible.
- **Expected**
  - A "collapse/expand" control for the agent's action trace (globally and/or per step).
  - Default behavior could be "collapsed after completion" with a one-line summary (e.g., "Searched KB, searched OpenAlex, generated answer").
  - When expanding an action/step, show **more detail**, such as:
    - the **exact query** used (KB query / OpenAlex query)
    - **counts** (e.g., KB hits, papers retrieved)
    - brief **tool output preview** (first N lines / first N papers)
    - optional **timing** (start/end or duration)
    - identifiers useful for debugging (e.g., KB collection name `kb_{name}`, step id)
