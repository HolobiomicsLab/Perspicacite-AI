# Perspicacité v2 TODO

## Available Resource
packages_to_use/Perspicacite-AI-release Contains previous version. We should search there first for methods to adapt to the new version

## Codebase Assessment (from CODEBASE_ASSESSMENT.md)

### Critical (Fixed)
- [x] Fix broken CLI server entry point (`perspicacite serve`)

### Duplication to Fix
- [x] Unify BibTeX parsing (CLI uses bibtexparser, Web UI uses regex)
- [x] Clarify PDF download logic (`get_pdf_with_fallback` vs `get_content_with_fallback`) - both needed, documented
- [x] Consolidate chunking systems - Created `chunking_advanced.py` with v1 strategies (semantic, section-aware)
- [x] Pick one agentic path (orchestrator vs `RAGEngine` + `AgenticRAGMode`)

### Unused/Dead Code to Clean Up, make sure to double check that it is really not used.
- [x] ~~Remove or wire up `get_content_with_fallback`~~ - Kept both, clarified documentation
- [x] Remove `KBBuilder` - Deleted (was not used anywhere)
- [x] Remove `PaperAssessor`, `QueryRefiner`, `RelevanceAssessment` - Deleted (were not used anywhere)
- [x] Remove stub tools (`WebSearchTool`, `FetchPDFTool`, `CitationNetworkTool`) - Deleted
- [x] Delete `unified_agentic.py.bak` - Deleted
- [ ] Remove or implement `GoogleScholarSearch` - Kept (functional with SciLEx fallback, may be useful)
- [ ] Remove or implement `resolve_doi` / `resolve_dois_batch` - Kept (functional CrossRef resolver, may be useful)

### Integration Gaps
- [ ] Wire Elsevier API into actual KB enrichment path
- [x] Fix bare `except:` in `agentic_chat_stream` - Fixed to catch specific exceptions (JSONDecodeError, binascii.Error, UnicodeDecodeError)

## Short Term
- [x] the remove chat history button in UI removes all the history. The should be a way to choose what to remove
- [ ] Complete agentic chunking implementation using LLM client (currently falls back to semantic)
- [ ] test semantic chunking
- [x] Alternative endpoint scihub
- [x] UI tune to be more like modern AI chatbot interface
- [x] publisher api download
- [x] properly test publisher api (see `tests/test_publisher_api_live.py`, run `pytest -m live`)
- [x] tool to create KB from bibtex file (`perspicacite create-kb NAME --from-bibtex file.bib`)
    - [x] link KB creation tool to UI
- [x] Comprehensive readme to guide users who doesn't have a lot of computer knowledge.
- [x] a way do delete chat history in UI
- [x] make sure I can use scilex to search - Integrated SciLExAdapter as primary search, OpenAlex as fallback
- [ ] Add citation network tool to planner (expose _get_citation_network as a tool)
- [ ] Consider user preference for download strategy: quick answer (high threshold, low max) vs comprehensive survey (low threshold, high max)
- [ ] make sure the advanced and profond works exactly as in v1
- [ ] adding papers from bibtex to an existing KB
## Medium Term
- [ ] MCP support
- [ ] the webapp is monolithic, maybe break it down to multiple files for better readability
- [ ] add hovering function to chat history, which shows three dots. When clicking on it, show functions such as pin, delete, edit title etc.

## Long Term

- [ ] Agent swarm type of system
- [ ] Skills support
- [ ] See if you can convert the faiss kb `/home/tjiang/repos/ScienceGuide/knowledge_bases/MetaboSciGuide_v1`

## Maybe
- [ ] Mouse hovering on the selected KB gives detailed description of the KB