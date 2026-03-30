# Perspicacité v2 TODO

## Available Resource
packages_to_use/Perspicacite-AI-release Contains previous version. We should search there first for methods to adapt to the new version

## Short Term

- [x] Alternative endpoint scihub
- [x] UI tune to be more like modern AI chatbot interface
- [x] publisher api download
- [x] properly test publisher api (see `tests/test_publisher_api_live.py`, run `pytest -m live`)
- [x] tool to create KB from bibtex file (`perspicacite create-kb NAME --from-bibtex file.bib`)
    - [x] link KB creation tool to UI
- [x] Comprehensive readme to guide users who doesn't have a lot of computer knowledge.
- [x] a way do delete chat history in UI
- [ ] make sure I can use scilex to search, now we are using alternative openalex search
- [ ] make sure the advanced and profond works exactly as in v1
## Medium Term
- [ ] MCP support
- [ ] the webapp is monolithic, maybe break it down to multiple files for better readability

## Long Term

- [ ] Agent swarm type of system
- [ ] Skills support
- [ ] See if you can convert the faiss kb `/home/tjiang/repos/ScienceGuide/knowledge_bases/MetaboSciGuide_v1`

## Maybe
- [ ] Mouse hovering on the selected KB gives detailed description of the KB