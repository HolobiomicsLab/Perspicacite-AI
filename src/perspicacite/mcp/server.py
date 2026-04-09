"""MCP server implementation for Perspicacité v2.

Exposes scientific literature tools via the Model Context Protocol
so external agent systems (e.g., Mimosa-AI) can discover and use them.

Tools exposed:
- search_literature: Search academic databases
- get_paper_content: Fetch full text + structured sections
- get_paper_references: Extract cited references from a paper
- list_knowledge_bases: List all KBs
- search_knowledge_base: Semantic search within a KB
- create_knowledge_base: Create a new KB
- add_papers_to_kb: Add papers to a KB
- generate_report: Synthesize a research report from a KB
"""

from __future__ import annotations

import json
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.mcp.server")

try:
    from fastmcp import FastMCP

    mcp = FastMCP("perspicacite")
except ImportError:
    mcp = None


# =============================================================================
# Shared State
# =============================================================================


class MCPState:
    """Shared state object set during server startup.

    Provides MCP tools with access to Perspicacité internals
    (session store, vector store, embedding provider, config, etc.)
    without coupling to the web app module.
    """

    def __init__(self) -> None:
        self.session_store: Any = None
        self.vector_store: Any = None
        self.embedding_provider: Any = None
        self.config: Any = None
        self.llm_client: Any = None
        self.pdf_parser: Any = None
        self.tool_registry: Any = None
        self.initialized: bool = False

    async def initialize(self, config: Any) -> None:
        """Initialize all components from config."""
        if self.initialized:
            return

        from perspicacite.llm import AsyncLLMClient, LiteLLMEmbeddingProvider
        from perspicacite.memory.session_store import SessionStore
        from perspicacite.pipeline.parsers.pdf import PDFParser
        from perspicacite.retrieval import ChromaVectorStore
        from pathlib import Path

        self.config = config

        # LLM client
        self.llm_client = AsyncLLMClient(config.llm)

        # Embedding provider
        self.embedding_provider = LiteLLMEmbeddingProvider(
            model=config.knowledge_base.embedding_model,
        )

        # Vector store
        self.vector_store = ChromaVectorStore(
            persist_dir="./chroma_db",
            embedding_provider=self.embedding_provider,
        )

        # Session store
        db_path = Path("./data/perspicacite.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_store = SessionStore(db_path)
        await self.session_store.init_db()

        # PDF parser
        self.pdf_parser = PDFParser()

        # Tool registry for RAG engine
        from perspicacite.rag.tools import ToolRegistry
        self.tool_registry = ToolRegistry()

        self.initialized = True
        logger.info("mcp_state_initialized")


mcp_state = MCPState()


# =============================================================================
# Helper: JSON response builder
# =============================================================================


def _json_ok(data: dict[str, Any]) -> str:
    """Build a success JSON response."""
    return json.dumps({"success": True, **data}, ensure_ascii=False, default=str)


def _json_error(message: str, **extra: Any) -> str:
    """Build an error JSON response."""
    return json.dumps({"success": False, "error": message, **extra}, default=str)


def _require_state() -> MCPState | str:
    """Check that MCP state is initialized. Returns state or error string."""
    if not mcp_state.initialized:
        return _json_error("MCP server not initialized")
    return mcp_state


# =============================================================================
# Tool 1: search_literature
# =============================================================================


@mcp.tool
async def search_literature(
    query: str,
    max_results: int = 20,
    year_min: int | None = None,
    year_max: int | None = None,
    article_type: str | None = None,
    databases: list[str] | None = None,
) -> str:
    """
    Search academic databases for scientific papers matching a query.

    Args:
        query: Search query (keywords, phrases, or natural language)
        max_results: Maximum number of results to return (1-50)
        year_min: Earliest publication year (inclusive)
        year_max: Latest publication year (inclusive)
        article_type: Filter by type ("review", "article", "conference")
        databases: Databases to search. Options: semantic_scholar, openalex, pubmed, arxiv

    Returns:
        JSON with list of papers including title, authors, year, doi, abstract.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    from perspicacite.search.scilex_adapter import SciLExAdapter

    try:
        adapter = SciLExAdapter()
        papers = await adapter.search(
            query=query,
            max_results=max_results,
            year_min=year_min,
            year_max=year_max,
            apis=databases or ["semantic_scholar", "openalex", "pubmed"],
            article_type=article_type,
        )

        # Convert Paper models to dicts
        results = []
        for p in papers:
            pd = {
                "id": p.id,
                "title": p.title,
                "year": p.year,
                "doi": p.doi,
                "abstract": p.abstract,
                "journal": p.journal,
                "citation_count": p.citation_count,
                "source": str(p.source) if p.source else None,
                "url": p.url,
            }
            if p.authors:
                pd["authors"] = [
                    a.family if hasattr(a, "family") and a.family else str(a)
                    for a in p.authors
                ]
            results.append(pd)

        logger.info("mcp_search_literature", query=query, results=len(results))
        return _json_ok({"query": query, "total_results": len(results), "papers": results})

    except Exception as e:
        logger.error("mcp_search_literature_error", error=str(e))
        return _json_error(f"Search failed: {e}")


# =============================================================================
# Tool 2: get_paper_content
# =============================================================================


@mcp.tool
async def get_paper_content(
    doi: str,
    include_sections: bool = True,
) -> str:
    """
    Fetch full text and structured sections for a paper by DOI.

    Uses a unified pipeline: discovers sources via OpenAlex/Unpaywall, then
    tries PMC JATS XML, arXiv HTML, publisher PDF, and abstract in priority order.

    Args:
        doi: Paper DOI (e.g., "10.1234/example")
        include_sections: Whether to include section breakdowns

    Returns:
        JSON with content_type, full_text_length, sections, and references.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    import httpx
    from perspicacite.pipeline.download import retrieve_paper_content

    try:
        pdf_config = state.config.pdf_download
        pdf_kwargs: dict[str, Any] = {}
        if pdf_config:
            pdf_kwargs = {
                "unpaywall_email": pdf_config.unpaywall_email,
                "alternative_endpoint": pdf_config.alternative_endpoint,
                "wiley_tdm_token": pdf_config.wiley_tdm_token,
                "aaas_api_key": pdf_config.aaas_api_key,
                "rsc_api_key": pdf_config.rsc_api_key,
                "springer_api_key": pdf_config.springer_api_key,
            }

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            result = await retrieve_paper_content(
                doi,
                http_client=client,
                pdf_parser=state.pdf_parser,
                **pdf_kwargs,
            )

        if result.success and result.content_type in ("structured", "full_text"):
            resp: dict[str, Any] = {
                "doi": doi,
                "content_type": result.content_type,
                "content_source": result.content_source,
                "full_text_length": len(result.full_text or ""),
            }
            if include_sections and result.sections:
                resp["sections"] = result.sections
            if result.references:
                resp["references_count"] = len(result.references)
                resp["references"] = result.references
            return _json_ok(resp)

        if result.content_type == "abstract":
            return _json_ok({
                "doi": doi,
                "content_type": "abstract",
                "content_source": result.content_source,
                "abstract": result.abstract,
                "note": "Full text not available; returning abstract only",
            })

        return _json_error(f"Could not retrieve content for DOI: {doi}")

    except Exception as e:
        logger.error("mcp_get_paper_content_error", doi=doi, error=str(e))
        return _json_error(f"Content retrieval failed: {e}")


# =============================================================================
# Tool 3: get_paper_references
# =============================================================================


@mcp.tool
async def get_paper_references(
    doi: str,
) -> str:
    """
    Get the list of cited references from a paper.

    Extracts references from JATS XML when available via PMC Open Access.
    Falls back to discovery metadata for non-PMC papers.

    Args:
        doi: Paper DOI

    Returns:
        JSON with list of referenced papers (doi, title, authors, year).
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    import httpx

    try:
        from perspicacite.pipeline.download.unified import _load_cached_references

        # Try loading cached refs first (from a previous content fetch)
        refs = _load_cached_references(doi)
        if refs:
            return _json_ok({"doi": doi, "references": refs, "total": len(refs)})

        # No cached refs — fetch content through unified pipeline to populate cache
        from perspicacite.pipeline.download import retrieve_paper_content

        pdf_config = getattr(state.config, "pdf_download", None)
        pdf_kwargs = {}
        if pdf_config:
            for key in (
                "unpaywall_email", "wiley_tdm_token", "elsevier_api_key",
                "aaas_api_key", "rsc_api_key", "springer_api_key",
            ):
                val = getattr(pdf_config, key, None)
                if val:
                    pdf_kwargs[key] = val

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            await retrieve_paper_content(
                doi, http_client=client, pdf_parser=state.pdf_parser, **pdf_kwargs,
            )

        # Try cache again after content fetch
        refs = _load_cached_references(doi)
        if refs:
            return _json_ok({"doi": doi, "references": refs, "total": len(refs)})

        return _json_ok({
            "doi": doi,
            "references": [],
            "total": 0,
            "note": "References not available — JATS XML extraction only works for PMC Open Access papers",
        })

    except Exception as e:
        logger.error("mcp_get_paper_references_error", doi=doi, error=str(e))
        return _json_error(f"Reference retrieval failed: {e}")


# =============================================================================
# Tool 4: list_knowledge_bases (DEFERRED: get_citation_network)
# =============================================================================


@mcp.tool
async def list_knowledge_bases() -> str:
    """
    List all available knowledge bases.

    Returns:
        JSON with list of KBs including name, description, paper/chunk counts.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        kbs = await state.session_store.list_kbs()
        result = []
        for kb in kbs:
            result.append({
                "name": kb.name,
                "description": kb.description,
                "paper_count": kb.paper_count,
                "chunk_count": kb.chunk_count,
                "created_at": str(kb.created_at) if hasattr(kb, "created_at") else None,
            })
        return _json_ok({"knowledge_bases": result})
    except Exception as e:
        logger.error("mcp_list_kbs_error", error=str(e))
        return _json_error(f"Failed to list KBs: {e}")


# =============================================================================
# Tool 5: search_knowledge_base
# =============================================================================


@mcp.tool
async def search_knowledge_base(
    query: str,
    kb_name: str = "default",
    top_k: int = 5,
) -> str:
    """
    Search within a specific knowledge base using semantic similarity.

    Args:
        query: Search query
        kb_name: Knowledge base name
        top_k: Number of top results to return

    Returns:
        JSON with matching chunks including paper title, section, relevance score.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        from perspicacite.models.kb import chroma_collection_name_for_kb
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

        collection_name = chroma_collection_name_for_kb(kb_name)

        # Verify KB exists
        kb_meta = await state.session_store.get_kb_metadata(kb_name)
        if not kb_meta:
            return _json_error(f"Knowledge base '{kb_name}' not found")

        dkb = DynamicKnowledgeBase(
            state.vector_store,
            state.embedding_provider,
            config=KnowledgeBaseConfig(
                vector_size=state.embedding_provider.dimension,
            ),
        )
        dkb.collection_name = collection_name
        dkb._initialized = True

        results = await dkb.search(query, top_k=top_k)

        chunks = []
        for r in results:
            meta = r.metadata if hasattr(r, "metadata") else {}
            chunks.append({
                "paper_id": meta.get("paper_id"),
                "title": meta.get("title"),
                "section": meta.get("section"),
                "chunk_text": r.text if hasattr(r, "text") else str(r),
                "relevance_score": r.score if hasattr(r, "score") else None,
                "doi": meta.get("doi"),
            })

        return _json_ok({
            "query": query,
            "kb_name": kb_name,
            "results": chunks,
        })

    except Exception as e:
        logger.error("mcp_search_kb_error", kb_name=kb_name, error=str(e))
        return _json_error(f"KB search failed: {e}")


# =============================================================================
# Tool 6: create_knowledge_base
# =============================================================================


@mcp.tool
async def create_knowledge_base(
    name: str,
    description: str = "",
) -> str:
    """
    Create a new empty knowledge base.

    Args:
        name: KB name (alphanumeric, hyphens, underscores)
        description: Optional description

    Returns:
        JSON with created KB details.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        from perspicacite.models.kb import (
            ChunkConfig,
            KnowledgeBase,
            chroma_collection_name_for_kb,
        )

        collection_name = chroma_collection_name_for_kb(name)

        # Check for duplicate
        existing = await state.session_store.get_kb_metadata(name)
        if existing:
            return _json_error(f"Knowledge base '{name}' already exists")

        # Create ChromaDB collection
        await state.vector_store.create_collection(collection_name)

        # Save metadata
        kb = KnowledgeBase(
            name=name,
            description=description or f"Created via MCP",
            collection_name=collection_name,
            embedding_model=state.embedding_provider.model_name,
            chunk_config=ChunkConfig(
                chunk_size=state.config.knowledge_base.chunk_size,
                chunk_overlap=state.config.knowledge_base.chunk_overlap,
            ),
        )
        await state.session_store.save_kb_metadata(kb)

        logger.info("mcp_create_kb", name=name)
        return _json_ok({
            "name": name,
            "description": kb.description,
            "collection_name": collection_name,
            "paper_count": 0,
            "chunk_count": 0,
        })

    except Exception as e:
        logger.error("mcp_create_kb_error", name=name, error=str(e))
        return _json_error(f"KB creation failed: {e}")


# =============================================================================
# Tool 7: add_papers_to_kb
# =============================================================================


@mcp.tool
async def add_papers_to_kb(
    kb_name: str,
    papers: list[dict],
) -> str:
    """
    Add papers to a knowledge base with automatic PDF download and indexing.

    Each paper dict should have at least 'title' and optionally 'doi', 'year',
    'authors', 'abstract', 'citations'.

    Args:
        kb_name: Target knowledge base name
        papers: List of paper dicts to add

    Returns:
        JSON with counts of added/skipped papers and PDF download stats.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        import hashlib
        import httpx
        from perspicacite.models.kb import chroma_collection_name_for_kb
        from perspicacite.models.papers import Author, Paper, PaperSource
        from perspicacite.pipeline.download.fallback import get_pdf_with_fallback
        from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

        collection_name = chroma_collection_name_for_kb(kb_name)
        kb_meta = await state.session_store.get_kb_metadata(kb_name)
        if not kb_meta:
            return _json_error(f"Knowledge base '{kb_name}' not found")

        # Convert paper dicts to Paper models
        paper_models: list[Paper] = []
        for pd in papers:
            paper_id = pd.get("doi") or hashlib.md5(
                pd.get("title", "").encode()
            ).hexdigest()[:12]

            authors = []
            for a in pd.get("authors", []):
                if isinstance(a, str):
                    authors.append(Author(family=a, given="", name=a))
                elif isinstance(a, dict):
                    authors.append(Author(
                        family=a.get("family", ""),
                        given=a.get("given", ""),
                        name=a.get("name", ""),
                    ))

            paper = Paper(
                id=paper_id,
                title=pd.get("title", ""),
                authors=authors,
                year=pd.get("year"),
                doi=pd.get("doi"),
                abstract=pd.get("abstract"),
                citation_count=pd.get("citations"),
                journal=pd.get("journal"),
                url=pd.get("url"),
                pdf_url=pd.get("pdf_url"),
                source=PaperSource.WEB_SEARCH,
                keywords=pd.get("keywords", []),
                metadata=pd.get("metadata", {}),
            )
            paper_models.append(paper)

        # Download PDFs
        pdf_stats = {"attempted": 0, "success": 0, "failed": 0}
        pdf_config = state.config.pdf_download
        pdf_kwargs: dict[str, Any] = {}
        if pdf_config:
            pdf_kwargs = {
                "unpaywall_email": pdf_config.unpaywall_email,
                "alternative_endpoint": pdf_config.alternative_endpoint,
                "wiley_tdm_token": pdf_config.wiley_tdm_token,
                "aaas_api_key": pdf_config.aaas_api_key,
                "rsc_api_key": pdf_config.rsc_api_key,
                "springer_api_key": pdf_config.springer_api_key,
            }

        from perspicacite.pipeline.download import retrieve_paper_content

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            for paper in paper_models:
                if not paper.doi:
                    continue
                pdf_stats["attempted"] += 1
                try:
                    result = await retrieve_paper_content(
                        paper.doi,
                        url=paper.url,
                        http_client=client,
                        pdf_parser=state.pdf_parser,
                        **pdf_kwargs,
                    )
                    if result.success and result.full_text:
                        paper.full_text = result.full_text
                        pdf_stats["success"] += 1
                        continue
                    pdf_stats["failed"] += 1
                except Exception:
                    pdf_stats["failed"] += 1

        # Add to vector store
        dkb_config = KnowledgeBaseConfig(
            vector_size=state.embedding_provider.dimension,
            chunk_size=state.config.knowledge_base.chunk_size,
            chunk_overlap=state.config.knowledge_base.chunk_overlap,
            chunking_method=state.config.knowledge_base.chunking_method,
        )
        dkb = DynamicKnowledgeBase(
            state.vector_store,
            state.embedding_provider,
            config=dkb_config,
        )
        dkb.collection_name = collection_name
        dkb._initialized = True

        chunks_added = await dkb.add_papers(paper_models, include_full_text=True)

        # Update metadata
        kb_meta.paper_count = (kb_meta.paper_count or 0) + len(paper_models)
        kb_meta.chunk_count = (kb_meta.chunk_count or 0) + chunks_added
        await state.session_store.save_kb_metadata(kb_meta)

        logger.info(
            "mcp_add_papers",
            kb_name=kb_name,
            papers=len(paper_models),
            chunks=chunks_added,
        )

        return _json_ok({
            "kb_name": kb_name,
            "added_papers": len(paper_models),
            "added_chunks": chunks_added,
            "pdf_download": pdf_stats,
        })

    except Exception as e:
        logger.error("mcp_add_papers_error", kb_name=kb_name, error=str(e))
        return _json_error(f"Failed to add papers: {e}")


# =============================================================================
# Tool 8: generate_report
# =============================================================================


@mcp.tool
async def generate_report(
    query: str,
    kb_name: str = "default",
    mode: str = "advanced",
    max_papers: int = 10,
) -> str:
    """
    Generate a synthesized research report from a knowledge base.

    Uses Perspicacité's RAG pipeline (retrieval + LLM synthesis) to answer
    a research question using papers in the specified KB.

    Args:
        query: Research question to answer
        kb_name: Knowledge base to query
        mode: RAG mode - "basic" (fast), "advanced" (query expansion), or "profound" (multi-cycle)
        max_papers: Maximum papers to reference in the report

    Returns:
        JSON with the report text, cited sources, and metadata.
    """
    state = _require_state()
    if isinstance(state, str):
        return state

    try:
        from perspicacite.models.kb import chroma_collection_name_for_kb
        from perspicacite.rag.engine import RAGEngine
        from perspicacite.models.rag import RAGRequest, RAGMode

        collection_name = chroma_collection_name_for_kb(kb_name)
        kb_meta = await state.session_store.get_kb_metadata(kb_name)
        if not kb_meta:
            return _json_error(f"Knowledge base '{kb_name}' not found")

        engine = RAGEngine(
            llm_client=state.llm_client,
            vector_store=state.vector_store,
            embedding_provider=state.embedding_provider,
            tool_registry=state.tool_registry,
            config=state.config,
        )

        # Collect full response from streaming generator
        report_text = ""
        sources: list[dict] = []

        mode_map = {
            "basic": RAGMode.BASIC,
            "advanced": RAGMode.ADVANCED,
            "profound": RAGMode.PROFOUND,
        }
        rag_mode = mode_map.get(mode, RAGMode.ADVANCED)

        rag_request = RAGRequest(
            query=query,
            kb_name=kb_name,
            mode=rag_mode,
        )

        async for event in engine.query_stream(rag_request):
            if event.event == "content":
                import json as _json
                payload = _json.loads(event.data)
                report_text += payload.get("delta", "")
            elif event.event == "source":
                import json as _json
                src = _json.loads(event.data)
                sources.append({
                    "title": src.get("title"),
                    "authors": src.get("authors"),
                    "year": src.get("year"),
                    "doi": src.get("doi"),
                    "relevance_score": src.get("relevance_score"),
                    "section": src.get("section"),
                })

        logger.info("mcp_generate_report", query=query, kb_name=kb_name, mode=mode)

        return _json_ok({
            "query": query,
            "kb_name": kb_name,
            "mode": mode,
            "report": report_text,
            "sources": sources,
            "papers_used": len(sources),
        })

    except Exception as e:
        logger.error("mcp_generate_report_error", query=query, error=str(e))
        return _json_error(f"Report generation failed: {e}")


# =============================================================================
# Resource
# =============================================================================


@mcp.resource("perspicacite://info")
async def get_info() -> str:
    """Perspicacité capabilities and status."""
    return json.dumps({
        "name": "Perspicacité v2",
        "description": "AI-powered scientific literature research assistant",
        "tools": [
            "search_literature",
            "get_paper_content",
            "get_paper_references",
            "list_knowledge_bases",
            "search_knowledge_base",
            "create_knowledge_base",
            "add_papers_to_kb",
            "generate_report",
        ],
        "initialized": mcp_state.initialized,
    })
