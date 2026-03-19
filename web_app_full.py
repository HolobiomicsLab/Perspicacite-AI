#!/usr/bin/env python3
"""
Perspicacité v2 - Web Interface with TRUE Agentic RAG

Features:
- LLM-driven orchestration (not fixed pipeline)
- Intent-based routing
- Session-scoped knowledge bases
- Conversation context
- Streaming responses with transparency
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from perspicacite.memory.session_store import SessionStore
from perspicacite.models.kb import KnowledgeBase, ChunkConfig

# Configure logging with file output
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"web_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("perspicacite.web")

# Pydantic models for API
class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat endpoint - NOW WITH CONVERSATION SUPPORT."""
    query: str = Field(..., description="Current research question")
    messages: List[ChatMessage] = Field(default_factory=list, description="Conversation history")
    session_id: Optional[str] = Field(default=None, description="Session ID for persistence")
    kb_name: Optional[str] = Field(default=None, description="Knowledge base to search first")
    stream: bool = Field(default=True, description="Stream the response")
    max_papers: int = Field(default=3, ge=1, le=10)


class KBCreateRequest(BaseModel):
    """Request to create a knowledge base."""
    name: str = Field(..., pattern=r"^[a-zA-Z0-9 _-]+$", min_length=1, max_length=100)
    description: Optional[str] = None


class PaperData(BaseModel):
    """Paper data from chat results, for adding to KB."""
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    citations: Optional[int] = None


class KBAddPapersRequest(BaseModel):
    """Request to add papers to a knowledge base."""
    papers: List[PaperData]


class ChatResponse(BaseModel):
    """Response chunk for streaming."""
    type: str = Field(..., description="thinking, tool_call, tool_result, answer, error")
    content: Optional[str] = None
    message: Optional[str] = None
    step: Optional[str] = None
    tool: Optional[str] = None
    description: Optional[str] = None
    result_summary: Optional[str] = None
    details: Optional[str] = None
    session_id: Optional[str] = None


# Global state
class AppState:
    """Application state with agentic orchestrator."""
    
    def __init__(self):
        self.llm_client = None
        self.embedding_provider = None
        self.vector_store = None
        self.orchestrator = None
        self.session_store: Optional[SessionStore] = None
        self.pdf_downloader = None
        self.pdf_parser = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return
        
        logger.info("Initializing Perspicacité v2 Agentic System...")
        
        # Load config
        from perspicacite.config.loader import load_config
        from perspicacite.llm import AsyncLLMClient, LiteLLMEmbeddingProvider
        from perspicacite.retrieval import ChromaVectorStore
        from perspicacite.rag.agentic import AgenticOrchestrator, LLMAdapter
        from perspicacite.rag.tools import ToolRegistry, LotusSearchTool
        
        config = load_config()
        
        # Initialize LLM
        self.llm_client = AsyncLLMClient(config.llm)
        logger.info("LLM client initialized")
        
        # Initialize embeddings
        self.embedding_provider = LiteLLMEmbeddingProvider(
            model=config.knowledge_base.embedding_model,
        )
        logger.info("Embedding provider initialized")
        
        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            persist_dir="./chroma_db",
            embedding_provider=self.embedding_provider
        )
        logger.info("Vector store initialized")
        
        # Store config for later use
        self.config = config
        
        # Initialize tool registry (LOTUS deactivated for now)
        tool_registry = ToolRegistry()
        logger.info("Tool registry initialized (LOTUS deactivated)")
        
        # Create LLM adapter for agentic components
        llm_adapter = LLMAdapter(
            client=self.llm_client,
            model=config.llm.default_model,
            provider=config.llm.default_provider
        )
        
        # Initialize agentic orchestrator
        self.orchestrator = AgenticOrchestrator(
            llm_client=llm_adapter,
            tool_registry=tool_registry,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            max_iterations=5
        )
        logger.info("Agentic orchestrator initialized")
        
        # Initialize session store (SQLite for KB metadata)
        db_path = Path("./data/perspicacite.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_store = SessionStore(db_path)
        await self.session_store.init_db()
        logger.info("Session store initialized")

        # Initialize PDF downloader and parser
        from perspicacite.pipeline.download import PDFDownloader
        from perspicacite.pipeline.parsers.pdf import PDFParser
        self.pdf_downloader = PDFDownloader()
        self.pdf_parser = PDFParser()
        logger.info("PDF downloader and parser initialized")

        self.initialized = True
        logger.info("System initialization complete!")


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await app_state.initialize()
    yield
    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(title="Perspicacité v2 - True Agentic RAG", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface."""
    return HTMLResponse(content=HTML_TEMPLATE)


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with true agentic orchestration.
    
    Uses LLM-driven planning, not fixed workflow.
    """
    if not app_state.initialized:
        await app_state.initialize()
    
    if request.stream:
        return StreamingResponse(
            agentic_chat_stream(request),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming not implemented for simplicity
        return {"error": "Non-streaming not supported. Use stream=true"}


async def agentic_chat_stream(request: ChatRequest):
    """
    Stream chat responses using true agentic orchestration.
    
    Yields SSE events with thinking steps, tool calls, and final answer.
    """
    try:
        logger.info(f"Chat request: {request.query[:50]}...")
        
        # Convert messages to dict format for orchestrator
        conversation_history = [
            {"role": m.role, "content": m.content}
            for m in request.messages
        ] if request.messages else None
        
        # Use the agentic orchestrator
        async for event in app_state.orchestrator.chat(
            query=request.query,
            session_id=request.session_id,
            kb_name=request.kb_name,
            stream=True
        ):
            # Convert to SSE format
            data = json.dumps(event)
            yield f"data: {data}\n\n"
        
        # End of stream
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        logger.error(f"Error in chat stream: {e}", exc_info=True)
        error_data = json.dumps({
            "type": "error",
            "message": str(e)
        })
        yield f"data: {error_data}\n\n"


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    provider_info = {}
    if app_state.initialized and app_state.config:
        llm_config = app_state.config.llm
        provider_info = {
            "default_provider": llm_config.default_provider,
            "default_model": llm_config.default_model,
            "available_providers": list(llm_config.providers.keys())
        }
    
    return {
        "status": "healthy" if app_state.initialized else "initializing",
        "initialized": app_state.initialized,
        "timestamp": datetime.now().isoformat(),
        "llm": provider_info
    }


# ── KB CRUD routes ──────────────────────────────────────────────────

@app.get("/api/kb")
async def list_knowledge_bases():
    """List all knowledge bases."""
    if not app_state.session_store:
        return []
    kbs = await app_state.session_store.list_kbs()
    results = []
    for kb in kbs:
        stats = await app_state.vector_store.get_collection_stats(kb.collection_name)
        results.append({
            "name": kb.name,
            "description": kb.description,
            "paper_count": kb.paper_count,
            "chunk_count": stats.get("count", 0),
            "created_at": kb.created_at.isoformat() if kb.created_at else None,
        })
    return results


@app.post("/api/kb")
async def create_knowledge_base(request: KBCreateRequest):
    """Create a new knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}
    
    safe_name = request.name.replace(" ", "_")
    collection_name = f"kb_{safe_name}"
    
    existing = await app_state.session_store.get_kb_metadata(request.name)
    if existing:
        return {"error": f"Knowledge base '{request.name}' already exists"}
    
    await app_state.vector_store.create_collection(collection_name)
    
    kb = KnowledgeBase(
        name=request.name,
        description=request.description,
        collection_name=collection_name,
        embedding_model=app_state.embedding_provider.model_name,
        chunk_config=ChunkConfig(),
    )
    await app_state.session_store.save_kb_metadata(kb)
    logger.info(f"Created KB: {request.name} (collection: {collection_name})")
    
    return {
        "name": kb.name,
        "description": kb.description,
        "collection_name": collection_name,
        "paper_count": 0,
        "chunk_count": 0,
    }


@app.get("/api/kb/{name}")
async def get_knowledge_base(name: str):
    """Get knowledge base details."""
    if not app_state.session_store:
        return {"error": "System not initialized"}
    
    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}
    
    stats = await app_state.vector_store.get_collection_stats(kb.collection_name)
    return {
        "name": kb.name,
        "description": kb.description,
        "paper_count": kb.paper_count,
        "chunk_count": stats.get("count", 0),
        "embedding_model": kb.embedding_model,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
    }


@app.delete("/api/kb/{name}")
async def delete_knowledge_base(name: str):
    """Delete a knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}
    
    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}
    
    try:
        await app_state.vector_store.delete_collection(kb.collection_name)
    except Exception:
        pass  # Collection may not exist in ChromaDB
    
    # Delete metadata from SQLite
    import aiosqlite
    async with aiosqlite.connect(app_state.session_store.db_path) as db:
        await db.execute("DELETE FROM kb_metadata WHERE name = ?", (name,))
        await db.commit()
    
    logger.info(f"Deleted KB: {name}")
    return {"deleted": name}


@app.post("/api/kb/{name}/papers")
async def add_papers_to_kb(name: str, request: KBAddPapersRequest):
    """Add papers to a knowledge base with deduplication and optional PDF download."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    from perspicacite.models.papers import Paper, Author, PaperSource
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
    from perspicacite.pipeline.download import get_open_access_url

    # Convert PaperData dicts to Paper models with deduplication check
    papers_to_add = []
    skipped_duplicates = []
    download_stats = {"attempted": 0, "success": 0, "failed": 0}

    for pd in request.papers:
        import hashlib
        paper_id = pd.doi if pd.doi else f"generated:{hashlib.md5(pd.title.encode()).hexdigest()[:12]}"

        # Check if paper already exists in this KB
        exists = await app_state.vector_store.paper_exists(kb.collection_name, paper_id)
        if exists:
            skipped_duplicates.append({
                "title": pd.title,
                "paper_id": paper_id,
                "doi": pd.doi,
            })
            continue

        authors = [Author(name=a) for a in pd.authors]
        paper = Paper(
            id=paper_id,
            title=pd.title,
            authors=authors,
            year=pd.year,
            doi=pd.doi,
            abstract=pd.abstract,
            citation_count=pd.citations,
            source=PaperSource.WEB_SEARCH,
        )

        # Try to download full text if DOI available
        full_text = None
        if pd.doi and app_state.pdf_downloader and app_state.pdf_parser:
            download_stats["attempted"] += 1
            try:
                # Try Unpaywall for OA PDF URL
                pdf_url = await get_open_access_url(pd.doi)
                if pdf_url:
                    # Download and parse PDF
                    pdf_bytes = await app_state.pdf_downloader.download(pdf_url)
                    if pdf_bytes and len(pdf_bytes) > 1000:
                        parsed = await app_state.pdf_parser.parse(pdf_bytes)
                        if parsed and parsed.text:
                            full_text = parsed.text
                            download_stats["success"] += 1
                            logger.info(f"Downloaded full text for: {pd.title[:50]}...")
                        else:
                            download_stats["failed"] += 1
                    else:
                        download_stats["failed"] += 1
                else:
                    download_stats["failed"] += 1
            except Exception as e:
                logger.warning(f"PDF download failed for {pd.title[:50]}: {e}")
                download_stats["failed"] += 1

        paper.full_text = full_text
        papers_to_add.append(paper)

    if not papers_to_add:
        logger.info(f"All {len(skipped_duplicates)} papers already exist in KB '{name}'")
        return {
            "added_papers": 0,
            "added_chunks": 0,
            "skipped_duplicates": len(skipped_duplicates),
            "kb": name,
        }

    # Use DynamicKnowledgeBase to add papers to the collection
    dkb = DynamicKnowledgeBase(
        vector_store=app_state.vector_store,
        embedding_service=app_state.embedding_provider,
    )
    # Override with the real KB collection
    dkb.collection_name = kb.collection_name
    dkb._initialized = True

    # Add papers with full text if available
    added = await dkb.add_papers(papers_to_add, include_full_text=True)

    # Update metadata counts only for new papers
    kb.paper_count += len(papers_to_add)
    kb.chunk_count += added
    await app_state.session_store.save_kb_metadata(kb)

    logger.info(f"Added {len(papers_to_add)} papers ({added} chunks) to KB '{name}', skipped {len(skipped_duplicates)} duplicates. PDF stats: {download_stats}")
    return {
        "added_papers": len(papers_to_add),
        "added_chunks": added,
        "skipped_duplicates": len(skipped_duplicates),
        "duplicates": skipped_duplicates,
        "pdf_download": download_stats,
        "kb": name,
    }


@app.post("/api/kb/{name}/bibtex")
async def add_bibtex_to_kb(name: str, request: Request):
    """Upload a BibTeX file and add papers to a knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}

    try:
        body = await request.json()
        bibtex_content = body.get("bibtex", "")
    except Exception:
        return {"error": "Invalid request body"}

    if not bibtex_content.strip():
        return {"error": "BibTeX content is empty"}

    # Parse BibTeX entries
    from perspicacite.models.papers import Paper, Author, PaperSource
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
    from perspicacite.pipeline.download import get_open_access_url
    import re

    def parse_bibtex_entry(entry: str) -> dict | None:
        """Parse a single BibTeX entry."""
        # Extract entry type and key
        type_key_match = re.match(r'@(\w+)\s*\{\s*([^,]+),', entry)
        if not type_key_match:
            return None

        entry_type = type_key_match.group(1).lower()
        if entry_type != "article" and entry_type != "inproceedings" and entry_type != "book":
            return None

        parsed = {
            "title": "",
            "authors": [],
            "year": None,
            "doi": "",
            "abstract": "",
        }

        # Extract fields
        fields = {
            "title": r'title\s*=\s*\{([^}]+)\}',
            "author": r'author\s*=\s*\{([^}]+)\}',
            "year": r'year\s*=\s*\{([^}]+)\}',
            "doi": r'doi\s*=\s*\{([^}]+)\}',
            "abstract": r'abstract\s*=\s*\{([^}]+)\}',
        }

        for field, pattern in fields.items():
            match = re.search(pattern, entry, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if field == "author":
                    # Parse authors - BibTeX uses "and" to separate authors
                    parsed["authors"] = [a.strip() for a in value.split(" and ")]
                elif field == "year":
                    try:
                        parsed["year"] = int(value)
                    except ValueError:
                        pass
                else:
                    parsed[field] = value

        # Clean title of LaTeX braces
        parsed["title"] = parsed["title"].replace("{", "").replace("}", "")

        return parsed if parsed["title"] else None

    # Split BibTeX into entries and parse each
    entries = re.split(r'\n(?=@)', bibtex_content)
    parsed_entries = []
    for entry in entries:
        if entry.strip():
            parsed = parse_bibtex_entry(entry)
            if parsed:
                parsed_entries.append(parsed)

    if not parsed_entries:
        return {"error": "No valid paper entries found in BibTeX file"}

    # Convert to Paper models with PDF download
    papers_to_add = []
    download_stats = {"attempted": 0, "success": 0, "failed": 0}

    for entry in parsed_entries:
        import hashlib
        paper_id = entry["doi"] if entry["doi"] else f"generated:{hashlib.md5(entry['title'].encode()).hexdigest()[:12]}"

        # Check if paper already exists
        exists = await app_state.vector_store.paper_exists(kb.collection_name, paper_id)
        if exists:
            continue

        authors = [Author(name=a) for a in entry["authors"]]
        paper = Paper(
            id=paper_id,
            title=entry["title"],
            authors=authors,
            year=entry["year"],
            doi=entry["doi"],
            abstract=entry["abstract"],
            source=PaperSource.BIBTEX,
        )

        # Try to download full text if DOI available
        full_text = None
        if entry["doi"] and app_state.pdf_downloader and app_state.pdf_parser:
            download_stats["attempted"] += 1
            try:
                pdf_url = await get_open_access_url(entry["doi"])
                if pdf_url:
                    pdf_bytes = await app_state.pdf_downloader.download(pdf_url)
                    if pdf_bytes and len(pdf_bytes) > 1000:
                        parsed = await app_state.pdf_parser.parse(pdf_bytes)
                        if parsed and parsed.text:
                            full_text = parsed.text
                            download_stats["success"] += 1
            except Exception as e:
                logger.warning(f"PDF download failed for {entry['title'][:50]}: {e}")
                download_stats["failed"] += 1

        paper.full_text = full_text
        papers_to_add.append(paper)

    if not papers_to_add:
        return {
            "message": "All papers already exist in KB",
            "added_papers": 0,
            "kb": name,
        }

    # Add papers to KB
    dkb = DynamicKnowledgeBase(
        vector_store=app_state.vector_store,
        embedding_service=app_state.embedding_provider,
    )
    dkb.collection_name = kb.collection_name
    dkb._initialized = True

    added = await dkb.add_papers(papers_to_add, include_full_text=True)

    # Update metadata
    kb.paper_count += len(papers_to_add)
    kb.chunk_count += added
    await app_state.session_store.save_kb_metadata(kb)

    logger.info(f"Added {len(papers_to_add)} papers from BibTeX ({added} chunks) to KB '{name}'. PDF stats: {download_stats}")
    return {
        "added_papers": len(papers_to_add),
        "added_chunks": added,
        "pdf_download": download_stats,
        "kb": name,
    }


# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perspicacité v2 - True Agentic RAG</title>
    <style>
        /* ScienceGuide-inspired styling */
        @import url('https://fonts.googleapis.com/css2?family=Chivo:wght@400;600;700&family=Libre+Franklin:wght@300;400;500;600&family=Quicksand:wght@400;500;600&display=swap');

        :root {
            --bg-main: #E9F2F4;
            --bg-card: #ffffff;
            --primary: #1b4479;
            --primary-light: #2d5a9e;
            --secondary: #04919A;
            --accent: #61CE70;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #1b4479;
            --input-border: #1b4479;
            --shadow: 0 4px 20px rgba(27, 68, 121, 0.15);
            --radius: 0.35rem;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Libre Franklin', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-main);
            min-height: 100vh;
            padding: 20px;
            color: var(--text-main);
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Chivo', sans-serif;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 20px;
            height: calc(100vh - 40px);
        }

        .main-panel {
            background: var(--bg-card);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid rgba(27, 68, 121, 0.1);
        }

        .header {
            background: var(--primary);
            color: white;
            padding: 20px 24px;
        }

        .header h1 {
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 4px;
            color: white;
        }

        .header p {
            opacity: 0.85;
            font-size: 13px;
            font-weight: 300;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .message {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: var(--radius);
            line-height: 1.6;
            animation: fadeIn 0.3s ease;
            font-size: 14px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
            background: var(--primary);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.assistant {
            align-self: flex-start;
            background: #f8fafc;
            color: var(--text-main);
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 4px;
        }

        .input-container {
            padding: 20px 24px;
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 12px;
            background: #fafbfc;
        }

        .input-container input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid var(--input-border);
            border-radius: var(--radius);
            font-size: 15px;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            background: white;
        }

        .input-container input:focus {
            border-color: var(--secondary);
            box-shadow: 0 0 0 3px rgba(4, 145, 154, 0.1);
        }

        .input-container button {
            padding: 12px 28px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: var(--radius);
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            font-family: 'Chivo', sans-serif;
            transition: background 0.2s, transform 0.15s;
        }

        .input-container button:hover:not(:disabled) {
            background: var(--primary-light);
            transform: translateY(-1px);
        }

        .input-container button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .panel {
            background: var(--bg-card);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 18px;
            border: 1px solid rgba(27, 68, 121, 0.1);
        }

        .panel h3 {
            color: var(--primary);
            margin-bottom: 14px;
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .thinking-step {
            padding: 10px 12px;
            margin-bottom: 8px;
            border-radius: var(--radius);
            font-size: 13px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
            animation: slideIn 0.3s ease;
            border-left: 3px solid transparent;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-8px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .thinking-step.analyzing {
            background: #e0f2fe;
            color: #0369a1;
            border-left-color: #0ea5e9;
        }

        .thinking-step.planning {
            background: #fef3c7;
            color: #92400e;
            border-left-color: #f59e0b;
        }

        .thinking-step.tool {
            background: #d1fae5;
            color: #065f46;
            border-left-color: var(--accent);
        }

        .thinking-step.result {
            background: #e0e7ff;
            color: #3730a3;
            border-left-color: #6366f1;
        }

        .thinking-step.complete {
            background: #f0fdf4;
            color: #166534;
            border-left-color: var(--accent);
        }

        .thinking-step .icon {
            font-size: 15px;
            flex-shrink: 0;
        }

        .thinking-step .content {
            flex: 1;
        }

        .thinking-step .details {
            font-size: 12px;
            opacity: 0.8;
            margin-top: 4px;
        }

        .session-info {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 8px;
        }

        .loading {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid #e2e8f0;
            border-top: 2px solid var(--secondary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }

        .status-indicator.ready {
            background: var(--accent);
        }

        .status-indicator.initializing {
            background: #f59e0b;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .intent-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: var(--radius);
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .intent-badge.natural_products_only {
            background: #dcfce7;
            color: #166534;
        }

        .intent-badge.papers_only {
            background: #dbeafe;
            color: #1e40af;
        }

        .intent-badge.combined_research {
            background: #fef3c7;
            color: #92400e;
        }

        .intent-badge.follow_up {
            background: #fce7f3;
            color: #9d174d;
        }

        /* KB Panel */
        .kb-select {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid var(--input-border);
            border-radius: var(--radius);
            font-size: 13px;
            outline: none;
            background: white;
            margin-bottom: 10px;
            font-family: inherit;
            color: var(--text-main);
        }
        .kb-select:focus {
            border-color: var(--secondary);
            box-shadow: 0 0 0 3px rgba(4, 145, 154, 0.1);
        }

        .kb-info {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 10px;
            padding: 8px 10px;
            background: #f8fafc;
            border-radius: var(--radius);
            border: 1px solid #e2e8f0;
        }

        .kb-create-toggle {
            font-size: 12px;
            color: var(--secondary);
            cursor: pointer;
            border: none;
            background: none;
            padding: 4px 0;
            font-weight: 500;
        }

        .kb-create-form {
            display: none;
            margin-top: 10px;
            padding: 12px;
            background: #f8fafc;
            border-radius: var(--radius);
            border: 1px solid #e2e8f0;
        }
        .kb-create-form.visible { display: block; }

        .kb-create-form input {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #cbd5e1;
            border-radius: var(--radius);
            font-size: 13px;
            margin-bottom: 8px;
            outline: none;
            font-family: inherit;
        }
        .kb-create-form input:focus {
            border-color: var(--secondary);
        }

        .kb-create-form button {
            width: 100%;
            padding: 8px;
            background: var(--secondary);
            color: white;
            border: none;
            border-radius: var(--radius);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            font-family: 'Chivo', sans-serif;
        }
        .kb-create-form button:hover { background: #037a82; }

        .kb-delete-btn {
            font-size: 11px;
            color: #dc2626;
            cursor: pointer;
            border: none;
            background: none;
            padding: 4px 0;
            font-weight: 500;
            margin-left: 8px;
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(27, 68, 121, 0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.visible { display: flex; }
        .modal {
            background: white;
            padding: 24px;
            border-radius: var(--radius);
            max-width: 400px;
            width: 90%;
            box-shadow: var(--shadow);
        }
        .modal h3 { margin-bottom: 12px; color: var(--primary); font-family: 'Chivo', sans-serif; }
        .modal p { margin-bottom: 12px; color: var(--text-muted); font-size: 14px; }
        .modal input {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid var(--input-border);
            border-radius: var(--radius);
            margin-bottom: 12px;
            font-size: 14px;
            font-family: inherit;
            outline: none;
        }
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
        .modal-buttons button {
            padding: 8px 16px;
            border-radius: var(--radius);
            font-size: 13px;
            cursor: pointer;
            font-family: 'Chivo', sans-serif;
            font-weight: 500;
        }
        .modal-cancel { background: #e2e8f0; border: none; color: var(--text-main); }
        .modal-confirm { background: #dc2626; color: white; border: none; }
        .modal-confirm:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Thinking panel collapse */
        .thinking-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .thinking-collapse-btn {
            font-size: 11px;
            color: var(--secondary);
            cursor: pointer;
            border: none;
            background: none;
            padding: 2px 6px;
            font-weight: 500;
        }
        .thinking-step.collapsed .step-details { display: none; }
        .thinking-step .step-summary {
            display: none;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }
        .thinking-step.collapsed .step-summary { display: block; }
        .thinking-step .chevron {
            cursor: pointer;
            font-size: 10px;
            margin-right: 4px;
            transition: transform 0.2s;
        }
        .thinking-step.collapsed .chevron { transform: rotate(-90deg); }

        /* Papers found section */
        .papers-found {
            margin-top: 14px;
            padding: 14px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: var(--radius);
            font-size: 13px;
        }
        .papers-found h4 {
            color: #166534;
            margin-bottom: 10px;
            font-size: 13px;
            font-family: 'Chivo', sans-serif;
        }
        .paper-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #dcfce7;
        }
        .paper-item:last-of-type { border-bottom: none; }
        .paper-item input[type="checkbox"] {
            margin-top: 2px;
            flex-shrink: 0;
            accent-color: var(--secondary);
        }
        .paper-item label {
            font-size: 12px;
            color: var(--text-main);
            line-height: 1.5;
            cursor: pointer;
        }
        .paper-item .paper-meta {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .add-to-kb-btn {
            margin-top: 10px;
            padding: 8px 16px;
            background: var(--secondary);
            color: white;
            border: none;
            border-radius: var(--radius);
            font-size: 12px;
            cursor: pointer;
            font-family: 'Chivo', sans-serif;
            font-weight: 500;
        }
        .add-to-kb-btn:hover { background: #037a82; }
        .add-to-kb-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 12px 20px;
            background: var(--primary);
            color: white;
            border-radius: var(--radius);
            font-size: 13px;
            z-index: 1000;
            animation: toastIn 0.3s ease, toastOut 0.3s ease 2.7s;
            box-shadow: var(--shadow);
        }
        @keyframes toastIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes toastOut { from { opacity: 1; } to { opacity: 0; } }

        /* Query info in thinking steps */
        .query-info {
            font-size: 11px;
            color: var(--secondary);
            margin-top: 4px;
        }
        .query-info code {
            background: #e0f2fe;
            padding: 1px 5px;
            border-radius: 3px;
            font-family: 'Quicksand', monospace;
        }

        /* Code blocks in messages */
        .message code {
            background: #e2e8f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Quicksand', monospace;
            font-size: 0.9em;
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
            .side-panel {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-panel">
            <div class="header">
                <h1>🔬 Perspicacité v2</h1>
                <p>Literature AI assistant — KB-first retrieval with web fallback</p>
            </div>
            
            <div class="chat-container" id="chat-container">
                <div class="message assistant">
                    Hello! I'm Perspicacité — an AI literature assistant. I will:
                    <br><br>
                    • Search your selected Knowledge Base first (curated by you)<br>
                    • Fall back to web literature search (OpenAlex) when your KB is insufficient<br>
                    • Ground answers in retrieved papers and maintain conversation context<br>
                    • Let you curate: add selected retrieved papers back into your KB<br><br>
                    Try: "What are key metabolites in jasmine?" or "Summarize feature-based molecular networking (FBMN)"
                </div>
            </div>
            
            <div class="input-container">
                <input 
                    type="text" 
                    id="query-input" 
                    placeholder="Ask your research question..."
                    onkeypress="if(event.key==='Enter') sendQuery()"
                >
                <button id="send-btn" onclick="sendQuery()">Send</button>
            </div>
        </div>
        
        <div class="side-panel">
            <div class="panel">
                <h3>📚 Knowledge Base</h3>
                <select id="kb-select" class="kb-select" onchange="selectKB(this.value)">
                    <option value="">No KB (web search only)</option>
                </select>
                <div id="kb-info" class="kb-info" style="display:none;"></div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <button class="kb-create-toggle" onclick="toggleCreateKB()">+ Create new KB</button>
                    <button id="kb-delete-btn" class="kb-delete-btn" style="display:none;" onclick="showDeleteKBDialog(selectedKb)">Delete KB</button>
                </div>
                <div id="kb-create-form" class="kb-create-form">
                    <input id="kb-name-input" type="text" placeholder="Name (e.g. FBMN papers)">
                    <input id="kb-desc-input" type="text" placeholder="Description (optional)">
                    <button onclick="createKB()">Create</button>
                </div>
            </div>
            
            <div class="panel">
                <h3>
                    <span id="status-dot" class="status-indicator initializing"></span>
                    System Status
                </h3>
                <div id="system-status">Initializing...</div>
                <div class="session-info" id="session-info"></div>
            </div>
            
            <div class="panel">
                <div class="thinking-header">
                    <h3 style="margin:0;">🧠 Agent Thinking</h3>
                    <button id="thinking-collapse-btn" class="thinking-collapse-btn" onclick="toggleThinkingCollapse()">Collapse all</button>
                </div>
                <div id="thinking-panel">
                    <div style="color: #94a3b8; font-size: 13px;">
                        Agent thinking will appear here...
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>📊 Detected Intent</h3>
                <div id="intent-panel">
                    <div style="color: #94a3b8; font-size: 13px;">
                        Intent will be detected automatically...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let messages = [];
        let isProcessing = false;
        let selectedKb = null;
        let lastFoundPapers = [];
        
        // Check system status on load
        async function checkStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                
                const statusDot = document.getElementById('status-dot');
                const statusText = document.getElementById('system-status');
                
                if (data.initialized) {
                    statusDot.className = 'status-indicator ready';
                    statusText.innerHTML = '✓ Ready';
                    loadKBs();
                } else {
                    statusDot.className = 'status-indicator initializing';
                    statusText.innerHTML = '⏳ Initializing...';
                    setTimeout(checkStatus, 2000);
                }
            } catch (e) {
                document.getElementById('system-status').innerHTML = '❌ Error: ' + e.message;
            }
        }
        
        checkStatus();
        
        // KB management
        async function loadKBs() {
            try {
                const resp = await fetch('/api/kb');
                const kbs = await resp.json();
                const select = document.getElementById('kb-select');
                const currentVal = select.value;
                select.innerHTML = '<option value="">No KB (web search only)</option>';
                for (const kb of kbs) {
                    const opt = document.createElement('option');
                    opt.value = kb.name;
                    opt.textContent = `${kb.name} (${kb.paper_count} papers)`;
                    if (kb.description) opt.title = kb.description;
                    select.appendChild(opt);
                }
                if (currentVal) select.value = currentVal;
            } catch (e) {
                console.error('Failed to load KBs:', e);
            }
        }
        
        function selectKB(name) {
            selectedKb = name || null;
            const infoDiv = document.getElementById('kb-info');
            const deleteBtn = document.getElementById('kb-delete-btn');
            if (selectedKb) {
                fetch(`/api/kb/${selectedKb}`).then(r => r.json()).then(data => {
                    if (data.error) {
                        infoDiv.style.display = 'none';
                        deleteBtn.style.display = 'none';
                        return;
                    }
                    infoDiv.style.display = 'block';
                    infoDiv.innerHTML = `<strong>${data.name}</strong><br>` +
                        (data.description ? data.description + '<br>' : '') +
                        `${data.paper_count} papers, ${data.chunk_count} chunks`;
                });
                deleteBtn.style.display = 'inline';
            } else {
                infoDiv.style.display = 'none';
                deleteBtn.style.display = 'none';
            }
        }
        
        function toggleCreateKB() {
            document.getElementById('kb-create-form').classList.toggle('visible');
        }
        
        async function createKB() {
            const name = document.getElementById('kb-name-input').value.trim();
            const desc = document.getElementById('kb-desc-input').value.trim();
            if (!name) return;
            
            try {
                const resp = await fetch('/api/kb', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ name, description: desc || null })
                });
                const data = await resp.json();
                if (!resp.ok) {
                    const msg = data.detail
                        ? (Array.isArray(data.detail) ? data.detail.map(d => d.msg || d).join('; ') : data.detail)
                        : (data.error || 'Unknown error');
                    showToast('Error: ' + msg);
                    return;
                }
                if (data.error) {
                    showToast('Error: ' + data.error);
                    return;
                }
                showToast('KB "' + name + '" created');
                document.getElementById('kb-name-input').value = '';
                document.getElementById('kb-desc-input').value = '';
                document.getElementById('kb-create-form').classList.remove('visible');
                await loadKBs();
                document.getElementById('kb-select').value = name;
                selectKB(name);
            } catch (e) {
                showToast('Error creating KB: ' + e.message);
            }
        }
        
        async function sendQuery() {
            if (isProcessing) return;
            
            const input = document.getElementById('query-input');
            const query = input.value.trim();
            
            if (!query) return;
            
            isProcessing = true;
            input.value = '';
            input.disabled = true;
            document.getElementById('send-btn').disabled = true;
            
            // Add user message
            addMessage('user', query);
            messages.push({role: 'user', content: query});
            
            // Clear thinking panel
            document.getElementById('thinking-panel').innerHTML = '';
            document.getElementById('intent-panel').innerHTML = '<div style="color: #94a3b8;">Analyzing...</div>';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        query: query,
                        messages: messages.slice(0, -1),
                        session_id: sessionId,
                        kb_name: selectedKb,
                        stream: true
                    })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let assistantMessage = '';
                let assistantDiv = null;
                
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    const lines = text.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'thinking') {
                                addThinkingStep(data.message, 'analyzing', data.details);
                            } else if (data.type === 'tool_call') {
                                addThinkingStep(
                                    `Using tool: ${data.tool}`,
                                    'tool',
                                    data.description,
                                    data.query || ''
                                );
                            } else if (data.type === 'tool_result') {
                                addThinkingStep(
                                    `Result from ${data.step}`,
                                    'result',
                                    data.result_summary
                                );
                            } else if (data.type === 'answer') {
                                if (!assistantDiv) {
                                    assistantDiv = addMessage('assistant', '');
                                }
                                assistantMessage = data.content;
                                assistantDiv.innerHTML = formatMessage(assistantMessage);
                                sessionId = data.session_id;
                                
                                document.getElementById('session-info').textContent = 
                                    `Session: ${sessionId.slice(0, 8)}...`;
                            } else if (data.type === 'papers_found' && data.papers && data.papers.length > 0) {
                                lastFoundPapers = data.papers;
                                if (assistantDiv) {
                                    showPapersCuration(assistantDiv, data.papers);
                                }
                            }
                            
                            // Update intent display if available
                            if (data.details && data.details.includes('Intent:')) {
                                const intentMatch = data.details.match(/Intent: (\\w+)/);
                                if (intentMatch) {
                                    showIntent(intentMatch[1]);
                                }
                            }
                        }
                    }
                }
                
                if (assistantMessage) {
                    messages.push({role: 'assistant', content: assistantMessage});
                }
                
            } catch (error) {
                addMessage('assistant', '❌ Error: ' + error.message);
            }
            
            isProcessing = false;
            input.disabled = false;
            document.getElementById('send-btn').disabled = false;
            input.focus();
        }
        
        function addMessage(role, content) {
            const container = document.getElementById('chat-container');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = formatMessage(content);
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }
        
        function formatMessage(content) {
            // Simple markdown-like formatting
            return content
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code style="background: #e2e8f0; padding: 2px 4px; border-radius: 4px;">$1</code>');
        }

        let thinkingCollapsed = false;

        function addThinkingStep(message, type, details, query) {
            const panel = document.getElementById('thinking-panel');

            // Remove "waiting" message if present
            if (panel.children.length === 1 && panel.children[0].style.color === 'rgb(148, 163, 184)') {
                panel.innerHTML = '';
            }

            const icons = {
                analyzing: '🧠',
                planning: '📋',
                tool: '🔧',
                result: '📄',
                complete: '✅'
            };

            const div = document.createElement('div');
            div.className = `thinking-step ${type}${thinkingCollapsed ? ' collapsed' : ''}`;

            let queryInfo = '';
            if (query) {
                queryInfo = `<div class="query-info">Query: <code>${query}</code></div>`;
            }

            let stepDetails = '';
            if (details || queryInfo) {
                stepDetails = `
                    <div class="step-details">
                        ${queryInfo}
                        ${details ? `<div class="details">${details}</div>` : ''}
                    </div>
                    <div class="step-summary">${message}${query ? ` — Query: ${query}` : ''}</div>
                `;
            }

            div.innerHTML = `
                <span class="chevron" onclick="toggleStepCollapse(this.parentElement)">▼</span>
                <span class="icon">${icons[type] || '•'}</span>
                <div class="content">
                    <div>${message}</div>
                    ${stepDetails}
                </div>
            `;

            panel.appendChild(div);
            panel.scrollTop = panel.scrollHeight;
        }
        
        function showIntent(intent) {
            const panel = document.getElementById('intent-panel');
            const displayNames = {
                'NATURAL_PRODUCTS_ONLY': 'Natural Products Only',
                'PAPERS_ONLY': 'Papers Only',
                'COMBINED_RESEARCH': 'Combined Research',
                'FOLLOW_UP': 'Follow-up Question',
                'CLARIFICATION': 'Clarification',
                'ANALYSIS': 'Analysis',
                'UNKNOWN': 'Unknown'
            };
            
            const classNames = {
                'NATURAL_PRODUCTS_ONLY': 'natural_products_only',
                'PAPERS_ONLY': 'papers_only',
                'COMBINED_RESEARCH': 'combined_research',
                'FOLLOW_UP': 'follow_up'
            };
            
            panel.innerHTML = `
                <span class="intent-badge ${classNames[intent] || 'combined_research'}">
                    ${displayNames[intent] || intent}
                </span>
            `;
        }
        
        // Paper curation
        function showPapersCuration(parentDiv, papers) {
            const section = document.createElement('div');
            section.className = 'papers-found';
            
            let html = '<h4>📄 Papers Found (' + papers.length + ')</h4>';
            papers.forEach((p, i) => {
                const authors = (p.authors || []).join(', ');
                const year = p.year || '?';
                const citations = p.citations != null ? ` | Cited: ${p.citations}` : '';
                html += `
                    <div class="paper-item">
                        <input type="checkbox" id="paper-${i}" checked data-index="${i}">
                        <label for="paper-${i}">
                            ${p.title}
                            <div class="paper-meta">${authors} (${year})${citations}</div>
                        </label>
                    </div>`;
            });
            
            const disabled = selectedKb ? '' : 'disabled title="Select a KB first"';
            const kbLabel = selectedKb ? `Add selected to "${selectedKb}"` : 'Select a KB first';
            html += `<button class="add-to-kb-btn" onclick="addToKB(this)" ${disabled}>${kbLabel}</button>`;
            
            section.innerHTML = html;
            parentDiv.appendChild(section);
            
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }
        
        async function addToKB(btn) {
            if (!selectedKb || !lastFoundPapers.length) return;
            
            const section = btn.closest('.papers-found');
            const checkboxes = section.querySelectorAll('input[type="checkbox"]');
            const selected = [];
            checkboxes.forEach(cb => {
                if (cb.checked) {
                    const p = lastFoundPapers[parseInt(cb.dataset.index)];
                    selected.push({
                        title: p.title,
                        authors: p.authors || [],
                        year: p.year,
                        doi: p.doi,
                        abstract: p.abstract,
                        citations: p.citations
                    });
                }
            });
            
            if (!selected.length) {
                showToast('No papers selected');
                return;
            }
            
            btn.disabled = true;
            btn.textContent = 'Adding...';
            
            try {
                const resp = await fetch(`/api/kb/${selectedKb}/papers`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ papers: selected })
                });
                const data = await resp.json();
                if (data.error) {
                    showToast('Error: ' + data.error);
                    btn.disabled = false;
                    btn.textContent = `Add selected to "${selectedKb}"`;
                    return;
                }
                btn.textContent = `✓ Added ${data.added_papers} papers`;
                showToast(`Added ${data.added_papers} papers to "${selectedKb}"`);
                loadKBs();
            } catch (e) {
                showToast('Error: ' + e.message);
                btn.disabled = false;
                btn.textContent = `Add selected to "${selectedKb}"`;
            }
        }
        
        function showToast(message) {
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.textContent = message;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }

        // Delete KB modal
        let kbToDelete = '';

        // Thinking panel collapse

        function toggleThinkingCollapse() {
            thinkingCollapsed = !thinkingCollapsed;
            document.querySelectorAll('.thinking-step').forEach(el => {
                el.classList.toggle('collapsed', thinkingCollapsed);
            });
            document.getElementById('thinking-collapse-btn').textContent = thinkingCollapsed ? 'Expand all' : 'Collapse all';
        }

        function toggleStepCollapse(el) {
            el.classList.toggle('collapsed');
        }

        function showDeleteKBDialog(kbName) {
            kbToDelete = kbName;
            document.getElementById('delete-kb-name').textContent = kbName;
            document.getElementById('delete-kb-name2').textContent = kbName;
            document.getElementById('delete-kb-input').value = '';
            document.getElementById('delete-kb-confirm').disabled = true;
            document.getElementById('delete-modal').classList.add('visible');
        }

        function hideDeleteKBDialog() {
            document.getElementById('delete-modal').classList.remove('visible');
            kbToDelete = '';
        }

        function checkDeleteKBInput() {
            const input = document.getElementById('delete-kb-input').value;
            document.getElementById('delete-kb-confirm').disabled = input !== kbToDelete;
        }

        async function confirmDeleteKB() {
            if (kbToDelete && document.getElementById('delete-kb-input').value === kbToDelete) {
                try {
                    const resp = await fetch(`/api/kb/${kbToDelete}`, { method: 'DELETE' });
                    const data = await resp.json();
                    if (data.error) {
                        showToast('Error: ' + data.error);
                        return;
                    }
                    showToast('KB "' + kbToDelete + '" deleted');
                    hideDeleteKBDialog();
                    selectedKb = null;
                    document.getElementById('kb-select').value = '';
                    document.getElementById('kb-info').style.display = 'none';
                    await loadKBs();
                } catch (e) {
                    showToast('Error deleting KB: ' + e.message);
                }
            }
        }
    </script>

    <!-- Delete KB Confirmation Modal -->
    <div id="delete-modal" class="modal-overlay">
        <div class="modal">
            <h3>Delete Knowledge Base</h3>
            <p>This will permanently delete <strong id="delete-kb-name"></strong> and all its papers. This cannot be undone.</p>
            <p>Type <strong id="delete-kb-name2"></strong> to confirm:</p>
            <input type="text" id="delete-kb-input" placeholder="Type KB name to confirm" oninput="checkDeleteKBInput()">
            <div class="modal-buttons">
                <button class="modal-cancel" onclick="hideDeleteKBDialog()">Cancel</button>
                <button class="modal-confirm" id="delete-kb-confirm" onclick="confirmDeleteKB()" disabled>Delete</button>
            </div>
        </div>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
