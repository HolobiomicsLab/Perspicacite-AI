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
import base64
import asyncio
import logging
import uuid
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
from perspicacite.models.kb import KnowledgeBase, ChunkConfig, chroma_collection_name_for_kb

# Configure logging with file output
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"web_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
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
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for persistent chat thread"
    )
    kb_name: Optional[str] = Field(default=None, description="Knowledge base to search first")
    mode: str = Field(default="basic", description="RAG mode: basic, advanced, profound, agentic")
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


# =============================================================================
# SECTION 2: Application State
# =============================================================================

class AppState:
    """Application state with agentic orchestrator and RAG engine."""

    def __init__(self):
        self.llm_client = None
        self.embedding_provider = None
        self.vector_store = None
        self.orchestrator = None
        self.rag_engine = None  # Multi-mode RAG engine
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
            persist_dir="./chroma_db", embedding_provider=self.embedding_provider
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
            provider=config.llm.default_provider,
        )

        # Initialize agentic orchestrator
        self.orchestrator = AgenticOrchestrator(
            llm_client=llm_adapter,
            tool_registry=tool_registry,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            max_iterations=5,
        )
        logger.info("Agentic orchestrator initialized")

        # Initialize RAG engine for multi-mode support
        from perspicacite.rag.engine import RAGEngine

        self.rag_engine = RAGEngine(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            tool_registry=tool_registry,
            config=config,
        )
        logger.info("RAG engine initialized (supports all modes)")

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


# =============================================================================
# SECTION 3: Helper Functions
# =============================================================================

def _get_pdf_fallback_kwargs(pdf_config) -> dict:
    """Build keyword args for get_pdf_with_fallback from PDFDownloadConfig."""
    if not pdf_config:
        return {}
    return {
        "alternative_endpoint": pdf_config.alternative_endpoint,
        "unpaywall_email": pdf_config.unpaywall_email,
        "wiley_tdm_token": pdf_config.wiley_tdm_token,
        "aaas_api_key": pdf_config.aaas_api_key,
        "rsc_api_key": pdf_config.rsc_api_key,
        "springer_api_key": pdf_config.springer_api_key,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    await app_state.initialize()
    yield
    # Cleanup
    logger.info("Shutting down...")


app = FastAPI(title="Perspicacité v2 - True Agentic RAG", lifespan=lifespan)


# =============================================================================
# SECTION 4: Web UI Route
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        content = template_path.read_text()
    else:
        content = "<h1>Perspicacité v2</h1><p>Template not found. Please ensure templates/index.html exists.</p>"
    return HTMLResponse(content=content)


# =============================================================================
# SECTION 5: Chat API Routes
# =============================================================================

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with true agentic orchestration.

    Uses LLM-driven planning, not fixed workflow.
    """
    if not app_state.initialized:
        await app_state.initialize()

    # Get or create conversation for persistence
    conversation_id = request.conversation_id
    if app_state.session_store:
        if conversation_id:
            # Verify conversation exists
            conv = await app_state.session_store.get_conversation(conversation_id)
            if not conv:
                conversation_id = None  # Will create new below

        if not conversation_id:
            # Create new conversation
            session_id = request.session_id or str(uuid.uuid4())
            kb_name = request.kb_name or "default"
            # Use first 30 chars of query as title
            title = request.query[:30] + "..." if len(request.query) > 30 else request.query

            conv = await app_state.session_store.create_conversation(
                session_id=session_id,
                kb_name=kb_name,
                title=title,
            )
            conversation_id = conv.id
            logger.info(f"Created new conversation: {conversation_id} for session {session_id}")

    if request.stream:
        return StreamingResponse(
            agentic_chat_stream(request, conversation_id), media_type="text/event-stream"
        )
    else:
        # Non-streaming not implemented for simplicity
        return {"error": "Non-streaming not supported. Use stream=true"}


async def agentic_chat_stream(request: ChatRequest, conversation_id: Optional[str] = None):
    """
    Stream chat responses using selected RAG mode.

    Routes to appropriate handler based on request.mode:
    - agentic: Uses AgenticOrchestrator (intent-based, tool use)
    - basic/advanced/profound: Uses RAGEngine with respective mode

    Yields SSE events with thinking steps, tool calls, and final answer.
    """
    from perspicacite.models.messages import Message

    # Save user message to conversation if we have one
    if conversation_id and app_state.session_store:
        try:
            await app_state.session_store.add_message(
                conversation_id, Message(role="user", content=request.query)
            )
        except Exception as e:
            logger.warning(f"Failed to save user message: {e}")

    assistant_content = ""

    try:
        logger.info(f"Chat request: {request.query[:50]}... | Mode: {request.mode}")

        # Route based on selected mode
        if request.mode == "agentic":
            # Use agentic orchestrator for full agentic behavior
            async for event in _stream_agentic(request, conversation_id):
                # Collect assistant content for saving
                if event.startswith("data:"):
                    try:
                        data = json.loads(event[5:].strip())
                        if data.get("type") == "answer":
                            content_b64 = data.get("content_b64")
                            if content_b64:
                                assistant_content = base64.b64decode(content_b64).decode("utf-8")
                    except:
                        pass
                yield event
        else:
            # Use RAGEngine for other modes (basic, advanced, profound)
            async for event in _stream_rag_mode(request, conversation_id):
                # Collect assistant content for saving
                if event.startswith("data:"):
                    try:
                        data = json.loads(event[5:].strip())
                        if data.get("type") == "answer":
                            content_b64 = data.get("content_b64")
                            if content_b64:
                                assistant_content = base64.b64decode(content_b64).decode("utf-8")
                    except:
                        pass
                yield event

    except Exception as e:
        logger.error(f"Error in chat stream: {e}", exc_info=True)
        error_data = json.dumps({"type": "error", "message": str(e)})
        yield f"data: {error_data}\n\n"

    # Save assistant message to conversation
    if conversation_id and app_state.session_store and assistant_content:
        try:
            await app_state.session_store.add_message(
                conversation_id, Message(role="assistant", content=assistant_content)
            )
            logger.info(f"Saved conversation messages to {conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to save assistant message: {e}")


async def _stream_agentic(request: ChatRequest, conversation_id: Optional[str] = None):
    """Stream using AgenticOrchestrator."""
    async for event in app_state.orchestrator.chat(
        query=request.query, session_id=request.session_id, kb_name=request.kb_name, stream=True
    ):
        # Large answer bodies as JSON strings are fragile over chunked HTTP (mid-string
        # splits → client JSON.parse "Unterminated string"). Ship answer text as base64.
        if event.get("type") == "answer":
            content = event.get("content") or ""
            safe = {
                "type": "answer",
                "session_id": event.get("session_id"),
                "conversation_id": conversation_id,
                "content_b64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            }
            data = json.dumps(safe, separators=(",", ":"))
        else:
            data = json.dumps(event, separators=(",", ":"))
        yield f"data: {data}\n\n"

    # End of stream
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_rag_mode(request: ChatRequest, conversation_id: Optional[str] = None):
    """Stream using RAGEngine with selected mode (basic, advanced, profound)."""
    from perspicacite.models.rag import RAGRequest as RAGReq, RAGMode

    # Map string mode to RAGMode enum
    mode_map = {
        "basic": RAGMode.BASIC,
        "advanced": RAGMode.ADVANCED,
        "profound": RAGMode.PROFOUND,
    }
    rag_mode = mode_map.get(request.mode, RAGMode.BASIC)

    logger.info(f"Using RAGEngine with mode: {rag_mode.value}")

    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Create RAG request
    rag_request = RAGReq(
        query=request.query, kb_name=request.kb_name or "default", mode=rag_mode, stream=True
    )

    # Execute using RAGEngine streaming
    full_answer = ""
    sources = []

    try:
        async for event in app_state.rag_engine.query_stream(rag_request):
            if event.event == "status":
                # Forward status updates
                yield f"data: {json.dumps({'type': 'thinking', 'message': json.loads(event.data)['message']})}\n\n"

            elif event.event == "source":
                # Collect sources
                source_data = json.loads(event.data)
                sources.append(source_data)
                # Also forward to UI for display
                yield f"data: {json.dumps({'type': 'source', 'source': source_data})}\n\n"

            elif event.event == "content":
                # Accumulate answer content
                delta = json.loads(event.data)["delta"]
                full_answer += delta

            elif event.event == "done":
                # Send final answer as base64
                safe = {
                    "type": "answer",
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "content_b64": base64.b64encode(full_answer.encode("utf-8")).decode("ascii"),
                    "sources": sources,
                }
                yield f"data: {json.dumps(safe, separators=(',', ':'))}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            elif event.event == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': event.data})}\n\n"
                return

    except Exception as e:
        logger.error(f"RAG engine error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': f'Error in {rag_mode.value} mode: {str(e)}'})}\n\n"

    # End of stream (fallback if no done event)
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# =============================================================================
# SECTION 6: Health & System Routes
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    provider_info = {}
    if app_state.initialized and app_state.config:
        llm_config = app_state.config.llm
        provider_info = {
            "default_provider": llm_config.default_provider,
            "default_model": llm_config.default_model,
            "available_providers": list(llm_config.providers.keys()),
        }

    return {
        "status": "healthy" if app_state.initialized else "initializing",
        "initialized": app_state.initialized,
        "timestamp": datetime.now().isoformat(),
        "llm": provider_info,
    }


# ── Conversation routes ─────────────────────────────────────────────


# =============================================================================
# SECTION 7: Conversation Management Routes
# =============================================================================

@app.get("/api/conversations")
async def list_conversations(session_id: Optional[str] = None):
    """List all conversations (optionally filtered by session_id)."""
    if not app_state.session_store:
        return []

    # If no session_id provided, return all conversations
    if session_id:
        conversations = await app_state.session_store.list_conversations(session_id)
    else:
        # Get all conversations from all sessions
        import aiosqlite

        async with aiosqlite.connect(app_state.session_store.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall("SELECT * FROM conversations ORDER BY updated_at DESC")
            conversations = [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "kb_name": r["kb_name"],
                    "session_id": r["session_id"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    return conversations


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a specific conversation with all messages."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    conversation = await app_state.session_store.get_conversation(conv_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": conversation.id,
        "title": conversation.title,
        "kb_name": conversation.kb_name,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in conversation.messages
        ],
    }


@app.post("/api/conversations")
async def create_conversation(request: dict):
    """Create a new conversation."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    session_id = request.get("session_id", "default")
    kb_name = request.get("kb_name", "default")
    title = request.get("title", "New Conversation")

    conversation = await app_state.session_store.create_conversation(
        session_id=session_id,
        kb_name=kb_name,
        title=title,
    )

    return {
        "id": conversation.id,
        "title": conversation.title,
        "kb_name": conversation.kb_name,
        "session_id": session_id,
    }


@app.post("/api/conversations/{conv_id}/messages")
async def add_message(conv_id: str, request: dict):
    """Add a message to a conversation."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    from perspicacite.models.messages import Message

    message = Message(
        role=request.get("role", "user"),
        content=request.get("content", ""),
    )

    await app_state.session_store.add_message(conv_id, message)

    return {"status": "ok"}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    success = await app_state.session_store.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"status": "deleted", "conversation_id": conv_id}


@app.delete("/api/conversations")
async def delete_all_conversations():
    """Delete all conversations for the current user."""
    if not app_state.session_store:
        raise HTTPException(status_code=503, detail="Session store not available")

    count = await app_state.session_store.delete_all_conversations()
    return {"status": "deleted", "count": count}


# ── KB CRUD routes ──────────────────────────────────────────────────


# =============================================================================
# SECTION 8: Knowledge Base Routes
# =============================================================================

@app.get("/api/kb")
async def list_knowledge_bases():
    """List all knowledge bases."""
    if not app_state.session_store:
        return []
    kbs = await app_state.session_store.list_kbs()
    results = []
    for kb in kbs:
        stats = await app_state.vector_store.get_collection_stats(kb.collection_name)
        results.append(
            {
                "name": kb.name,
                "description": kb.description,
                "paper_count": kb.paper_count,
                "chunk_count": stats.get("count", 0),
                "created_at": kb.created_at.isoformat() if kb.created_at else None,
            }
        )
    return results


@app.post("/api/kb")
async def create_knowledge_base(request: KBCreateRequest):
    """Create a new knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}

    collection_name = chroma_collection_name_for_kb(request.name)

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
    from perspicacite.pipeline.download import get_pdf_with_fallback

    # Convert PaperData dicts to Paper models with deduplication check
    papers_to_add = []
    skipped_duplicates = []
    download_stats = {"attempted": 0, "success": 0, "failed": 0}

    pdf_config = app_state.config.pdf_download if app_state.config else None
    pdf_kw = _get_pdf_fallback_kwargs(pdf_config)

    for pd in request.papers:
        import hashlib

        paper_id = (
            pd.doi if pd.doi else f"generated:{hashlib.md5(pd.title.encode()).hexdigest()[:12]}"
        )

        # Check if paper already exists in this KB
        exists = await app_state.vector_store.paper_exists(kb.collection_name, paper_id)
        if exists:
            skipped_duplicates.append(
                {
                    "title": pd.title,
                    "paper_id": paper_id,
                    "doi": pd.doi,
                }
            )
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
                # Try Unpaywall first, then alternative endpoint
                pdf_bytes = await get_pdf_with_fallback(pd.doi, **pdf_kw)
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

    logger.info(
        f"Added {len(papers_to_add)} papers ({added} chunks) to KB '{name}', skipped {len(skipped_duplicates)} duplicates. PDF stats: {download_stats}"
    )
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

    # Parse BibTeX entries using bibtexparser (same as CLI)
    from perspicacite.models.papers import Paper, Author, PaperSource
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
    from perspicacite.pipeline.download import get_pdf_with_fallback
    from perspicacite.pipeline.bibtex_kb import (
        load_bibtex_entries,
        entries_to_papers,
        ALLOWED_ENTRY_TYPES,
    )
    import io

    # Use bibtexparser to parse the BibTeX content
    try:
        bibtex_file = io.StringIO(bibtex_content)
        bibtex_file.path = None  # type: ignore
        entries = load_bibtex_entries(bibtex_file)  # type: ignore
        papers = entries_to_papers(entries)
    except Exception as e:
        logger.error(f"BibTeX parsing failed: {e}")
        return {"error": f"Failed to parse BibTeX: {str(e)}"}

    if not papers:
        return {"error": "No valid paper entries found in BibTeX file"}

    # Process papers with deduplication and PDF download
    papers_to_add = []
    download_stats = {"attempted": 0, "success": 0, "failed": 0}

    pdf_config = app_state.config.pdf_download if app_state.config else None
    pdf_kw = _get_pdf_fallback_kwargs(pdf_config)

    for paper in papers:
        # Use DOI as ID if available, otherwise generate from title
        paper_id = paper.doi if paper.doi else paper.id

        # Check if paper already exists
        exists = await app_state.vector_store.paper_exists(kb.collection_name, paper_id)
        if exists:
            continue

        # Ensure source is set to BIBTEX
        paper.source = PaperSource.BIBTEX

        # Try to download full text if DOI available
        if paper.doi and app_state.pdf_downloader and app_state.pdf_parser:
            download_stats["attempted"] += 1
            try:
                pdf_bytes = await get_pdf_with_fallback(paper.doi, **pdf_kw)
                if pdf_bytes and len(pdf_bytes) > 1000:
                    parsed = await app_state.pdf_parser.parse(pdf_bytes)
                    if parsed and parsed.text:
                        paper.full_text = parsed.text
                        download_stats["success"] += 1
            except Exception as e:
                logger.warning(f"PDF download failed for {paper.title[:50]}: {e}")
                download_stats["failed"] += 1

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

    logger.info(
        f"Added {len(papers_to_add)} papers from BibTeX ({added} chunks) to KB '{name}'. PDF stats: {download_stats}"
    )
    return {
        "added_papers": len(papers_to_add),
        "added_chunks": added,
        "pdf_download": download_stats,
        "kb": name,
    }




# =============================================================================
# SECTION 9: Main Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
