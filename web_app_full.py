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
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
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
    
    collection_name = f"kb_{request.name}"
    
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
    """Add papers to a knowledge base."""
    if not app_state.session_store:
        return {"error": "System not initialized"}
    
    kb = await app_state.session_store.get_kb_metadata(name)
    if not kb:
        return {"error": f"Knowledge base '{name}' not found"}
    
    from perspicacite.models.papers import Paper, Author, PaperSource
    from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase
    
    # Convert PaperData dicts to Paper models
    papers = []
    for pd in request.papers:
        import hashlib
        paper_id = pd.doi if pd.doi else f"generated:{hashlib.md5(pd.title.encode()).hexdigest()[:12]}"
        authors = [Author(name=a) for a in pd.authors]
        papers.append(Paper(
            id=paper_id,
            title=pd.title,
            authors=authors,
            year=pd.year,
            doi=pd.doi,
            abstract=pd.abstract,
            citation_count=pd.citations,
            source=PaperSource.WEB_SEARCH,
        ))
    
    # Use DynamicKnowledgeBase to add papers to the collection
    dkb = DynamicKnowledgeBase(
        vector_store=app_state.vector_store,
        embedding_service=app_state.embedding_provider,
    )
    # Override with the real KB collection
    dkb.collection_name = kb.collection_name
    dkb._initialized = True
    
    added = await dkb.add_papers(papers, include_full_text=False)
    
    # Update metadata counts
    kb.paper_count += len(papers)
    kb.chunk_count += added
    await app_state.session_store.save_kb_metadata(kb)
    
    logger.info(f"Added {len(papers)} papers ({added} chunks) to KB '{name}'")
    return {"added_papers": len(papers), "added_chunks": added, "kb": name}


# HTML Template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perspicacité v2 - True Agentic RAG</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            height: calc(100vh - 40px);
        }
        
        .main-panel {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.assistant {
            align-self: flex-start;
            background: #f1f5f9;
            color: #1e293b;
        }
        
        .input-container {
            padding: 20px;
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 10px;
        }
        
        .input-container input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .input-container input:focus {
            border-color: #667eea;
        }
        
        .input-container button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .input-container button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .input-container button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .side-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 20px;
        }
        
        .panel h3 {
            color: #1e293b;
            margin-bottom: 15px;
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .thinking-step {
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            font-size: 13px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .thinking-step.analyzing {
            background: #dbeafe;
            color: #1e40af;
        }
        
        .thinking-step.planning {
            background: #fef3c7;
            color: #92400e;
        }
        
        .thinking-step.tool {
            background: #d1fae5;
            color: #065f46;
        }
        
        .thinking-step.result {
            background: #e0e7ff;
            color: #3730a3;
        }
        
        .thinking-step.complete {
            background: #f3e8ff;
            color: #6b21a8;
        }
        
        .thinking-step .icon {
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .thinking-step .content {
            flex: 1;
        }
        
        .thinking-step .details {
            font-size: 11px;
            opacity: 0.8;
            margin-top: 4px;
        }
        
        .session-info {
            font-size: 12px;
            color: #64748b;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
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
            background: #10b981;
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
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
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
            padding: 8px 10px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
            background: white;
            margin-bottom: 10px;
        }
        .kb-select:focus { border-color: #667eea; }
        
        .kb-info {
            font-size: 12px;
            color: #64748b;
            margin-bottom: 10px;
            padding: 6px 8px;
            background: #f8fafc;
            border-radius: 6px;
        }
        
        .kb-create-toggle {
            font-size: 12px;
            color: #667eea;
            cursor: pointer;
            border: none;
            background: none;
            padding: 4px 0;
            text-decoration: underline;
        }
        
        .kb-create-form {
            display: none;
            margin-top: 8px;
            padding: 10px;
            background: #f8fafc;
            border-radius: 8px;
        }
        .kb-create-form.visible { display: block; }
        
        .kb-create-form input {
            width: 100%;
            padding: 6px 8px;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            font-size: 12px;
            margin-bottom: 6px;
            outline: none;
        }
        .kb-create-form input:focus { border-color: #667eea; }
        
        .kb-create-form button {
            width: 100%;
            padding: 6px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
        }
        .kb-create-form button:hover { background: #5a6fd6; }
        
        /* Papers found section */
        .papers-found {
            margin-top: 12px;
            padding: 12px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 10px;
            font-size: 13px;
        }
        .papers-found h4 {
            color: #166534;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .paper-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 6px 0;
            border-bottom: 1px solid #dcfce7;
        }
        .paper-item:last-of-type { border-bottom: none; }
        .paper-item input[type="checkbox"] { margin-top: 3px; flex-shrink: 0; }
        .paper-item label {
            font-size: 12px;
            color: #1e293b;
            line-height: 1.4;
            cursor: pointer;
        }
        .paper-item .paper-meta {
            font-size: 11px;
            color: #64748b;
        }
        
        .add-to-kb-btn {
            margin-top: 8px;
            padding: 6px 14px;
            background: #16a34a;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
        }
        .add-to-kb-btn:hover { background: #15803d; }
        .add-to-kb-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        /* Toast notification */
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            background: #1e293b;
            color: white;
            border-radius: 8px;
            font-size: 13px;
            z-index: 1000;
            animation: toastIn 0.3s ease, toastOut 0.3s ease 2.7s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        @keyframes toastIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes toastOut { from { opacity: 1; } to { opacity: 0; } }
        
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
                <button class="kb-create-toggle" onclick="toggleCreateKB()">+ Create new KB</button>
                <div id="kb-create-form" class="kb-create-form">
                    <input id="kb-name-input" type="text" placeholder="Name (letters, numbers, -, _)" pattern="^[a-zA-Z0-9_-]+$">
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
                <h3>🧠 Agent Thinking</h3>
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
            if (selectedKb) {
                fetch(`/api/kb/${selectedKb}`).then(r => r.json()).then(data => {
                    if (data.error) {
                        infoDiv.style.display = 'none';
                        return;
                    }
                    infoDiv.style.display = 'block';
                    infoDiv.innerHTML = `<strong>${data.name}</strong><br>` +
                        (data.description ? data.description + '<br>' : '') +
                        `${data.paper_count} papers, ${data.chunk_count} chunks`;
                });
            } else {
                infoDiv.style.display = 'none';
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
                                    data.description
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
        
        function addThinkingStep(message, type, details) {
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
            div.className = `thinking-step ${type}`;
            div.innerHTML = `
                <span class="icon">${icons[type] || '•'}</span>
                <div class="content">
                    <div>${message}</div>
                    ${details ? `<div class="details">${details}</div>` : ''}
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
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
