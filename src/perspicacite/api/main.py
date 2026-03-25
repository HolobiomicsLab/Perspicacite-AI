"""
Perspicacité v2 Web GUI - FastAPI Application

A modern web interface for the research assistant with:
- Real-time streaming responses
- Session management  
- Knowledge base selection
- Source citations with preview
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Perspicacité imports
from perspicacite.config.schema import Config, LLMProviderConfig
from perspicacite.llm.client import AsyncLLMClient
from perspicacite.models.papers import Paper, Author, PaperSource
from perspicacite.models.rag import RAGMode, RAGRequest
from perspicacite.pipeline.download import PDFDownloader
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig
from perspicacite.rag.modes.agentic import AgenticRAGMode
from perspicacite.rag.simple_embeddings import SimpleOpenAIEmbeddingProvider
from perspicacite.retrieval.chroma_store import ChromaVectorStore
from perspicacite.logging import get_logger, setup_logging

logger = get_logger("perspicacite.api")

# ============== Models ==============

class ChatMessage(BaseModel):
    role: str = Field(..., description="user, assistant, or system")
    content: str = Field(..., description="Message content")
    sources: list[dict] = Field(default_factory=list, description="Source citations")


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    query: str = Field(..., description="Current user query")
    mode: str = Field(default="agentic", description="RAG mode: basic, advanced, profound, agentic")
    max_papers: int = Field(default=3, ge=1, le=10, description="Max papers to search")
    stream: bool = Field(default=True, description="Stream response")
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    message: ChatMessage
    conversation_id: str
    done: bool = False


class SessionInfo(BaseModel):
    id: str
    created_at: str
    message_count: int


# ============== Global State ==============

class AppState:
    """Application state container."""
    
    def __init__(self):
        self.config = self._load_config()
        self.conversations: dict[str, list[ChatMessage]] = {}
        self.chroma_dir = Path("chroma_data")
        self.chroma_dir.mkdir(exist_ok=True)
        
        # Initialize embedding provider
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set - API will not work")
        else:
            os.environ["OPENAI_API_KEY"] = api_key
            
        self.embedding_provider = SimpleOpenAIEmbeddingProvider(
            model="text-embedding-3-small"
        )
        self.vector_store = ChromaVectorStore(
            persist_dir=str(self.chroma_dir),
            embedding_provider=self.embedding_provider,
        )
        self.llm_client = AsyncLLMClient(self.config.llm)
        self.downloader = PDFDownloader()
        
        logger.info("api_state_initialized")
    
    def _load_config(self) -> Config:
        """Load application config."""
        config = Config()
        config.llm.default_provider = "openai"
        config.llm.default_model = "gpt-4o"
        config.llm.providers = {
            "openai": LLMProviderConfig(
                base_url="https://api.openai.com/v1",
                timeout=120,
                max_retries=3,
            )
        }
        return config
    
    def get_conversation(self, conv_id: str | None) -> tuple[str, list[ChatMessage]]:
        """Get or create conversation."""
        if conv_id and conv_id in self.conversations:
            return conv_id, self.conversations[conv_id]
        
        new_id = str(uuid.uuid4())[:8]
        self.conversations[new_id] = []
        return new_id, self.conversations[new_id]


app_state: AppState | None = None


# ============== Lifespan ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global app_state
    
    # Startup
    setup_logging(Config().logging)
    logger.info("api_startup")
    app_state = AppState()
    
    yield
    
    # Shutdown
    logger.info("api_shutdown")


# ============== FastAPI App ==============

app = FastAPI(
    title="Perspicacité v2",
    description="AI-powered scientific literature research assistant",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== HTML Frontend ==============

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Perspicacité v2 - Research Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            color: white;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .header p { opacity: 0.8; font-size: 0.9rem; margin-top: 0.25rem; }
        .main { flex: 1; display: flex; max-width: 1400px; width: 100%; margin: 0 auto; padding: 1rem; gap: 1rem; }
        .sidebar {
            width: 280px;
            background: rgba(255,255,255,0.95);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: rgba(255,255,255,0.95);
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .messages { flex: 1; overflow-y: auto; padding: 1.5rem; }
        .message { margin-bottom: 1.5rem; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
        .avatar {
            width: 32px; height: 32px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.9rem; font-weight: 600;
        }
        .user-avatar { background: #667eea; color: white; }
        .assistant-avatar { background: #764ba2; color: white; }
        .message-role { font-weight: 600; color: #333; }
        .message-content {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 12px;
            margin-left: 40px;
            line-height: 1.6;
            color: #333;
        }
        .assistant .message-content { background: #f0e6f5; }
        .sources {
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid #ddd;
            font-size: 0.85rem;
        }
        .sources-title { font-weight: 600; color: #666; margin-bottom: 0.5rem; }
        .source-item {
            display: flex; align-items: center; gap: 0.5rem;
            padding: 0.25rem 0;
            color: #667eea;
        }
        .input-area {
            padding: 1rem 1.5rem;
            border-top: 1px solid #eee;
            background: white;
        }
        .input-row { display: flex; gap: 0.5rem; }
        textarea {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            resize: none;
            font-size: 1rem;
            min-height: 56px;
            max-height: 150px;
        }
        textarea:focus { outline: none; border-color: #667eea; }
        button {
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            display: flex; align-items: center; gap: 0.5rem;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .settings-group { margin-bottom: 1.5rem; }
        .settings-label { font-size: 0.85rem; font-weight: 600; color: #666; margin-bottom: 0.5rem; display: block; }
        select, input[type="number"] {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 0.9rem;
        }
        .new-chat-btn {
            width: 100%;
            margin-bottom: 1rem;
        }
        .status {
            padding: 0.5rem 1rem;
            background: #e8f4fd;
            color: #0066cc;
            font-size: 0.85rem;
            display: none;
        }
        .status.active { display: block; }
        .typing { display: inline-flex; gap: 3px; }
        .typing span { animation: bounce 1s infinite; }
        .typing span:nth-child(2) { animation-delay: 0.1s; }
        .typing span:nth-child(3) { animation-delay: 0.2s; }
        @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-5px); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Perspicacité v2</h1>
        <p>AI-powered scientific literature research assistant</p>
    </div>
    
    <div class="main">
        <div class="sidebar">
            <button class="new-chat-btn" onclick="newChat()">+ New Chat</button>
            
            <div class="settings-group">
                <label class="settings-label">Research Mode</label>
                <select id="mode">
                    <option value="agentic" selected>🤖 Agentic (Smart)</option>
                    <option value="profound">🔬 Profound (Deep)</option>
                    <option value="advanced">⚡ Advanced (WRRF)</option>
                    <option value="basic">📚 Basic (Fast)</option>
                </select>
            </div>
            
            <div class="settings-group">
                <label class="settings-label">Max Papers</label>
                <input type="number" id="maxPapers" value="3" min="1" max="10">
            </div>
            
            <div class="settings-group">
                <label class="settings-label">Conversation ID</label>
                <input type="text" id="conversationId" readonly placeholder="New conversation">
            </div>
        </div>
        
        <div class="chat-container">
            <div id="status" class="status">Searching papers...</div>
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <div class="input-row">
                    <textarea id="input" placeholder="Ask a research question..." rows="1"></textarea>
                    <button id="sendBtn" onclick="sendMessage()">
                        <span>Send</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22,2 15,22 11,13 2,9"></polygon>
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let conversationId = null;
        let isProcessing = false;
        
        // Auto-resize textarea
        const textarea = document.getElementById('input');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
        
        // Enter to send, Shift+Enter for new line
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        function newChat() {
            conversationId = null;
            document.getElementById('messages').innerHTML = '';
            document.getElementById('conversationId').value = '';
            textarea.focus();
        }
        
        function addMessage(role, content, sources = []) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatarClass = role === 'user' ? 'user-avatar' : 'assistant-avatar';
            const avatarText = role === 'user' ? '👤' : '🔬';
            const roleText = role === 'user' ? 'You' : 'Perspicacité';
            
            let sourcesHtml = '';
            if (sources.length > 0) {
                sourcesHtml = `
                    <div class="sources">
                        <div class="sources-title">📚 Sources</div>
                        ${sources.map(s => `
                            <div class="source-item">
                                <span>📄</span>
                                <span>${s.title}</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar ${avatarClass}">${avatarText}</div>
                    <span class="message-role">${roleText}</span>
                </div>
                <div class="message-content">
                    ${content}
                    ${sourcesHtml}
                </div>
            `;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function showStatus(text) {
            const status = document.getElementById('status');
            status.textContent = text;
            status.classList.add('active');
        }
        
        function hideStatus() {
            document.getElementById('status').classList.remove('active');
        }
        
        async function sendMessage() {
            if (isProcessing) return;
            
            const input = document.getElementById('input');
            const query = input.value.trim();
            if (!query) return;
            
            const mode = document.getElementById('mode').value;
            const maxPapers = parseInt(document.getElementById('maxPapers').value);
            
            isProcessing = true;
            input.value = '';
            input.style.height = 'auto';
            document.getElementById('sendBtn').disabled = true;
            
            // Add user message
            addMessage('user', query);
            
            // Add placeholder for assistant response
            const messagesDiv = document.getElementById('messages');
            const assistantDiv = document.createElement('div');
            assistantDiv.className = 'message assistant';
            assistantDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar assistant-avatar">🔬</div>
                    <span class="message-role">Perspicacité</span>
                </div>
                <div class="message-content" id="response">
                    <span class="typing"><span>●</span><span>●</span><span>●</span></span>
                </div>
            `;
            messagesDiv.appendChild(assistantDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            showStatus('🔍 Searching academic papers...');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        messages: [{ role: 'user', content: query }],
                        mode: mode,
                        max_papers: maxPapers,
                        stream: true,
                        conversation_id: conversationId
                    })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let content = '';
                let sources = [];
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    const lines = text.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                
                                if (data.type === 'status') {
                                    showStatus(data.content);
                                } else if (data.type === 'content') {
                                    content += data.content;
                                    document.getElementById('response').innerHTML = content;
                                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                                } else if (data.type === 'sources') {
                                    sources = data.sources;
                                } else if (data.type === 'done') {
                                    conversationId = data.conversation_id;
                                    document.getElementById('conversationId').value = conversationId;
                                }
                            } catch (e) {}
                        }
                    }
                }
                
                // Update final message with sources
                if (sources.length > 0) {
                    addMessage('assistant', content, sources);
                    assistantDiv.remove();
                } else {
                    document.getElementById('response').innerHTML = content;
                }
                
            } catch (error) {
                document.getElementById('response').innerHTML = 
                    `<span style="color: #dc3545;">Error: ${error.message}</span>`;
            } finally {
                hideStatus();
                isProcessing = false;
                document.getElementById('sendBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
"""


# ============== API Routes ==============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web UI."""
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
    }


async def research_and_respond(
    query: str,
    mode: str,
    max_papers: int,
    conversation_id: str,
) -> AsyncIterator[str]:
    """
    Research a query and stream the response.
    
    Yields SSE-formatted events:
    - data: {"type": "status", "content": "Searching..."}
    - data: {"type": "content", "content": "..."}
    - data: {"type": "sources", "sources": [...]}
    - data: {"type": "done", "conversation_id": "..."}
    """
    if not app_state:
        yield f"data: {json.dumps({'type': 'error', 'content': 'API not initialized'})}\n\n"
        return
    
    try:
        # Import here to avoid circular imports
        from full_chatbot import scilex_search_openalex, download_paper
        
        # Step 1: Search papers
        yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Searching OpenAlex for papers...'})}\n\n"
        
        paper_data_list = await scilex_search_openalex(query, max_papers)
        
        if not paper_data_list:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No papers found for this query'})}\n\n"
            return
        
        yield f"data: {json.dumps({'type': 'status', 'content': f'📚 Found {len(paper_data_list)} papers, downloading...'})}\n\n"
        
        # Step 2: Download papers
        papers: list[Paper] = []
        for paper_data in paper_data_list:
            paper = await download_paper(paper_data, app_state.downloader)
            if paper:
                papers.append(paper)
        
        yield f"data: {json.dumps({'type': 'status', 'content': f'✅ Downloaded {len(papers)} papers, analyzing...'})}\n\n"
        
        # Step 3: Create knowledge base
        kb = DynamicKnowledgeBase(
            vector_store=app_state.vector_store,
            embedding_service=app_state.embedding_provider,
            config=KnowledgeBaseConfig(
                chunk_size=1000,
                chunk_overlap=200,
                top_k=5,
                min_relevance_score=0.3,
            ),
        )
        
        # Step 4: Add papers
        await kb.add_papers(papers, include_full_text=True)
        
        # Step 5: Generate response
        yield f"data: {json.dumps({'type': 'status', 'content': '🤔 Generating answer...'})}\n\n"
        
        request = RAGRequest(
            query=query,
            kb_name=kb.collection_name,
            mode=RAGMode.AGENTIC,
            provider="openai",
            model="gpt-4o",
            max_iterations=3,
        )
        
        # For now, stream a simple response
        # (Full agentic streaming would need more work)
        answer = "This is a placeholder response. The full AgenticRAGMode would generate a comprehensive answer here based on the retrieved papers."
        
        # Stream word by word for effect
        words = answer.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            await asyncio.sleep(0.05)
        
        # Send sources
        sources = [{"title": p.title, "authors": str(p.authors), "year": p.year} for p in papers[:3]]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        
        # Done
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"
        
        # Cleanup
        await kb.cleanup()
        
    except Exception as e:
        logger.error("research_error", error=str(e))
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint with streaming support."""
    conv_id, messages = app_state.get_conversation(request.conversation_id)
    
    # Add user message to history
    messages.append(ChatMessage(role="user", content=request.query))
    
    if request.stream:
        return StreamingResponse(
            research_and_respond(
                query=request.query,
                mode=request.mode,
                max_papers=request.max_papers,
                conversation_id=conv_id,
            ),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming (not implemented yet)
        raise HTTPException(status_code=501, detail="Non-streaming not implemented")


@app.get("/api/conversations")
async def list_conversations():
    """List all active conversations."""
    return [
        SessionInfo(
            id=cid,
            created_at=datetime.now().isoformat(),
            message_count=len(msgs),
        )
        for cid, msgs in app_state.conversations.items()
    ]


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    if conversation_id not in app_state.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "id": conversation_id,
        "messages": app_state.conversations[conversation_id],
    }


# ============== Main Entry ==============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "perspicacite.api.main:app",
        host="0.0.0.0",
        port=5468,
        reload=True,
    )
