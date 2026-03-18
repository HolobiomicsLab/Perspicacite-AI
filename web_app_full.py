"""
Perspicacité v2 - Full Web GUI with Real Research Pipeline

Run: .venv/bin/python web_app_full.py
Open: http://localhost:5468
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import AsyncIterator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx

# Perspicacité imports
from perspicacite.config.schema import Config, LLMProviderConfig
from perspicacite.llm.client import AsyncLLMClient
from perspicacite.models.papers import Paper, Author, PaperSource
from perspicacite.models.rag import RAGMode, RAGRequest
from perspicacite.pipeline.download import PDFDownloader
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig
from perspicacite.rag.modes.agentic import AgenticRAGMode
from perspicacite.rag.tools import ToolRegistry, KBSearchTool
from perspicacite.rag.simple_embeddings import SimpleOpenAIEmbeddingProvider
from perspicacite.retrieval.chroma_store import ChromaVectorStore
from perspicacite.logging import get_logger, setup_logging

logger = get_logger("perspicacite.web")

# ============== Models ==============

class ChatRequest(BaseModel):
    query: str = Field(..., description="Research question")
    mode: str = Field(default="agentic", description="RAG mode")
    max_papers: int = Field(default=3, ge=1, le=5)
    stream: bool = Field(default=True)


# ============== Global State ==============

class AppState:
    """Application state with initialized components."""
    
    def __init__(self):
        # Check API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not set!")
            raise RuntimeError("Please set OPENAI_API_KEY environment variable")
        
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Setup config
        self.config = Config()
        self.config.llm.default_provider = "openai"
        self.config.llm.default_model = "gpt-4o"
        self.config.llm.providers = {
            "openai": LLMProviderConfig(
                base_url="https://api.openai.com/v1",
                timeout=120,
                max_retries=3,
            )
        }
        
        # Initialize components
        self.chroma_dir = Path("chroma_data")
        self.chroma_dir.mkdir(exist_ok=True)
        
        self.embedding_provider = SimpleOpenAIEmbeddingProvider(
            model="text-embedding-3-small"
        )
        self.vector_store = ChromaVectorStore(
            persist_dir=str(self.chroma_dir),
            embedding_provider=self.embedding_provider,
        )
        self.llm_client = AsyncLLMClient(self.config.llm)
        self.downloader = PDFDownloader()
        
        logger.info("app_state_initialized")


# Initialize state
try:
    app_state = AppState()
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    app_state = None


# ============== OpenAlex Search ==============

async def search_openalex(query: str, max_results: int = 5) -> list[dict]:
    """Search papers using OpenAlex API."""
    import urllib.parse
    
    query_encoded = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/works"
        f"?search={query_encoded}"
        f"&per-page={max_results * 2}"
        f"&mailto=user@example.com"
    )
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
    
    results = []
    for work in data.get("results", [])[:max_results]:
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)
        
        year = None
        pub_date = work.get("publication_date", "")
        if pub_date:
            try:
                year = int(pub_date.split("-")[0])
            except (ValueError, IndexError):
                pass
        
        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        
        # Get PDF URL
        pdf_url = None
        open_access = work.get("open_access", {})
        if open_access.get("is_oa") and open_access.get("oa_url"):
            pdf_url = open_access["oa_url"]
        
        paper = {
            "id": work.get("id", "").split("/")[-1] if work.get("id") else str(uuid.uuid4()),
            "title": work.get("display_name", "Unknown"),
            "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
            "year": str(year) if year else "",
            "abstract": work.get("abstract", "") or "",
            "doi": doi,
            "pdf_url": pdf_url or "",
        }
        
        results.append(paper)
    
    return results


async def download_paper(paper_data: dict, downloader: PDFDownloader) -> Paper | None:
    """Download a paper and create a Paper object."""
    title = paper_data.get("title", "Unknown")
    authors_str = paper_data.get("authors", "")
    year = paper_data.get("year", "")
    abstract = paper_data.get("abstract", "")
    doi = paper_data.get("doi", "")
    pdf_url = paper_data.get("pdf_url", "")
    paper_id = paper_data.get("id", str(uuid.uuid4()))
    
    full_text = None
    
    # Try to download PDF if URL available
    if pdf_url and pdf_url.startswith("http"):
        pdf_bytes = await downloader.download(pdf_url)
        
        if pdf_bytes and len(pdf_bytes) > 1000:
            # Simplified text extraction
            full_text = f"PDF content ({len(pdf_bytes)} bytes)\n\n"
            full_text += f"Title: {title}\n\nAbstract:\n{abstract}"
    
    # Parse authors
    authors = []
    if authors_str:
        for name in authors_str.split(", "):
            name = name.strip()
            if name and name != "et al.":
                authors.append(Author(name=name))
    
    return Paper(
        id=paper_id,
        title=title,
        authors=authors,
        year=int(year) if year and year.isdigit() else None,
        abstract=abstract,
        doi=doi,
        source=PaperSource.WEB_SEARCH,
        full_text=full_text or f"Title: {title}\n\nAbstract:\n{abstract}",
        metadata={"pdf_url": pdf_url},
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
        .main {
            flex: 1;
            display: flex;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 1rem;
            gap: 1rem;
        }
        .sidebar {
            width: 260px;
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
            font-size: 0.9rem;
        }
        .user-avatar { background: #667eea; color: white; }
        .assistant-avatar { background: #764ba2; color: white; }
        .message-content {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 12px;
            margin-left: 40px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
        .assistant .message-content { background: #f0e6f5; }
        .input-area {
            padding: 1rem;
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
            font-family: inherit;
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
        }
        button:hover:not(:disabled) { opacity: 0.9; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .settings-group { margin-bottom: 1.5rem; }
        .settings-label { font-size: 0.85rem; font-weight: 600; color: #666; margin-bottom: 0.5rem; display: block; }
        select, input[type="number"] {
            width: 100%;
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .status-bar {
            padding: 0.75rem 1rem;
            background: #e8f4fd;
            color: #0066cc;
            font-size: 0.85rem;
            display: none;
            border-bottom: 1px solid #cce5ff;
        }
        .status-bar.active { display: block; }
        .typing { display: inline-flex; gap: 3px; }
        .typing span { animation: bounce 1s infinite; }
        @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-5px); } }
        .sources { margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #ddd; }
        .sources-title { font-weight: 600; color: #666; margin-bottom: 0.5rem; font-size: 0.9rem; }
        .source-item { 
            display: flex; align-items: flex-start; gap: 0.5rem; 
            padding: 0.5rem; margin: 0.25rem 0;
            background: rgba(102,126,234,0.1);
            border-radius: 8px;
            font-size: 0.85rem;
        }
        .source-num { 
            background: #667eea; color: white; 
            width: 20px; height: 20px; 
            border-radius: 50%; 
            display: flex; align-items: center; justify-content: center;
            font-size: 0.75rem; font-weight: 600;
            flex-shrink: 0;
        }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Perspicacité v2</h1>
        <p>AI-powered scientific literature research assistant</p>
    </div>
    
    <div class="main">
        <div class="sidebar">
            <div class="settings-group">
                <label class="settings-label">Research Mode</label>
                <select id="mode">
                    <option value="agentic">🧠 Agentic (Deep)</option>
                    <option value="standard">📚 Standard</option>
                    <option value="quick">⚡ Quick</option>
                </select>
            </div>
            <div class="settings-group">
                <label class="settings-label">Max Papers</label>
                <input type="number" id="maxPapers" value="3" min="1" max="5">
            </div>
            <button onclick="newChat()" style="width:100%">+ New Chat</button>
        </div>
        
        <div class="chat-container">
            <div id="statusBar" class="status-bar"></div>
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <div class="input-row">
                    <textarea id="input" placeholder="Ask a research question..." rows="1"></textarea>
                    <button id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;
        
        const textarea = document.getElementById('input');
        const messagesDiv = document.getElementById('messages');
        const statusBar = document.getElementById('statusBar');
        const sendBtn = document.getElementById('sendBtn');
        
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
        
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        function newChat() {
            messagesDiv.innerHTML = '';
            statusBar.classList.remove('active');
            textarea.focus();
        }
        
        function showStatus(text) {
            statusBar.textContent = text;
            statusBar.classList.add('active');
        }
        
        function hideStatus() {
            statusBar.classList.remove('active');
        }
        
        function addMessage(role, content, sources = []) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatarClass = role === 'user' ? 'user-avatar' : 'assistant-avatar';
            const avatarText = role === 'user' ? '👤' : '🔬';
            const roleText = role === 'user' ? 'You' : 'Perspicacité';
            
            let sourcesHtml = '';
            if (sources && sources.length > 0) {
                sourcesHtml = `<div class="sources"><div class="sources-title">📚 Sources</div>` + 
                    sources.map((s, i) => `
                        <div class="source-item">
                            <span class="source-num">${i+1}</span>
                            <div>
                                <div style="font-weight:500">${s.title}</div>
                                ${s.authors ? `<div style="color:#666;font-size:0.8rem">${s.authors}${s.year ? ', ' + s.year : ''}</div>` : ''}
                            </div>
                        </div>
                    `).join('') + 
                    `</div>`;
            }
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar ${avatarClass}">${avatarText}</div>
                    <span style="font-weight:600">${roleText}</span>
                </div>
                <div class="message-content">${escapeHtml(content)}${sourcesHtml}</div>
            `;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function sendMessage() {
            if (isProcessing) return;
            
            const query = textarea.value.trim();
            if (!query) return;
            
            isProcessing = true;
            textarea.value = '';
            textarea.style.height = 'auto';
            sendBtn.disabled = true;
            
            addMessage('user', query);
            
            // Create placeholder for assistant response
            const assistantDiv = document.createElement('div');
            assistantDiv.className = 'message assistant';
            assistantDiv.innerHTML = `
                <div class="message-header">
                    <div class="avatar assistant-avatar">🔬</div>
                    <span style="font-weight:600">Perspicacité</span>
                </div>
                <div class="message-content" id="responseContent">
                    <span class="typing"><span>●</span><span>●</span><span>●</span></span>
                </div>
            `;
            messagesDiv.appendChild(assistantDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        mode: document.getElementById('mode').value,
                        max_papers: parseInt(document.getElementById('maxPapers').value),
                        stream: true
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let content = '';
                let sources = [];
                const responseContent = document.getElementById('responseContent');
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    const lines = text.split('\\n');
                    
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'status') {
                                showStatus(data.content);
                            } 
                            else if (data.type === 'content') {
                                content += data.content;
                                responseContent.textContent = content;
                                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                            } 
                            else if (data.type === 'sources') {
                                sources = data.sources;
                            }
                            else if (data.type === 'error') {
                                responseContent.innerHTML = `<span class="error">Error: ${data.content}</span>`;
                            }
                        } catch (e) {}
                    }
                }
                
                // Replace with final message including sources
                assistantDiv.remove();
                if (content) {
                    addMessage('assistant', content, sources);
                }
                hideStatus();
                
            } catch (error) {
                const responseContent = document.getElementById('responseContent');
                if (responseContent) {
                    responseContent.innerHTML = `<span class="error">Error: ${error.message}</span>`;
                }
                hideStatus();
            } finally {
                isProcessing = false;
                sendBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""


# ============== FastAPI App ==============

app = FastAPI(title="Perspicacité v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=HTML_TEMPLATE)


@app.get("/health")
async def health():
    return {
        "status": "healthy" if app_state else "error",
        "version": "2.0.0",
        "openai_configured": bool(app_state),
    }


async def research_pipeline(request: ChatRequest) -> AsyncIterator[str]:
    """
    Full research pipeline with real perspicacite components.
    """
    if not app_state:
        yield f"data: {json.dumps({'type': 'error', 'content': 'API not initialized - check OPENAI_API_KEY'})}\n\n"
        return
    
    try:
        # Step 1: Search OpenAlex
        yield f"data: {json.dumps({'type': 'status', 'content': '🔍 Searching OpenAlex for papers...'})}\n\n"
        
        paper_data_list = await search_openalex(request.query, request.max_papers)
        
        if not paper_data_list:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No papers found for this query'})}\n\n"
            return
        
        yield f"data: {json.dumps({'type': 'status', 'content': f'📚 Found {len(paper_data_list)} papers'})}\n\n"
        await asyncio.sleep(0.5)
        
        # Step 2: Download papers
        yield f"data: {json.dumps({'type': 'status', 'content': '📥 Downloading PDFs...'})}\n\n"
        
        papers: list[Paper] = []
        for i, paper_data in enumerate(paper_data_list):
            paper = await download_paper(paper_data, app_state.downloader)
            if paper:
                papers.append(paper)
            yield f"data: {json.dumps({'type': 'status', 'content': f'📥 Downloaded {i+1}/{len(paper_data_list)} papers...'})}\n\n"
        
        if not papers:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Failed to download any papers'})}\n\n"
            return
        
        # Step 3: Create knowledge base
        yield f"data: {json.dumps({'type': 'status', 'content': '🧠 Creating knowledge base...'})}\n\n"
        
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
        doc_count = await kb.add_papers(papers, include_full_text=True)
        yield f"data: {json.dumps({'type': 'status', 'content': f'✅ Added {doc_count} document chunks'})}\n\n"
        
        # Step 5: Generate answer with AgenticRAG
        yield f"data: {json.dumps({'type': 'status', 'content': '🤔 Analyzing and generating answer...'})}\n\n"
        
        rag_mode = RAGMode.AGENTIC if request.mode == "agentic" else RAGMode.STANDARD
        
        rag_request = RAGRequest(
            query=request.query,
            kb_name=kb.collection_name,
            mode=rag_mode,
            provider="openai",
            model="gpt-4o",
            max_iterations=2 if request.mode == "agentic" else 1,
        )
        
        tools = ToolRegistry()
        kb_tool = KBSearchTool(app_state.vector_store, app_state.embedding_provider)
        tools.register(kb_tool)
        
        # Run AgenticRAG
        agentic = AgenticRAGMode(app_state.config)
        response = await agentic.execute(
            request=rag_request,
            llm=app_state.llm_client,
            vector_store=app_state.vector_store,
            embedding_provider=app_state.embedding_provider,
            tools=tools,
        )
        
        # Stream the answer word by word
        answer = response.answer
        words = answer.split()
        
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            await asyncio.sleep(0.02)  # Small delay for streaming effect
        
        # Send sources
        sources = []
        for src in response.sources[:5]:
            sources.append({
                "title": src.title,
                "authors": src.authors,
                "year": src.year,
            })
        
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        # Cleanup
        await kb.cleanup()
        
    except Exception as e:
        logger.error("pipeline_error", error=str(e))
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint with streaming."""
    return StreamingResponse(
        research_pipeline(request),
        media_type="text/event-stream",
    )


# ============== Main ==============

if __name__ == "__main__":
    if not app_state:
        print("❌ Error: Failed to initialize. Check OPENAI_API_KEY.")
        print("   export OPENAI_API_KEY='your-key'")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 Perspicacité v2 - Full Research Pipeline")
    print("=" * 60)
    print("Open: http://localhost:5468")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=5468)
