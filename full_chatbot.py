"""
Full Perspicacité v2 + SciLEx Chatbot

This script demonstrates the complete integration using actual perspicacite classes:
- Config, AsyncLLMClient
- LiteLLMEmbeddingProvider
- ChromaVectorStore
- DynamicKnowledgeBase
- AgenticRAGMode
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml

# Perspicacité imports
from perspicacite.config.schema import Config, LLMConfig, LLMProviderConfig
from perspicacite.llm.client import AsyncLLMClient
from perspicacite.rag.simple_embeddings import SimpleOpenAIEmbeddingProvider
from perspicacite.models.papers import Paper, Author, PaperSource
from perspicacite.models.rag import RAGMode, RAGRequest
from perspicacite.pipeline.download import PDFDownloader
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig
from perspicacite.rag.modes.agentic import AgenticRAGMode
from perspicacite.rag.tools import ToolRegistry, KBSearchTool
from perspicacite.retrieval.chroma_store import ChromaVectorStore


# ============== SciLEx Integration ==============

async def scilex_search_openalex(query: str, max_results: int = 5) -> list[dict]:
    """
    Search papers using OpenAlex API directly (SciLEx-style search).
    
    This avoids the config file complexity of SciLEx CLI.
    """
    import urllib.parse
    
    print(f"   🌐 Searching OpenAlex API...")
    
    # Build OpenAlex query
    query_encoded = urllib.parse.quote(query)
    url = (
        f"https://api.openalex.org/works"
        f"?search={query_encoded}"
        f"&per-page={max_results * 2}"  # Get extra for filtering
        f"&mailto=user@example.com"
    )
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        if response.status_code != 200:
            print(f"   ⚠️ API error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
        response.raise_for_status()
        data = response.json()
    
    results = []
    for work in data.get("results", []):
        # Extract authors
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)
        
        # Extract year
        year = None
        pub_date = work.get("publication_date", "")
        if pub_date:
            try:
                year = int(pub_date.split("-")[0])
            except (ValueError, IndexError):
                pass
        
        # Get PDF URL
        pdf_url = None
        open_access = work.get("open_access", {})
        if open_access.get("is_oa") and open_access.get("oa_url"):
            pdf_url = open_access["oa_url"]
        
        # Get DOI
        doi = work.get("doi", "")
        if doi:
            doi = doi.replace("https://doi.org/", "")
        
        paper = {
            "id": work.get("id", "").split("/")[-1] if work.get("id") else str(uuid.uuid4()),
            "title": work.get("display_name", "Unknown"),
            "authors": ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
            "year": str(year) if year else "",
            "abstract": work.get("abstract", "") or "",
            "doi": doi,
            "pdf_url": pdf_url or "",
            "citations": str(work.get("cited_by_count", 0)),
        }
        
        results.append(paper)
    
    print(f"   ✅ Found {len(results)} papers from OpenAlex")
    return results[:max_results]


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
        print(f"   📥 Downloading PDF...")
        pdf_bytes = await downloader.download(pdf_url)
        
        if pdf_bytes and len(pdf_bytes) > 1000:
            print(f"   ✅ Downloaded {len(pdf_bytes) / 1024 / 1024:.1f} MB")
            # Extract text from PDF (simplified - would use pdfplumber)
            full_text = f"PDF content extracted ({len(pdf_bytes)} bytes)\n\n"
            full_text += f"Title: {title}\n\nAbstract:\n{abstract}"
        else:
            print(f"   ⚠️ PDF download failed or too small")
    
    # Parse authors into Author objects
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


# ============== Main Chatbot Class ==============

class PerspicaciteChatbot:
    """
    Full Perspicacité v2 chatbot with OpenAlex integration.
    """
    
    def __init__(
        self,
        openai_api_key: str | None = None,
        chroma_dir: str = "./chroma_data",
    ):
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config = self._create_config()
        self.llm_client = AsyncLLMClient(self.config.llm)
        self.embedding_provider = SimpleOpenAIEmbeddingProvider(model="text-embedding-3-small")
        self.vector_store = ChromaVectorStore(
            persist_dir=str(self.chroma_dir),
            embedding_provider=self.embedding_provider,
        )
        self.downloader = PDFDownloader()
        
        print(f"🤖 Perspicacité v2 Chatbot initialized")
        print(f"   📊 ChromaDB: {self.chroma_dir}")
        print(f"   🧠 Embedding: text-embedding-3-small")
        print(f"   🤖 LLM: gpt-4o")
    
    def _create_config(self) -> Config:
        """Create configuration for the chatbot."""
        config = Config()
        
        # Configure LLM for OpenAI
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
    
    async def search_and_add_papers(
        self,
        query: str,
        max_results: int = 5,
    ) -> tuple[DynamicKnowledgeBase, list[Paper]]:
        """
        Search papers with OpenAlex and add to a new knowledge base.
        
        Returns:
            (DynamicKnowledgeBase, list of Paper objects)
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        # Search OpenAlex
        paper_data_list = await scilex_search_openalex(query, max_results)
        
        if not paper_data_list:
            raise RuntimeError("No papers found in search results")
        
        # Download papers
        papers: list[Paper] = []
        for i, paper_data in enumerate(paper_data_list[:max_results], 1):
            print(f"\n   [{i}/{min(len(paper_data_list), max_results)}] {paper_data.get('title', 'Unknown')[:60]}...")
            paper = await download_paper(paper_data, self.downloader)
            if paper:
                papers.append(paper)
        
        print(f"\n   ✅ Successfully prepared {len(papers)} papers")
        
        # Create knowledge base
        kb_config = KnowledgeBaseConfig(
            chunk_size=1000,
            chunk_overlap=200,
            top_k=5,
            min_relevance_score=0.5,
        )
        
        kb = DynamicKnowledgeBase(
            vector_store=self.vector_store,
            embedding_service=self.embedding_provider,
            config=kb_config,
        )
        
        # Add papers to KB
        await kb.add_papers(papers, include_full_text=True)
        
        return kb, papers
    
    async def answer(
        self,
        query: str,
        kb: DynamicKnowledgeBase,
        mode: RAGMode = RAGMode.AGENTIC,
    ) -> str:
        """
        Answer a question using the knowledge base.
        
        Args:
            query: The question to answer
            kb: The knowledge base to search
            mode: RAG mode (AGENTIC for full agentic research)
        
        Returns:
            Answer string with sources
        """
        print(f"\n🤔 Answering question with {mode.value} mode...")
        
        # Create request
        request = RAGRequest(
            query=query,
            kb_name=kb.collection_name,
            mode=mode,
            provider="openai",
            model="gpt-4o",
            max_iterations=3,
        )
        
        # Create tools registry
        tools = ToolRegistry()
        kb_tool = KBSearchTool(self.vector_store, self.embedding_provider)
        tools.register(kb_tool)
        
        # Execute based on mode
        if mode == RAGMode.AGENTIC:
            agentic = AgenticRAGMode(self.config)
            response = await agentic.execute(
                request=request,
                llm=self.llm_client,
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                tools=tools,
            )
        else:
            # Use simpler RAG for other modes
            raise NotImplementedError(f"Mode {mode} not yet implemented")
        
        # Format answer with sources
        answer = response.answer
        
        if response.sources:
            answer += "\n\n📚 Sources:\n"
            for i, source in enumerate(response.sources[:5], 1):
                citation = source.to_citation()
                answer += f"   {i}. {citation} {source.title[:80]}...\n"
        
        return answer
    
    async def chat(
        self,
        query: str,
        max_papers: int = 5,
    ) -> str:
        """
        Complete chat flow: search papers → add to KB → answer question.
        
        Args:
            query: The research question
            max_papers: Maximum number of papers to fetch
        
        Returns:
            Answer with citations
        """
        # Step 1: Search and add papers
        kb, papers = await self.search_and_add_papers(query, max_papers)
        
        try:
            # Step 2: Answer the question
            answer = await self.answer(query, kb, mode=RAGMode.AGENTIC)
            return answer
        finally:
            # Cleanup
            await kb.cleanup()


# ============== Main ==============

async def main():
    """Run the chatbot."""
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    
    # Create chatbot
    chatbot = PerspicaciteChatbot(openai_api_key=api_key)
    
    # Research question
    question = "What is feature-based molecular networking and how is it used in metabolomics?"
    
    print(f"\n{'='*70}")
    print(f"🧬 Research Question: {question}")
    print(f"{'='*70}")
    
    # Get answer
    try:
        answer = await chatbot.chat(question, max_papers=3)
        print(f"\n{'='*70}")
        print("📝 ANSWER:")
        print(f"{'='*70}")
        print(answer)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
