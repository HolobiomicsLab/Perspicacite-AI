"""Basic RAG Mode - Exact implementation from release package.

Basic RAG performs simple retrieval and generation:
- Single query (no rephrasing)
- Vector similarity search
- Basic document selection
- Direct response generation (no refinement)
"""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.basic")


class BasicRAGMode(BaseRAGMode):
    """
    Basic RAG Mode - Exact port from release package core/core.py
    
    Characteristics:
    - Single query retrieval (no query expansion)
    - Vector-based similarity search only
    - No hybrid retrieval (BM25)
    - No response refinement
    - Fastest mode, suitable for simple factual queries
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.initial_docs = config.knowledge_base.default_top_k * 3  # 30 default
        self.final_max_docs = 5
        self.max_docs_per_source = 2

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """
        Execute Basic RAG - single query, direct retrieval, no refinement.
        
        Ported from: core/core.py::retrieve_documents() and get_response()
        """
        logger.info("basic_rag_start", query=request.query)

        # Step 1: Generate query embedding
        query_embedding = await embedding_provider.embed([request.query])
        
        # Step 2: Retrieve documents using vector search only
        # This is the "basic" approach - no hybrid retrieval
        logger.info("basic_retrieve_documents", query=request.query[:100])
        
        raw_results = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=self.initial_docs,
        )
        
        logger.info("basic_retrieved_raw", count=len(raw_results))

        # Step 3: Basic document selection (no WRRF, no re-ranking)
        # Simple deduplication by source
        selected_documents = []
        source_counts = {}
        
        for doc in raw_results:
            # Extract source citation
            citation = self._get_doc_citation(doc)
            
            # Limit docs per source
            source_counts[citation] = source_counts.get(citation, 0) + 1
            if source_counts[citation] > self.max_docs_per_source:
                continue
                
            selected_documents.append(doc)
            
            if len(selected_documents) >= self.final_max_docs:
                break
        
        logger.info("basic_selected_docs", count=len(selected_documents))

        # Step 4: Generate response (no refinement for Basic mode)
        answer = await self._generate_response(
            query=request.query,
            documents=selected_documents,
            llm=llm,
            request=request,
        )

        # Step 5: Prepare sources
        sources = self._prepare_sources(selected_documents)

        logger.info("basic_rag_complete", sources=len(sources))

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.BASIC,
            iterations=1,
            web_search_used=False,
        )

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute Basic RAG with streaming output."""
        yield StreamEvent.status("Basic RAG: Retrieving documents...")
        
        # Delegate to non-streaming for core logic
        response = await self.execute(
            request, llm, vector_store, embedding_provider, tools
        )
        
        yield StreamEvent.status("Basic RAG: Generating answer...")
        
        # Stream the answer word by word
        words = response.answer.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield StreamEvent.content(chunk)
        
        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="basic",
            iterations=1,
        )

    def _get_doc_citation(self, doc: Any) -> str:
        """Extract citation from document."""
        if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'metadata'):
            meta = doc.chunk.metadata
            if hasattr(meta, 'citation'):
                return meta.citation
            if hasattr(meta, 'title'):
                return meta.title
        if isinstance(doc, dict):
            return doc.get('citation', doc.get('source', 'Unknown'))
        return 'Unknown'

    def _format_documents_for_prompt(self, documents: list[Any]) -> str:
        """Format documents for inclusion in prompt."""
        formatted = []
        
        for i, doc in enumerate(documents, 1):
            # Extract text content
            if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'text'):
                text = doc.chunk.text
            elif hasattr(doc, 'content'):
                text = str(doc.content)
            else:
                text = str(doc)
            
            # Extract citation
            citation = self._get_doc_citation(doc)
            
            formatted.append(f"[{i}] Source: {citation}\n{text}")
        
        return "\n\n---\n\n".join(formatted)

    async def _generate_response(
        self,
        query: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
    ) -> str:
        """Generate response without refinement (Basic mode)."""
        
        if not documents:
            return "No relevant documents found to answer your question."
        
        # Format context
        context = self._format_documents_for_prompt(documents)
        
        # System prompt (simplified from release package)
        system_prompt = """You are a scientific research assistant. Answer the user's question based on the provided documents.

Guidelines:
- Base your answer on the provided documents
- Be concise and direct
- Include citations like [1], [2], etc.
- If documents don't contain sufficient information, say so"""

        # Build user message
        user_message = f"""Documents:
{context}

Question: {query}

Provide a clear answer based on the documents above."""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            )
            return response
        except Exception as e:
            logger.error("basic_response_generation_error", error=str(e))
            return f"Error generating response: {e}"

    def _prepare_sources(self, documents: list[Any]) -> list[SourceReference]:
        """Prepare source references from documents."""
        seen = set()
        sources = []
        
        for doc in documents:
            # Extract metadata
            if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'metadata'):
                meta = doc.chunk.metadata
                title = getattr(meta, 'title', 'Untitled')
                authors = getattr(meta, 'authors', [])
                year = getattr(meta, 'year', None)
                doi = getattr(meta, 'doi', None)
            elif isinstance(doc, dict):
                title = doc.get('title', doc.get('source', 'Unknown'))
                authors = doc.get('authors', [])
                year = doc.get('year')
                doi = doc.get('doi')
            else:
                continue

            # Deduplicate by title
            if title in seen:
                continue
            seen.add(title)

            # Format authors
            authors_str = None
            if authors:
                if isinstance(authors, list):
                    authors_str = ", ".join(str(a) for a in authors[:3])
                    if len(authors) > 3:
                        authors_str += " et al."
                else:
                    authors_str = str(authors)

            sources.append(SourceReference(
                title=title,
                authors=authors_str,
                year=year,
                doi=doi,
                relevance_score=getattr(doc, 'score', 0.0),
            ))

        return sources
