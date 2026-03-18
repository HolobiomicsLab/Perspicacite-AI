"""Standard RAG mode - hybrid search + reranking."""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.standard")


class StandardRAGMode(BaseRAGMode):
    """Standard RAG - hybrid search + reranking."""

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """Execute standard RAG with hybrid search."""
        # Get mode config
        mode_config = self.config.rag_modes.standard

        # Embed query
        query_embedding = await embedding_provider.embed([request.query])

        # Vector search
        vector_results = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=20,
        )

        # In full implementation, would also do BM25 and fuse
        # For now, use vector results
        results = vector_results[:10]

        # Build context
        context = "\n\n".join(
            f"[Document {i+1}] {r.chunk.text}"
            for i, r in enumerate(results)
        )

        # Generate answer
        messages = self._build_messages(request.query, context)

        answer = await llm.complete(
            messages=messages,
            model=request.model,
            provider=request.provider,
        )

        # Convert to sources
        sources = [
            SourceReference(
                title=r.chunk.metadata.title or "Untitled",
                authors=r.chunk.metadata.authors,
                year=r.chunk.metadata.year,
                doi=r.chunk.metadata.doi,
                relevance_score=r.score,
            )
            for r in results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.STANDARD,
            iterations=1,
        )

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute standard RAG with streaming."""
        yield StreamEvent.status("Searching knowledge base...")

        query_embedding = await embedding_provider.embed([request.query])
        results = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=10,
        )

        yield StreamEvent.status(f"Found {len(results)} documents")

        context = "\n\n".join(
            f"[Document {i+1}] {r.chunk.text}"
            for i, r in enumerate(results)
        )

        messages = self._build_messages(request.query, context)

        # Yield sources
        for r in results:
            yield StreamEvent(
                event="source",
                data=f'{{"title": "{r.chunk.metadata.title or "Untitled"}", "score": {r.score:.3f}}}',
            )

        # Stream content
        async for chunk in llm.stream(
            messages=messages,
            model=request.model,
            provider=request.provider,
        ):
            yield StreamEvent.content(chunk)

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="standard",
            iterations=1,
        )
