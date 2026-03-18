"""Quick RAG mode - single-pass vector search."""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent

logger = get_logger("perspicacite.rag.modes.quick")


class QuickRAGMode:
    """Quick RAG - single-pass vector search."""

    def __init__(self, config: Config):
        self.config = config

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """Execute quick RAG."""
        # Embed query
        query_embedding = await embedding_provider.embed([request.query])

        # Search
        results = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=5,
        )

        # Build context
        context = "\n\n".join(
            f"[Document {i+1}] {r.chunk.text}"
            for i, r in enumerate(results)
        )

        # Generate answer
        messages = [
            {"role": "system", "content": "Answer based on the provided documents."},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {request.query}"},
        ]

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
            mode=RAGMode.QUICK,
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
        """Execute quick RAG with streaming."""
        # Yield status
        yield StreamEvent.status("Searching knowledge base...")

        # Search
        query_embedding = await embedding_provider.embed([request.query])
        results = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=5,
        )

        yield StreamEvent.status(f"Found {len(results)} documents")

        # Stream generation
        context = "\n\n".join(
            f"[Document {i+1}] {r.chunk.text}"
            for i, r in enumerate(results)
        )

        messages = [
            {"role": "system", "content": "Answer based on the provided documents."},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {request.query}"},
        ]

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

        # Done
        yield StreamEvent.done(
            conversation_id="",  # Would be set from session
            tokens_used=0,
            mode="quick",
            iterations=1,
        )
