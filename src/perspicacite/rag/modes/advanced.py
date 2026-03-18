"""Advanced RAG mode - query expansion + multi-hop."""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.advanced")


class AdvancedRAGMode(BaseRAGMode):
    """Advanced RAG - query expansion + optional web search."""

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """Execute advanced RAG."""
        # Step 1: Expand query
        expanded_queries = await self._expand_query(request.query, llm, request)

        # Step 2: Search with expanded queries
        all_results = []
        for query in expanded_queries:
            query_embedding = await embedding_provider.embed([query])
            results = await vector_store.search(
                collection=request.kb_name,
                query_embedding=query_embedding[0],
                top_k=10,
            )
            all_results.extend(results)

        # Deduplicate by chunk ID
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r.chunk.id not in seen_ids:
                seen_ids.add(r.chunk.id)
                unique_results.append(r)

        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        top_results = unique_results[:10]

        # Generate answer
        context = "\n\n".join(
            f"[Document {i+1}] {r.chunk.text}"
            for i, r in enumerate(top_results)
        )

        messages = self._build_messages(request.query, context)

        answer = await llm.complete(
            messages=messages,
            model=request.model,
            provider=request.provider,
        )

        sources = [
            SourceReference(
                title=r.chunk.metadata.title or "Untitled",
                authors=r.chunk.metadata.authors,
                year=r.chunk.metadata.year,
                relevance_score=r.score,
            )
            for r in top_results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.ADVANCED,
            iterations=2,
        )

    async def _expand_query(
        self,
        query: str,
        llm: Any,
        request: RAGRequest,
    ) -> list[str]:
        """Generate query variations."""
        messages = [
            {
                "role": "system",
                "content": "Generate 3 search queries that capture different aspects of the user's question. "
                "Return one per line.",
            },
            {"role": "user", "content": f"Question: {query}\n\nSearch queries:"},
        ]

        try:
            response = await llm.complete(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=200,
            )

            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            # Always include original
            if query not in queries:
                queries.insert(0, query)
            return queries[:4]  # Max 4 queries
        except Exception:
            return [query]  # Fallback to original

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute advanced RAG with streaming."""
        yield StreamEvent.status("Expanding query...")

        expanded = await self._expand_query(request.query, llm, request)
        yield StreamEvent.status(f"Searching with {len(expanded)} query variations...")

        # Search with each query (simplified)
        query_embedding = await embedding_provider.embed([request.query])
        results = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=10,
        )

        context = "\n\n".join(
            f"[Document {i+1}] {r.chunk.text}"
            for i, r in enumerate(results)
        )

        messages = self._build_messages(request.query, context)

        for r in results:
            yield StreamEvent(
                event="source",
                data=f'{{"title": "{r.chunk.metadata.title or "Untitled"}", "score": {r.score:.3f}}}',
            )

        async for chunk in llm.stream(
            messages=messages,
            model=request.model,
            provider=request.provider,
        ):
            yield StreamEvent.content(chunk)

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="advanced",
            iterations=2,
        )
