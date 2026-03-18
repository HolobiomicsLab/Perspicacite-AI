"""Citation RAG mode - citation network analysis."""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.citation")


class CitationRAGMode(BaseRAGMode):
    """Citation RAG - analyze citation network."""

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """Execute citation RAG."""
        # Step 1: Initial search for seed papers
        query_embedding = await embedding_provider.embed([request.query])
        seed_papers = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=5,
        )

        # Step 2: Expand via citation network (placeholder)
        # In full implementation, would fetch citations
        all_papers = list(seed_papers)

        # Step 3: Score by centrality + relevance
        all_papers.sort(key=lambda x: x.score, reverse=True)
        top_papers = all_papers[:10]

        # Step 4: Generate literature review
        context = "\n\n".join(
            f"[Paper {i+1}] {p.chunk.metadata.title or 'Untitled'}\n{p.chunk.text}"
            for i, p in enumerate(top_papers)
        )

        messages = [
            {
                "role": "system",
                "content": "Provide a literature review synthesizing these papers. "
                "Identify key themes, seminal works, and research gaps.",
            },
            {"role": "user", "content": f"Papers:\n{context}\n\nResearch question: {request.query}"},
        ]

        answer = await llm.complete(
            messages=messages,
            model=request.model,
            provider=request.provider,
        )

        sources = [
            SourceReference(
                title=p.chunk.metadata.title or "Untitled",
                authors=p.chunk.metadata.authors,
                year=p.chunk.metadata.year,
                doi=p.chunk.metadata.doi,
                relevance_score=p.score,
            )
            for p in top_papers
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.CITATION,
            iterations=3,
        )

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute citation RAG with streaming."""
        yield StreamEvent.status("Finding seed papers...")

        query_embedding = await embedding_provider.embed([request.query])
        results = await vector_store.search(
            collection=request.kb_name,
            query_embedding=query_embedding[0],
            top_k=10,
        )

        yield StreamEvent.status(f"Analyzing citation network...")

        context = "\n\n".join(
            f"[Paper {i+1}] {r.chunk.metadata.title or 'Untitled'}\n{r.chunk.text}"
            for i, r in enumerate(results)
        )

        messages = [
            {
                "role": "system",
                "content": "Provide a literature review with citations.",
            },
            {"role": "user", "content": f"Papers:\n{context}\n\nResearch question: {request.query}"},
        ]

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
            mode="citation",
            iterations=3,
        )
