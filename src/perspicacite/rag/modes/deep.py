"""Deep RAG mode - multi-cycle research with planning."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode

logger = get_logger("perspicacite.rag.modes.deep")


@dataclass
class ResearchStep:
    """A single research step."""

    query: str
    purpose: str
    documents: list[Any] = field(default_factory=list)
    analysis: str = ""
    confidence: float = 0.0
    success: bool = False


class DeepRAGMode(BaseRAGMode):
    """Deep RAG - multi-cycle research with planning and reflection."""

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """Execute deep RAG."""
        max_iterations = request.max_iterations or self.config.rag_modes.deep.max_iterations

        # Create research plan
        plan = await self._create_research_plan(request.query, llm, request)

        all_documents = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.debug("deep_research_iteration", iteration=iteration)

            # Execute research steps
            for step in plan:
                query_embedding = await embedding_provider.embed([step.query])
                results = await vector_store.search(
                    collection=request.kb_name,
                    query_embedding=query_embedding[0],
                    top_k=5,
                )

                if results:
                    step.documents = results
                    step.success = True
                    all_documents.extend(results)

            # Check if we should continue
            if iteration >= max_iterations:
                break

            # Adjust plan for next iteration
            plan = await self._adjust_plan(plan, request.query, llm, request)

        # Deduplicate and sort documents
        seen_ids = set()
        unique_docs = []
        for doc in all_documents:
            if doc.chunk.id not in seen_ids:
                seen_ids.add(doc.chunk.id)
                unique_docs.append(doc)

        unique_docs.sort(key=lambda x: x.score, reverse=True)
        top_docs = unique_docs[:10]

        # Generate final answer
        context = "\n\n".join(
            f"[Document {i+1}] {d.chunk.text}"
            for i, d in enumerate(top_docs)
        )

        messages = self._build_messages(
            request.query,
            context,
            "Provide a comprehensive answer based on these research findings.",
        )

        answer = await llm.complete(
            messages=messages,
            model=request.model,
            provider=request.provider,
        )

        sources = [
            SourceReference(
                title=d.chunk.metadata.title or "Untitled",
                authors=d.chunk.metadata.authors,
                year=d.chunk.metadata.year,
                relevance_score=d.score,
            )
            for d in top_docs
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.DEEP,
            iterations=iteration,
            research_plan=[s.purpose for s in plan],
        )

    async def _create_research_plan(
        self,
        query: str,
        llm: Any,
        request: RAGRequest,
    ) -> list[ResearchStep]:
        """Create initial research plan."""
        messages = [
            {
                "role": "system",
                "content": "Break down this research question into 2-3 specific sub-questions. "
                "For each, provide a search query and purpose.",
            },
            {"role": "user", "content": f"Research question: {query}"},
        ]

        try:
            response = await llm.complete(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=500,
            )

            # Simple parsing - create steps from response
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            steps = []

            for i, line in enumerate(lines[:3]):
                steps.append(
                    ResearchStep(
                        query=f"{query} {line}",
                        purpose=line,
                    )
                )

            return steps if steps else [ResearchStep(query=query, purpose="Main research")]

        except Exception:
            return [ResearchStep(query=query, purpose="Main research")]

    async def _adjust_plan(
        self,
        current_plan: list[ResearchStep],
        original_query: str,
        llm: Any,
        request: RAGRequest,
    ) -> list[ResearchStep]:
        """Adjust research plan based on findings."""
        # For now, return same plan
        # Full implementation would analyze gaps
        return current_plan

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute deep RAG with streaming."""
        yield StreamEvent.status("Creating research plan...")

        plan = await self._create_research_plan(request.query, llm, request)
        yield StreamEvent.status(f"Research plan: {len(plan)} steps")

        for i, step in enumerate(plan):
            yield StreamEvent(
                event="plan",
                data=f'{{"step_index": {i}, "total_steps": {len(plan)}, "query": "{step.query}", "purpose": "{step.purpose}"}}',
            )

        # Simplified - just do one search
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
            mode="deep",
            iterations=1,
        )
