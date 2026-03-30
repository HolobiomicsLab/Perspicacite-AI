"""Basic RAG Mode - Exact implementation from release package.

Basic RAG performs simple retrieval and generation:
- Single query (no rephrasing)
- Vector similarity search with optional hybrid retrieval
- Basic document selection
- Direct response generation (no refinement)
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.prompts import (
    DEFAULT_SYSTEM_PROMPT,
)
from perspicacite.retrieval.hybrid import hybrid_retrieval
from perspicacite.rag.utils import (
    format_references,
    prepare_sources,
    get_doc_citation,
    format_documents_for_prompt,
    get_system_prompt,
)

logger = get_logger("perspicacite.rag.modes.basic")


class BasicRAGMode(BaseRAGMode):
    """
    Basic RAG Mode - Exact port from release package core/core.py

    Characteristics:
    - Single query retrieval (no query expansion)
    - Vector-based similarity search with optional hybrid retrieval
    - No response refinement
    - Fastest mode, suitable for simple factual queries
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.initial_docs = config.knowledge_base.default_top_k * 3  # 30 default
        self.final_max_docs = 5
        self.max_docs_per_source = 2

        # Enable hybrid retrieval by default for better retrieval quality
        rag_settings = getattr(config.rag_modes, "basic", None)
        if rag_settings is None:
            rag_settings = {}
        elif hasattr(rag_settings, "model_dump"):
            rag_settings = rag_settings.model_dump()
        elif hasattr(rag_settings, "dict"):
            rag_settings = rag_settings.dict()

        self.use_hybrid = rag_settings.get("use_hybrid", True)

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
        logger.info("basic_rag_start", query=request.query, use_hybrid=self.use_hybrid)

        # Step 1: Generate query embedding
        query_embedding = await embedding_provider.embed([request.query])

        # Step 2: Retrieve documents using vector search
        logger.info("basic_retrieve_documents", query=request.query[:100])

        raw_results = await vector_store.search(
            collection=chroma_collection_name_for_kb(request.kb_name),
            query_embedding=query_embedding[0],
            top_k=self.initial_docs,
        )

        logger.info("basic_retrieved_raw", count=len(raw_results))

        # Step 3: Apply hybrid retrieval if enabled
        if self.use_hybrid and raw_results and llm is not None:
            try:
                logger.info("basic_applying_hybrid", query=request.query[:100])

                # Extract vector scores
                vector_scores = [getattr(r, "score", 0.5) for r in raw_results]

                # Apply hybrid retrieval
                hybrid_results = await hybrid_retrieval(
                    query=request.query,
                    documents=raw_results,
                    vector_scores=vector_scores,
                    use_llm_weights=True,
                    llm=llm,
                )

                # Replace results with hybrid-scored versions
                raw_results = [doc for doc, _ in hybrid_results]
                for doc, hybrid_score in hybrid_results:
                    doc.score = hybrid_score

                logger.info("basic_hybrid_applied", num_results=len(raw_results))

            except Exception as e:
                logger.warning("basic_hybrid_error", error=str(e))

        # Step 4: Basic document selection (deduplication by source)
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

        # Step 5: Prepare sources using utility function
        sources = prepare_sources(selected_documents, max_docs=self.final_max_docs)

        # Step 6: Append references section to answer using utility function
        if sources:
            references = format_references(sources)
            answer = answer.strip() + "\n\n" + references

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
        """Execute Basic RAG with true streaming output."""
        yield StreamEvent.status("Basic RAG: Retrieving documents...")

        # Step 1: Generate query embedding and retrieve documents
        query_embedding = await embedding_provider.embed([request.query])

        raw_results = await vector_store.search(
            collection=chroma_collection_name_for_kb(request.kb_name),
            query_embedding=query_embedding[0],
            top_k=self.initial_docs,
        )

        # Apply hybrid retrieval if enabled
        if self.use_hybrid and raw_results and llm is not None:
            try:
                from perspicacite.retrieval.hybrid import hybrid_retrieval

                vector_scores = [getattr(r, "score", 0.5) for r in raw_results]
                hybrid_results = await hybrid_retrieval(
                    query=request.query,
                    documents=raw_results,
                    vector_scores=vector_scores,
                    use_llm_weights=True,
                    llm=llm,
                )
                raw_results = [doc for doc, _ in hybrid_results]
                for doc, hybrid_score in hybrid_results:
                    doc.score = hybrid_score
            except Exception as e:
                logger.warning("basic_hybrid_error", error=str(e))

        # Step 2: Basic document selection
        selected_documents = []
        source_counts = {}

        for doc in raw_results:
            citation = get_doc_citation(doc)
            source_counts[citation] = source_counts.get(citation, 0) + 1
            if source_counts[citation] > self.max_docs_per_source:
                continue
            selected_documents.append(doc)
            if len(selected_documents) >= self.final_max_docs:
                break

        # Step 3: Prepare sources
        sources = prepare_sources(selected_documents, max_docs=self.final_max_docs)
        for source in sources:
            yield StreamEvent.source(source)

        # Step 4: Stream the response generation
        if not selected_documents:
            yield StreamEvent.content("No relevant documents found to answer your question.")
            yield StreamEvent.done(
                conversation_id="",
                tokens_used=0,
                mode="basic",
                iterations=1,
            )
            return

        yield StreamEvent.status("Basic RAG: Generating response...")

        context = format_documents_for_prompt(selected_documents)
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {request.query}"},
        ]

        # Stream the LLM response
        full_response = ""
        try:
            async for chunk in llm.stream(
                messages=messages,
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            ):
                full_response += chunk
                yield StreamEvent.content(chunk)
        except Exception as e:
            logger.error("basic_streaming_error", error=str(e))
            # Fall back to non-streaming
            answer = await self._generate_response(
                query=request.query,
                documents=selected_documents,
                llm=llm,
                request=request,
            )
            yield StreamEvent.content(answer)
            full_response = answer

        # Append references section after streaming completes
        if sources:
            references = format_references(sources)
            yield StreamEvent.content("\n\n" + references)

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="basic",
            iterations=1,
        )

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

        # Format context using utility function
        context = format_documents_for_prompt(documents)

        # Build user prompt with context
        template = f"""Based on the following research documents, please answer this question:

Question: {query}

Documents:
{context}

---

Instructions:
- Provide a comprehensive answer with clear sections
- Use markdown formatting (headings ##, bullet points -, minimal bold **)
- Base your answer on the documents provided
- Number of documents: {len(documents)}
- Unique sources: {len(set(get_doc_citation(d) for d in documents))}"""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": template},
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
