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
from perspicacite.rag.conversation_helpers import (
    build_user_message_with_history,
    compute_retrieval_query,
    format_conversation_block,
)
from perspicacite.rag.query_scope import resolve_paper_scope_for_query
from perspicacite.rag.utils import (
    format_references,
    prepare_sources,
    get_doc_citation,
    format_documents_for_prompt,
    format_paper_results_for_prompt,
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
        self.use_two_pass = getattr(config.knowledge_base, "use_two_pass", True)

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
        logger.info("basic_rag_start", query=request.query, use_hybrid=self.use_hybrid, use_two_pass=self.use_two_pass)

        collection = chroma_collection_name_for_kb(request.kb_name)
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined  # type: ignore[misc]
        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = min(5, getattr(request, "max_papers_retrieval", None) or 5)

        if self.use_two_pass:
            # Two-pass retrieval — identify papers, then fetch all their chunks
            from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

            dkb = DynamicKnowledgeBase(
                vector_store=vector_store,
                embedding_service=embedding_provider,
            )
            dkb.collection_name = collection
            dkb._initialized = True

            paper_results = await dkb.search_two_pass(
                retrieval_query,
                top_k=self.final_max_docs,
                paper_scope=scope,
                max_papers_cap=cap,
            )
            logger.info("basic_two_pass", papers=len(paper_results))
        else:
            # Legacy chunk-level retrieval (no two-pass)
            from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

            dkb = DynamicKnowledgeBase(
                vector_store=vector_store,
                embedding_service=embedding_provider,
            )
            dkb.collection_name = collection
            dkb._initialized = True

            chunk_results = await dkb.search(retrieval_query, top_k=self.final_max_docs)
            paper_results = []
            for r in chunk_results:
                meta = r.get("metadata")
                paper_results.append({
                    "paper_id": getattr(meta, "paper_id", None) if meta else None,
                    "paper_score": r.get("score", 0.0),
                    "title": getattr(meta, "title", None) if meta else None,
                    "authors": getattr(meta, "authors", None) if meta else None,
                    "year": getattr(meta, "year", None) if meta else None,
                    "doi": getattr(meta, "doi", None) if meta else None,
                    "full_text": r.get("text", ""),
                })
            logger.info("basic_chunk_retrieval", chunks=len(chunk_results))

        # Build sources from paper results
        sources = []
        for p in paper_results:
            sources.append(SourceReference(
                title=p.get("title") or "Untitled",
                authors=p.get("authors"),
                year=p.get("year"),
                doi=p.get("doi"),
                relevance_score=p.get("paper_score", 0.0),
            ))

        # Step 2: Generate response using full paper context
        if paper_results:
            context = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
        else:
            context = "No relevant papers found."

        answer = await self._generate_response_from_context(
            query=request.query,
            context=context,
            llm=llm,
            request=request,
            num_papers=len(paper_results),
            preamble=scope.scope_note,
        )

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

        collection = chroma_collection_name_for_kb(request.kb_name)
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined  # type: ignore[misc]
        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = min(5, getattr(request, "max_papers_retrieval", None) or 5)

        if self.use_two_pass:
            from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

            dkb = DynamicKnowledgeBase(
                vector_store=vector_store,
                embedding_service=embedding_provider,
            )
            dkb.collection_name = collection
            dkb._initialized = True

            paper_results = await dkb.search_two_pass(
                retrieval_query,
                top_k=self.final_max_docs,
                paper_scope=scope,
                max_papers_cap=cap,
            )
        else:
            from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

            dkb = DynamicKnowledgeBase(
                vector_store=vector_store,
                embedding_service=embedding_provider,
            )
            dkb.collection_name = collection
            dkb._initialized = True

            chunk_results = await dkb.search(retrieval_query, top_k=self.final_max_docs)
            paper_results = []
            for r in chunk_results:
                meta = r.get("metadata")
                paper_results.append({
                    "paper_id": getattr(meta, "paper_id", None) if meta else None,
                    "paper_score": r.get("score", 0.0),
                    "title": getattr(meta, "title", None) if meta else None,
                    "authors": getattr(meta, "authors", None) if meta else None,
                    "year": getattr(meta, "year", None) if meta else None,
                    "doi": getattr(meta, "doi", None) if meta else None,
                    "full_text": r.get("text", ""),
                })

        # Prepare sources
        sources = []
        for p in paper_results:
            sources.append(SourceReference(
                title=p.get("title") or "Untitled",
                authors=p.get("authors"),
                year=p.get("year"),
                doi=p.get("doi"),
                relevance_score=p.get("paper_score", 0.0),
            ))
        for source in sources:
            yield StreamEvent.source(source)

        if not paper_results:
            yield StreamEvent.content("No relevant documents found to answer your question.")
            yield StreamEvent.done(
                conversation_id="",
                tokens_used=0,
                mode="basic",
                iterations=1,
            )
            return

        yield StreamEvent.status("Basic RAG: Generating response...")

        context = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        user_body = f"Documents:\n{context}\n\nQuestion: {request.query}"
        if scope.scope_note:
            user_body = f"{scope.scope_note}\n\n{user_body}"
        user_content = build_user_message_with_history(history_block=hist, body=user_body)
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_content},
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
            answer = await self._generate_response_from_context(
                query=request.query,
                context=context,
                llm=llm,
                request=request,
                num_papers=len(paper_results),
                preamble=scope.scope_note,
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

    async def _generate_response_from_context(
        self,
        query: str,
        context: str,
        llm: Any,
        request: RAGRequest,
        num_papers: int = 0,
        *,
        preamble: str | None = None,
    ) -> str:
        """Generate response from pre-formatted paper context."""
        template = f"""Based on the following research papers, please answer this question:

Question: {query}

{context}

---

Instructions:
- Provide a comprehensive answer with clear sections
- Use markdown formatting (headings ##, bullet points -, minimal bold **)
- Base your answer on the papers provided
- Number of papers: {num_papers}
- IMPORTANT: Extract ALL specific names, tools, methods, chemicals, organisms, or other entities mentioned in the papers that are relevant to the question. Do NOT say "specific names are not listed" if the papers contain them.
- Be specific and concrete — cite specific tools, software, methods, or findings by name rather than giving vague generalizations."""
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        body = template
        if preamble:
            body = f"{preamble}\n\n{body}"
        user_content = build_user_message_with_history(history_block=hist, body=body)

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": user_content},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.15,
            )
            return response
        except Exception as e:
            logger.error("basic_response_generation_error", error=str(e))
            return f"Error generating response: {e}"
