"""Advanced RAG Mode - Exact implementation from release package.

Advanced RAG adds:
- Query rephrasing/expansion (generate_similar_queries)
- Hybrid retrieval (vector + BM25-inspired scoring)
- WRRF scoring for multi-query fusion
- Optional response refinement
"""

import math
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, SourceReference, StreamEvent
from perspicacite.rag.modes.base import BaseRAGMode
from perspicacite.rag.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    MANDATORY_PROMPT,
    get_mandatory_prompt,
    FORMAT_PROMPT,
    GENERATE_SIMILAR_QUERIES_PROMPT,
    EVALUATE_RESPONSE_PROMPT,
    REFINE_RESPONSE_SYSTEM_PROMPT,
    REFINE_RESPONSE_HUMAN_PROMPT_SUFFIX,
    FOCUS_INSTRUCTIONS_PROMPT,
)
from perspicacite.models.kb import chroma_collection_name_for_kb
from perspicacite.rag.conversation_helpers import compute_retrieval_query, format_conversation_block
from perspicacite.rag.query_scope import merge_scope_with_candidates, resolve_paper_scope_for_query
from perspicacite.retrieval.hybrid import hybrid_retrieval
from perspicacite.rag.relevancy import assess_query_complexity, reorder_documents_by_relevance
from perspicacite.rag.utils import (
    format_references,
    prepare_sources,
    get_doc_citation,
    format_documents_for_prompt,
    format_paper_results_for_prompt,
)
from perspicacite.rag.wrrf_v1 import doc_page_content, select_wrrf_merged_documents

logger = get_logger("perspicacite.rag.modes.advanced")


class AdvancedRAGMode(BaseRAGMode):
    """
    Advanced RAG Mode - Exact port from release package core/core.py

    Characteristics:
    - Query rephrasing using LLM (generate_similar_queries)
    - Multiple query execution
    - WRRF (Weighted Reciprocal Rank Fusion) scoring
    - Hybrid retrieval support (when enabled)
    - Optional response refinement (if use_refinement=True)
    """

    def __init__(self, config: Any):
        super().__init__(config)
        rag_settings = getattr(config.rag_modes, "advanced", None)

        # Handle both dict and Pydantic model
        if rag_settings is None:
            rag_settings = {}
        elif hasattr(rag_settings, "model_dump"):
            # Pydantic v2 model
            rag_settings = rag_settings.model_dump()
        elif hasattr(rag_settings, "dict"):
            # Pydantic v1 model
            rag_settings = rag_settings.dict()

        self.initial_docs = 150  # From release package
        self.final_max_docs = 5
        self.max_docs_per_source = 1
        self.rephrases = 3  # Number of additional queries to generate
        self.use_refinement = rag_settings.get("enable_reflection", False)
        self.use_hybrid = rag_settings.get("use_hybrid", True)  # Enable hybrid retrieval by default
        self.use_two_pass = getattr(config.knowledge_base, "use_two_pass", True)

        # WRRF constants from release package
        self.wrrf_k = 60

        # Sigmoid parameters for score normalization
        self.pth = 0.8  # threshold
        self.stp = 30  # steepness
        # v1 get_response: relevancy reorder + focus instructions
        self.use_relevancy_optimization = rag_settings.get("use_relevancy_optimization", True)
        # v1 profonde: refinement_iterations clamped 1–3; core refine_response default 2
        self.refinement_iterations = max(1, min(int(rag_settings.get("refinement_iterations", 2)), 3))
        self.evaluator_model = rag_settings.get("evaluator_model")
        self.evaluator_provider = rag_settings.get("evaluator_provider")
        # v1 get_response: max length for mandatory + DEFAULT_SYSTEM_PROMPT before focus instructions
        llm_cfg = getattr(config, "llm", None)
        self.max_context_window = int(
            getattr(llm_cfg, "max_context_window", None) or 10000
        )

    def _truncate_combined_prompt_base(self, combined_prompt: str) -> str:
        """core/core.py get_response: truncate mandatory + system only (before focus)."""
        if len(combined_prompt) <= self.max_context_window:
            return combined_prompt
        logger.warning(
            "advanced_prompt_truncated",
            original_len=len(combined_prompt),
            limit=self.max_context_window,
        )
        return combined_prompt[: self.max_context_window]

    @staticmethod
    def _clean_refined_response(text: str) -> str:
        """Approximate v1 LLMWrapper._clean_response when refinement exits early on high score."""
        t = (text or "").strip()
        if t.startswith("```"):
            if t.startswith("```json"):
                t = t.split("```json", 1)[1]
            else:
                t = t.split("```", 1)[1]
            if t.rstrip().endswith("```"):
                t = t.rsplit("```", 1)[0]
        return t.strip()

    def _v1_question_line(self, request: RAGRequest) -> str:
        """core/core.py get_response question field."""
        original_query = request.query
        refined = getattr(request, "refined_query", None) or request.query
        return (
            f"User original question: {original_query}\n"
            f"User refined question (based on conversation history): {refined}"
        )

    def _v1_metadata_dict(self, doc: Any) -> dict[str, Any]:
        if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
            m = doc.chunk.metadata
            if hasattr(m, "model_dump"):
                d = m.model_dump()
            elif isinstance(m, dict):
                d = dict(m)
            else:
                d = {
                    "citation": getattr(m, "citation", None) or getattr(m, "title", ""),
                    "url": getattr(m, "url", "") or "",
                    "chunk": str(getattr(m, "year", "") or getattr(m, "chunk", "")),
                }
            d.setdefault("citation", get_doc_citation(doc))
            d.setdefault("url", "")
            d.setdefault("chunk", "")
            return d
        return {"citation": "Unknown", "url": "", "chunk": ""}

    def _v1_context_and_citations(self, documents: list[Any]) -> tuple[str, str]:
        """Build context_with_citations + citation_list like core/core.py get_response."""
        context_with_citations = ""
        source_counter: Counter[tuple[str, str]] = Counter()
        for doc in documents:
            text = doc_page_content(doc)
            md = self._v1_metadata_dict(doc)
            cit = md.get("citation") or "Unknown"
            url = md.get("url") or ""
            context_with_citations += f"{text}, Citation: {cit})\n\n"
            source_counter[(url, cit)] += 1

        citations: list[str] = []
        for (url, citation), count in source_counter.items():
            chunk = "Unknown date"
            for d in documents:
                m = self._v1_metadata_dict(d)
                if m.get("url") == url and m.get("citation") == citation:
                    chunk = str(m.get("chunk", "Unknown date"))
                    break
            citations.append(f"{citation}. Accessed on: {chunk} (url: {url}). [{count} docs]")
        citation_list = "\n".join(citations)
        return context_with_citations, citation_list

    def _build_v1_answer_chat_chunk_docs(
        self,
        request: RAGRequest,
        documents: list[Any],
    ) -> tuple[str, str, float]:
        """
        Same prompt structure as core/core.py get_response for chunk-level docs
        (streaming path; no two-pass).
        """
        kb_title = getattr(request, "kb_name", None) or "Perspicacité"
        kb_scope = getattr(request, "kb_scope", None) or "scientific research and education"
        mandatory = get_mandatory_prompt(kb_title, kb_scope)
        combined_prompt = self._truncate_combined_prompt_base(
            mandatory + "\n" + DEFAULT_SYSTEM_PROMPT
        )
        if self.use_relevancy_optimization:
            combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        context_with_citations, citation_list = self._v1_context_and_citations(documents)
        n_docs = len(documents)
        n_sources = len({get_doc_citation(d) for d in documents})
        question = self._v1_question_line(request)
        user_template = f"""System prompt: {combined_prompt}
Context: {context_with_citations}
Format: {FORMAT_PROMPT}
Question: {question}

Additional information:
- Total documents used: {n_docs}
- Unique sources: {n_sources}
Sources:
{citation_list}
"""
        if self.use_relevancy_optimization:
            qc = assess_query_complexity(getattr(request, "refined_query", None) or request.query)
            temperature = 0.7 if qc > 0.7 else 0.3
        else:
            temperature = 0.3
        return combined_prompt, user_template, temperature

    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """
        Execute Advanced RAG with query rephrasing and WRRF scoring.

        Ported from: core/core.py::retrieve_documents() and get_response()
        """
        logger.info("advanced_rag_start", query=request.query)

        kb_collection = chroma_collection_name_for_kb(request.kb_name)
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined  # type: ignore[misc]
        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            kb_collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = min(5, getattr(request, "max_papers_retrieval", None) or 5)

        # Step 1: Generate similar/rephrased queries
        # This is the key difference from Basic mode
        logger.info("advanced_generate_queries", original=retrieval_query[:100])

        all_queries = await self._generate_similar_queries(
            original_query=retrieval_query, llm=llm, number=self.rephrases
        )

        logger.info("advanced_queries_generated", count=len(all_queries), queries=all_queries)

        # Step 2: Retrieve documents for all queries with WRRF scoring
        # This uses the WRRF (Weighted Reciprocal Rank Fusion) algorithm
        logger.info("advanced_wrrf_retrieval", num_queries=len(all_queries))

        selected_documents = await self._wrrf_retrieval(
            queries=all_queries,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            kb_name=chroma_collection_name_for_kb(request.kb_name),
            llm=llm,
        )

        logger.info("advanced_selected_docs", count=len(selected_documents), use_two_pass=self.use_two_pass)

        retrieval_docs_for_refine = list(selected_documents)
        if self.use_relevancy_optimization and selected_documents:
            selected_documents = reorder_documents_by_relevance(
                getattr(request, "refined_query", None) or request.query,
                selected_documents,
            )

        # Step 2b: Two-pass — fetch full text for selected papers
        paper_results = []
        if self.use_two_pass:
            from perspicacite.rag.utils import deduplicate_chunk_overlaps

            paper_ids = []
            paper_scores: dict[str, float] = {}
            paper_meta: dict[str, Any] = {}
            for doc in selected_documents:
                meta = getattr(doc, "chunk", None)
                if meta and hasattr(meta, "metadata"):
                    pid = getattr(meta.metadata, "paper_id", None)
                    if pid and pid not in paper_ids:
                        paper_ids.append(pid)
                        score = getattr(doc, "wrrf_score", getattr(doc, "score", 0.5))
                        paper_scores[pid] = score
                        paper_meta[pid] = meta.metadata

            if paper_ids:
                ordered = merge_scope_with_candidates(
                    paper_ids, paper_scores, scope, cap
                )
                all_chunks = await vector_store.get_chunks_by_paper_ids(
                    chroma_collection_name_for_kb(request.kb_name), ordered
                )
                deduped = deduplicate_chunk_overlaps(all_chunks)
                # Group by paper
                from collections import OrderedDict
                grouped: OrderedDict[str, list] = OrderedDict()
                for d in deduped:
                    grouped.setdefault(d["paper_id"], []).append(d)
                for pid in ordered:
                    chunks_list = grouped.get(pid, [])
                    full_text = " ".join(c["text"] for c in chunks_list)
                    meta = paper_meta.get(pid)
                    paper_results.append({
                        "paper_id": pid,
                        "paper_score": paper_scores[pid],
                        "title": getattr(meta, "title", None),
                        "authors": getattr(meta, "authors", None),
                        "year": getattr(meta, "year", None),
                        "doi": getattr(meta, "doi", None),
                        "chunks": chunks_list,
                        "full_text": full_text,
                    })

        # Step 3: Generate response using full paper context
        if paper_results:
            context = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
        else:
            context = format_documents_for_prompt(selected_documents)

        answer = await self._generate_response_from_context(
            query=request.query,
            context=context,
            llm=llm,
            request=request,
            num_papers=len(paper_results),
            source_documents=retrieval_docs_for_refine if not paper_results else None,
            paper_results=paper_results,
            preamble=scope.scope_note,
        )

        # Step 4: Apply refinement if enabled (Advanced mode feature)
        if self.use_refinement and not self._is_streaming(request):
            logger.info("advanced_apply_refinement")
            answer = await self._refine_response(
                response=answer,
                query=request.query,
                context=context,
                documents=retrieval_docs_for_refine,
                llm=llm,
                request=request,
                max_iterations=self.refinement_iterations,
                eval_model=getattr(request, "evaluator_model", None),
                eval_provider=getattr(request, "evaluator_provider", None),
            )

        # Step 5: Prepare sources
        if paper_results:
            sources = []
            for p in paper_results:
                from perspicacite.models.rag import SourceReference
                sources.append(SourceReference(
                    title=p.get("title") or "Untitled",
                    authors=p.get("authors"),
                    year=p.get("year"),
                    doi=p.get("doi"),
                    relevance_score=p.get("paper_score", 0.0),
                ))
        else:
            sources = self._prepare_sources(selected_documents)

        # Step 6: Append references section to answer
        if sources:
            references = self._format_references(sources)
            answer = answer.strip() + "\n\n" + references

        logger.info("advanced_rag_complete", sources=len(sources), refined=self.use_refinement)

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.ADVANCED,
            iterations=1,
            web_search_used=False,
        )

    def _advanced_paper_stream_messages(
        self,
        request: RAGRequest,
        paper_results: list[dict[str, Any]],
        *,
        preamble: str | None,
    ) -> tuple[list[dict[str, str]], float, bool]:
        """Messages for two-pass paper streaming (aligned with ``_generate_response_from_context``)."""
        kb_title = getattr(request, "kb_name", None) or "Perspicacité"
        kb_scope = getattr(request, "kb_scope", None) or "scientific research and education"
        mandatory = get_mandatory_prompt(kb_title, kb_scope)
        combined_prompt = self._truncate_combined_prompt_base(
            mandatory + "\n" + DEFAULT_SYSTEM_PROMPT
        )
        if self.use_relevancy_optimization:
            combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        context_with_citations = ""
        sc: Counter[tuple[str, str]] = Counter()
        for p in paper_results:
            text = (p.get("full_text") or "")[:12000]
            cit = p.get("title") or p.get("doi") or "Unknown"
            doi = p.get("doi") or ""
            url = f"https://doi.org/{doi}" if doi else ""
            context_with_citations += f"{text}, Citation: {cit})\n\n"
            sc[(url, str(cit))] += 1
        citations: list[str] = []
        for (url, citation), count in sc.items():
            year = ""
            for p in paper_results:
                if (p.get("title") or p.get("doi")) == citation or str(p.get("doi")) == citation:
                    year = str(p.get("year") or "")
                    break
            citations.append(f"{citation}. Accessed on: {year} (url: {url}). [{count} docs]")
        citation_list = "\n".join(citations)
        n_docs = len(paper_results)
        n_sources = len(sc)

        question = self._v1_question_line(request)
        template = f"""System prompt: {combined_prompt}
Context: {context_with_citations}
Format: {FORMAT_PROMPT}
Question: {question}

Additional information:
- Total documents used: {n_docs}
- Unique sources: {n_sources}
Sources:
{citation_list}
"""
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        if hist:
            template = f"{hist}\n\n---\n\n{template}"
        if preamble:
            template = f"{preamble.strip()}\n\n{template}"

        req_model = getattr(request, "model", "") or ""
        is_o_series = req_model.startswith("o") or "gpt-5" in req_model
        if self.use_relevancy_optimization:
            qc = assess_query_complexity(getattr(request, "refined_query", None) or request.query)
            temperature = 0.7 if qc > 0.7 else 0.3
        else:
            temperature = 0.3

        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "user", "content": template},
        ]
        return messages, temperature, is_o_series

    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute Advanced RAG with true streaming output."""
        import json

        yield StreamEvent.status("Advanced RAG: Generating query variations...")

        kb_collection = chroma_collection_name_for_kb(request.kb_name)
        retrieval_query, refined = await compute_retrieval_query(request, llm)
        if refined:
            request.refined_query = refined  # type: ignore[misc]
        scope = await resolve_paper_scope_for_query(
            retrieval_query,
            kb_collection,
            vector_store,
            max_papers_override=getattr(request, "max_papers_retrieval", None),
        )
        cap = min(5, getattr(request, "max_papers_retrieval", None) or 5)

        # Step 1: Generate similar/rephrased queries
        all_queries = await self._generate_similar_queries(
            original_query=retrieval_query, llm=llm, number=self.rephrases
        )

        yield StreamEvent.status(
            f"Advanced RAG: Searching with {len(all_queries)} query variations..."
        )

        # Step 2: Retrieve documents using WRRF
        selected_documents = await self._wrrf_retrieval(
            queries=all_queries,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            kb_name=kb_collection,
            llm=llm,
        )

        if self.use_relevancy_optimization and selected_documents:
            selected_documents = reorder_documents_by_relevance(
                getattr(request, "refined_query", None) or request.query,
                selected_documents,
            )

        paper_results: list[dict[str, Any]] = []
        if self.use_two_pass and selected_documents:
            from perspicacite.rag.utils import deduplicate_chunk_overlaps

            paper_ids: list[str] = []
            paper_scores: dict[str, float] = {}
            paper_meta: dict[str, Any] = {}
            for doc in selected_documents:
                meta = getattr(doc, "chunk", None)
                if meta and hasattr(meta, "metadata"):
                    pid = getattr(meta.metadata, "paper_id", None)
                    if pid and pid not in paper_ids:
                        paper_ids.append(pid)
                        score = getattr(doc, "wrrf_score", getattr(doc, "score", 0.5))
                        paper_scores[pid] = score
                        paper_meta[pid] = meta.metadata
            if paper_ids:
                ordered = merge_scope_with_candidates(
                    paper_ids, paper_scores, scope, cap
                )
                all_chunks = await vector_store.get_chunks_by_paper_ids(
                    kb_collection, ordered
                )
                deduped = deduplicate_chunk_overlaps(all_chunks)
                from collections import OrderedDict

                grouped: OrderedDict[str, list] = OrderedDict()
                for d in deduped:
                    grouped.setdefault(d["paper_id"], []).append(d)
                for pid in ordered:
                    chunks_list = grouped.get(pid, [])
                    full_text = " ".join(c["text"] for c in chunks_list)
                    meta = paper_meta.get(pid)
                    paper_results.append({
                        "paper_id": pid,
                        "paper_score": paper_scores[pid],
                        "title": getattr(meta, "title", None),
                        "authors": getattr(meta, "authors", None),
                        "year": getattr(meta, "year", None),
                        "doi": getattr(meta, "doi", None),
                        "chunks": chunks_list,
                        "full_text": full_text,
                    })

        # Step 3: Prepare sources
        if paper_results:
            sources = []
            for p in paper_results:
                sources.append(
                    SourceReference(
                        title=p.get("title") or "Untitled",
                        authors=p.get("authors"),
                        year=p.get("year"),
                        doi=p.get("doi"),
                        relevance_score=p.get("paper_score", 0.0),
                    )
                )
        else:
            sources = self._prepare_sources(selected_documents)
        for source in sources:
            yield StreamEvent.source(source)

        # Step 4: Stream the response generation
        if not selected_documents:
            yield StreamEvent.content("No relevant documents found to answer your question.")
            yield StreamEvent.done(
                conversation_id="",
                tokens_used=0,
                mode="advanced",
                iterations=1,
            )
            return

        yield StreamEvent.status("Advanced RAG: Generating response...")

        if paper_results:
            messages, temperature, is_o_series = self._advanced_paper_stream_messages(
                request, paper_results, preamble=scope.scope_note
            )
        else:
            combined_prompt, user_template, temperature = self._build_v1_answer_chat_chunk_docs(
                request, selected_documents
            )
            messages = [
                {"role": "system", "content": combined_prompt},
                {"role": "user", "content": user_template},
            ]
            model = getattr(request, "model", "") or ""
            is_o_series = model.startswith("o") or "gpt-5" in model

        # Stream the LLM response (v1 get_response structure; no refinement when streaming)
        full_response = ""
        try:
            stream_kwargs: dict[str, Any] = {
                "messages": messages,
                "model": request.model,
                "provider": request.provider,
                "max_tokens": 2000,
            }
            if not is_o_series:
                stream_kwargs["temperature"] = temperature
            async for chunk in llm.stream(**stream_kwargs):
                full_response += chunk
                yield StreamEvent.content(chunk)
        except Exception as e:
            logger.error("advanced_streaming_error", error=str(e))
            # Fall back to non-streaming (same v1 template as execute)
            if paper_results:
                ctx = format_paper_results_for_prompt(paper_results, max_chars_per_paper=4000)
                answer = await self._generate_response_from_context(
                    query=request.query,
                    context=ctx,
                    llm=llm,
                    request=request,
                    num_papers=len(paper_results),
                    paper_results=paper_results,
                    preamble=scope.scope_note,
                )
            else:
                answer = await self._generate_response_from_context(
                    query=request.query,
                    context="",
                    llm=llm,
                    request=request,
                    source_documents=selected_documents,
                )
            yield StreamEvent.content(answer)
            full_response = answer

        # Add references using utility function
        if sources:
            references = format_references(sources)
            yield StreamEvent.content("\n\n" + references)

        yield StreamEvent.done(
            conversation_id="",
            tokens_used=0,
            mode="advanced",
            iterations=1,
        )

    async def _generate_similar_queries(
        self,
        original_query: str,
        llm: Any,
        number: int = 3,
    ) -> list[str]:
        """
        Generate similar/rephrased queries using LLM.

        Ported from: core/core.py::generate_similar_queries()

        Returns list including original query + generated variations.
        """
        queries = [original_query]  # Always include original

        if not number or number <= 0:
            return queries

        for i in range(number):
            # Build context with already generated queries
            additional_queries_content = f"Original Query: {original_query}."
            additional_queries_content += "".join(
                [f" Additional Q{j + 1}: {query}" for j, query in enumerate(queries[1:])]
            )

            prompt = """Rephrase slightly the question based on the original query that is not the same as the additional ones. 
Use scientific language. Your answer should be just one phrase. 
Don't deviate the topic of the queries and questions. Do not use bullet points or numbering."""

            try:
                response = await llm.complete(
                    messages=[
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": f"Queries already used: {additional_queries_content}",
                        },
                    ],
                    temperature=0.7,
                    max_tokens=100,
                )

                # Clean and add the generated query
                new_query = response.strip()
                if new_query and new_query not in queries:
                    queries.append(new_query)
                    logger.debug("advanced_generated_query", query=new_query[:100])

            except Exception as e:
                logger.warning("advanced_query_generation_error", error=str(e))
                break

        return queries

    async def _wrrf_retrieval(
        self,
        queries: list[str],
        vector_store: Any,
        embedding_provider: Any,
        kb_name: str,
        llm: Any = None,
    ) -> list[Any]:
        """
        Retrieve documents using WRRF (Weighted Reciprocal Rank Fusion).

        Ported from: core/core.py::retrieve_documents() - the multi-query branch

        WRRF formula: score = sum(normalized_score / (k + rank))

        If use_hybrid is enabled, also applies BM25 scoring to combine with vector scores.
        """
        rankings = {}  # doc_id -> {query_idx: rank}
        scores_per_query = {}  # query_idx -> {doc_id: score}
        documents_info = {}  # doc_id -> document

        # Process each query
        for q_idx, query in enumerate(queries):
            logger.debug("advanced_wrrf_processing_query", idx=q_idx, query=query[:100])

            # Get query embedding and search
            query_embedding = await embedding_provider.embed([query])
            results = await vector_store.search(
                collection=kb_name,
                query_embedding=query_embedding[0],
                top_k=self.initial_docs,
            )

            scores_per_query[q_idx] = {}

            # v1: hybrid for every query when advanced_mode + use_hybrid
            if self.use_hybrid and results and llm is not None:
                try:
                    logger.info("advanced_applying_hybrid", query=query[:100])

                    vector_scores = [getattr(r, "score", 0.5) for r in results]

                    hybrid_results = await hybrid_retrieval(
                        query=query,
                        documents=results,
                        vector_scores=vector_scores,
                        use_llm_weights=True,
                        llm=llm,
                    )

                    results = [doc for doc, _ in hybrid_results]
                    for doc, hybrid_score in hybrid_results:
                        doc.score = hybrid_score

                    logger.info("advanced_hybrid_applied", num_results=len(results))

                except Exception as e:
                    logger.warning("advanced_hybrid_error", error=str(e))

            # Process results for this query
            for rank, doc in enumerate(results, start=1):
                doc_id = doc_page_content(doc)

                # Get relevance score (normalized)
                score = getattr(doc, "score", 0.5)

                # Apply sigmoid normalization (from release package)
                # norm_score = 1 / (1 + exp(-(score - pth) * stp))
                norm_score = 1 / (1 + math.exp(-(score - self.pth) * self.stp))

                if doc_id not in rankings:
                    rankings[doc_id] = {}
                    documents_info[doc_id] = doc

                rankings[doc_id][q_idx] = rank
                scores_per_query[q_idx][doc_id] = norm_score

            logger.debug("advanced_wrrf_query_processed", idx=q_idx, docs=len(results))

        # Calculate WRRF scores
        wrrf_scores = {}
        for doc_id in rankings:
            total_score = 0
            for q_idx, rank in rankings[doc_id].items():
                norm_score = scores_per_query[q_idx][doc_id]
                # WRRF formula: weighted reciprocal rank fusion
                total_score += norm_score / (self.wrrf_k + rank)
            wrrf_scores[doc_id] = total_score

        sorted_docs = sorted(wrrf_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_docs:
            logger.warning("advanced_wrrf_no_documents")
            return []

        selected_documents = select_wrrf_merged_documents(
            sorted_docs,
            documents_info,
            self.final_max_docs,
            self.max_docs_per_source,
        )

        logger.info(
            "advanced_wrrf_selected",
            total_docs=len(sorted_docs),
            selected=len(selected_documents),
            hybrid_used=self.use_hybrid,
        )

        return selected_documents

    def _get_doc_citation(self, doc: Any) -> str:
        """Extract citation from document."""
        # Use utility function
        from perspicacite.rag.utils import get_doc_citation

        return get_doc_citation(doc)

    async def _generate_response(
        self,
        query: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
    ) -> str:
        """Generate response with optional relevancy optimization using v1 prompts."""

        if not documents:
            return "No relevant documents found to answer your question."

        # Format context using utility function
        context = format_documents_for_prompt(documents)

        # Use KB-specific mandatory prompt if KB info available (v1 compatibility)
        kb_title = getattr(request, "kb_name", "Perspicacité")
        kb_scope = getattr(request, "kb_scope", "scientific research and education")
        
        if kb_title and kb_scope:
            mandatory = get_mandatory_prompt(kb_title, kb_scope)
        else:
            mandatory = MANDATORY_PROMPT

        # Use exact prompts from release package
        combined_prompt = mandatory + "\n" + DEFAULT_SYSTEM_PROMPT

        # Add focus instructions (from relevancy optimization in original)
        combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        template = f"""System prompt: {combined_prompt}
Context: {context}
Format: {FORMAT_PROMPT}
Question: {query}

Additional information:
- Total documents used: {len(documents)}
- Unique sources: {len(set(get_doc_citation(d) for d in documents))}

Provide a comprehensive answer based on the documents above."""

        try:
            response = await llm.complete(
                messages=[
                    {"role": "system", "content": combined_prompt},
                    {"role": "user", "content": template},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            )
            return response
        except Exception as e:
            logger.error("advanced_response_generation_error", error=str(e))
            return f"Error generating response: {e}"

    async def _generate_response_from_context(
        self,
        query: str,
        context: str,
        llm: Any,
        request: RAGRequest,
        num_papers: int = 0,
        source_documents: list[Any] | None = None,
        paper_results: list[dict[str, Any]] | None = None,
        *,
        preamble: str | None = None,
    ) -> str:
        """core/core.py get_response template; two-pass uses same structure with paper dicts."""
        if (
            (not context or context == "No relevant papers found.")
            and not paper_results
            and not source_documents
        ):
            return "No relevant documents found to answer your question."

        kb_title = getattr(request, "kb_name", None) or "Perspicacité"
        kb_scope = getattr(request, "kb_scope", None) or "scientific research and education"
        mandatory = get_mandatory_prompt(kb_title, kb_scope)
        combined_prompt = self._truncate_combined_prompt_base(
            mandatory + "\n" + DEFAULT_SYSTEM_PROMPT
        )
        if self.use_relevancy_optimization:
            combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT

        if paper_results:
            context_with_citations = ""
            sc = Counter()
            for p in paper_results:
                text = (p.get("full_text") or "")[:12000]
                cit = p.get("title") or p.get("doi") or "Unknown"
                doi = p.get("doi") or ""
                url = f"https://doi.org/{doi}" if doi else ""
                context_with_citations += f"{text}, Citation: {cit})\n\n"
                sc[(url, str(cit))] += 1
            citations: list[str] = []
            for (url, citation), count in sc.items():
                year = ""
                for p in paper_results:
                    if (p.get("title") or p.get("doi")) == citation or str(p.get("doi")) == citation:
                        year = str(p.get("year") or "")
                        break
                citations.append(f"{citation}. Accessed on: {year} (url: {url}). [{count} docs]")
            citation_list = "\n".join(citations)
            n_docs = len(paper_results)
            n_sources = len(sc)
        elif source_documents:
            context_with_citations, citation_list = self._v1_context_and_citations(source_documents)
            n_docs = len(source_documents)
            n_sources = len({get_doc_citation(d) for d in source_documents})
        else:
            context_with_citations = context
            citation_list = ""
            n_docs = num_papers or 1
            n_sources = 1

        question = self._v1_question_line(request)
        template = f"""System prompt: {combined_prompt}
Context: {context_with_citations}
Format: {FORMAT_PROMPT}
Question: {question}

Additional information:
- Total documents used: {n_docs}
- Unique sources: {n_sources}
Sources:
{citation_list}
"""
        hist = format_conversation_block(getattr(request, "conversation_history", None))
        if hist:
            template = f"{hist}\n\n---\n\n{template}"
        if preamble:
            template = f"{preamble.strip()}\n\n{template}"

        model = getattr(request, "model", "") or ""
        is_o_series = model.startswith("o") or "gpt-5" in model
        # v1 get_response: complexity-based temperature only if use_relevancy_optimization
        if self.use_relevancy_optimization:
            qc = assess_query_complexity(getattr(request, "refined_query", None) or request.query)
            temperature = 0.7 if qc > 0.7 else 0.3
        else:
            temperature = 0.3
        if is_o_series:
            try:
                return await llm.complete(
                    messages=[
                        {"role": "system", "content": combined_prompt},
                        {"role": "user", "content": template},
                    ],
                    model=request.model,
                    provider=request.provider,
                    max_tokens=2000,
                )
            except Exception as e:
                logger.error("advanced_response_generation_error", error=str(e))
                return f"Error generating response: {e}"

        try:
            return await llm.complete(
                messages=[
                    {"role": "system", "content": combined_prompt},
                    {"role": "user", "content": template},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=temperature,
            )
        except Exception as e:
            logger.error("advanced_response_generation_error", error=str(e))
            return f"Error generating response: {e}"

    async def _refine_response(
        self,
        response: str,
        query: str,
        context: str = "",
        documents: list[Any] | None = None,
        llm: Any = None,
        request: RAGRequest | None = None,
        max_iterations: int | None = None,
        eval_model: str | None = None,
        eval_provider: str | None = None,
    ) -> str:
        """
        Refine response through iterative evaluation.

        Ported from: core/core.py::refine_response()
        """
        mi = max_iterations if max_iterations is not None else self.refinement_iterations
        current_response = response

        for i in range(mi):
            # Evaluate current response
            feedback = await self._evaluate_response(
                response=current_response,
                query=query,
                documents=documents,
                llm=llm,
                request=request,
                eval_model=eval_model,
                eval_provider=eval_provider,
            )

            # Check if response is good enough
            overall_score = feedback.get("overall_score", 0)
            if overall_score >= 8:
                logger.info("advanced_refinement_complete", score=overall_score, iteration=i + 1)
                return self._clean_refined_response(current_response)

            # Generate improved response
            current_response = await self._improve_response(
                response=current_response,
                query=query,
                documents=documents,
                feedback=feedback,
                llm=llm,
                request=request,
            )

        logger.info("advanced_refinement_max_iterations", iterations=mi)
        # v1 refine_response: final evaluate_response (result logged; return is last draft)
        try:
            final_fb = await self._evaluate_response(
                response=current_response,
                query=query,
                documents=documents,
                llm=llm,
                request=request,
                eval_model=eval_model,
                eval_provider=eval_provider,
            )
            logger.info("advanced_refinement_final_eval", overall_score=final_fb.get("overall_score"))
        except Exception as e:
            logger.debug("advanced_refinement_final_eval_skipped", error=str(e))
        return current_response

    def _resolve_evaluator_model_provider(
        self,
        request: RAGRequest | None,
        eval_model: str | None = None,
        eval_provider: str | None = None,
    ) -> tuple[str | None, str | None]:
        """v1 evaluator_llm: optional separate model/provider for evaluation calls."""
        m = eval_model or (getattr(request, "evaluator_model", None) if request else None)
        m = m or self.evaluator_model
        p = eval_provider or (getattr(request, "evaluator_provider", None) if request else None)
        p = p or self.evaluator_provider
        return m, p

    async def _evaluate_response(
        self,
        response: str,
        query: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest | None = None,
        eval_model: str | None = None,
        eval_provider: str | None = None,
    ) -> dict:
        """Evaluate response quality using v1 EVALUATE_RESPONSE_PROMPT."""

        # Format documents for context (v1 evaluate_response: all docs)
        doc_texts = []
        for doc in documents or []:
            if isinstance(doc, dict) and "full_text" in doc:
                text = str(doc.get("full_text") or "")
                citation = doc.get("title") or doc.get("doi") or "Unknown"
            elif hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                text = doc.chunk.text
                citation = get_doc_citation(doc)
            else:
                text = str(doc)
                citation = "Unknown"
            doc_texts.append(f"[Citation: {citation}]\n{text}")
        
        doc_content = "\n\n---\n\n".join(doc_texts) if doc_texts else "No documents"

        user_message = f"""Response to evaluate:
{response}

Original query:
{query}

Source documents:
{doc_content}

Evaluate the response according to the criteria and return a valid JSON."""

        try:
            eval_kw: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": EVALUATE_RESPONSE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.0,
                "max_tokens": 800,
            }
            em, ep = self._resolve_evaluator_model_provider(request, eval_model, eval_provider)
            if em and request is not None:
                eval_kw["model"] = em
                eval_kw["provider"] = ep or request.provider
            elif request is not None:
                eval_kw["model"] = request.model
                eval_kw["provider"] = request.provider
            result = await llm.complete(**eval_kw)

            import json
            import re

            # Extract JSON (handle markdown code blocks)
            result = result.strip()
            if result.startswith("```json"):
                result = result.split("```json")[1]
            if result.endswith("```"):
                result = result.rsplit("```", 1)[0]
            
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(result)

        except Exception as e:
            logger.error("advanced_evaluation_error", error=str(e))
            return {"overall_score": 5}  # Neutral score on error

    async def _improve_response(
        self,
        response: str,
        query: str,
        documents: list[Any],
        feedback: dict,
        llm: Any,
        request: RAGRequest,
    ) -> str:
        """Generate improved response based on feedback using v1 format."""

        # v1 refine_response: [{citation}]: first 300 chars per document, all docs
        doc_summaries = []
        for j, doc in enumerate(documents or [], 1):
            if isinstance(doc, dict) and "full_text" in doc:
                content = str(doc.get("full_text") or "")
                citation = doc.get("title") or doc.get("doi") or f"Source {j}"
            elif hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
                content = doc.chunk.text
                citation = get_doc_citation(doc)
            else:
                content = str(doc)
                citation = f"Source {j}"
            excerpt = content[:300] + ("..." if len(content) > 300 else "")
            doc_summaries.append(f"[{citation}]: {excerpt}")
        doc_context = "\n\n".join(doc_summaries)

        user_message = f"""Original query: {query}

Previous response:
{response}

Feedback from evaluator:
- Overall score: {feedback.get('overall_score', 'Not provided')}
- Relevance: {feedback.get('relevance', {}).get('feedback', 'Not provided')}
- Accuracy: {feedback.get('accuracy', {}).get('feedback', 'Not provided')}
- Completeness: {feedback.get('completeness', {}).get('feedback', 'Not provided')}
- Entities Recall: {feedback.get('entities_recall', {}).get('feedback', 'Not provided')}
- Faithfulness: {feedback.get('faithfulness', {}).get('feedback', 'Not provided')}

Missing key points: {', '.join(feedback.get('completeness', {}).get('missing_key_points', ['None provided']))}
Missing entities: {', '.join(feedback.get('entities_recall', {}).get('missing_entities', ['None provided']))}
Unfaithful statements: {', '.join(feedback.get('faithfulness', {}).get('unfaithful_statements', ['None provided']))}

Source documents contain the following information:
{doc_context}

Provide an improved response that addresses all the feedback points while staying strictly faithful to the source documents."""

        try:
            return await llm.complete(
                messages=[
                    {"role": "system", "content": REFINE_RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message + REFINE_RESPONSE_HUMAN_PROMPT_SUFFIX},
                ],
                model=request.model,
                provider=request.provider,
                max_tokens=2000,
                temperature=0.3,
            )
        except Exception as e:
            logger.error("advanced_improvement_error", error=str(e))
            return response  # Return original on error

    def _is_streaming(self, request: RAGRequest) -> bool:
        """Check if request is for streaming (placeholder)."""
        return False

    def _prepare_sources(self, documents: list[Any]) -> list[SourceReference]:
        """Prepare source references from documents using utility function."""
        # Use utility function with Advanced-specific max_docs
        return prepare_sources(documents, max_docs=self.final_max_docs)

    def _format_references(self, sources: list[SourceReference]) -> str:
        """Format sources as a references section using utility function."""
        return format_references(sources)
