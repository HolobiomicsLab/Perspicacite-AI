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
    FORMAT_PROMPT,
    GENERATE_SIMILAR_QUERIES_PROMPT,
    EVALUATE_RESPONSE_PROMPT,
    REFINE_RESPONSE_SYSTEM_PROMPT,
    REFINE_RESPONSE_HUMAN_PROMPT_SUFFIX,
    FOCUS_INSTRUCTIONS_PROMPT,
)
from perspicacite.retrieval.hybrid import hybrid_retrieval

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
        rag_settings = getattr(config.rag_modes, 'advanced', {})
        
        self.initial_docs = 150  # From release package
        self.final_max_docs = 5
        self.max_docs_per_source = 1
        self.rephrases = 3  # Number of additional queries to generate
        self.use_refinement = rag_settings.get('enable_reflection', False)
        self.use_hybrid = rag_settings.get('use_hybrid', False)  # Enable hybrid retrieval
        
        # WRRF constants from release package
        self.wrrf_k = 60
        
        # Sigmoid parameters for score normalization
        self.pth = 0.8  # threshold
        self.stp = 30   # steepness

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

        # Step 1: Generate similar/rephrased queries
        # This is the key difference from Basic mode
        logger.info("advanced_generate_queries", original=request.query[:100])
        
        all_queries = await self._generate_similar_queries(
            original_query=request.query,
            llm=llm,
            number=self.rephrases
        )
        
        logger.info("advanced_queries_generated", count=len(all_queries), queries=all_queries)

        # Step 2: Retrieve documents for all queries with WRRF scoring
        # This uses the WRRF (Weighted Reciprocal Rank Fusion) algorithm
        logger.info("advanced_wrrf_retrieval", num_queries=len(all_queries))
        
        selected_documents = await self._wrrf_retrieval(
            queries=all_queries,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            kb_name=request.kb_name,
            llm=llm,
        )
        
        logger.info("advanced_selected_docs", count=len(selected_documents))

        # Step 3: Generate response (with optional refinement)
        answer = await self._generate_response(
            query=request.query,
            documents=selected_documents,
            llm=llm,
            request=request,
        )

        # Step 4: Apply refinement if enabled (Advanced mode feature)
        if self.use_refinement and not self._is_streaming(request):
            logger.info("advanced_apply_refinement")
            answer = await self._refine_response(
                response=answer,
                query=request.query,
                documents=selected_documents,
                llm=llm,
                request=request,
            )

        # Step 5: Prepare sources
        sources = self._prepare_sources(selected_documents)

        logger.info("advanced_rag_complete", sources=len(sources), refined=self.use_refinement)

        return RAGResponse(
            answer=answer,
            sources=sources,
            mode=RAGMode.ADVANCED,
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
        """Execute Advanced RAG with streaming output."""
        yield StreamEvent.status("Advanced RAG: Generating query variations...")
        
        # Delegate to non-streaming for core logic
        response = await self.execute(
            request, llm, vector_store, embedding_provider, tools
        )
        
        yield StreamEvent.status("Advanced RAG: Generating answer...")
        
        # Stream the answer word by word
        words = response.answer.split()
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield StreamEvent.content(chunk)
        
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
                [f" Additional Q{j+1}: {query}" for j, query in enumerate(queries[1:])]
            )

            prompt = """Rephrase slightly the question based on the original query that is not the same as the additional ones. 
Use scientific language. Your answer should be just one phrase. 
Don't deviate the topic of the queries and questions. Do not use bullet points or numbering."""

            try:
                response = await llm.complete(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Queries already used: {additional_queries_content}"},
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
            
            # Apply hybrid retrieval if enabled (first query only to avoid redundancy)
            if self.use_hybrid and q_idx == 0 and results and llm is not None:
                try:
                    logger.info("advanced_applying_hybrid", query=query[:100])
                    
                    # Extract vector scores
                    vector_scores = [getattr(r, 'score', 0.5) for r in results]
                    
                    # Apply hybrid retrieval
                    hybrid_results = await hybrid_retrieval(
                        query=query,
                        documents=results,
                        vector_scores=vector_scores,
                        use_llm_weights=True,
                        llm=llm,
                    )
                    
                    # Replace results with hybrid-scored versions
                    results = [doc for doc, _ in hybrid_results]
                    for doc, hybrid_score in hybrid_results:
                        doc.score = hybrid_score
                    
                    logger.info("advanced_hybrid_applied", num_results=len(results))
                    
                except Exception as e:
                    logger.warning("advanced_hybrid_error", error=str(e))
            
            # Process results for this query
            for rank, doc in enumerate(results, start=1):
                # Use content as doc_id for deduplication
                doc_id = self._get_doc_content_hash(doc)
                
                # Get relevance score (normalized)
                score = getattr(doc, 'score', 0.5)
                
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

        # Sort by WRRF score
        sorted_docs = sorted(wrrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_docs:
            logger.warning("advanced_wrrf_no_documents")
            return []

        # Select final documents with source diversity
        selected_documents = []
        source_counter = Counter()

        for doc_id, score in sorted_docs:
            if len(selected_documents) >= self.final_max_docs:
                break

            doc = documents_info[doc_id]
            source = self._get_doc_citation(doc)

            if source_counter[source] >= self.max_docs_per_source:
                continue

            # Attach the WRRF score to the document
            doc.wrrf_score = score
            selected_documents.append(doc)
            source_counter[source] += 1

        logger.info("advanced_wrrf_selected", 
                   total_docs=len(sorted_docs), 
                   selected=len(selected_documents),
                   unique_sources=len(source_counter),
                   hybrid_used=self.use_hybrid)

        return selected_documents

    def _get_doc_content_hash(self, doc: Any) -> str:
        """Get a hash/id for a document for deduplication."""
        if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'text'):
            return hash(doc.chunk.text[:200])  # Use first 200 chars as ID
        if hasattr(doc, 'content'):
            return hash(str(doc.content)[:200])
        return hash(str(doc))

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
        """Generate response with optional relevancy optimization."""
        
        if not documents:
            return "No relevant documents found to answer your question."
        
        # Format context
        context = self._format_documents_for_prompt(documents)
        
        # Use exact prompts from release package
        combined_prompt = MANDATORY_PROMPT + "\n" + DEFAULT_SYSTEM_PROMPT
        
        # Add focus instructions (from relevancy optimization in original)
        combined_prompt = combined_prompt + "\n" + FOCUS_INSTRUCTIONS_PROMPT
        
        template = f"""System prompt: {combined_prompt}
Context: {context}
Format: {FORMAT_PROMPT}
Question: {query}

Additional information:
- Total documents used: {len(documents)}
- Unique sources: {len(set(self._get_doc_citation(d) for d in documents))}

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

    async def _refine_response(
        self,
        response: str,
        query: str,
        documents: list[Any],
        llm: Any,
        request: RAGRequest,
        max_iterations: int = 2,
    ) -> str:
        """
        Refine response through iterative evaluation.
        
        Ported from: core/core.py::refine_response()
        """
        current_response = response

        for i in range(max_iterations):
            # Evaluate current response
            feedback = await self._evaluate_response(
                response=current_response,
                query=query,
                documents=documents,
                llm=llm,
            )

            # Check if response is good enough
            overall_score = feedback.get("overall_score", 0)
            if overall_score >= 8:
                logger.info("advanced_refinement_complete", score=overall_score, iteration=i+1)
                return current_response

            # Generate improved response
            current_response = await self._improve_response(
                response=current_response,
                query=query,
                documents=documents,
                feedback=feedback,
                llm=llm,
                request=request,
            )

        logger.info("advanced_refinement_max_iterations", iterations=max_iterations)
        return current_response

    async def _evaluate_response(
        self,
        response: str,
        query: str,
        documents: list[Any],
        llm: Any,
    ) -> dict:
        """Evaluate response quality."""
        
        system_prompt = """Evaluate the response based on:
1. Relevance - Does it address the query?
2. Accuracy - Is it factually correct based on documents?
3. Completeness - Does it cover key points?
4. Faithfulness - Does it stick to document content?

Respond in JSON format:
{
    "overall_score": 0-10,
    "relevance": {"score": 0-10, "feedback": "..."},
    "accuracy": {"score": 0-10, "feedback": "..."},
    "completeness": {"score": 0-10, "feedback": "...", "missing_key_points": [...]},
    "faithfulness": {"score": 0-10, "feedback": "...", "unfaithful_statements": [...]}
}"""

        try:
            result = await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nResponse: {response}"},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            
            import json
            import re
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
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
        """Generate improved response based on feedback."""
        
        system_prompt = """You are an expert at improving responses based on evaluation feedback.

Prioritize fixing these issues in order:
1. Faithfulness issues - Remove unsupported content
2. Relevance issues - Ensure direct query addressing
3. Accuracy issues - Fix factual errors
4. Completeness - Add missing information from sources

Do not invent information not present in the sources."""

        user_message = f"""Original query: {query}

Previous response:
{response}

Feedback:
- Overall score: {feedback.get('overall_score')}
- Relevance: {feedback.get('relevance', {}).get('feedback')}
- Accuracy: {feedback.get('accuracy', {}).get('feedback')}
- Completeness: {feedback.get('completeness', {}).get('feedback')}
- Missing points: {feedback.get('completeness', {}).get('missing_key_points', [])}
- Faithfulness: {feedback.get('faithfulness', {}).get('feedback')}
- Unfaithful statements: {feedback.get('faithfulness', {}).get('unfaithful_statements', [])}

Provide an improved response."""

        try:
            return await llm.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
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

    def _get_doc_excerpt(self, doc: Any, max_len: int = 200) -> str:
        """Get a short excerpt from a document for the refinement prompt."""
        if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'text'):
            return doc.chunk.text[:max_len]
        elif hasattr(doc, 'content'):
            return str(doc.content)[:max_len]
        return str(doc)[:max_len]

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
                relevance_score=getattr(doc, 'wrrf_score', getattr(doc, 'score', 0.0)),
            ))

        return sources
