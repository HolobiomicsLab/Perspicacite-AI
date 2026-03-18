"""DeepResearch mode: End-to-end dynamic KB building workflow.

User Query → SciLEx Search → PDF Download → Relevance Assessment → Dynamic KB → Answer
"""

from dataclasses import dataclass
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper
from perspicacite.pipeline.download import PDFDownloader
from perspicacite.rag.assessment import PaperAssessor, QueryRefiner, RelevanceAssessment
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase

logger = get_logger("perspicacite.rag.deep_research")


@dataclass
class DeepResearchResult:
    """Result of deep research workflow."""

    query: str
    answer: str
    papers_found: int
    papers_relevant: int
    papers_in_kb: int
    iterations: int
    search_history: list[str]
    assessments: list[RelevanceAssessment]


class DeepResearchMode:
    """
    DeepResearch: Full pipeline for comprehensive research.

    This mode:
    1. Searches SciLEx for relevant papers
    2. Downloads PDFs for promising results
    3. Assesses relevance using LLM
    4. Builds dynamic knowledge base from relevant papers
    5. Generates answer from knowledge base

    Unlike standard modes, this performs its own document retrieval
    and quality assessment rather than relying on pre-indexed documents.
    """

    def __init__(
        self,
        scilex_searcher: Any,  # SciLexSearchInterface
        pdf_downloader: PDFDownloader,
        paper_assessor: PaperAssessor,
        query_refiner: QueryRefiner,
        kb_factory: Any,  # DynamicKBFactory
        llm_client: Any,
        max_search_iterations: int = 3,
        min_relevant_papers: int = 3,
    ):
        self.scilex_searcher = scilex_searcher
        self.pdf_downloader = pdf_downloader
        self.paper_assessor = paper_assessor
        self.query_refiner = query_refiner
        self.kb_factory = kb_factory
        self.llm_client = llm_client

        self.max_iterations = max_search_iterations
        self.min_relevant = min_relevant_papers

    async def execute(
        self,
        query: str,
        search_params: dict[str, Any] | None = None,
    ) -> DeepResearchResult:
        """
        Execute full deep research workflow.

        Args:
            query: Research question
            search_params: Additional SciLEx search parameters

        Returns:
            Deep research result with answer and metadata
        """
        logger.info("deep_research_start", query=query)

        current_query = query
        search_history = []
        all_relevant_papers: list[Paper] = []
        all_assessments: list[RelevanceAssessment] = []

        # Search, download, assess loop
        for iteration in range(self.max_iterations):
            logger.info("search_iteration", iteration=iteration + 1, query=current_query)
            search_history.append(current_query)

            # 1. Search SciLEx
            papers = await self._search_papers(current_query, search_params)
            logger.info("papers_found", count=len(papers))

            if not papers:
                # Try to refine query
                refined = await self.query_refiner.refine_query(
                    current_query, all_assessments
                )
                if refined and refined != current_query:
                    current_query = refined
                    continue
                break

            # 2. Download PDFs for papers without full text
            papers = await self._download_pdfs(papers)

            # 3. Assess relevance
            relevant, assessments = await self.paper_assessor.batch_assess(
                query, papers, relevance_threshold=0.6
            )
            all_assessments.extend(assessments)

            logger.info("relevance_assessment", relevant=len(relevant), total=len(papers))

            all_relevant_papers.extend(relevant)

            # Check if we have enough relevant papers
            if len(all_relevant_papers) >= self.min_relevant:
                break

            # Refine query for next iteration
            refined = await self.query_refiner.refine_query(query, all_assessments)
            if refined and refined != current_query:
                current_query = refined
            else:
                break

        # 4. Build dynamic knowledge base and generate answer
        if all_relevant_papers:
            answer = await self._generate_from_kb(query, all_relevant_papers)
        else:
            answer = self._generate_no_papers_response(query)

        result = DeepResearchResult(
            query=query,
            answer=answer,
            papers_found=sum(1 for a in all_assessments),
            papers_relevant=len(all_relevant_papers),
            papers_in_kb=len(set(a.paper_id for a in all_assessments if a.is_relevant)),
            iterations=len(search_history),
            search_history=search_history,
            assessments=all_assessments,
        )

        logger.info(
            "deep_research_complete",
            papers_found=result.papers_found,
            papers_relevant=result.papers_relevant,
            iterations=result.iterations,
        )

        return result

    async def _search_papers(
        self,
        query: str,
        params: dict[str, Any] | None,
    ) -> list[Paper]:
        """Search for papers using SciLEx."""
        search_params = {
            "query": query,
            "topn": 10,
            "source": "hybrid",
            "with_full_text": False,
        }
        if params:
            search_params.update(params)

        try:
            response = await self.scilex_searcher.search(**search_params)
            return response.get("papers", [])
        except Exception as e:
            logger.error("search_error", error=str(e))
            return []

    async def _download_pdfs(self, papers: list[Paper]) -> list[Paper]:
        """Download PDFs for papers that don't have full text."""
        import asyncio

        papers_needing_download = [p for p in papers if not p.full_text and p.pdf_url]

        if not papers_needing_download:
            return papers

        logger.info("downloading_pdfs", count=len(papers_needing_download))

        # Download in parallel
        download_tasks = [
            self.pdf_downloader.download_and_parse(p.pdf_url)
            for p in papers_needing_download
        ]
        results = await asyncio.gather(*download_tasks, return_exceptions=True)

        # Update papers with downloaded content
        updated_papers = list(papers)  # Copy
        for paper, result in zip(papers_needing_download, results):
            if isinstance(result, Exception):
                logger.warning("pdf_download_failed", paper_id=paper.id, error=str(result))
                continue

            # Find paper in list and update
            for i, p in enumerate(updated_papers):
                if p.id == paper.id:
                    updated_papers[i] = Paper(
                        id=paper.id,
                        title=paper.title,
                        authors=paper.authors,
                        year=paper.year,
                        doi=paper.doi,
                        abstract=paper.abstract,
                        pdf_url=paper.pdf_url,
                        full_text=result.content,  # From download result
                        metadata=paper.metadata,
                    )
                    break

        return updated_papers

    async def _generate_from_kb(
        self,
        query: str,
        papers: list[Paper],
    ) -> str:
        """Generate answer using dynamic knowledge base."""
        # Create and populate knowledge base
        async with self.kb_factory.create_kb() as kb:
            await kb.add_papers(papers, include_full_text=True)

            # Search for relevant context
            contexts = await kb.search(query, top_k=5)

            if not contexts:
                return self._generate_no_papers_response(query)

            # Generate answer
            context_text = "\n\n---\n\n".join(
                f"[Source: {c['metadata'].get('title', 'Unknown')}]"
                f"\n{c['text'][:1000]}"  # Truncate long chunks
                for c in contexts
            )

            prompt = f"""Research Question: {query}

Relevant information from academic papers:

{context_text}

Provide a comprehensive answer based on the above sources. Cite specific papers where applicable.

Answer:"""

            try:
                answer = await self.llm_client.complete(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a research assistant synthesizing information from academic papers.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model="claude-3-5-sonnet-20241022",
                    temperature=0.3,
                    max_tokens=2000,
                )
                return answer

            except Exception as e:
                logger.error("answer_generation_error", error=str(e))
                return f"Error generating answer: {e}"

    def _generate_no_papers_response(self, query: str) -> str:
        """Generate response when no relevant papers found."""
        return (
            f"I couldn't find sufficient relevant academic literature to answer: \"{query}\"\n\n"
            "Suggestions:\n"
            "- Try rephrasing your question with more specific technical terms\n"
            "- Check if your topic has sufficient published research\n"
            "- Consider broadening your search criteria"
        )
