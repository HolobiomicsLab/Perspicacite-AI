"""Document relevance assessment for agentic RAG.

This module provides LLM-based assessment of paper relevance,
ported and adapted from v1 profonde.py.
"""

from dataclasses import dataclass
from typing import Any

from perspicacite.logging import get_logger
from perspicacite.models.papers import Paper

logger = get_logger("perspicacite.rag.assessment")


@dataclass
class RelevanceAssessment:
    """Result of relevance assessment."""

    paper_id: str
    is_relevant: bool
    relevance_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    key_findings: list[str]
    missing_information: list[str]


class PaperAssessor:
    """
    Assess paper relevance to research query.

    Uses LLM to evaluate if a paper contains relevant information
    and extracts key findings.
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    async def assess_relevance(
        self,
        query: str,
        paper: Paper,
        max_chars: int = 8000,
    ) -> RelevanceAssessment:
        """
        Assess if paper is relevant to query.

        Args:
            query: Research query
            paper: Paper to assess (with full_text)
            max_chars: Max characters to include from paper

        Returns:
            Relevance assessment
        """
        if not paper.full_text:
            logger.warning("assess_no_full_text", paper_id=paper.id)
            return RelevanceAssessment(
                paper_id=paper.id,
                is_relevant=False,
                relevance_score=0.0,
                confidence=1.0,
                reasoning="No full text available",
                key_findings=[],
                missing_information=["Full text"],
            )

        # Truncate if needed
        text = paper.full_text[:max_chars]
        if len(paper.full_text) > max_chars:
            text += "\n\n[Content truncated...]"

        # Build assessment prompt
        prompt = self._build_assessment_prompt(query, paper, text)

        try:
            response = await self.llm_client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant evaluating paper relevance. "
                        "Respond in JSON format only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model="claude-3-5-haiku-20241022",  # Use cheaper model for assessment
                temperature=0.0,  # Deterministic
            )

            # Parse response
            assessment = self._parse_assessment_response(response, paper.id)
            logger.info(
                "assessment_complete",
                paper_id=paper.id,
                is_relevant=assessment.is_relevant,
                score=assessment.relevance_score,
            )
            return assessment

        except Exception as e:
            logger.error("assessment_error", paper_id=paper.id, error=str(e))
            # Return conservative assessment on error
            return RelevanceAssessment(
                paper_id=paper.id,
                is_relevant=False,
                relevance_score=0.0,
                confidence=0.0,
                reasoning=f"Assessment failed: {e}",
                key_findings=[],
                missing_information=[],
            )

    async def batch_assess(
        self,
        query: str,
        papers: list[Paper],
        relevance_threshold: float = 0.6,
    ) -> tuple[list[Paper], list[RelevanceAssessment]]:
        """
        Assess multiple papers and return relevant ones.

        Args:
            query: Research query
            papers: Papers to assess
            relevance_threshold: Minimum score to be considered relevant

        Returns:
            Tuple of (relevant papers, all assessments)
        """
        import asyncio

        # Assess in parallel
        tasks = [self.assess_relevance(query, p) for p in papers]
        assessments = await asyncio.gather(*tasks)

        # Filter relevant papers
        relevant = []
        for paper, assessment in zip(papers, assessments):
            if assessment.is_relevant and assessment.relevance_score >= relevance_threshold:
                relevant.append(paper)

        logger.info(
            "batch_assessment_complete",
            total=len(papers),
            relevant=len(relevant),
            threshold=relevance_threshold,
        )

        return relevant, assessments

    def _build_assessment_prompt(
        self,
        query: str,
        paper: Paper,
        text: str,
    ) -> str:
        """Build the assessment prompt."""
        return f"""Research Question: {query}

Paper Details:
- Title: {paper.title}
- Authors: {paper.authors}
- Year: {paper.year or "Unknown"}
- Abstract: {paper.abstract or "Not available"}

Full Text:
{text[:4000]}

Evaluate this paper's relevance to the research question.

Respond in JSON format:
{{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "key_findings": ["finding 1", "finding 2"],
    "missing_information": ["what's missing"]
}}

Guidelines:
- is_relevant: Does this paper directly address the research question?
- relevance_score: How relevant (0=not at all, 1=perfect match)?
- confidence: How certain are you (0=uncertain, 1=very certain)?
- key_findings: Specific findings related to the question
- missing_information: What info would be needed for a complete answer?"""

    def _parse_assessment_response(
        self,
        response: str,
        paper_id: str,
    ) -> RelevanceAssessment:
        """Parse LLM assessment response."""
        import json
        import re

        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)

            return RelevanceAssessment(
                paper_id=paper_id,
                is_relevant=data.get("is_relevant", False),
                relevance_score=float(data.get("relevance_score", 0.0)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=data.get("reasoning", ""),
                key_findings=data.get("key_findings", []),
                missing_information=data.get("missing_information", []),
            )

        except json.JSONDecodeError as e:
            logger.error("assessment_parse_error", response=response[:200], error=str(e))
            # Fallback parsing
            is_relevant = "relevant" in response.lower() and "not relevant" not in response.lower()
            return RelevanceAssessment(
                paper_id=paper_id,
                is_relevant=is_relevant,
                relevance_score=0.5 if is_relevant else 0.0,
                confidence=0.3,
                reasoning="Failed to parse structured response",
                key_findings=[],
                missing_information=[],
            )


class QueryRefiner:
    """
    Refine search query based on failed results.

    When no relevant papers are found, generates an improved query.
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    async def refine_query(
        self,
        original_query: str,
        papers_assessed: list[RelevanceAssessment],
        max_attempts: int = 3,
    ) -> str | None:
        """
        Generate refined query based on assessment results.

        Args:
            original_query: Original search query
            papers_assessed: Assessments of searched papers
            max_attempts: Maximum refinement attempts

        Returns:
            Refined query or None if no improvement possible
        """
        # Analyze why papers were irrelevant
        reasons = []
        for a in papers_assessed:
            if not a.is_relevant:
                reasons.append(a.reasoning)

        if not reasons:
            return None  # All papers were relevant

        prompt = f"""Original Query: {original_query}

Previous Search Results:
{chr(10).join(f"- {r}" for r in reasons[:5])}

The previous search didn't find highly relevant papers.
Generate a refined search query that might yield better results.

Guidelines:
- Use more specific technical terms
- Try alternative keywords
- Consider broader or narrower scope
- Use synonyms for key concepts

Respond with ONLY the refined query, no explanation."""

        try:
            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model="claude-3-5-haiku-20241022",
                temperature=0.3,
                max_tokens=100,
            )

            refined = response.strip().strip('"')

            # Don't return if it's the same
            if refined.lower() == original_query.lower():
                return None

            logger.info("query_refined", original=original_query, refined=refined)
            return refined

        except Exception as e:
            logger.error("refine_error", error=str(e))
            return None
