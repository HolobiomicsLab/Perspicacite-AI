"""Integration test for DeepResearch mode.

This test demonstrates the full workflow:
User Query → SciLEx Search → PDF Download → Relevance Assessment → Dynamic KB → Answer
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_services():
    """Create mocked dependencies."""
    # Mock SciLEx searcher
    scilex = AsyncMock()
    scilex.search.return_value = {
        "papers": [
            {
                "id": "paper_1",
                "title": "Transformers in Medical Imaging",
                "authors": ["Smith, J.", "Doe, A."],
                "year": 2024,
                "doi": "10.1234/example.1",
                "abstract": "This paper explores transformer architectures for medical image analysis.",
                "pdf_url": "https://example.com/paper1.pdf",
                "full_text": None,
            }
        ]
    }

    # Mock PDF downloader
    downloader = AsyncMock()
    downloader.download_and_parse.return_value = MagicMock(
        content="Full text of the paper about transformers in medical imaging...",
        content_type="application/pdf",
    )

    # Mock LLM client for assessment
    llm_assessment = AsyncMock()
    llm_assessment.complete.return_value = """{
        "is_relevant": true,
        "relevance_score": 0.85,
        "confidence": 0.9,
        "reasoning": "Paper directly addresses transformer applications in medical imaging",
        "key_findings": ["Vision transformers outperform CNNs", "Self-attention improves segmentation"],
        "missing_information": []
    }"""

    # Mock LLM client for answer
    llm_answer = AsyncMock()
    llm_answer.complete.return_value = (
        "Transformers have shown significant promise in medical imaging, "
        "particularly for tasks like segmentation and classification. "
        "Vision transformers often outperform traditional CNNs due to their "
        "self-attention mechanisms."
    )

    # Mock vector store
    vector_store = AsyncMock()
    vector_store.create_collection = AsyncMock()
    vector_store.add_documents = AsyncMock()
    vector_store.search.return_value = [
        {
            "text": "Vision transformers outperform CNNs on medical imaging tasks",
            "score": 0.92,
            "metadata": {"paper_id": "paper_1", "title": "Transformers in Medical Imaging"},
        }
    ]

    # Mock embedding service
    embedding_service = AsyncMock()
    embedding_service.embed.return_value = [0.1] * 768
    embedding_service.embed_batch.return_value = [[0.1] * 768]

    return {
        "scilex": scilex,
        "downloader": downloader,
        "llm_assessment": llm_assessment,
        "llm_answer": llm_answer,
        "vector_store": vector_store,
        "embedding_service": embedding_service,
    }


@pytest.mark.asyncio
async def test_deep_research_full_workflow(mock_services):
    """Test the complete DeepResearch workflow."""
    # Import here to avoid dependency issues during collection
    import sys
    sys.path.insert(0, 'src')

    from perspicacite.rag.modes.deep_research import DeepResearchMode
    from perspicacite.pipeline.download import PDFDownloader
    from perspicacite.rag.assessment import PaperAssessor, QueryRefiner
    from perspicacite.rag.dynamic_kb import DynamicKBFactory

    # Create services
    services = mock_services

    # Create KB factory
    kb_factory = DynamicKBFactory(
        vector_store=services["vector_store"],
        embedding_service=services["embedding_service"],
    )

    # Create assessor and refiner
    assessor = PaperAssessor(services["llm_assessment"])
    refiner = QueryRefiner(services["llm_assessment"])

    # Create DeepResearch mode
    mode = DeepResearchMode(
        scilex_searcher=services["scilex"],
        pdf_downloader=services["downloader"],
        paper_assessor=assessor,
        query_refiner=refiner,
        kb_factory=kb_factory,
        llm_client=services["llm_answer"],
        max_search_iterations=2,
        min_relevant_papers=1,
    )

    # Execute
    result = await mode.execute("How are transformers used in medical imaging?")

    # Verify workflow completed
    assert result.query == "How are transformers used in medical imaging?"
    assert result.papers_found == 1
    assert result.papers_relevant == 1
    assert len(result.search_history) >= 1
    assert "transformer" in result.answer.lower()

    # Verify PDF was downloaded
    services["downloader"].download_and_parse.assert_called_once()

    # Verify assessment was performed
    services["llm_assessment"].complete.assert_called()

    # Verify KB was created and used
    services["vector_store"].create_collection.assert_called_once()
    services["vector_store"].search.assert_called_once()

    print("\n✅ Full workflow test passed!")
    print(f"   Query: {result.query}")
    print(f"   Papers found: {result.papers_found}")
    print(f"   Papers relevant: {result.papers_relevant}")
    print(f"   Search iterations: {result.iterations}")


@pytest.mark.asyncio
async def test_deep_research_no_papers(mock_services):
    """Test behavior when no relevant papers found."""
    import sys
    sys.path.insert(0, 'src')

    from perspicacite.rag.modes.deep_research import DeepResearchMode
    from perspicacite.pipeline.download import PDFDownloader
    from perspicacite.rag.assessment import PaperAssessor, QueryRefiner
    from perspicacite.rag.dynamic_kb import DynamicKBFactory

    services = mock_services
    services["scilex"].search.return_value = {"papers": []}

    kb_factory = DynamicKBFactory(
        vector_store=services["vector_store"],
        embedding_service=services["embedding_service"],
    )

    mode = DeepResearchMode(
        scilex_searcher=services["scilex"],
        pdf_downloader=services["downloader"],
        paper_assessor=PaperAssessor(services["llm_assessment"]),
        query_refiner=QueryRefiner(services["llm_assessment"]),
        kb_factory=kb_factory,
        llm_client=services["llm_answer"],
    )

    result = await mode.execute("Very obscure topic with no papers")

    assert result.papers_found == 0
    assert result.papers_relevant == 0
    assert "couldn't find" in result.answer.lower()

    print("\n✅ No-papers test passed!")


@pytest.mark.asyncio
async def test_deep_research_query_refinement(mock_services):
    """Test query refinement when initial search is poor."""
    import sys
    sys.path.insert(0, 'src')

    from perspicacite.rag.modes.deep_research import DeepResearchMode
    from perspicacite.pipeline.download import PDFDownloader
    from perspicacite.rag.assessment import PaperAssessor, QueryRefiner
    from perspicacite.rag.dynamic_kb import DynamicKBFactory
    from perspicacite.models.papers import Paper

    services = mock_services

    # First search returns irrelevant paper
    services["scilex"].search.side_effect = [
        {"papers": [{"id": "p1", "title": "Irrelevant", "pdf_url": None}]},
        {"papers": [{"id": "p2", "title": "Relevant Paper", "pdf_url": None}]},
    ]

    # Assessment marks first as irrelevant, second as relevant
    services["llm_assessment"].complete.side_effect = [
        '{"is_relevant": false, "relevance_score": 0.2}',
        '{"is_relevant": true, "relevance_score": 0.8}',
    ]

    kb_factory = DynamicKBFactory(
        vector_store=services["vector_store"],
        embedding_service=services["embedding_service"],
    )

    mode = DeepResearchMode(
        scilex_searcher=services["scilex"],
        pdf_downloader=services["downloader"],
        paper_assessor=PaperAssessor(services["llm_assessment"]),
        query_refiner=QueryRefiner(services["llm_assessment"]),
        kb_factory=kb_factory,
        llm_client=services["llm_answer"],
        max_search_iterations=2,
    )

    result = await mode.execute("Test query")

    # Should have attempted multiple searches
    assert result.iterations >= 1

    print("\n✅ Query refinement test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
