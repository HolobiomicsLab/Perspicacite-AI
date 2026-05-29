"""Unit tests for _filter_known_papers seed logic (Issue 3)."""
from unittest.mock import MagicMock


def _make_paper(paper_id: str, doi: str | None = None) -> MagicMock:
    p = MagicMock()
    p.id = paper_id
    p.doi = doi
    return p


def _make_survey_mode() -> object:
    """Instantiate LiteratureSurveyRAGMode with a minimal mock config."""
    from perspicacite.rag.modes.literature_survey import LiteratureSurveyRAGMode
    cfg = MagicMock()
    cfg.rag_modes.literature_survey = None   # triggers dict fallback in __init__
    return LiteratureSurveyRAGMode(cfg)


def test_all_new_papers_returned_unchanged():
    mode = _make_survey_mode()
    papers = [_make_paper(f"new_{i}") for i in range(10)]
    known_ids = {"known_a", "known_b"}
    result = mode._filter_known_papers(papers, known_ids)
    assert result == papers


def test_all_known_returns_seed_known_max_papers():
    mode = _make_survey_mode()
    papers = [_make_paper(f"known_{i}") for i in range(10)]
    known_ids = {f"known_{i}" for i in range(10)}
    result = mode._filter_known_papers(papers, known_ids)
    # Should return at most seed_known_max papers (default 5)
    assert 1 <= len(result) <= mode.seed_known_max
    # All returned papers must be from the original list
    assert all(p in papers for p in result)


def test_mixed_returns_only_new_papers():
    """When there are new papers, known ones should NOT be included."""
    mode = _make_survey_mode()
    new_papers = [_make_paper(f"new_{i}") for i in range(3)]
    known_papers = [_make_paper(f"known_{i}") for i in range(5)]
    known_ids = {f"known_{i}" for i in range(5)}
    result = mode._filter_known_papers(new_papers + known_papers, known_ids)
    assert result == new_papers


def test_empty_input_returns_empty():
    mode = _make_survey_mode()
    assert mode._filter_known_papers([], {"a", "b"}) == []


def test_no_known_ids_returns_all():
    mode = _make_survey_mode()
    papers = [_make_paper(f"p_{i}") for i in range(5)]
    assert mode._filter_known_papers(papers, set()) == papers
