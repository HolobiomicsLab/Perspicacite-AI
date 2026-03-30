"""Unit tests for BibTeX → Paper parsing (no embeddings)."""

from pathlib import Path

import pytest

from perspicacite.pipeline.bibtex_kb import (
    entries_to_papers,
    load_bibtex_entries,
    normalize_bibtex_doi,
    sanitize_kb_display_name,
)


def test_normalize_bibtex_doi():
    assert normalize_bibtex_doi("10.1000/abc") == "10.1000/abc"
    assert normalize_bibtex_doi("https://doi.org/10.1000/abc") == "10.1000/abc"
    assert normalize_bibtex_doi("doi:10.1000/abc") == "10.1000/abc"


def test_sanitize_kb_display_name():
    assert sanitize_kb_display_name("My Papers") == "My_Papers"
    assert sanitize_kb_display_name("safe-name_01") == "safe-name_01"


def test_load_sample_download_try_bib():
    path = Path(__file__).resolve().parent.parent / "sample_download_try.bib"
    if not path.exists():
        pytest.skip("sample_download_try.bib missing")
    entries = load_bibtex_entries(path)
    assert len(entries) >= 2
    papers = entries_to_papers(entries)
    assert len(papers) >= 2
    dois = {p.doi for p in papers if p.doi}
    assert "10.48550/arXiv.1706.03762" in dois or any(
        "1706.03762" in (d or "") for d in dois
    )
