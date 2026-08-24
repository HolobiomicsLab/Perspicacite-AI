"""Google Scholar arXiv links must yield a usable DOI.

Scholar returns arXiv landing pages with no DOI anywhere in the URL, so
``_extract_doi_from_url`` used to return None and the ingest pipeline
dropped the paper under ``no_doi``. On a live 2026-08-22 query, 3 of 10
Scholar hits were arXiv URLs lost this way. Every arXiv id has a
registered DOI of the form ``10.48550/arXiv.<id>``, which the rest of
the codebase (``pipeline.arxiv_ids.parse_arxiv_doi``) already reads.
"""

from perspicacite.pipeline.arxiv_ids import parse_arxiv_doi
from perspicacite.search.google_scholar_playwright import _extract_doi_from_url


def test_arxiv_abs_url_yields_datacite_doi():
    doi = _extract_doi_from_url("https://arxiv.org/abs/2606.12950")
    assert doi == "10.48550/arXiv.2606.12950"


def test_arxiv_pdf_url_with_version_drops_the_version():
    doi = _extract_doi_from_url("https://arxiv.org/pdf/2602.03128v2")
    assert doi == "10.48550/arXiv.2602.03128"


def test_derived_doi_round_trips_through_the_arxiv_parser():
    doi = _extract_doi_from_url("https://arxiv.org/abs/2605.14892")
    assert parse_arxiv_doi(doi) == "2605.14892"


def test_publisher_doi_extraction_is_unaffected():
    doi = _extract_doi_from_url("https://dl.acm.org/doi/abs/10.1145/3770854.3785692")
    assert doi == "10.1145/3770854.3785692"


def test_doi_org_url_still_wins():
    doi = _extract_doi_from_url("https://doi.org/10.1021/acs.jnatprod.3c00468")
    assert doi == "10.1021/acs.jnatprod.3c00468"


def test_non_arxiv_url_without_a_doi_returns_none():
    assert _extract_doi_from_url("https://openreview.net/forum?id=372FjQy1cF") is None


def test_iop_view_suffix_is_not_glued_onto_the_doi():
    """IOP landing pages append /meta after the DOI; it must not survive."""
    doi = _extract_doi_from_url(
        "https://iopscience.iop.org/article/10.1088/1742-6596/513/3/032027/meta"
    )
    assert doi == "10.1088/1742-6596/513/3/032027"


def test_slashes_inside_a_real_doi_suffix_are_kept():
    doi = _extract_doi_from_url(
        "https://www.techrxiv.org/doi/10.36227/techrxiv.176689868.81376258/v1"
    )
    assert doi == "10.36227/techrxiv.176689868.81376258/v1"


if __name__ == "__main__":
    for u in ("https://arxiv.org/abs/2606.12950", "https://arxiv.org/pdf/2602.03128v2"):
        print(u, "->", _extract_doi_from_url(u))


def test_biorxiv_version_and_view_tail_are_stripped():
    """``…691830v1.abstract`` is a URL, not a DOI — Crossref 404s on it.

    Seen live on 2026-08-22: three of seven DOIs a Google Scholar query
    selected for ingest carried this tail and would have entered the KB
    unresolvable.
    """
    doi = _extract_doi_from_url(
        "https://www.biorxiv.org/content/10.64898/2025.12.02.691830v1.abstract"
    )
    assert doi == "10.64898/2025.12.02.691830"


def test_biorxiv_multi_part_view_tail_is_stripped():
    doi = _extract_doi_from_url(
        "https://www.biorxiv.org/content/10.64898/2025.12.02.691830v1.full.pdf"
    )
    assert doi == "10.64898/2025.12.02.691830"


def test_biorxiv_bare_version_tail_is_stripped():
    doi = _extract_doi_from_url(
        "https://www.biorxiv.org/content/10.64898/2025.12.02.691830v1"
    )
    assert doi == "10.64898/2025.12.02.691830"


def test_medrxiv_eight_digit_id_and_hyphenated_view_are_stripped():
    doi = _extract_doi_from_url(
        "https://www.medrxiv.org/content/10.1101/2024.05.05.24306789v2.full-text"
    )
    assert doi == "10.1101/2024.05.05.24306789"


def test_a_dotted_doi_suffix_that_is_not_a_preprint_id_is_left_alone():
    """The tail-strip must never touch an ordinary versioned DOI suffix."""
    doi = _extract_doi_from_url("https://pubs.acs.org/doi/10.1021/acs.jnatprod.3c00468")
    assert doi == "10.1021/acs.jnatprod.3c00468"
