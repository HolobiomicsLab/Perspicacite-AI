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


if __name__ == "__main__":
    for u in ("https://arxiv.org/abs/2606.12950", "https://arxiv.org/pdf/2602.03128v2"):
        print(u, "->", _extract_doi_from_url(u))
