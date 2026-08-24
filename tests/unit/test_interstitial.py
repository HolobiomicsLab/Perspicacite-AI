"""Two-sided tests for bot-wall / proof-of-work interstitial detection."""

from perspicacite.pipeline.download.interstitial import (
    SCAN_PREFIX_BYTES,
    interstitial_marker,
)

# A minimal but genuine PDF byte stream: header, one object, trailer.
REAL_PDF_BYTES = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n%%EOF\n"

# Shape of the PMC proof-of-work page observed where a PDF was expected.
PMC_POW_PAGE = (
    b'<html><head><script src="/static/cloudpmc-viewer-pow/challenge.js"></script>'
    b"</head><body>Verifying your browser</body></html>"
)

# A legitimate publisher landing page: HTML, and it says "download".
ARTICLE_LANDING_PAGE = (
    b"<html><head><title>An article about proof of work protocols</title></head>"
    b'<body><a href="/doi/pdf/10.1000/xyz">Download PDF</a>'
    b"<p>Please wait while the figures load.</p></body></html>"
)


def test_fires_on_proof_of_work_body_served_as_pdf() -> None:
    """A challenge body declared application/pdf is reported as an interstitial."""
    marker = interstitial_marker(
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/pdf/main.pdf",
        "application/pdf",
        PMC_POW_PAGE,
    )
    assert marker == "pmc_proof_of_work"


def test_fires_on_bot_wall_final_host() -> None:
    """A response whose final host is a known bot wall is reported."""
    marker = interstitial_marker(
        "https://validate.perfdrive.com/abc?ssa=1",
        "text/html",
        b"<html><body>checking</body></html>",
    )
    assert marker == "perfdrive_bot_wall"


def test_fires_on_bot_wall_subdomain() -> None:
    """Host matching covers subdomains of a known challenge host."""
    marker = interstitial_marker(
        "https://eu.validate.perfdrive.com/abc", "text/html", b"<html></html>"
    )
    assert marker == "perfdrive_bot_wall"


def test_stays_clean_on_look_alike_host() -> None:
    """A host merely ending in the wall's registrable name is not a wall."""
    url = "https://notvalidate.perfdrive.com.example.org/a.pdf"
    assert interstitial_marker(url, "application/pdf", b"<html></html>") is None


def test_stays_clean_on_real_pdf_bytes() -> None:
    """A body starting with the PDF magic is never an interstitial."""
    url = "https://example.org/a.pdf"
    assert interstitial_marker(url, "application/pdf", REAL_PDF_BYTES) is None


def test_pdf_magic_wins_over_every_other_signal() -> None:
    """PDF magic bytes beat both the host table and the body table."""
    body = REAL_PDF_BYTES + b"cloudpmc-viewer-pow"
    assert interstitial_marker("https://validate.perfdrive.com/a.pdf", "text/html", body) is None


def test_stays_clean_on_article_landing_page_saying_download() -> None:
    """A real HTML landing page containing 'download' does not fire."""
    assert interstitial_marker(
        "https://journals.example.org/doi/10.1000/xyz", "text/html", ARTICLE_LANDING_PAGE
    ) is None


def test_stays_clean_on_empty_body() -> None:
    """An empty body carries no marker; emptiness is the caller's problem."""
    assert interstitial_marker("https://example.org/a.pdf", "application/pdf", b"") is None


def test_marker_beyond_scan_prefix_is_not_searched() -> None:
    """The prefix bound is a contract: a token past the window is not detected."""
    padding = b"<html>" + b"a" * SCAN_PREFIX_BYTES
    assert interstitial_marker(
        "https://example.org/a.pdf", "application/pdf", padding + b"cloudpmc-viewer-pow"
    ) is None


def test_marker_inside_scan_prefix_is_searched() -> None:
    """The same token just inside the window still fires."""
    padding = b"a" * (SCAN_PREFIX_BYTES - len(b"cloudpmc-viewer-pow"))
    assert interstitial_marker(
        "https://example.org/a.pdf", "application/pdf", padding + b"cloudpmc-viewer-pow"
    ) == "pmc_proof_of_work"


def test_marker_match_is_case_insensitive() -> None:
    """Publishers vary the casing of the challenge identifier."""
    assert interstitial_marker(
        "https://example.org/a.pdf", "application/pdf", b"<div id='CloudPMC-Viewer-POW'>"
    ) == "pmc_proof_of_work"


if __name__ == "__main__":
    test_fires_on_proof_of_work_body_served_as_pdf()
    test_stays_clean_on_real_pdf_bytes()
    test_marker_beyond_scan_prefix_is_not_searched()
    print("test_interstitial.py smoke checks passed")
