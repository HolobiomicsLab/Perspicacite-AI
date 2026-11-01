"""Two-sided tests for bot-wall / proof-of-work interstitial detection."""

from perspicacite.pipeline.download.interstitial import (
    SCAN_PREFIX_BYTES,
    interstitial_marker,
)

# A minimal but genuine PDF byte stream: header, one object, trailer.
REAL_PDF_BYTES = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n%%EOF\n"

# Representative Cloudflare page: both discriminating tokens occur beyond the
# old 4 KB scan window, and the full response is approximately 5.7 KB.
CLOUDFLARE_PAGE_SIZE_BYTES = 5_700
_CLOUDFLARE_PAGE_PREFIX = (
    b"<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Just a moment...</title></head><body>"
)
_CLOUDFLARE_PAGE_SUFFIX = (
    b'<script src="/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1"></script>'
    b'<input type="hidden" name="cf-chl-response"></body></html>'
)
CLOUDFLARE_CHALLENGE_PAGE = (
    _CLOUDFLARE_PAGE_PREFIX
    + b"x"
    * (CLOUDFLARE_PAGE_SIZE_BYTES - len(_CLOUDFLARE_PAGE_PREFIX) - len(_CLOUDFLARE_PAGE_SUFFIX))
    + _CLOUDFLARE_PAGE_SUFFIX
)

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


def test_fires_on_representative_cloudflare_challenge() -> None:
    """A full-size Cloudflare challenge is detected past the old scan window."""
    assert len(CLOUDFLARE_CHALLENGE_PAGE) == CLOUDFLARE_PAGE_SIZE_BYTES
    marker = interstitial_marker(
        "https://publisher.example/article.pdf",
        "text/html; charset=UTF-8",
        CLOUDFLARE_CHALLENGE_PAGE,
    )
    assert marker == "cloudflare_challenge"


def test_each_cloudflare_challenge_token_is_discriminating() -> None:
    """Either Cloudflare-internal token identifies the same challenge family."""
    for token in (b"/cdn-cgi/challenge-platform/", b"cf-chl-"):
        assert (
            interstitial_marker(
                "https://publisher.example/article.pdf",
                "text/html",
                b"<html><body>" + token + b"</body></html>",
            )
            == "cloudflare_challenge"
        )


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


def test_real_pdf_mentioning_cloudflare_is_not_classified() -> None:
    """Cloudflare challenge strings in genuine PDF bytes do not trigger."""
    body = REAL_PDF_BYTES + b"Cloudflare /cdn-cgi/challenge-platform/ cf-chl-response"
    assert interstitial_marker("https://example.org/a.pdf", "application/pdf", body) is None


def test_generic_cloudflare_word_is_not_a_marker() -> None:
    """The provider name alone is ordinary article text, not a challenge."""
    body = b"<html><article>Cloudflare operates a global network.</article></html>"
    assert interstitial_marker("https://example.org/article", "text/html", body) is None


def test_stays_clean_on_article_landing_page_saying_download() -> None:
    """A real HTML landing page containing 'download' does not fire."""
    assert (
        interstitial_marker(
            "https://journals.example.org/doi/10.1000/xyz", "text/html", ARTICLE_LANDING_PAGE
        )
        is None
    )


def test_stays_clean_on_empty_body() -> None:
    """An empty body carries no marker; emptiness is the caller's problem."""
    assert interstitial_marker("https://example.org/a.pdf", "application/pdf", b"") is None


def test_marker_beyond_scan_prefix_is_not_searched() -> None:
    """The prefix bound is a contract: a token past the window is not detected."""
    padding = b"<html>" + b"a" * SCAN_PREFIX_BYTES
    assert (
        interstitial_marker(
            "https://example.org/a.pdf", "application/pdf", padding + b"cloudpmc-viewer-pow"
        )
        is None
    )


def test_marker_inside_scan_prefix_is_searched() -> None:
    """The same token just inside the window still fires."""
    padding = b"a" * (SCAN_PREFIX_BYTES - len(b"cloudpmc-viewer-pow"))
    assert (
        interstitial_marker(
            "https://example.org/a.pdf", "application/pdf", padding + b"cloudpmc-viewer-pow"
        )
        == "pmc_proof_of_work"
    )


def test_marker_match_is_case_insensitive() -> None:
    """Publishers vary the casing of the challenge identifier."""
    assert (
        interstitial_marker(
            "https://example.org/a.pdf", "application/pdf", b"<div id='CloudPMC-Viewer-POW'>"
        )
        == "pmc_proof_of_work"
    )


if __name__ == "__main__":
    test_fires_on_proof_of_work_body_served_as_pdf()
    test_stays_clean_on_real_pdf_bytes()
    test_marker_beyond_scan_prefix_is_not_searched()
    print("test_interstitial.py smoke checks passed")
