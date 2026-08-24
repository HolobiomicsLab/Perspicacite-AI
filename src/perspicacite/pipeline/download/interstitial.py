"""Detection of bot-wall / proof-of-work pages served where a PDF was expected.

Publishers increasingly answer an automated PDF fetch with HTTP 200 and a small
challenge page. Without this check the pipeline stores the challenge as if it
were the paper. This module is a leaf: no repo-internal imports beyond the
logger, so any download backend can call it.
"""

from urllib.parse import urlsplit

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.pipeline.download.interstitial")

# Every real PDF starts with this signature (ISO 32000-1, 7.5.2). Its presence
# settles the question before any heuristic runs.
PDF_MAGIC = b"%PDF"

# Bytes of the body examined for challenge markers. Interstitials measured in
# the 2026-08 publisher probe were 1-2 KB in total, so their marker is well
# inside this window; the bound keeps a multi-MB real PDF cheap to check and
# stops article text far from the head from ever matching.
SCAN_PREFIX_BYTES = 4096

# (host suffix, marker name). The FINAL response host is matched, so a wall is
# caught even when the publisher URL that started the fetch looks legitimate.
BOT_WALL_HOST_SUFFIXES: tuple[tuple[str, str], ...] = (
    # Radware Bot Manager challenge host. Observed 2026-08 as the final URL of
    # an IOP (iopscience.iop.org) PDF fetch answered HTTP 200 text/html.
    ("validate.perfdrive.com", "perfdrive_bot_wall"),
)

# (lowercased body token, marker name). Tokens must be discriminating: a word a
# real article could contain ("download", "please wait") is never enough.
BODY_CHALLENGE_MARKERS: tuple[tuple[bytes, str], ...] = (
    # PMC proof-of-work viewer page. Observed 2026-08 where a PMC PDF was
    # expected; the body carries this script/element identifier.
    (b"cloudpmc-viewer-pow", "pmc_proof_of_work"),
    # Radware Bot Manager challenge body references its own validation host.
    # Same 2026-08 IOP probe as the host row above.
    (b"validate.perfdrive.com", "perfdrive_bot_wall"),
)


def _host_wall_marker(url: str) -> str | None:
    """Marker for a URL whose host is a known bot-wall host, else None.

    Matches the host exactly or as a parent suffix, so subdomains of a
    challenge host are covered without matching unrelated look-alike domains.
    """
    host = urlsplit(url).hostname
    if not host:
        return None
    host = host.lower()
    for suffix, marker in BOT_WALL_HOST_SUFFIXES:
        if host == suffix or host.endswith(f".{suffix}"):
            return marker
    return None


def _body_wall_marker(body: bytes) -> str | None:
    """Marker for a body whose scanned prefix holds a challenge token, else None.

    Only the first SCAN_PREFIX_BYTES bytes are searched; a token appearing
    later in the response is deliberately not detected.
    """
    prefix = body[:SCAN_PREFIX_BYTES].lower()
    for token, marker in BODY_CHALLENGE_MARKERS:
        if token in prefix:
            return marker
    return None


def interstitial_marker(url: str, content_type: str, body: bytes) -> str | None:
    """Name the bot wall this response came from, or None if it is not one.

    Inputs: `url` is the FINAL response URL after redirects, `content_type` the
    declared response content type (used for diagnostics only), `body` the raw
    response bytes.

    Returns the short marker name of the signal that fired, so the caller can
    log which wall it hit. Returns None - not False, not "" - when no
    interstitial was detected; None means only that, and the caller must still
    validate that the body really is a usable document.
    """
    if body.startswith(PDF_MAGIC):
        return None
    marker = _host_wall_marker(url) or _body_wall_marker(body)
    if marker is None:
        return None
    logger.warning(
        "interstitial_detected",
        url=url,
        marker=marker,
        content_type=content_type,
        size=len(body),
    )
    return marker


if __name__ == "__main__":
    assert interstitial_marker("https://x.org/a.pdf", "application/pdf", b"%PDF-1.7 body") is None
    assert interstitial_marker("https://x.org/a.pdf", "application/pdf", b"") is None
    assert (
        interstitial_marker("https://x.org/a.pdf", "application/pdf", b"<html>cloudPMC-viewer-pow")
        == "pmc_proof_of_work"
    )
    assert (
        interstitial_marker("https://validate.perfdrive.com/x", "text/html", b"<html>hi")
        == "perfdrive_bot_wall"
    )
    assert interstitial_marker("https://x.org/a.html", "text/html", b"<a>Download PDF</a>") is None
    print("interstitial.py smoke checks passed")
