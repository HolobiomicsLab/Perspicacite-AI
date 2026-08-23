"""Live cookie validation: an unexpired cookie is not a working cookie.

The offline freshness report read "ok" for every publisher on a jar that
could not actually fetch a single PDF — the sessions were fine by date and
refused at the edge. These tests pin the classifier that tells those two
situations apart, because the remedies are opposite: re-import cookies for
a dead session, and do not bother for an edge block.
"""
import asyncio
import time

import click
import pytest

from perspicacite.cli import _parse_probe_options
from perspicacite.pipeline.download.cookies import (
    classify_probe_response,
    jar_has_live_cookie,
    probe_cookie_urls,
)


class _Cookie:
    def __init__(self, domain, expires):
        self.domain = domain
        self.expires = expires


def test_pdf_bytes_mean_the_session_was_accepted():
    assert classify_probe_response(200, "application/pdf", b"%PDF-1.7\n") == (
        "authenticated", None)


def test_html_body_is_a_paywall_not_a_success():
    """The canonical dead-session symptom: HTTP 200 carrying a login page."""
    status, detail = classify_probe_response(
        200, "text/html", b"<!DOCTYPE html><html>Sign in</html>")
    assert status == "paywalled"
    assert detail


def test_forbidden_is_blocked_not_paywalled():
    """403 means the request never reached a session check."""
    assert classify_probe_response(403, None, b"")[0] == "blocked"
    assert classify_probe_response(401, None, b"")[0] == "blocked"


def test_server_error_is_not_mistaken_for_a_cookie_problem():
    assert classify_probe_response(500, None, b"")[0] == "error"


def test_unexpected_content_type_is_reported_not_guessed():
    assert classify_probe_response(200, "application/json", b"{}")[0] == "error"


def test_expired_cookie_does_not_count_as_live():
    jar = [_Cookie("pubs.acs.org", int(time.time()) - 10)]
    assert not jar_has_live_cookie(jar, "pubs.acs.org")


def test_session_cookie_without_expiry_counts_as_live():
    assert jar_has_live_cookie([_Cookie("nature.com", None)], "nature.com")


def test_domain_matching_is_a_substring_of_the_cookie_host():
    jar = [_Cookie(".www.nature.com", int(time.time()) + 3600)]
    assert jar_has_live_cookie(jar, "nature.com")
    assert not jar_has_live_cookie(jar, "wiley.com")


def test_no_probe_urls_means_no_network_calls():
    assert asyncio.run(probe_cookie_urls(cookies_path=None, probe_urls={})) == []


def test_missing_jar_reports_no_cookies_without_fetching():
    """Never report a domain as blocked when we simply had no cookie for it."""
    results = asyncio.run(probe_cookie_urls(
        cookies_path=None, probe_urls={"pubs.acs.org": "https://example.invalid/x.pdf"}))
    assert [r.status for r in results] == ["no_cookies"]


def test_probe_option_parsing():
    assert _parse_probe_options(("a.com=https://a/x.pdf",)) == {"a.com": "https://a/x.pdf"}


def test_probe_option_without_url_is_rejected():
    with pytest.raises(click.BadParameter):
        _parse_probe_options(("a.com",))
