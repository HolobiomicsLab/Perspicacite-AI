"""A throttled backend must be distinguishable from an empty result set.

Regression test for the failure mode where every SciLEx backend answered
429 and ``search_to_kb`` reported ``candidates=0`` — indistinguishable
from a query that genuinely matched no literature. The log lines below
are the ones SciLEx actually emitted on 2026-08-21, when OpenAlex's
daily credit budget was exhausted.
"""

import logging

from perspicacite.search.scilex_adapter import SciLExAdapter, _QuotaLogCapture

OPENALEX_THROTTLE = (
    "OpenAlex API throttled (429). Server Retry-After: 16758s (attempt 1/3)"
)
SEMANTIC_SCHOLAR_THROTTLE = (
    "SemanticScholar API throttled (429). Waiting 4s before retry "
    "(attempt 2/3). Strategy: exponential backoff"
)
PUBMED_QUOTA = "PubMed API: Only 3 requests remaining in current period!"


def _emit(capture: _QuotaLogCapture, message: str) -> None:
    """Push one warning record through the capture handler."""
    capture.emit(
        logging.LogRecord(
            name="root",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )
    )


def test_captures_throttled_provider_with_retry_after():
    capture = _QuotaLogCapture()
    _emit(capture, OPENALEX_THROTTLE)
    assert capture.throttled_providers == {"OpenAlex": 16758}


def test_captures_throttled_provider_without_retry_after():
    capture = _QuotaLogCapture()
    _emit(capture, SEMANTIC_SCHOLAR_THROTTLE)
    assert capture.throttled_providers == {"SemanticScholar": None}


def test_keeps_longest_retry_after_across_repeated_throttles():
    capture = _QuotaLogCapture()
    _emit(capture, "OpenAlex API throttled (429). Server Retry-After: 10s")
    _emit(capture, OPENALEX_THROTTLE)
    _emit(capture, "OpenAlex API throttled (429). Server Retry-After: 5s")
    assert capture.throttled_providers == {"OpenAlex": 16758}


def test_quota_and_throttle_signals_are_independent():
    capture = _QuotaLogCapture()
    _emit(capture, PUBMED_QUOTA)
    _emit(capture, OPENALEX_THROTTLE)
    assert capture.last_remaining == 3
    assert "OpenAlex" in capture.throttled_providers


def test_unthrottled_run_records_nothing():
    capture = _QuotaLogCapture()
    _emit(capture, "Collected 25 articles from OpenAlex")
    assert capture.throttled_providers == {}


def test_warning_names_providers_and_longest_wait():
    warning = SciLExAdapter._build_throttle_warning(
        {"OpenAlex": 16758, "SemanticScholar": None}
    )
    assert warning["kind"] == "rate_limit_blocked"
    assert warning["providers"] == ["OpenAlex", "SemanticScholar"]
    assert warning["retry_after_s"] == 16758
    assert "429" in warning["advice"]


def test_warning_tolerates_absent_retry_after():
    warning = SciLExAdapter._build_throttle_warning({"SemanticScholar": None})
    assert warning["retry_after_s"] is None


if __name__ == "__main__":
    capture = _QuotaLogCapture()
    _emit(capture, OPENALEX_THROTTLE)
    print("throttled:", capture.throttled_providers)
    print("warning:", SciLExAdapter._build_throttle_warning(capture.throttled_providers))
