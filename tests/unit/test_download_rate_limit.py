"""Tests for the per-host politeness layer (offline, no network)."""

import asyncio
import time

import httpx
import pytest

from perspicacite.pipeline.download import rate_limit
from perspicacite.pipeline.download.rate_limit import (
    DEFAULT_HOST_POLICY,
    UNKNOWN_HOST_KEY,
    HostLimiter,
    HostPolicy,
    RateLimited,
    get_host_limiter,
    host_key_for_url,
    parse_retry_after,
    policy_for_host,
    polite_get,
)

# Tiny waits keep the timing assertions honest without slowing the suite.
TEST_INTERVAL_S = 0.05
TEST_RETRY_AFTER_S = 0.06
# Slack allowed on monotonic comparisons (coarse clocks on loaded CI).
TIMING_TOLERANCE_S = 0.005


def _register_policy(monkeypatch, host: str, policy: HostPolicy) -> None:
    """Add a temporary row to the host policy table for one test."""
    monkeypatch.setitem(rate_limit.HOST_POLICIES, host, policy)


class TestHostKey:
    """Normalisation of URLs into limiter keys."""

    def test_sibling_subdomains_collapse_to_one_key(self):
        """www./api. of one host share a key, hence one budget."""
        assert host_key_for_url("https://www.biorxiv.org/x.full.pdf") == "biorxiv.org"
        assert host_key_for_url("https://api.biorxiv.org/details/y") == "biorxiv.org"

    def test_different_hosts_do_not_collapse(self):
        """A lookalike host keeps its own key."""
        assert host_key_for_url("https://biorxiv.org.evil.test/x") != "biorxiv.org"
        assert host_key_for_url("https://www.arxiv.org/abs/1") == "arxiv.org"

    def test_registry_suffix_keeps_three_labels(self):
        """co.uk-style suffixes are not treated as the organisation."""
        assert host_key_for_url("https://www.ebi.ac.uk/x") == "ebi.ac.uk"

    def test_hostless_url_is_explicitly_unknown(self):
        """A URL with no host gets the unknown key, never an empty budget."""
        assert host_key_for_url("not-a-url") == UNKNOWN_HOST_KEY

    def test_ip_literal_keeps_full_host(self):
        """IP literals are not collapsed into a shared numeric key."""
        assert host_key_for_url("http://127.0.0.1:8000/x") == "127.0.0.1"


class TestPolicyTable:
    """Host policy lookup."""

    def test_measured_host_has_its_own_row(self):
        """bioRxiv is serialised because it was measured to throttle us."""
        assert policy_for_host("biorxiv.org").max_concurrency == 1

    def test_other_hosts_use_the_default(self):
        """Any host absent from the table uses the documented default."""
        assert policy_for_host("example.com") is DEFAULT_HOST_POLICY


class TestParseRetryAfter:
    """Retry-After header parsing."""

    def test_numeric_seconds_are_parsed(self):
        """A plain numeric header yields the seconds requested."""
        assert parse_retry_after({"Retry-After": "30"}) == 30.0

    def test_garbage_is_treated_as_absent(self):
        """An HTTP-date (or junk) header falls back to backoff."""
        assert parse_retry_after({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}) is None

    def test_missing_header_is_none(self):
        """No header at all yields None."""
        assert parse_retry_after({}) is None

    def test_negative_seconds_are_rejected(self):
        """A negative delay is not a usable wait."""
        assert parse_retry_after({"retry-after": "-5"}) is None


class TestLimiterRegistry:
    """One limiter, one shared budget per host."""

    async def test_sibling_urls_share_one_limiter(self):
        """Both subdomains resolve to the same limiter object."""
        first = get_host_limiter("https://www.biorxiv.org/a.full.pdf")
        second = get_host_limiter("https://api.biorxiv.org/b")
        assert first is second

    async def test_unrelated_host_gets_its_own_limiter(self):
        """A genuinely different host does not share the budget."""
        first = get_host_limiter("https://www.biorxiv.org/a")
        assert first is not get_host_limiter("https://arxiv.org/abs/1")


class TestConcurrencyCap:
    """The per-host semaphore."""

    async def test_second_slot_blocks_when_concurrency_is_one(self):
        """A serialised host admits only one in-flight request."""
        limiter = HostLimiter(
            "solo.example",
            HostPolicy(max_concurrency=1, min_interval_s=0.0, max_attempts=1),
        )
        async with limiter.slot():
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(limiter.acquire(), timeout=TEST_INTERVAL_S)

    async def test_second_slot_admitted_when_concurrency_is_two(self):
        """The same wait succeeds when the policy allows two in flight."""
        limiter = HostLimiter(
            "pair.example",
            HostPolicy(max_concurrency=2, min_interval_s=0.0, max_attempts=1),
        )
        async with limiter.slot():
            await asyncio.wait_for(limiter.acquire(), timeout=TEST_INTERVAL_S)
            limiter.release()


class TestPoliteGet:
    """polite_get retry, backoff and error semantics."""

    async def test_retry_after_is_honoured_then_success(self, respx_mock, monkeypatch):
        """A 429 with Retry-After is waited out, then the 200 is returned."""
        _register_policy(
            monkeypatch, "throttled.test",
            HostPolicy(max_concurrency=1, min_interval_s=0.0, max_attempts=3),
        )
        route = respx_mock.get("https://throttled.test/paper.pdf").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": str(TEST_RETRY_AFTER_S)}),
                httpx.Response(200, content=b"body"),
            ],
        )
        started = time.monotonic()
        async with httpx.AsyncClient() as client:
            response = await polite_get(client, "https://throttled.test/paper.pdf")
        assert response.status_code == 200
        assert route.call_count == 2
        assert time.monotonic() - started >= TEST_RETRY_AFTER_S - TIMING_TOLERANCE_S

    async def test_garbage_retry_after_falls_back_to_backoff(self, respx_mock, monkeypatch):
        """An unparseable Retry-After still produces an exponential wait."""
        monkeypatch.setattr(rate_limit, "BASE_BACKOFF_S", TEST_INTERVAL_S)
        _register_policy(
            monkeypatch, "garbage.test",
            HostPolicy(max_concurrency=1, min_interval_s=0.0, max_attempts=2),
        )
        respx_mock.get("https://garbage.test/x").mock(
            side_effect=[
                httpx.Response(503, headers={"Retry-After": "soon-ish"}),
                httpx.Response(200, content=b"body"),
            ],
        )
        started = time.monotonic()
        async with httpx.AsyncClient() as client:
            response = await polite_get(client, "https://garbage.test/x")
        assert response.status_code == 200
        assert time.monotonic() - started >= TEST_INTERVAL_S - TIMING_TOLERANCE_S

    async def test_exhausted_attempts_raise_rate_limited(self, respx_mock, monkeypatch):
        """A permanently throttled host raises instead of returning the 429."""
        monkeypatch.setattr(rate_limit, "BASE_BACKOFF_S", TEST_INTERVAL_S)
        _register_policy(
            monkeypatch, "always429.test",
            HostPolicy(max_concurrency=1, min_interval_s=0.0, max_attempts=2),
        )
        route = respx_mock.get("https://www.always429.test/x").mock(
            return_value=httpx.Response(429, text="Too many requests"),
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(RateLimited) as excinfo:
                await polite_get(client, "https://www.always429.test/x")
        assert excinfo.value.host == "always429.test"
        assert excinfo.value.attempts == 2
        assert excinfo.value.retry_after_s is None
        assert route.call_count == 2

    async def test_client_error_is_returned_without_retry(self, respx_mock, monkeypatch):
        """A 404 is a real answer: return it, do not spend attempts on it."""
        _register_policy(
            monkeypatch, "missing.test",
            HostPolicy(max_concurrency=1, min_interval_s=0.0, max_attempts=3),
        )
        route = respx_mock.get("https://missing.test/x").mock(
            return_value=httpx.Response(404),
        )
        async with httpx.AsyncClient() as client:
            response = await polite_get(client, "https://missing.test/x")
        assert response.status_code == 404
        assert route.call_count == 1

    async def test_forbidden_is_returned_without_retry(self, respx_mock, monkeypatch):
        """403 resembles throttling but is not retryable either."""
        _register_policy(
            monkeypatch, "denied.test",
            HostPolicy(max_concurrency=1, min_interval_s=0.0, max_attempts=3),
        )
        route = respx_mock.get("https://denied.test/x").mock(
            return_value=httpx.Response(403),
        )
        async with httpx.AsyncClient() as client:
            response = await polite_get(client, "https://denied.test/x")
        assert response.status_code == 403
        assert route.call_count == 1

    async def test_concurrent_calls_are_spaced_by_min_interval(self, respx_mock, monkeypatch):
        """Two coroutines on one host cannot send back to back."""
        _register_policy(
            monkeypatch, "spaced.test",
            HostPolicy(max_concurrency=2, min_interval_s=TEST_INTERVAL_S, max_attempts=1),
        )
        sent_at: list[float] = []

        def _record(request: httpx.Request) -> httpx.Response:
            """Timestamp each send and answer 200."""
            sent_at.append(time.monotonic())
            return httpx.Response(200, content=b"body")

        respx_mock.get(url__regex=r"https://\w+\.spaced\.test/.*").mock(side_effect=_record)
        async with httpx.AsyncClient() as client:
            await asyncio.gather(
                polite_get(client, "https://www.spaced.test/a"),
                polite_get(client, "https://api.spaced.test/b"),
            )
        assert len(sent_at) == 2
        assert abs(sent_at[1] - sent_at[0]) >= TEST_INTERVAL_S - TIMING_TOLERANCE_S
