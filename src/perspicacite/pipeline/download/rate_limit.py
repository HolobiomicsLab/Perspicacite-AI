"""Per-host politeness for the download package.

Hosts throttle. Without a shared budget the pipeline opens many parallel
requests to one publisher, collects 429s and stores the throttled body as
if it were an article. This module gives every host one limiter: a
concurrency cap, a minimum spacing between sends, and a cooldown that a
429 sets so sibling coroutines wait too.

The mechanism is host-agnostic. Host-specific tuning lives only in the
``HOST_POLICIES`` data table below; every other host uses the default.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from perspicacite.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, MutableMapping

    import httpx

logger = get_logger("perspicacite.pipeline.download.rate_limit")

# Four in flight per host: enough to keep a batch moving, low enough that a
# single publisher never sees a burst it reads as scraping.
DEFAULT_MAX_CONCURRENCY = 4
# 5 sends/second/host. An order of magnitude under the ~30/s headroom
# integrations/zotero.py documents as the point where 429s start.
DEFAULT_MIN_INTERVAL_S = 0.2
# Same budget as pipeline/external/http.py's max_retries default: one try
# plus two retries is where a throttled host either recovers or is down.
DEFAULT_MAX_ATTEMPTS = 3

# First backoff wait, matching pipeline/external/http.py's initial backoff.
BASE_BACKOFF_S = 1.0
# Doubling, as in pipeline/external/http.py and integrations/zotero.py.
BACKOFF_MULTIPLIER = 2.0
# Ceiling on any single wait, including a Retry-After the host asks for.
# integrations/zotero.py uses the same 60s cap; longer stalls a batch.
MAX_BACKOFF_S = 60.0

# Statuses worth trying again: explicit throttling plus transient server
# faults. Mirrors the retry set in integrations/zotero.py.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

# A URL with no parseable host still gets a limiter under this key. Unknown
# is an explicit state, never "unlimited".
UNKNOWN_HOST_KEY = "unknown-host"

# Labels that belong to a registry rather than an organisation, so
# "ebi.ac.uk" must keep three labels while "biorxiv.org" keeps two.
MULTIPART_SUFFIX_LABELS = frozenset({"ac", "co", "com", "edu", "gov", "net", "org"})
# Labels kept for an ordinary registrable domain, e.g. "biorxiv.org".
REGISTRABLE_LABEL_COUNT = 2
# Labels kept when the second-to-last one is a registry label.
MULTIPART_LABEL_COUNT = 3


@dataclass(frozen=True)
class HostPolicy:
    """Politeness budget for one host.

    - ``max_concurrency``: simultaneous in-flight requests allowed.
    - ``min_interval_s``: minimum spacing between two sends.
    - ``max_attempts``: total tries before ``RateLimited`` is raised.
    """

    max_concurrency: int
    min_interval_s: float
    max_attempts: int


DEFAULT_HOST_POLICY = HostPolicy(
    max_concurrency=DEFAULT_MAX_CONCURRENCY,
    min_interval_s=DEFAULT_MIN_INTERVAL_S,
    max_attempts=DEFAULT_MAX_ATTEMPTS,
)

# Host suffix -> policy. Add a row only for a host measured to throttle us;
# everything else uses DEFAULT_HOST_POLICY. Transport politeness only, never
# content or domain logic.
HOST_POLICIES: dict[str, HostPolicy] = {
    # bioRxiv answers 429 with a 17-byte text/plain body on .full.pdf and
    # .source.xml (measured 2026-08), so serialise and space its requests.
    "biorxiv.org": HostPolicy(max_concurrency=1, min_interval_s=1.0, max_attempts=4),
}


class RateLimited(Exception):  # noqa: N818 - name fixed by the download API
    """Raised when a host kept answering with a retryable status.

    Attributes: ``host`` (normalised limiter key), ``url``, ``attempts``
    spent, and ``retry_after_s`` last requested by the host (None if it
    never sent a parseable Retry-After).
    """

    def __init__(self, url: str, attempts: int, retry_after_s: float | None) -> None:
        """Record the throttled `url`, the `attempts` spent, `retry_after_s`."""
        host = host_key_for_url(url)
        super().__init__(f"rate limited by {host} after {attempts} attempts: {url}")
        self.host = host
        self.url = url
        self.attempts = attempts
        self.retry_after_s = retry_after_s


def _is_ip_literal(host: str) -> bool:
    """Whether `host` is an IPv4/IPv6 literal rather than a domain name."""
    return ":" in host or host.replace(".", "").isdigit()


def _registrable_domain(host: str) -> str:
    """Collapse `host` to its registrable domain so subdomains share it."""
    if _is_ip_literal(host):
        return host
    labels = host.split(".")
    if len(labels) <= REGISTRABLE_LABEL_COUNT:
        return host
    if labels[-REGISTRABLE_LABEL_COUNT] in MULTIPART_SUFFIX_LABELS:
        return ".".join(labels[-MULTIPART_LABEL_COUNT:])
    return ".".join(labels[-REGISTRABLE_LABEL_COUNT:])


def _configured_suffix(host: str) -> str | None:
    """Longest HOST_POLICIES suffix matching `host`, or None."""
    for suffix in sorted(HOST_POLICIES, key=len, reverse=True):
        if host == suffix or host.endswith(f".{suffix}"):
            return suffix
    return None


def host_key_for_url(url: str) -> str:
    """Normalised limiter key for `url`.

    Returns the configured host suffix when one matches, else the
    registrable domain, so www./api. siblings share one budget. Returns
    UNKNOWN_HOST_KEY when `url` carries no host.
    """
    host = (urlsplit(url).hostname or "").strip(".").lower()
    if not host:
        return UNKNOWN_HOST_KEY
    return _configured_suffix(host) or _registrable_domain(host)


def policy_for_host(host_key: str) -> HostPolicy:
    """Policy for a normalised `host_key`, falling back to the default."""
    return HOST_POLICIES.get(host_key, DEFAULT_HOST_POLICY)


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds requested by a Retry-After header in `headers`.

    Returns None when the header is absent, non-numeric (HTTP-date form) or
    negative, so callers fall back to exponential backoff.
    """
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _backoff_seconds(attempt: int) -> float:
    """Capped exponential backoff for a 1-based `attempt` number."""
    return min(BASE_BACKOFF_S * BACKOFF_MULTIPLIER ** (attempt - 1), MAX_BACKOFF_S)


def _throttle_wait_seconds(attempt: int, retry_after_s: float | None) -> float:
    """Seconds to hold a host after a retryable status on `attempt`."""
    if retry_after_s is None:
        return _backoff_seconds(attempt)
    return min(retry_after_s, MAX_BACKOFF_S)


class HostLimiter:
    """Concurrency cap, send spacing and cooldown for one host key."""

    def __init__(self, host: str, policy: HostPolicy) -> None:
        """Build a limiter for `host` governed by `policy`."""
        self.host = host
        self.policy = policy
        self._semaphore = asyncio.Semaphore(policy.max_concurrency)
        self._lock = asyncio.Lock()
        self._next_send_at = 0.0  # monotonic deadline

    async def _reserve_send(self) -> float:
        """Claim the next send slot; returns seconds to wait for it."""
        async with self._lock:
            now = time.monotonic()
            send_at = max(now, self._next_send_at)
            self._next_send_at = send_at + self.policy.min_interval_s
            return send_at - now

    async def acquire(self) -> None:
        """Wait for a concurrency slot, then for the host's spacing."""
        await self._semaphore.acquire()
        wait_s = await self._reserve_send()
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    def release(self) -> None:
        """Return the concurrency slot taken by `acquire`."""
        self._semaphore.release()

    async def start_cooldown(self, seconds: float) -> None:
        """Push this host's next send out by `seconds` for every caller."""
        async with self._lock:
            self._next_send_at = max(self._next_send_at, time.monotonic() + seconds)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one send slot for the duration of the block."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()


# Limiters hold asyncio primitives, which bind to the loop that first awaits
# them, so each running loop keeps its own registry and drops it with itself.
_REGISTRIES: MutableMapping[
    asyncio.AbstractEventLoop, dict[str, HostLimiter]
] = WeakKeyDictionary()


def get_host_limiter(url: str) -> HostLimiter:
    """Shared limiter for `url`'s host, created once per host and loop."""
    registry = _REGISTRIES.setdefault(asyncio.get_running_loop(), {})
    host_key = host_key_for_url(url)
    limiter = registry.get(host_key)
    if limiter is None:
        limiter = HostLimiter(host_key, policy_for_host(host_key))
        registry[host_key] = limiter
    return limiter


async def polite_get(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    """GET `url` on the caller's `client`, respecting its host's budget.

    Retries only RETRYABLE_STATUSES, honouring Retry-After when parseable
    and backing off exponentially otherwise; every wait also holds sibling
    coroutines on that host. Client errors such as 403/404 are returned
    unchanged on the first attempt. Raises RateLimited when the last
    attempt is still throttled, so a 429 body never reaches the caller.
    """
    limiter = get_host_limiter(url)
    retry_after_s: float | None = None
    for attempt in range(1, limiter.policy.max_attempts + 1):
        async with limiter.slot():
            response = await client.get(url, **kwargs)
        if response.status_code not in RETRYABLE_STATUSES:
            return response
        retry_after_s = parse_retry_after(response.headers)
        wait_s = _throttle_wait_seconds(attempt, retry_after_s)
        await limiter.start_cooldown(wait_s)
        logger.warning(
            "polite_get_throttled",
            host=limiter.host, url=url, status=response.status_code,
            attempt=attempt, wait_s=wait_s,
        )
    raise RateLimited(url, limiter.policy.max_attempts, retry_after_s)


if __name__ == "__main__":
    SMOKE_INTERVAL_S = 0.05  # tiny spacing so the smoke check stays fast

    async def _check_spacing() -> None:
        """Two sequential slots on one host are spaced by min_interval_s."""
        limiter = HostLimiter(
            "smoke.example",
            HostPolicy(max_concurrency=1, min_interval_s=SMOKE_INTERVAL_S, max_attempts=1),
        )
        started = time.monotonic()
        async with limiter.slot():
            pass
        async with limiter.slot():
            pass
        assert time.monotonic() - started >= SMOKE_INTERVAL_S

    assert host_key_for_url("https://www.biorxiv.org/a") == "biorxiv.org"
    assert host_key_for_url("https://api.biorxiv.org/b") == "biorxiv.org"
    assert host_key_for_url("https://sub.example.co.uk/x") == "example.co.uk"
    assert host_key_for_url("not-a-url") == UNKNOWN_HOST_KEY
    assert parse_retry_after({"Retry-After": "12"}) == 12.0
    assert parse_retry_after({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}) is None
    assert parse_retry_after({}) is None
    assert policy_for_host("biorxiv.org").max_concurrency == 1
    assert policy_for_host("example.com") is DEFAULT_HOST_POLICY
    assert _throttle_wait_seconds(3, None) == BASE_BACKOFF_S * BACKOFF_MULTIPLIER**2
    assert _throttle_wait_seconds(1, MAX_BACKOFF_S * 10) == MAX_BACKOFF_S
    asyncio.run(_check_spacing())
    print("rate_limit.py smoke checks passed")
