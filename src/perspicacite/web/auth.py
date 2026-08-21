"""Bearer-token guard for the HTTP API.

The API hands out more than data: ``/api/llm/proxy`` streams an arbitrary
prompt through the operator's own API keys, and the KB routes read and write
every knowledge base on the machine. That is fine on loopback, where the only
caller is the operator, and unsafe on any interface a second machine can
reach.

So the guard has two halves:

* **Start time** — refuse to bind a non-loopback interface unless a token is
  configured. This is what protects the default install; a config that would
  publish an unauthenticated LLM gateway to the LAN now fails loudly instead
  of serving.
* **Request time** — once a token is configured, require it as a bearer
  credential on ``/api/*``.

With no token and a loopback bind (the default) nothing changes, so the
local-first workflow is untouched.
"""

from __future__ import annotations

import ipaddress
import secrets
from typing import Any

# Loopback names that never resolve off-machine. Anything else is treated as
# reachable, including "" and "*", which uvicorn expands to all interfaces.
_LOOPBACK_NAMES = {"localhost", "localhost.localdomain"}

# Unauthenticated by design: the operator's own browser loads these before it
# can present a credential, and a liveness probe must work without one.
_PUBLIC_API_PATHS = {"/api/health"}

_BEARER_PREFIX = "bearer "


class InsecureBindError(RuntimeError):
    """Raised when a bind would expose the API to the network without a token."""


def is_loopback_host(host: str | None) -> bool:
    """True when ``host`` can only be reached from this machine."""
    if host is None:
        return False
    candidate = host.strip()
    if not candidate:
        return False  # uvicorn treats "" as all interfaces
    if candidate.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(candidate.strip("[]")).is_loopback
    except ValueError:
        return False


def resolve_token(config: Any) -> str | None:
    """The configured API token, or None when the API is unauthenticated."""
    auth = getattr(config, "auth", None)
    if auth is None or not getattr(auth, "enabled", True):
        return None
    token = getattr(auth, "token", None)
    return token.strip() if isinstance(token, str) and token.strip() else None


def assert_bind_is_safe(host: str | None, config: Any) -> None:
    """Raise :class:`InsecureBindError` for a tokenless non-loopback bind.

    Set ``auth.allow_insecure_network_bind`` to accept the risk deliberately —
    the sane reason being a reverse proxy that terminates authentication in
    front of this process.
    """
    if is_loopback_host(host) or resolve_token(config) is not None:
        return
    auth = getattr(config, "auth", None)
    if auth is not None and getattr(auth, "allow_insecure_network_bind", False):
        return
    raise InsecureBindError(
        f"refusing to serve on {host!r} without an API token: this exposes "
        "/api/llm/proxy (which spends your LLM credits) and every knowledge "
        "base to anyone who can reach this host. Either bind 127.0.0.1, or "
        "set PERSPICACITE_AUTH_TOKEN, or set auth.allow_insecure_network_bind "
        "if something in front of this process already authenticates callers."
    )


def is_authorized(auth_header: str | None, expected_token: str) -> bool:
    """True when the Authorization header carries the expected bearer token."""
    if not auth_header or not auth_header.lower().startswith(_BEARER_PREFIX):
        return False
    presented = auth_header[len(_BEARER_PREFIX) :].strip()
    return secrets.compare_digest(presented, expected_token)


def requires_token(path: str) -> bool:
    """True when ``path`` may only be served to an authenticated caller."""
    return path.startswith("/api/") and path not in _PUBLIC_API_PATHS


if __name__ == "__main__":
    from types import SimpleNamespace

    assert is_loopback_host("127.0.0.1") and is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0") and not is_loopback_host("")
    open_cfg = SimpleNamespace(auth=SimpleNamespace(enabled=True, token=None))
    assert_bind_is_safe("127.0.0.1", open_cfg)
    try:
        assert_bind_is_safe("0.0.0.0", open_cfg)
        raise AssertionError("expected InsecureBindError")
    except InsecureBindError:
        pass
    assert is_authorized("Bearer s3cret", "s3cret")
    assert not is_authorized("Bearer wrong", "s3cret")
    assert requires_token("/api/kb") and not requires_token("/api/health")
    print("auth guard OK")
