"""The API guard: a tokenless bind stays on loopback, and /api/* needs the token.

The endpoint that makes this matter is ``/api/llm/proxy`` — it streams an
arbitrary prompt through the operator's own API keys, so an unauthenticated
copy on a routable interface is a free LLM gateway billed to them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from perspicacite.web.auth import (
    InsecureBindError,
    assert_bind_is_safe,
    is_authorized,
    is_loopback_host,
    requires_token,
    resolve_token,
)

TOKEN = "s3cret-token-value"


def _config(token=None, *, enabled=True, allow_insecure=False):
    return SimpleNamespace(
        auth=SimpleNamespace(
            enabled=enabled, token=token, allow_insecure_network_bind=allow_insecure
        )
    )


@pytest.mark.parametrize("host", ["127.0.0.1", "127.5.0.1", "::1", "localhost", "[::1]"])
def test_loopback_hosts_are_recognised(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "", "   ", "example.org", None])
def test_non_loopback_hosts_are_rejected(host):
    """An empty host is uvicorn's 'all interfaces', so it must not read as loopback."""
    assert not is_loopback_host(host)


def test_loopback_bind_without_token_is_allowed():
    """The default local-first workflow must keep working untouched."""
    assert_bind_is_safe("127.0.0.1", _config())


def test_network_bind_without_token_is_refused():
    with pytest.raises(InsecureBindError) as excinfo:
        assert_bind_is_safe("0.0.0.0", _config())
    assert "llm/proxy" in str(excinfo.value)


def test_network_bind_with_token_is_allowed():
    assert_bind_is_safe("0.0.0.0", _config(token=TOKEN))


def test_network_bind_allowed_when_explicitly_accepted():
    """The escape hatch exists for an authenticating reverse proxy in front."""
    assert_bind_is_safe("0.0.0.0", _config(allow_insecure=True))


def test_disabled_auth_does_not_smuggle_a_token_past_the_bind_guard():
    """auth.enabled=False means no enforcement, so it must not license a bind."""
    with pytest.raises(InsecureBindError):
        assert_bind_is_safe("0.0.0.0", _config(token=TOKEN, enabled=False))


def test_resolve_token_treats_blank_as_absent():
    assert resolve_token(_config(token="   ")) is None
    assert resolve_token(_config(token=TOKEN)) == TOKEN
    assert resolve_token(_config(token=TOKEN, enabled=False)) is None


@pytest.mark.parametrize(
    "header",
    [None, "", "Bearer", "Basic s3cret-token-value", "Bearer wrong", f"Bearer {TOKEN}x"],
)
def test_bad_credentials_are_rejected(header):
    assert not is_authorized(header, TOKEN)


@pytest.mark.parametrize("header", [f"Bearer {TOKEN}", f"bearer {TOKEN}", f"BEARER  {TOKEN} "])
def test_good_credentials_are_accepted(header):
    assert is_authorized(header, TOKEN)


def test_only_api_paths_require_the_token():
    assert requires_token("/api/kb/foo")
    assert requires_token("/api/llm/proxy")
    assert not requires_token("/api/health")  # liveness probes carry no credential
    assert not requires_token("/")
    assert not requires_token("/static/js/chat.js")


# --- middleware wiring -----------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """TestClient over the real app, with app_state.config swapped per test.

    Constructed without the context manager so the lifespan (which would build
    ChromaDB, the LLM client and the session store) never runs.
    """
    from perspicacite.web.app import app
    from perspicacite.web.state import app_state

    def _with(config, *, raise_server_exceptions=True):
        # Patch the singleton itself; app.py holds a reference to this object.
        monkeypatch.setattr(app_state, "config", config, raising=False)
        return TestClient(app, raise_server_exceptions=raise_server_exceptions)

    return _with


def test_api_call_is_rejected_without_the_token(client):
    r = client(_config(token=TOKEN)).post("/api/llm/proxy", json={"prompt": "hi"})
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_api_call_passes_the_guard_with_the_token(client):
    """A correct token reaches the route — 422 here is the body validator, not the guard."""
    c = client(_config(token=TOKEN))
    r = c.post("/api/llm/proxy", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 422


def test_api_is_open_when_no_token_is_configured(client):
    """Unchanged behaviour for the default loopback install."""
    r = client(_config()).post("/api/llm/proxy", json={})
    assert r.status_code == 422  # reached the route, guard inactive


def test_health_stays_reachable_while_a_token_is_set(client):
    """The guard must let an unauthenticated liveness probe through.

    The route itself needs a fully initialised AppState, which the lifespan
    never built here, so it errors — the point is only that the guard did not
    turn it into a 401 first.
    """
    c = client(_config(token=TOKEN), raise_server_exceptions=False)
    assert c.get("/api/health").status_code != 401
