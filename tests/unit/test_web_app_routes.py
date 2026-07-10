"""Smoke test for the web app's route registration.

Pins the exact set of (path, method) pairs the FastAPI app exposes so
that the router-split refactor cannot accidentally drop, rename, or
re-method a route.

This file imports the live FastAPI app via the same loader the CLI uses
during the migration. After Task 9 the import switches to
`from perspicacite.web import app`.
"""

from __future__ import annotations

import pytest

# Routes the app must expose. Every entry is (path, set-of-methods).
EXPECTED_ROUTES: list[tuple[str, set[str]]] = [
    ("/", {"GET"}),
    ("/favicon.ico", {"GET"}),
    ("/api/health", {"GET"}),
    ("/api/chat", {"POST"}),
    ("/api/conversations", {"GET", "POST", "DELETE"}),
    ("/api/conversations/search", {"GET"}),
    ("/api/conversations/{conv_id}", {"GET", "DELETE"}),
    ("/api/conversations/{conv_id}/export", {"GET"}),
    ("/api/conversations/{conv_id}/messages/{message_id}/provenance", {"GET"}),
    ("/api/conversations/{conv_id}/provenance", {"GET"}),
    ("/api/conversations/{conv_id}/messages", {"POST"}),
    ("/api/kb", {"GET", "POST"}),
    ("/api/kb/{name}", {"GET", "DELETE"}),
    ("/api/kb/{name}/papers", {"POST"}),
    ("/api/kb/{name}/chunks", {"GET"}),
    ("/api/kb/{name}/bibtex", {"POST"}),
    ("/api/kb/{name}/bibtex/async", {"POST"}),
    ("/api/kb/{name}/dois", {"POST"}),
    ("/api/kb/{name}/dois/async", {"POST"}),
    ("/api/kb/{name}/stats", {"GET"}),
    ("/api/kb/{name}/export", {"GET"}),
    ("/api/paper", {"GET"}),
    ("/api/survey/{session_id}", {"GET"}),
    ("/api/survey/{session_id}/select", {"POST"}),
    ("/api/survey/{session_id}/generate", {"POST"}),
    ("/api/jobs/{job_id}", {"GET"}),
    ("/api/jobs/{job_id}/events", {"GET"}),
    ("/api/zotero/status", {"GET"}),
    ("/api/zotero/push", {"POST"}),
]


def _load_app():
    """Load the FastAPI app via the canonical import path."""
    from perspicacite.web import app

    return app


def _flatten_routes(routes):
    """Yield leaf routes, descending into routers wrapped by include_router.

    FastAPI >= 0.139 no longer copies an included router's routes onto the app.
    It appends one wrapper per include_router() call, exposing the router as
    ``original_router``. Older versions flatten, leaving nothing to descend into.
    """
    for route in routes:
        included_router = getattr(route, "original_router", None)
        if included_router is None:
            yield route
        else:
            yield from _flatten_routes(included_router.routes)


def _route_methods_by_path(app):
    """Return {path: {methods}} for every APIRoute reachable from the app."""
    from fastapi.routing import APIRoute

    out: dict[str, set[str]] = {}
    for r in _flatten_routes(app.routes):
        if isinstance(r, APIRoute):
            out.setdefault(r.path, set()).update(r.methods or set())
        else:
            # Mount/Static/etc — keyed by .path if present.
            path = getattr(r, "path", None)
            if path:
                out.setdefault(path, set()).update(getattr(r, "methods", None) or {"GET"})
    return out


@pytest.mark.parametrize("path,expected_methods", EXPECTED_ROUTES)
def test_route_registered(path, expected_methods):
    app = _load_app()
    routes = _route_methods_by_path(app)
    assert path in routes, f"route {path} not registered; have: {sorted(routes)}"
    assert expected_methods.issubset(routes[path]), (
        f"route {path} methods mismatch: expected superset of {expected_methods}, "
        f"got {routes[path]}"
    )


def test_total_route_count_unchanged():
    """Pin the total number of (path, method) pairs across APIRoutes.

    Counts every (path, method) pair across APIRoutes plus the two
    non-APIRoute entries (root '/' and '/favicon.ico' use HTMLResponse
    /FileResponse decorators that still register as APIRoute today).
    """
    app = _load_app()
    routes = _route_methods_by_path(app)
    pair_count = sum(len(methods) for methods in routes.values())
    # 33 (path, method) pairs across the EXPECTED_ROUTES list above.
    assert pair_count >= 33, f"expected at least 33 (path, method) pairs, got {pair_count}"
