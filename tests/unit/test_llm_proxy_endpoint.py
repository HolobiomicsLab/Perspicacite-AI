"""POST /api/llm/proxy proxies a prompt through the configured LLM
provider with no RAG/KB awareness. Honours stage-tiering rules.
Used by external clients (e.g. Scriptorium) that want Perspicacité
to be their LLM gateway."""
import inspect
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _client():
    from perspicacite.web.app import app
    return TestClient(app)


@pytest.fixture
def stub_streaming():
    """Replace the proxy's _call_llm_streaming helper with a fixed
    sequence of chunks. Yielded chunks become the response body."""
    async def fake_gen(*args, **kwargs):
        yield "hel"
        yield "lo"
    with patch(
        "perspicacite.web.routers.llm_proxy._call_llm_streaming",
        new=fake_gen,
    ):
        yield


def test_llm_proxy_returns_streaming_text(stub_streaming):
    client = _client()
    with client.stream(
        "POST", "/api/llm/proxy",
        json={"prompt": "say hi", "model": "claude-haiku-4-5"},
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "hello" in body


def test_llm_proxy_validates_required_fields():
    client = _client()
    r = client.post("/api/llm/proxy", json={})
    assert r.status_code == 422


def test_llm_proxy_does_not_retrieve_or_touch_kb():
    """The proxy module must not import KB / retrieval modules. The
    point of the endpoint is to be a thin LLM gateway."""
    from perspicacite.web.routers import llm_proxy
    src = inspect.getsource(llm_proxy)
    forbidden = ["DynamicKnowledgeBase", "auto_route_kbs", "build_rag"]
    for term in forbidden:
        assert term not in src, (
            f"{term!r} found in llm_proxy.py — the proxy is meant to "
            "be a pure LLM gateway with no RAG/KB coupling."
        )


def test_llm_proxy_passes_stage_through(stub_streaming):
    """The 'stage' field should reach the underlying call."""
    captured: dict = {}

    async def capture_stream(*, prompt, model, stage, max_tokens, temperature):
        captured["prompt"] = prompt
        captured["model"] = model
        captured["stage"] = stage
        captured["max_tokens"] = max_tokens
        captured["temperature"] = temperature
        yield "x"

    with patch(
        "perspicacite.web.routers.llm_proxy._call_llm_streaming",
        new=capture_stream,
    ):
        client = _client()
        with client.stream(
            "POST", "/api/llm/proxy",
            json={"prompt": "x", "stage": "fast", "temperature": 0.3},
        ) as r:
            list(r.iter_text())
    assert captured["prompt"] == "x"
    assert captured["stage"] == "fast"
    assert captured["temperature"] == 0.3
