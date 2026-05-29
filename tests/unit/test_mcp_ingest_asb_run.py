"""MCP tool ingest_asb_run — thin wrapper around the orchestrator."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_ingest_asb_run_tool_calls_orchestrator():
    from perspicacite.mcp import server

    fake_result = {
        "kb_names": ["asb_bundle"],
        "skills_ingested": 1,
        "workflows_ingested": 2,
        "papers_ingested": 3,
        "failed": [],
        "workflow_dag": {"nodes": ["task_001"], "edges": []},
    }

    state = MagicMock()
    state.initialized = True
    state.session_store = MagicMock()

    with patch.object(server, "mcp_state", state), patch(
        "perspicacite.mcp.server.ingest_asb_run_pipeline",
        new=AsyncMock(return_value=fake_result),
    ):
        out_json = await server.ingest_asb_run(
            asb_run_dir="/tmp/fake_run",
            kb_name="asb_bundle",
            include=["skills", "workflows"],
            mode="composite",
        )

    out = json.loads(out_json)
    assert out["success"] is True
    assert out["ok"] is True  # legacy alias per A2
    assert out["skills_ingested"] == 1
    assert out["workflows_ingested"] == 2


@pytest.mark.asyncio
async def test_ingest_asb_run_tool_requires_state():
    from perspicacite.mcp import server

    state = MagicMock()
    state.initialized = False

    with patch.object(server, "mcp_state", state):
        out_json = await server.ingest_asb_run(
            asb_run_dir="/tmp/fake_run",
        )
    out = json.loads(out_json)
    assert out["success"] is False
    assert "not initialized" in out["error"].lower()


@pytest.mark.asyncio
async def test_ingest_asb_run_tool_handles_orchestrator_error():
    from perspicacite.mcp import server

    state = MagicMock()
    state.initialized = True

    with patch.object(server, "mcp_state", state), patch(
        "perspicacite.mcp.server.ingest_asb_run_pipeline",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        out_json = await server.ingest_asb_run(
            asb_run_dir="/tmp/fake_run",
        )
    out = json.loads(out_json)
    assert out["success"] is False
    assert "boom" in out["error"]


@pytest.mark.asyncio
async def test_ingest_asb_run_tool_default_include_is_both():
    """When include is None or empty, default to both skills + workflows."""
    from perspicacite.mcp import server

    captured = {}
    async def fake_orch(**kw):
        captured.update(kw)
        return {"kb_names": ["x"], "skills_ingested": 0, "workflows_ingested": 0,
                "papers_ingested": 0, "failed": [], "workflow_dag": None}

    state = MagicMock()
    state.initialized = True
    with patch.object(server, "mcp_state", state), patch(
        "perspicacite.mcp.server.ingest_asb_run_pipeline",
        new=fake_orch,
    ):
        await server.ingest_asb_run(asb_run_dir="/tmp/fake_run")
    assert set(captured.get("include", [])) == {"skills", "workflows"}


def test_ingest_asb_run_docstring_mentions_latency():
    """Per Phase B3 — all multi-second tools must surface latency."""
    import inspect

    from perspicacite.mcp import server
    doc = inspect.getdoc(server.ingest_asb_run) or ""
    assert "Latency" in doc
