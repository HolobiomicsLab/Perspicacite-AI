"""Unit tests for cancel_task MCP tool + registry integration."""
import json

import pytest

from perspicacite.mcp.server import cancel_task
from perspicacite.rag import cancellation as cr


@pytest.fixture(autouse=True)
async def _reset():
    await cr.reset_for_tests()
    yield
    await cr.reset_for_tests()


@pytest.mark.asyncio
async def test_cancel_task_marks_registry():
    out = await cancel_task("task-123")
    data = json.loads(out)
    assert data["ok"] is True
    assert data["task_id"] == "task-123"
    assert cr.is_cancelled("task-123") is True


@pytest.mark.asyncio
async def test_cancel_task_empty_id_rejected():
    out = await cancel_task("")
    data = json.loads(out)
    assert data["ok"] is False
    assert "missing" in data["error"]


@pytest.mark.asyncio
async def test_cancel_task_idempotent():
    await cancel_task("x")
    await cancel_task("x")
    assert cr.is_cancelled("x") is True


@pytest.mark.asyncio
async def test_agent_session_carries_task_id():
    """Regression: AgentSession.task_id is the field the agentic
    orchestrator's iteration-loop cancellation check reads.

    Before this field existed the check was dead code — getattr(...,
    None) always returned None and the loop never bailed out on a
    cancelled task. This test asserts the field is now part of the
    dataclass so the check has something to read.
    """
    from perspicacite.rag.agentic.orchestrator import AgentSession

    s = AgentSession(session_id="s1", task_id="task-xyz")
    assert s.task_id == "task-xyz"
    s2 = AgentSession(session_id="s2")
    assert s2.task_id is None  # default
