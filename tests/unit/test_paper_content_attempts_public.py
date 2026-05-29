"""Unit tests for PaperContent.attempts public field + MCP surface."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from perspicacite.mcp import server as mcp_server
from perspicacite.pipeline.download.base import PaperContent


def test_paper_content_attempts_is_field_not_property():
    """attempts is now a regular dataclass field, mutable directly."""
    pc = PaperContent(success=False, doi="10.1/x", content_type="none")
    assert pc.attempts == []
    pc.attempts.append({"source": "pmc", "status": "miss"})
    assert pc.attempts == [{"source": "pmc", "status": "miss"}]


def test_record_attempt_writes_to_attempts():
    pc = PaperContent(success=False, doi="10.1/x", content_type="none")
    pc.record_attempt("unpaywall", "miss", error="no oa url")
    assert pc.attempts == [{
        "source": "unpaywall",
        "status": "miss",
        "error": "no oa url",
    }]


def test_record_attempt_extras_merge():
    pc = PaperContent(success=False, doi="10.1/x", content_type="none")
    pc.record_attempt("wiley", "skip", reason="no api key")
    assert pc.attempts[0]["reason"] == "no api key"


def test_paper_content_default_attempts_are_independent():
    """Each PaperContent instance gets its own list (no shared mutable default)."""
    pc1 = PaperContent(success=False, doi="a", content_type="none")
    pc2 = PaperContent(success=False, doi="b", content_type="none")
    pc1.record_attempt("pmc", "miss")
    assert pc2.attempts == []


@pytest.fixture
def app_state_fixture():
    state = MagicMock()
    state.initialized = True
    state.config.pdf_download = None
    state.pdf_parser = MagicMock()
    with patch.object(mcp_server, "mcp_state", state):
        yield state


@pytest.mark.asyncio
async def test_mcp_get_paper_content_returns_attempts(app_state_fixture):
    pc = PaperContent(success=False, doi="10.1/x", content_type="none")
    pc.record_attempt("pmc", "miss")
    pc.record_attempt("unpaywall", "error", error="429")

    with patch(
        "perspicacite.pipeline.download.retrieve_paper_content",
        AsyncMock(return_value=pc),
    ):
        out = await mcp_server.get_paper_content(doi="10.1/x")
    data = json.loads(out)
    assert "attempts" in data
    assert len(data["attempts"]) == 2
    assert data["attempts"][0]["source"] == "pmc"
    assert data["attempts"][1]["error"] == "429"
