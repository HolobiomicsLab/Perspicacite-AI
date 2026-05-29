"""Unit tests for the PubMed quota log-scanner."""
import logging
from unittest.mock import AsyncMock, patch

import pytest

from perspicacite.search.scilex_adapter import (
    SciLExAdapter,
    _QuotaLogCapture,
)


def test_quota_capture_extracts_remaining():
    cap = _QuotaLogCapture()
    rec = logging.LogRecord(
        name="root", level=logging.WARNING, pathname="", lineno=0,
        msg="PubMed API: Only 2 requests remaining in current period!",
        args=(), exc_info=None,
    )
    cap.emit(rec)
    assert cap.last_remaining == 2


def test_quota_capture_ignores_unrelated_warnings():
    cap = _QuotaLogCapture()
    rec = logging.LogRecord(
        name="root", level=logging.WARNING, pathname="", lineno=0,
        msg="totally unrelated warning", args=(), exc_info=None,
    )
    cap.emit(rec)
    assert cap.last_remaining is None


def test_quota_capture_extracts_zero_remaining():
    cap = _QuotaLogCapture()
    rec = logging.LogRecord(
        name="root", level=logging.WARNING, pathname="", lineno=0,
        msg="Only 0 requests remaining in quota window",
        args=(), exc_info=None,
    )
    cap.emit(rec)
    assert cap.last_remaining == 0


@pytest.mark.asyncio
async def test_search_with_warnings_surfaces_quota():
    adapter = SciLExAdapter()

    async def fake_search(*args, **kwargs):
        adapter._last_dropped_apis = []
        adapter._last_quota_warning = {
            "kind": "rate_limit_low", "provider": "pubmed",
            "remaining": 2, "advice": "add NCBI_API_KEY",
        }
        return []

    with patch.object(adapter, "search", AsyncMock(side_effect=fake_search)):
        result = await adapter.search_with_warnings(query="q")
    assert len(result.warnings) == 1
    assert result.warnings[0]["kind"] == "rate_limit_low"
    assert result.warnings[0]["remaining"] == 2
