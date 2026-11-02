"""The DOI MCP tool enables public landing capture without a cookie jar."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from perspicacite.mcp import server
from perspicacite.pipeline import download
from perspicacite.pipeline.download import cookies
from perspicacite.pipeline.download.base import PaperContent


@pytest.mark.asyncio
async def test_mcp_doi_ingest_enables_public_landing_capture(monkeypatch):
    client = object()
    client_options = []

    @asynccontextmanager
    async def public_client(**kwargs):
        client_options.append(kwargs)
        yield client

    retrieve = AsyncMock(
        return_value=PaperContent(
            success=False,
            doi="10.1234/article",
            content_type="none",
        )
    )
    state = SimpleNamespace(
        initialized=True,
        config=SimpleNamespace(pdf_download=None),
        pdf_parser=object(),
        session_store=SimpleNamespace(get_kb_metadata=AsyncMock(return_value=object())),
        vector_store=SimpleNamespace(paper_exists=AsyncMock(return_value=False)),
    )
    monkeypatch.setattr(server, "mcp_state", state)
    monkeypatch.setattr(download, "retrieve_paper_content", retrieve)
    monkeypatch.setattr(cookies, "build_authenticated_client", public_client)

    result = json.loads(await server.add_dois_to_kb("test-kb", ["10.1234/article"]))

    assert result["success"] is True
    assert client_options == [{"cookies_path": None}]
    retrieve.assert_awaited_once()
    assert retrieve.await_args.kwargs["http_client"] is client
    assert retrieve.await_args.kwargs["enable_landing_capture"] is True
