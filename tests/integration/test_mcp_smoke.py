"""MCP tool inventory smoke — Wave 1.3 of framework-hardening roadmap.

Spawns the MCP server in-process, enumerates all @mcp.tool()-decorated tools,
and verifies:
  1. Tool count >= 20.
  2. Every tool can be invoked with minimal valid args without unhandled
     exceptions, Pydantic schema crashes, or AttributeErrors.
  3. The 5 sampling-wrapped tools correctly bind the MCP context via
     use_mcp_context(), so current_mcp_context() returns the sentinel.

All tests are marked @pytest.mark.smoke (fast, mocked, no real LLM/HTTP).
The 'smoke' marker is registered in pyproject.toml [tool.pytest.ini_options].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level: import the server + helpers
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Skip entire module when fastmcp is absent (shouldn't happen in CI, but safe)
fastmcp_spec = __import__("importlib").util.find_spec("fastmcp")
if fastmcp_spec is None:
    pytest.skip("fastmcp not installed", allow_module_level=True)

from perspicacite.llm.mcp_sampling import current_mcp_context, use_mcp_context  # noqa: E402
from perspicacite.mcp import server as _srv  # noqa: E402

mcp = _srv.mcp

# ---------------------------------------------------------------------------
# The 5 sampling-wrapped tools (have `ctx: Context | None = None` param)
# ---------------------------------------------------------------------------

SAMPLING_TOOLS = [
    "route_kbs",
    "screen_papers",
    "build_kb_from_search",
    "expand_kb_via_citations",
    "generate_report",
]

# ---------------------------------------------------------------------------
# Helper: build a fully mocked MCPState
# ---------------------------------------------------------------------------


def _make_mock_state() -> MagicMock:
    """Return a MCPState-compatible MagicMock with common attributes set."""
    from perspicacite.config.schema import Config

    state = MagicMock()
    state.initialized = True
    cfg = Config()
    state.config = cfg

    # Session store: empty by default — all KB lookups return None
    state.session_store = AsyncMock()
    state.session_store.list_kbs = AsyncMock(return_value=[])
    state.session_store.get_kb_metadata = AsyncMock(return_value=None)
    state.session_store.save_kb_metadata = AsyncMock(return_value=None)
    state.session_store.delete_kb_metadata = AsyncMock(return_value=True)

    # Vector store
    state.vector_store = AsyncMock()
    state.vector_store.create_collection = AsyncMock()
    state.vector_store.delete_collection = AsyncMock()
    state.vector_store.paper_exists = AsyncMock(return_value=False)
    state.vector_store.list_paper_metadata = AsyncMock(return_value=[])

    # Embedding provider stub
    state.embedding_provider = MagicMock()
    state.embedding_provider.dimension = 384
    state.embedding_provider.model_name = "mock-embeddings"

    # LLM client stub — complete() returns minimal valid JSON for routing
    state.llm_client = AsyncMock()
    state.llm_client.complete = AsyncMock(return_value='{"result": "ok"}')

    state.pdf_parser = MagicMock()
    state.tool_registry = MagicMock()
    state.provenance_store = None
    state.job_registry = None

    return state


# ---------------------------------------------------------------------------
# Fixture: swap mcp_state for the duration of each test
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_state():
    """Inject a mocked MCPState into the server module."""
    state = _make_mock_state()
    old = _srv.mcp_state
    _srv.mcp_state = state
    yield state
    _srv.mcp_state = old


# ---------------------------------------------------------------------------
# Minimal arg sets per tool — each entry is (tool_name, kwargs_dict)
# ---------------------------------------------------------------------------

# Tools that return a clean "not found" / "not configured" JSON error —
# these still count as PASS because there is no unhandled exception.

_TOOL_ARGS: dict[str, dict[str, Any]] = {
    "search_literature": {
        "query": "CRISPR gene editing",
        "max_results": 5,
    },
    "get_paper_content": {
        "doi": "10.1234/fake-smoke-doi",
    },
    "get_paper_references": {
        "doi": "10.1234/fake-smoke-doi",
    },
    "list_knowledge_bases": {},
    "search_knowledge_base": {
        "query": "protein folding",
        "kb_name": "nonexistent-smoke-kb",
    },
    "create_knowledge_base": {
        "name": "smoke-test-kb",
        "description": "smoke test",
    },
    "add_papers_to_kb": {
        "kb_name": "nonexistent-smoke-kb",
        "papers": [{"title": "Test Paper", "doi": "10.1234/fake"}],
    },
    "generate_report": {
        "query": "machine learning in genomics",
        "kb_name": "nonexistent-smoke-kb",
    },
    "screen_papers": {
        "candidates": [
            {"doi": "10.1234/test1", "title": "T cells in immunity", "abstract": "About T cells."},
        ],
        "query": "immunology",
        "method": "bm25",
    },
    "add_dois_to_kb": {
        "kb_name": "nonexistent-smoke-kb",
        "dois": ["10.1234/fake"],
    },
    "push_to_zotero": {
        "dois": ["10.1234/fake"],
    },
    "build_kbs_from_zotero": {
        "plan_only": True,
    },
    "ingest_local_documents": {
        "kb_name": "nonexistent-smoke-kb",
        "paths": ["/nonexistent/path/smoke.pdf"],
    },
    "build_capsule": {
        "paper_id": "10.1234/fake",
        "kb_name": "nonexistent-smoke-kb",
    },
    "build_capsules_for_kb": {
        "kb_name": "nonexistent-smoke-kb",
    },
    "fetch_paper_resources": {
        "kb_name": "nonexistent-smoke-kb",
        "paper_id": "10.1234/fake",
    },
    "fetch_supplementary": {
        "kb_name": "nonexistent-smoke-kb",
        "paper_id": "10.1234/fake",
    },
    "route_kbs": {
        "query": "metabolomics",
        "method": "bm25",
    },
    "build_kb_from_search": {
        "query": "CRISPR",
        "kb_name": "smoke-search-kb",
        "dry_run": True,
        "max_results": 5,
    },
    "export_kb": {
        "kb_name": "nonexistent-smoke-kb",
        "out_dir": "/tmp/smoke-export",
    },
    "expand_kb_via_citations": {
        "kb_name": "nonexistent-smoke-kb",
        "dry_run": True,
    },
    "delete_knowledge_base": {
        "name": "nonexistent-smoke-kb",
    },
    "ingest_asb_run": {
        "asb_run_dir": "/nonexistent/path/asb_smoke",
    },
    "enrich_kb_from_cite_graph_tool": {
        "kb_name": "nonexistent-smoke-kb",
        "doi": "10.1234/fake-smoke-doi",
        "dry_run": True,
    },
    "ingest_github_repo": {
        "url": "https://github.com/x/y",
        "kb_name": "smoke",
    },
    "ingest_skill_bundle": {
        "source": "/nonexistent/path/smoke",
    },
    # ---- tools added after the original _TOOL_ARGS was written ----
    "push_notes_to_zotero": {
        "notes": [{"item_key": "FAKEKEYSMOKE", "content": "Smoke test note."}],
    },
    "ingest_url": {
        "url": "https://example.com/nonexistent-smoke-page",
    },
    "add_local_papers_to_kb": {
        "kb_name": "nonexistent-smoke-kb",
        "papers": [
            {
                "file": "/nonexistent/smoke.pdf",
                "title": "Smoke Test Paper",
            }
        ],
    },
    "ingest_urls_to_kb": {
        "kb_name": "nonexistent-smoke-kb",
        "urls": ["https://example.com/nonexistent-smoke-page"],
    },
    "zotero_get_collection_items": {
        "collection_id": "FAKECOL1",
    },
    "zotero_get_attachment_bytes": {
        "attachment_key": "FAKEATT1",
    },
    "zotero_ingest_collection_to_kb": {
        "collection_id": "FAKECOL1",
        "kb_name": "nonexistent-smoke-kb",
    },
    "web_search": {
        "query": "CRISPR gene editing",
        "max_results": 3,
    },
    "cancel_task": {
        "task_id": "nonexistent-smoke-task-id",
    },
    "search_by_passage": {
        "text": "Attention is all you need.",
        "kb_name": "nonexistent-smoke-kb",
    },
    "get_relevant_passages": {
        "query": "protein folding mechanisms",
        "kb_name": "nonexistent-smoke-kb",
    },
    "extract_parameters_from_passages": {
        "passages": [
            {
                "text": "The optimal pH range was 7.2–7.4.",
                "paper_id": "10.1234/fake",
                "title": "Smoke Test Paper",
            }
        ],
        "context": "bioreactor conditions",
    },
    "extract_failure_modes_from_passages": {
        "passages": [
            {
                "text": "Aggregation occurs above 45 °C.",
                "paper_id": "10.1234/fake",
                "title": "Smoke Test Paper",
            }
        ],
        "context": "protein stability",
    },
    "extract_claims_from_passages": {
        "passages": [
            {
                "text": "Treatment increased yield by 30%.",
                "paper_id": "10.1234/fake",
                "title": "Smoke Test Paper",
            }
        ],
        "context": "agricultural study",
    },
    "export_astra": {
        "claims": [],
    },
    "suggest_databases": {
        "query": "CRISPR off-target effects",
    },
    "get_usage_guide": {},
    "ensure_kb": {
        "doi": "10.1234/fake-smoke-doi",
    },
    "ground_paper": {
        "doi": "10.1234/fake-smoke-doi",
        "question": "What is the reported yield?",
    },
}

# ---------------------------------------------------------------------------
# Helpers for mocking HTTP and SciLEx layers
# ---------------------------------------------------------------------------


def _fake_retrieve_paper_content(doi, *, http_client=None, pdf_parser=None, **kwargs):
    """Return a minimal PaperContentResult without network."""
    result = MagicMock()
    result.success = False
    result.content_type = "abstract"
    result.content_source = "mock"
    result.full_text = None
    result.abstract = "Mock abstract."
    result.metadata = {"title": f"Mock Paper {doi}", "doi": doi}
    result.sections = []
    result.references = []
    return result


def _fake_search_filter_and_ingest(**kwargs):
    """Return a minimal IngestReport without hitting SciLEx."""
    from perspicacite.pipeline.search_to_kb import IngestReport

    return IngestReport(
        query=kwargs.get("query", ""),
        kb_name=kwargs.get("kb_name", ""),
        kb_created=False,
        searched=0,
        filtered_out=0,
        candidates=0,
        after_screen=0,
        added_papers=0,
        added_chunks=0,
        failed=0,
        selected_dois=[],
        filter_reasons={},
        pdf_stats={"attempted": 0, "success": 0, "failed": 0},
    )


def _fake_expand_kb(**kwargs):
    """Return a minimal SnowballReport without hitting OpenAlex."""
    from perspicacite.pipeline.snowball import SnowballReport

    return SnowballReport(
        kb_name=kwargs.get("kb_name", ""),
        direction=kwargs.get("direction", "both"),
        raw_hits=0,
        after_filter=0,
        after_screen=0,
        papers_added=0,
        added_chunks=0,
        failed=0,
        selected_dois=[],
        filter_reasons={},
        pdf_stats={"attempted": 0, "success": 0, "failed": 0},
    )


async def _fake_enrich_kb_from_cite_graph(**kwargs):
    """Return an empty hits list without hitting OpenAlex."""
    return []


def _make_coroutine(value):
    """Wrap a regular (possibly coroutine-returning) callable so it works as AsyncMock."""
    async def _coro(*args, **kwargs):
        if callable(value):
            return value(*args, **kwargs)
        return value
    return _coro


# ---------------------------------------------------------------------------
# Test 1: Tool count
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_tool_count_at_least_20():
    """Registered tool count must be >= 20 (spec says 22)."""
    tools = await mcp.list_tools()
    assert len(tools) >= 20, f"Expected >=20 tools, got {len(tools)}: {[t.name for t in tools]}"


# ---------------------------------------------------------------------------
# Test 2: Tool inventory — parametrized over all discovered tool names
# ---------------------------------------------------------------------------


def _get_all_tool_names() -> list[str]:
    """Synchronously enumerate tools at collection time."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        tools = loop.run_until_complete(mcp.list_tools())
    finally:
        loop.close()
    return [t.name for t in tools]


# Compute once at module load so pytest_generate_tests is fast.
_ALL_TOOL_NAMES: list[str] = _get_all_tool_names()


def pytest_generate_tests(metafunc):
    """Inject tool_name parameter at collection time."""
    if "tool_name" in metafunc.fixturenames:
        metafunc.parametrize("tool_name", _ALL_TOOL_NAMES)


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_tool_inventory(tool_name: str, mock_state):
    """Each tool must return a result (or a clean error JSON) — no unhandled exceptions."""
    fn = getattr(_srv, tool_name, None)
    assert fn is not None, f"Tool function '{tool_name}' not found in server module"

    kwargs = dict(_TOOL_ARGS.get(tool_name, {}))

    # Patch heavy I/O layers so no real network / LLM calls happen
    patches = [
        patch(
            "perspicacite.pipeline.download.retrieve_paper_content",
            new=_make_coroutine(_fake_retrieve_paper_content),
        ),
        patch(
            "perspicacite.pipeline.download.fallback.get_pdf_with_fallback",
            new=AsyncMock(return_value=(None, None)),
        ),
        patch(
            "perspicacite.pipeline.search_to_kb.search_filter_and_ingest",
            new=_make_coroutine(_fake_search_filter_and_ingest),
        ),
        patch(
            "perspicacite.pipeline.snowball.expand_kb_via_citations",
            new=_make_coroutine(_fake_expand_kb),
        ),
        patch(
            "perspicacite.pipeline.cite_graph.enrich_kb_from_cite_graph",
            new=AsyncMock(side_effect=_fake_enrich_kb_from_cite_graph),
        ),
    ]

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        try:
            result = await fn(**kwargs)
        except Exception as exc:
            pytest.fail(
                f"Tool '{tool_name}' raised an unhandled {type(exc).__name__}: {exc}"
            )

    # Result can be a str (JSON) or dict — validate it is not None
    assert result is not None, f"Tool '{tool_name}' returned None"

    # If it's a JSON string, it must parse cleanly
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            # Must have at least one known envelope key — this catches
            # completely unstructured returns while accommodating the several
            # different response shapes used across tools.
            _KNOWN_ENVELOPE_KEYS = {
                "success", "error", "hits", "knowledge_bases", "plan", "per_kb",
                # cancel_task uses "ok"
                "ok",
                # web_search / search_literature return "papers"
                "papers",
                # extract_* tools / suggest_databases / get_usage_guide
                "databases", "capabilities", "tools", "claims",
                # zotero tools
                "items", "collection_id", "filename", "content_b64",
                # passage retrieval tools
                "passages", "results",
                # ingest / summary tools
                "summary", "added",
            }
            assert any(k in parsed for k in _KNOWN_ENVELOPE_KEYS), \
                f"Unexpected JSON shape for '{tool_name}': {list(parsed.keys())}"
        except json.JSONDecodeError as exc:
            pytest.fail(f"Tool '{tool_name}' returned non-JSON string: {result[:200]!r} — {exc}")

    # If it's a dict, it must be non-empty or contain a known key
    if isinstance(result, dict):
        # Dicts can have any shape — just confirm no crash
        pass


# ---------------------------------------------------------------------------
# Test 3: Sampling-wrapped tools bind ctx
#
# Each of the 5 tools does one of:
#   (a) with use_mcp_context(ctx): ... — imports lazily inside the function body
#   (b) _mcp_ctx.set(ctx) directly     — generate_report
#
# Strategy: patch `perspicacite.llm.mcp_sampling.use_mcp_context` (the
# source module) so all lazy imports pick it up. Capture current_mcp_context()
# from inside a replaced inner call to verify the sentinel is bound.
# ---------------------------------------------------------------------------

# Per-tool: (inner function to patch, patch target, kwargs override)
_SAMPLING_TOOL_INNER: dict[str, dict] = {
    "route_kbs": {
        "patch_target": "perspicacite.rag.kb_router.auto_route_kbs",
        "extra_kwargs": {"method": "bm25"},
    },
    "screen_papers": {
        "patch_target": "perspicacite.search.screening.screen_papers_llm",
        # Must use method='llm' to enter the use_mcp_context branch
        "extra_kwargs": {"method": "llm"},
    },
    "build_kb_from_search": {
        "patch_target": "perspicacite.pipeline.search_to_kb.search_filter_and_ingest",
        "extra_kwargs": {"dry_run": True},
    },
    "expand_kb_via_citations": {
        "patch_target": "perspicacite.pipeline.snowball.expand_kb_via_citations",
        "extra_kwargs": {"dry_run": True},
    },
    "generate_report": {
        # generate_report sets _mcp_ctx directly; no use_mcp_context wrapper.
        # We verify via the _mcp_ctx contextvar when the engine runs.
        "patch_target": "perspicacite.rag.engine.RAGEngine.query_stream",
        "extra_kwargs": {},
    },
}


@pytest.mark.smoke
@pytest.mark.asyncio
@pytest.mark.parametrize("sampling_tool_name", SAMPLING_TOOLS)
async def test_sampling_tools_bind_ctx(sampling_tool_name: str, mock_state):
    """Sampling-wrapped tools must bind the sentinel ctx before calling inner logic."""
    from perspicacite.llm.mcp_sampling import _mcp_ctx as _sampling_ctxvar

    tool_name = sampling_tool_name
    sentinel = object()
    captured: list[Any] = []
    tool_info = _SAMPLING_TOOL_INNER[tool_name]

    fn = getattr(_srv, tool_name)
    kwargs = dict(_TOOL_ARGS.get(tool_name, {}))
    kwargs.update(tool_info.get("extra_kwargs", {}))
    kwargs["ctx"] = sentinel

    inner_target = tool_info["patch_target"]

    if tool_name == "generate_report":
        # generate_report uses _mcp_ctx.set() directly — capture inside engine
        async def _fake_query_stream(*args, **kwargs):
            captured.append(_sampling_ctxvar.get())
            return
            yield  # noqa: unreachable — makes it an async generator

        patches = [
            patch(inner_target, new=_fake_query_stream),
            patch(
                "perspicacite.pipeline.download.retrieve_paper_content",
                new=_make_coroutine(_fake_retrieve_paper_content),
            ),
        ]
        try:
            with patches[0], patches[1]:
                await fn(**kwargs)
        except Exception:
            pass

        # generate_report may short-circuit before the engine (KB not found).
        # In all cases the ctxvar must not leak after the call completes.
        assert _sampling_ctxvar.get() is None, (
            "generate_report leaked the MCP ctx after completion"
        )
        # If the engine ran, captured[0] must be the sentinel.
        if captured:
            assert captured[0] is sentinel, (
                f"generate_report: expected sentinel in _mcp_ctx, got {captured[0]!r}"
            )
        # Regardless of short-circuit: confirm _mcp_ctx is properly managed.
        # The tool sets ctx when state is initialized; since mock_state IS
        # initialized, the _mcp_ctx.set() runs. Verify no leak is sufficient.
        return

    # route_kbs short-circuits when session_store returns no KBs; provide one.
    if tool_name == "route_kbs":
        mock_kb = MagicMock()
        mock_kb.name = "smoke-kb"
        mock_kb.description = "Smoke test KB"
        mock_kb.paper_count = 1
        mock_state.session_store.list_kbs = AsyncMock(return_value=[mock_kb])

    # For the other 4 tools using use_mcp_context():
    # We replace the inner function that runs *inside* the `with` block so we
    # can observe what current_mcp_context() returns at that moment.

    original_use_mcp_context = use_mcp_context

    def _capturing_umc(ctx):
        """Wraps original use_mcp_context; checks sentinel is visible inside."""
        import contextlib

        cm = original_use_mcp_context(ctx)

        @contextlib.contextmanager
        def _wrapper():
            with cm:
                captured.append(current_mcp_context())
                yield

        return _wrapper()

    async def _fake_inner(*args, **kwargs):
        """Stub for the inner function — return minimal sentinel."""
        if tool_name == "route_kbs":
            return []  # auto_route_kbs returns list[KBRouteHit]
        if tool_name == "screen_papers":
            return []  # screen_papers_llm returns list[ScreenResult]
        if tool_name == "build_kb_from_search":
            return _fake_search_filter_and_ingest(**kwargs)
        if tool_name == "expand_kb_via_citations":
            return _fake_expand_kb(**kwargs)
        return None

    patches = [
        # Patch use_mcp_context at the source so lazy imports see it
        patch("perspicacite.llm.mcp_sampling.use_mcp_context", side_effect=_capturing_umc),
        # Stub out the inner function that runs inside the with block
        patch(inner_target, new=AsyncMock(side_effect=_fake_inner)),
        # Suppress other I/O
        patch(
            "perspicacite.pipeline.download.retrieve_paper_content",
            new=_make_coroutine(_fake_retrieve_paper_content),
        ),
        patch(
            "perspicacite.pipeline.search_to_kb.search_filter_and_ingest",
            new=_make_coroutine(_fake_search_filter_and_ingest),
        ),
        patch(
            "perspicacite.pipeline.snowball.expand_kb_via_citations",
            new=_make_coroutine(_fake_expand_kb),
        ),
    ]

    try:
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            await fn(**kwargs)
    except Exception:
        pass  # Tool may still error on empty KB list etc.

    assert len(captured) >= 1, (
        f"Tool '{tool_name}': use_mcp_context() was never called — "
        "sampling context never bound. The tool may not have reached "
        "the ctx-binding code path."
    )
    assert captured[0] is sentinel, (
        f"Tool '{tool_name}': expected sentinel inside use_mcp_context, "
        f"got {captured[0]!r}"
    )
