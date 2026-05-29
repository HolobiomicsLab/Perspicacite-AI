"""Pin that user-facing multi-second MCP tools surface latency in
their docstrings. Calling LLM clients see these via tools/list and
should adjust HTTP timeouts accordingly."""
import inspect

import pytest

from perspicacite.mcp import server

# Tools that take more than ~3s in practice — they MUST advertise latency.
SLOW_TOOLS = [
    "search_literature",
    "generate_report",
    "add_dois_to_kb",
    "build_kb_from_search",
    "expand_kb_via_citations",
    "build_capsule",
    "build_capsules_for_kb",
    "fetch_paper_resources",
    "fetch_supplementary",
    "enrich_kb_from_cite_graph_tool",
    # 2026-05-16: ASB run ingest + github/skill-bundle ingest
    "ingest_asb_run",
    "ingest_github_repo",
    "ingest_skill_bundle",
]


@pytest.mark.parametrize("tool_name", SLOW_TOOLS)
def test_slow_mcp_tool_docstring_mentions_latency(tool_name):
    """Each slow tool's docstring should contain a 'Latency' note so
    LLM clients budgeting timeouts can see it via tools/list."""
    fn = getattr(server, tool_name, None)
    if fn is None:
        pytest.skip(f"Tool {tool_name!r} not present in this build")
    doc = inspect.getdoc(fn) or ""
    assert "Latency" in doc, (
        f"{tool_name} docstring must mention 'Latency' so LLM clients "
        "can budget timeouts. See docs/MCP.md.\n"
        f"Current docstring:\n{doc[:200]}"
    )
