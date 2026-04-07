#!/usr/bin/env python3
"""Live test scripts for Perspicacité MCP tools and JSON-RPC access.

These tests require a running Perspicacité server with real backends.
They are NOT part of the automated test suite — run manually.

Usage:
    # Start the server first:
    #   perspicacite serve --config config.yml

    # Run all live tests:
    #   python3 tests/test_mcp_live.py --base-url http://localhost:8000

    # Run individual tests:
    #   python3 tests/test_mcp_live.py --base-url http://localhost:8000 --test search
    #   python3 tests/test_mcp_live.py --base-url http://localhost:8000 --test content
    #   python3 tests/test_mcp_live.py --base-url http://localhost:8000 --test kb
    #   python3 tests/test_mcp_live.py --base-url http://localhost:8000 --test report
    #   python3 tests/test_mcp_live.py --base-url http://localhost:8000 --test jsonrpc
    #   python3 tests/test_mcp_live.py --base-url http://localhost:8000 --test nonstream
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"
MCP_PORT = 5001


def _json_post(url: str, data: dict, timeout: int = 120) -> dict:
    """POST JSON and return parsed response."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as e:
        return {"error": f"Connection failed: {e.reason}"}


def _json_rpc_call(tool_name: str, arguments: dict, timeout: int = 120) -> dict:
    """Call an MCP tool via JSON-RPC."""
    url = f"http://localhost:{MCP_PORT}/mcp"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    return _json_post(url, payload, timeout=timeout)


def _print_result(name: str, result: dict) -> None:
    """Print a test result with formatting."""
    success = result.get("success", "error" not in result)
    status = "PASS" if success else "FAIL"
    print(f"\n{'='*60}")
    print(f"[{status}] {name}")
    print(f"{'='*60}")
    # Truncate large fields for readability
    display = dict(result)
    if "full_text" in display and isinstance(display["full_text"], str) and len(display["full_text"]) > 200:
        display["full_text"] = display["full_text"][:200] + f"... ({len(result['full_text'])} chars total)"
    if "papers" in display and isinstance(display["papers"], list) and len(display["papers"]) > 3:
        display["papers"] = display["papers"][:3] + [f"... and {len(result['papers']) - 3} more"]
    if "report" in display and isinstance(display["report"], str) and len(display["report"]) > 300:
        display["report"] = display["report"][:300] + f"... ({len(result['report'])} chars total)"
    print(json.dumps(display, indent=2, ensure_ascii=False)[:2000])


# =========================================================================
# Test 1: search_literature
# =========================================================================

def test_search(base_url: str) -> None:
    """Test search_literature with real SciLex search."""
    print("\n>>> Test: search_literature (flash attention, 2023-2025)")

    # Via MCP tool (if MCP server running)
    result = _json_rpc_call("search_literature", {
        "query": "flash attention mechanisms",
        "max_results": 5,
        "year_min": 2023,
        "year_max": 2025,
        "databases": ["semantic_scholar", "openalex"],
    })

    _print_result("search_literature", result)

    if result.get("success"):
        papers = result.get("papers", [])
        print(f"\n  Found {len(papers)} papers")
        for p in papers[:3]:
            print(f"  - {p.get('title', '?')[:80]} ({p.get('year')}) [{p.get('doi', 'no DOI')}]")
    else:
        print(f"  Search returned error — MCP server may not be running on port {MCP_PORT}")
        print(f"  Error: {result.get('error', result)}")


# =========================================================================
# Test 2: get_paper_content — PMC paper
# =========================================================================

def test_content_pmc(base_url: str) -> None:
    """Test get_paper_content with a PMC paper (Nature Immunology)."""
    print("\n>>> Test: get_paper_content (PMC — bioRxiv preprint with published version)")

    # This bioRxiv DOI was published in Nature Immunology — should find via PMC
    doi = "10.1038/s41590-025-02241-4"
    result = _json_rpc_call("get_paper_content", {"doi": doi, "include_sections": True})

    _print_result("get_paper_content (PMC)", result)

    if result.get("success"):
        ct = result.get("content_type", "unknown")
        src = result.get("content_source", "unknown")
        length = result.get("full_text_length", 0)
        sections = result.get("sections")
        refs = result.get("references")
        print(f"\n  Content type: {ct}")
        print(f"  Source: {src}")
        print(f"  Full text length: {length}")
        if sections:
            print(f"  Sections ({len(sections)}): {list(sections.keys())[:5]}")
        if refs:
            print(f"  References: {len(refs)}")


# =========================================================================
# Test 3: get_paper_content — arXiv paper
# =========================================================================

def test_content_arxiv(base_url: str) -> None:
    """Test get_paper_content with an arXiv paper."""
    print("\n>>> Test: get_paper_content (arXiv)")

    doi = "10.48550/arXiv.2401.12345"
    result = _json_rpc_call("get_paper_content", {"doi": doi})

    _print_result("get_paper_content (arXiv)", result)

    if result.get("success"):
        ct = result.get("content_type", "unknown")
        src = result.get("content_source", "unknown")
        print(f"\n  Content type: {ct}")
        print(f"  Source: {src}")


# =========================================================================
# Test 4: KB lifecycle (create → add → search → report)
# =========================================================================

def test_kb_lifecycle(base_url: str) -> None:
    """Test full KB lifecycle: create, add papers, search, generate report."""
    kb_name = f"test_kb_{int(time.time())}"

    # Step 1: Create KB
    print(f"\n>>> Test: create_knowledge_base ({kb_name})")
    result = _json_rpc_call("create_knowledge_base", {"name": kb_name, "description": "Live test KB"})
    _print_result("create_knowledge_base", result)
    if not result.get("success"):
        print("  FAILED — skipping remaining KB tests")
        return

    # Step 2: Add papers
    print(f"\n>>> Test: add_papers_to_kb")
    papers = [
        {
            "title": "Attention Is All You Need",
            "doi": "10.5555/3295222.3295349",
            "year": 2017,
            "authors": ["Vaswani", "Shazeer"],
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "doi": "10.18653/v1/N19-1423",
            "year": 2019,
            "authors": ["Devlin", "Chang"],
            "abstract": "We introduce a new language representation model called BERT.",
        },
    ]
    result = _json_rpc_call("add_papers_to_kb", {"kb_name": kb_name, "papers": papers})
    _print_result("add_papers_to_kb", result)

    # Step 3: Search KB
    print(f"\n>>> Test: search_knowledge_base")
    result = _json_rpc_call("search_knowledge_base", {"query": "transformer attention", "kb_name": kb_name, "top_k": 3})
    _print_result("search_knowledge_base", result)

    # Step 4: List KBs
    print(f"\n>>> Test: list_knowledge_bases")
    result = _json_rpc_call("list_knowledge_bases", {})
    _print_result("list_knowledge_bases", result)

    # Step 5: Generate report (skip if no LLM — just show it was callable)
    print(f"\n>>> Test: generate_report")
    result = _json_rpc_call("generate_report", {
        "query": "What are the key transformer architectures?",
        "kb_name": kb_name,
        "mode": "basic",
    })
    _print_result("generate_report", result)


# =========================================================================
# Test 5: JSON-RPC access (validates /mcp endpoint)
# =========================================================================

def test_jsonrpc(base_url: str) -> None:
    """Test that MCP tools are callable via JSON-RPC POST to /mcp."""
    print(f"\n>>> Test: JSON-RPC access to MCP tools on port {MCP_PORT}")

    # Test 1: list_tools
    url = f"http://localhost:{MCP_PORT}/mcp"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    result = _json_post(url, payload)
    _print_result("tools/list (JSON-RPC)", result)

    # Test 2: Call list_knowledge_bases via JSON-RPC
    result = _json_rpc_call("list_knowledge_bases", {})
    _print_result("list_knowledge_bases (JSON-RPC)", result)

    if result.get("success"):
        print("\n  JSON-RPC access confirmed — MCP tools work via plain HTTP POST")
    else:
        print(f"\n  JSON-RPC failed — is the MCP server running on port {MCP_PORT}?")


# =========================================================================
# Test 6: Non-streaming /api/chat
# =========================================================================

def test_nonstream(base_url: str) -> None:
    """Test /api/chat with stream=False returns JSON."""
    print(f"\n>>> Test: /api/chat non-streaming")

    payload = {
        "query": "What is flash attention?",
        "mode": "basic",
        "stream": False,
        "max_papers": 2,
        "databases": ["semantic_scholar"],
    }

    result = _json_post(f"{base_url}/api/chat", payload, timeout=180)
    _print_result("/api/chat (non-streaming)", result)

    if "answer" in result and result["answer"]:
        print(f"\n  Answer length: {len(result['answer'])} chars")
        print(f"  Sources: {len(result.get('sources', []))}")
        print(f"  Papers found: {result.get('papers_found', 0)}")
        print(f"  Conversation ID: {result.get('conversation_id', 'N/A')}")
    elif "error" in result:
        print(f"  Error: {result['error']}")
    else:
        print("  Unexpected response format")


# =========================================================================
# Main
# =========================================================================

TESTS = {
    "search": ("Test search_literature", test_search),
    "content_pmc": ("Test get_paper_content (PMC)", test_content_pmc),
    "content_arxiv": ("Test get_paper_content (arXiv)", test_content_arxiv),
    "kb": ("Test KB lifecycle (create→add→search→report)", test_kb_lifecycle),
    "jsonrpc": ("Test JSON-RPC access to MCP tools", test_jsonrpc),
    "nonstream": ("Test /api/chat non-streaming", test_nonstream),
    "content": ("Test get_paper_content (PMC + arXiv)", lambda b: (test_content_pmc(b), test_content_arxiv(b))),
    "report": ("Test generate_report", lambda b: test_kb_lifecycle(b)),
}


def main():
    parser = argparse.ArgumentParser(description="Live tests for Perspicacité MCP tools")
    parser.add_argument("--base-url", default=BASE_URL, help="Perspicacité web app URL")
    parser.add_argument("--test", choices=list(TESTS.keys()), help="Run a specific test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.test:
        name, fn = TESTS[args.test]
        print(f"\nRunning: {name}")
        fn(args.base_url)
    elif args.all:
        for key, (name, fn) in TESTS.items():
            if key in ("content", "report"):
                continue  # skip aliases
            print(f"\nRunning: {name}")
            fn(args.base_url)
    else:
        print("Available tests:")
        for key, (name, _) in TESTS.items():
            print(f"  --test {key}: {name}")
        print(f"\nRun all: python3 {__file__} --all --base-url {BASE_URL}")


if __name__ == "__main__":
    main()
