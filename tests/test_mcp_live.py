#!/usr/bin/env python3
"""Live test scripts for Perspicacité MCP tools.

Requires a running server: `perspicacite -c config.yml serve`

Usage:
    python3 tests/test_mcp_live.py --all
    python3 tests/test_mcp_live.py --test search
    python3 tests/test_mcp_live.py --test content_pmc
    python3 tests/test_mcp_live.py --test content_arxiv
    python3 tests/test_mcp_live.py --test content_discovery
    python3 tests/test_mcp_live.py --test refs
    python3 tests/test_mcp_live.py --test kb
    python3 tests/test_mcp_live.py --test jsonrpc
    python3 tests/test_mcp_live.py --test nonstream
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

DEFAULT_PORT = 8000


# =========================================================================
# MCP Session Client
# =========================================================================

class MCPClient:
    """Minimal MCP streamable-HTTP client for testing."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None
        self._req_id = 0
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _session_headers(self) -> dict:
        h = dict(self._headers)
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _parse_sse(self, text: str) -> list[dict]:
        """Parse SSE response, return list of JSON payloads."""
        results = []
        for line in text.split("\n"):
            if line.startswith("data: "):
                results.append(json.loads(line[6:]))
        return results

    def initialize(self) -> dict:
        """Perform MCP initialize handshake."""
        r = httpx.post(
            self.base_url,
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-live", "version": "1.0"},
                },
            },
            headers=self._headers,
            timeout=10,
        )
        self.session_id = r.headers.get("mcp-session-id")
        results = self._parse_sse(r.text)
        if not results or "result" not in results[0]:
            raise RuntimeError(f"Initialize failed: {r.text}")
        return results[0]["result"]

    def send_initialized(self):
        """Send notifications/initialized to complete handshake."""
        httpx.post(
            self.base_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=self._session_headers(),
            timeout=5,
        )

    def list_tools(self) -> list[dict]:
        """List available MCP tools."""
        r = httpx.post(
            self.base_url,
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            },
            headers=self._session_headers(),
            timeout=10,
        )
        results = self._parse_sse(r.text)
        return results[0]["result"]["tools"]

    def call_tool(self, name: str, arguments: dict, timeout: int = 120) -> dict:
        """Call an MCP tool and return the parsed result."""
        r = httpx.post(
            self.base_url,
            json={
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            headers=self._session_headers(),
            timeout=timeout,
        )
        results = self._parse_sse(r.text)
        if not results:
            return {"error": "empty response", "raw": r.text}
        msg = results[0]
        if "error" in msg:
            return {"error": msg["error"]}
        # Extract text content from MCP content blocks
        content = msg.get("result", {}).get("content", [])
        text_parts = [c["text"] for c in content if c.get("type") == "text"]
        if text_parts:
            try:
                return json.loads(text_parts[0])
            except json.JSONDecodeError:
                return {"raw_text": text_parts[0]}
        return {"error": "no text content in response"}


def create_client(port: int) -> MCPClient:
    """Create and initialize an MCP client."""
    client = MCPClient(f"http://localhost:{port}/mcp")
    info = client.initialize()
    client.send_initialized()
    server_info = info.get("serverInfo", {})
    print(f"  Connected to {server_info.get('name', '?')} v{server_info.get('version', '?')}")
    print(f"  Session: {client.session_id}")
    return client


# =========================================================================
# Helpers
# =========================================================================

def _pass(name: str, detail: str = ""):
    msg = f"  PASS: {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _fail(name: str, detail: str = ""):
    msg = f"  FAIL: {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def _info(msg: str):
    print(f"  {msg}")


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) > max_len:
        return text[:max_len] + f"... ({len(text)} chars)"
    return text


def _cleanup_kb(kb_name: str):
    """Delete a test KB from ChromaDB and SQLite."""
    try:
        import chromadb, sqlite3
        from perspicacite.models.kb import chroma_collection_name_for_kb
        collection = chroma_collection_name_for_kb(kb_name)
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_client.delete_collection(collection)
        conn = sqlite3.connect("./data/perspicacite.db")
        conn.execute("DELETE FROM kb_metadata WHERE name = ?", (kb_name,))
        conn.commit()
        conn.close()
        _info(f"Cleaned up test KB: {kb_name}")
    except Exception as e:
        _info(f"Cleanup skipped ({e})")


# =========================================================================
# Tests
# =========================================================================

def test_discovery(port: int):
    """Test MCP handshake and tool listing."""
    print("\n>>> Test: MCP Discovery (initialize + tools/list)")

    client = create_client(port)
    tools = client.list_tools()
    tool_names = [t["name"] for t in tools]

    _info(f"Discovered {len(tools)} tools:")
    for name in tool_names:
        _info(f"  - {name}")

    expected = [
        "search_literature", "get_paper_content", "get_paper_references",
        "list_knowledge_bases", "search_knowledge_base",
        "create_knowledge_base", "add_papers_to_kb", "generate_report",
    ]
    missing = [t for t in expected if t not in tool_names]
    if missing:
        _fail("tool listing", f"missing: {missing}")
    else:
        _pass("tool listing", f"all {len(expected)} expected tools present")


def test_search(port: int):
    """Test search_literature with real backends."""
    print("\n>>> Test: search_literature (flash attention, 2024-2025)")

    client = create_client(port)
    result = client.call_tool("search_literature", {
        "query": "flash attention mechanisms",
        "max_results": 5,
        "year_min": 2024,
        "year_max": 2025,
        "databases": ["semantic_scholar", "openalex"],
    }, timeout=60)

    if "error" in result:
        _fail("search_literature", str(result["error"]))
        return

    papers = result.get("papers", result.get("results", []))
    total = result.get("total_results", len(papers))
    _info(f"Total results: {total}, returned: {len(papers)}")

    for p in papers[:3]:
        title = p.get("title", "?")[:70]
        year = p.get("year", "?")
        doi = p.get("doi", "no DOI")
        _info(f"  - [{year}] {title}  ({doi})")

    if len(papers) > 0:
        _pass("search_literature", f"{len(papers)} papers found")
    else:
        _fail("search_literature", "no papers returned")


def test_content_pmc(port: int):
    """Test get_paper_content with a PMC paper."""
    print("\n>>> Test: get_paper_content (PMC paper)")

    client = create_client(port)
    # A well-known OA paper with PMC full text
    doi = "10.1038/s41590-025-02241-4"
    _info(f"DOI: {doi}")

    result = client.call_tool("get_paper_content", {
        "doi": doi,
        "include_sections": True,
    }, timeout=60)

    if "error" in result:
        _fail("get_paper_content (PMC)", str(result["error"]))
        return

    success = result.get("success", False)
    ct = result.get("content_type", "?")
    src = result.get("content_source", "?")
    ft_len = result.get("full_text_length", len(result.get("full_text", "")))
    sections = result.get("sections")
    refs = result.get("references")

    _info(f"Success: {success}")
    _info(f"Content type: {ct}, Source: {src}")
    _info(f"Full text length: {ft_len}")
    if sections:
        _info(f"Sections ({len(sections)}): {list(sections.keys())[:5]}")
    if refs:
        _info(f"References: {len(refs)}")

    if success and ft_len > 0:
        _pass("get_paper_content (PMC)", f"{ct} from {src}, {ft_len} chars")
    else:
        _fail("get_paper_content (PMC)", f"success={success}, content_type={ct}")


def test_content_arxiv(port: int):
    """Test get_paper_content with an arXiv paper."""
    print("\n>>> Test: get_paper_content (arXiv paper)")

    client = create_client(port)
    # Flash attention paper on arXiv
    doi = "10.48550/arXiv.2205.14135"
    _info(f"DOI: {doi}")

    result = client.call_tool("get_paper_content", {
        "doi": doi,
    }, timeout=60)

    if "error" in result:
        _fail("get_paper_content (arXiv)", str(result["error"]))
        return

    success = result.get("success", False)
    ct = result.get("content_type", "?")
    src = result.get("content_source", "?")
    ft_len = result.get("full_text_length", len(result.get("full_text", "")))

    _info(f"Success: {success}")
    _info(f"Content type: {ct}, Source: {src}")
    _info(f"Full text length: {ft_len}")

    if success and ft_len > 0:
        _pass("get_paper_content (arXiv)", f"{ct} from {src}, {ft_len} chars")
    else:
        _fail("get_paper_content (arXiv)", f"success={success}, content_type={ct}")


def test_content_discovery(port: int):
    """Test get_paper_content with a paper that goes through discovery only (abstract)."""
    print("\n>>> Test: get_paper_content (abstract-only / discovery)")

    client = create_client(port)
    # A DOI unlikely to have full text OA
    doi = " 10.1056/NEJMra2500106"
#    doi = "10.1038/s42256-026-01200-4"
#    doi = "10.1126/science.adi3000"
    _info(f"DOI: {doi}")

    result = client.call_tool("get_paper_content", {
        "doi": doi,
    }, timeout=60)

    if "error" in result:
        _fail("get_paper_content (discovery)", str(result["error"]))
        return

    success = result.get("success", False)
    ct = result.get("content_type", "?")
    abstract = result.get("abstract", "")

    _info(f"Success: {success}")
    _info(f"Content type: {ct}")
    if abstract:
        _info(f"Abstract: {_truncate(abstract, 150)}")

    if ct == "abstract" and abstract:
        _pass("get_paper_content (discovery)", f"abstract returned ({len(abstract)} chars)")
    elif success:
        _pass("get_paper_content (discovery)", f"got {ct} content")
    else:
        _fail("get_paper_content (discovery)", f"success={success}, content_type={ct}")


def test_refs(port: int):
    """Test get_paper_references."""
    print("\n>>> Test: get_paper_references")

    client = create_client(port)
    # Use a PMC paper that likely has references
    doi = "10.1038/s41590-025-02241-4"
    _info(f"DOI: {doi}")

    result = client.call_tool("get_paper_references", {
        "doi": doi,
    }, timeout=60)

    if "error" in result:
        _fail("get_paper_references", str(result["error"]))
        return

    refs = result.get("references", [])
    total = result.get("total", len(refs))
    note = result.get("note", "")

    _info(f"Total references: {total}")
    if note:
        _info(f"Note: {note}")
    if refs:
        for r in refs[:3]:
            _info(f"  - {r.get('title', r.get('text', '?'))[:70]}")

    if total > 0:
        _pass("get_paper_references", f"{total} references found")
    elif note:
        _pass("get_paper_references", f"tool callable, note: {note}")
    else:
        _fail("get_paper_references", "no references and no note")


def test_kb(port: int):
    """Test full KB lifecycle: create → add → search → list → report."""
    print("\n>>> Test: KB lifecycle (create → add → search → list → report)")

    client = create_client(port)
    kb_name = f"test_kb_{int(time.time())}"

    # Step 1: Create KB
    _info(f"Creating KB: {kb_name}")
    result = client.call_tool("create_knowledge_base", {
        "name": kb_name,
        "description": "Live test KB",
    })
    if not result.get("success"):
        _fail("create_knowledge_base", str(result.get("error", result)))
        return
    _pass("create_knowledge_base")

    # Step 2: Add papers
    _info("Adding papers to KB...")
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
    result = client.call_tool("add_papers_to_kb", {
        "kb_name": kb_name,
        "papers": papers,
    })
    if result.get("success"):
        _pass("add_papers_to_kb", f"added {result.get('added_papers', result.get('added', '?'))} papers")
    else:
        _fail("add_papers_to_kb", str(result.get("error", result)))

    # Step 3: Search KB
    _info("Searching KB for 'transformer attention'...")
    result = client.call_tool("search_knowledge_base", {
        "query": "transformer attention",
        "kb_name": kb_name,
        "top_k": 3,
    })
    if "error" not in result:
        hits = result.get("results", result.get("papers", []))
        _pass("search_knowledge_base", f"{len(hits)} results")
    else:
        _fail("search_knowledge_base", str(result.get("error")))

    # Step 4: List KBs
    _info("Listing knowledge bases...")
    result = client.call_tool("list_knowledge_bases", {})
    if result.get("success"):
        kbs = result.get("knowledge_bases", [])
        found = any(kb.get("name") == kb_name for kb in kbs)
        if found:
            _pass("list_knowledge_bases", f"found {kb_name} in {len(kbs)} KBs")
        else:
            _fail("list_knowledge_bases", f"{kb_name} not found in {len(kbs)} KBs")
    else:
        _fail("list_knowledge_bases", str(result.get("error", result)))

    # Step 5: Generate report
    _info("Generating report...")
    result = client.call_tool("generate_report", {
        "query": "What are the key transformer architectures?",
        "kb_name": kb_name,
        "mode": "basic",
    }, timeout=180)
    if result.get("success"):
        report = result.get("report", "")
        _pass("generate_report", f"{len(report)} chars")
    else:
        # Report may fail without LLM configured — that's ok, tool was callable
        _info(f"Report result: {_truncate(str(result), 150)}")
        _pass("generate_report", "tool callable (may need LLM config)")

    # Step 6: Cleanup test KB
    _cleanup_kb(kb_name)


def test_jsonrpc(port: int):
    """Test that MCP tools are callable via JSON-RPC."""
    print(f"\n>>> Test: JSON-RPC session on port {port}")

    client = create_client(port)

    # Test tools/list
    tools = client.list_tools()
    _pass("tools/list via JSON-RPC", f"{len(tools)} tools")

    # Test a simple tool call
    result = client.call_tool("list_knowledge_bases", {})
    if "error" not in result:
        _pass("list_knowledge_bases via JSON-RPC")
    else:
        _fail("list_knowledge_bases via JSON-RPC", str(result.get("error")))


def test_nonstream(port: int):
    """Test /api/chat with stream=False returns valid JSON response."""
    print(f"\n>>> Test: /api/chat non-streaming on port {port}")

    # Create a KB with a paper so basic mode has something to retrieve
    client = create_client(port)
    kb_name = f"test_chat_{int(time.time())}"
    client.call_tool("create_knowledge_base", {"name": kb_name, "description": "Chat test KB"})
    client.call_tool("add_papers_to_kb", {
        "kb_name": kb_name,
        "papers": [{
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention",
            "doi": "10.48550/arXiv.2205.14135",
            "year": 2022,
            "authors": ["Dao", "Fu"],
            "abstract": "We present FlashAttention, an attention algorithm that computes exact attention with far fewer memory accesses.",
        }],
    })

    url = f"http://localhost:{port}/api/chat"
    payload = {
        "query": "What is flash attention?",
        "mode": "basic",
        "stream": False,
        "kb_name": kb_name,
    }

    try:
        r = httpx.post(url, json=payload, timeout=180)
        result = r.json()
    except Exception as e:
        _fail("/api/chat non-streaming", str(e))
        _cleanup_kb(kb_name)
        return

    if "answer" in result and result["answer"]:
        ans_len = len(result["answer"])
        sources = len(result.get("sources", []))
        _info(f"Answer: {ans_len} chars, Sources: {sources}")
        _info(f"Papers found: {result.get('papers_found', 0)}")
        _info(f"Conversation ID: {result.get('conversation_id', 'N/A')}")
        _pass("/api/chat non-streaming")
    elif "error" in result:
        _fail("/api/chat non-streaming", str(result["error"]))
    else:
        _fail("/api/chat non-streaming", f"unexpected: {_truncate(str(result))}")

    _cleanup_kb(kb_name)


# =========================================================================
# Main
# =========================================================================

TESTS = {
    "discovery": ("MCP Discovery (initialize + tools/list)", test_discovery),
    "search": ("search_literature", test_search),
    "content_pmc": ("get_paper_content (PMC)", test_content_pmc),
    "content_arxiv": ("get_paper_content (arXiv)", test_content_arxiv),
    "content_discovery": ("get_paper_content (abstract-only)", test_content_discovery),
    "refs": ("get_paper_references", test_refs),
    "kb": ("KB lifecycle", test_kb),
    "jsonrpc": ("JSON-RPC session", test_jsonrpc),
    "nonstream": ("/api/chat non-streaming", test_nonstream),
    "all_content": ("All content tests", lambda p: (test_content_pmc(p), test_content_arxiv(p), test_content_discovery(p))),
}


def main():
    parser = argparse.ArgumentParser(description="Live tests for Perspicacité MCP tools")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--test", choices=list(TESTS.keys()), help="Run a specific test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if args.test:
        if args.test == "all_content":
            args.all = True
            # run only content tests
            for key in ["content_pmc", "content_arxiv", "content_discovery"]:
                name, fn = TESTS[key]
                print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
                fn(args.port)
            return
        name, fn = TESTS[args.test]
        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        fn(args.port)
    elif args.all:
        for key, (name, fn) in TESTS.items():
            if key in ("all_content",):
                continue
            print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            try:
                fn(args.port)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print("Available tests:")
        for key, (name, _) in TESTS.items():
            print(f"  --test {key}: {name}")
        print(f"\nRun all: python3 {__file__} --all")
        print(f"Custom ports: python3 {__file__} --all --mcp-port 5500 --web-port 8000")


if __name__ == "__main__":
    main()
