"""
Test script for LOTUS natural products search tool.

Usage:
    .venv/bin/python test_lotus.py "quercetin"
    .venv/bin/python test_lotus.py "caffeine"
"""

import asyncio
import sys

# Direct import to avoid circular imports through rag package
sys.path.insert(0, "src")

# Import directly from file, not through package
import importlib.util
spec = importlib.util.spec_from_file_location("lotus", "src/perspicacite/rag/tools/lotus.py")
lotus_module = importlib.util.module_from_spec(spec)
sys.modules["lotus"] = lotus_module
spec.loader.exec_module(lotus_module)

LotusSearchTool = lotus_module.LotusSearchTool


async def test_simple_search(query: str):
    """Test simple name search."""
    print(f"\n{'='*60}")
    print(f"🔬 Testing LOTUS Simple Search: '{query}'")
    print(f"{'='*60}\n")
    
    tool = LotusSearchTool()
    results = await tool.execute(query=query, search_type="simple", max_results=3)
    
    print(results)


async def main():
    """Run tests."""
    # Test 1: Simple name search
    await test_simple_search("quercetin")
    
    # Test 2: Another compound
    await test_simple_search("paclitaxel")
    
    print(f"\n{'='*60}")
    print("✅ All tests completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Allow command line query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        asyncio.run(test_simple_search(query))
    else:
        asyncio.run(main())
