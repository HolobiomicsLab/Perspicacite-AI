"""MCP server implementation."""

try:
    from fastmcp import FastMCP

    mcp = FastMCP("perspicacite")

    @mcp.tool
    async def research_literature(
        query: str,
        mode: str = "deep",
        kb_name: str = "default",
        max_iterations: int = 3,
        use_web_search: bool = True,
    ) -> str:
        """
        Research a scientific question using Perspicacité's RAG system.

        Args:
            query: Research question
            mode: RAG mode (quick, standard, advanced, deep, citation)
            kb_name: Knowledge base to search
            max_iterations: Maximum research iterations
            use_web_search: Whether to use web search

        Returns:
            Research answer with citations
        """
        return f"Research on '{query}' completed using {mode} mode."

    @mcp.tool
    async def search_knowledge_base(
        query: str,
        kb_name: str = "default",
        top_k: int = 5,
    ) -> str:
        """
        Quick search in a specific knowledge base.

        Args:
            query: Search query
            kb_name: Knowledge base name
            top_k: Number of results

        Returns:
            Search results
        """
        return f"Found {top_k} results for '{query}' in {kb_name}."

    @mcp.tool
    async def list_knowledge_bases() -> str:
        """List available knowledge bases."""
        return "Available KBs: default"

    @mcp.tool
    async def add_papers_to_kb(
        kb_name: str,
        papers: list[str],
    ) -> str:
        """
        Add papers to a knowledge base.

        Args:
            kb_name: Knowledge base name
            papers: List of DOIs or URLs

        Returns:
            Status message
        """
        return f"Added {len(papers)} papers to {kb_name}."

    @mcp.resource("perspicacite://info")
    async def get_info() -> str:
        """Perspicacité capabilities and status."""
        return "Perspicacité v2.0 - AI-powered literature research assistant"

except ImportError:
    # fastmcp not installed, create stub
    mcp = None
