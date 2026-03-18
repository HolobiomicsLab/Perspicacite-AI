"""Tools for RAG modes."""

from typing import Any, Protocol

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.rag.tools")


class Tool(Protocol):
    """Protocol for tools."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    async def execute(self, **kwargs: Any) -> str: ...


class ToolRegistry:
    """Registry of tools available to RAG modes."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name)

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self.tools:
            raise KeyError(f"Tool not found: {name}")
        return self.tools[name]

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())


class KBSearchTool:
    """Tool to search knowledge base."""

    name = "kb_search"
    description = "Search the knowledge base for relevant documents"

    def __init__(self, vector_store, embedding_provider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    async def execute(
        self,
        query: str,
        kb_name: str = "default",
        top_k: int = 10,
    ) -> str:
        """Execute KB search."""
        from perspicacite.models.search import SearchFilters

        # Generate embedding
        embeddings = await self.embedding_provider.embed([query])

        # Search
        results = await self.vector_store.search(
            collection=kb_name,
            query_embedding=embeddings[0],
            top_k=top_k,
        )

        # Format results
        if not results:
            return "No relevant documents found."

        lines = [f"Found {len(results)} relevant documents:"]
        for r in results:
            lines.append(
                f"- {r.chunk.metadata.title or 'Untitled'} "
                f"(score: {r.score:.3f})"
            )

        return "\n".join(lines)


class WebSearchTool:
    """Tool to search the web."""

    name = "web_search"
    description = "Search academic databases on the web"

    async def execute(
        self,
        query: str,
        max_results: int = 10,
    ) -> str:
        """Execute web search."""
        # Placeholder - would integrate with SciLEx
        return f"Web search for '{query}' found {max_results} results."


class FetchPDFTool:
    """Tool to fetch and parse PDF."""

    name = "fetch_pdf"
    description = "Download and parse a PDF from URL"

    async def execute(self, url: str) -> str:
        """Fetch and parse PDF."""
        return f"Fetched PDF from {url}"


class CitationNetworkTool:
    """Tool to get citation network."""

    name = "citation_network"
    description = "Get papers citing or cited by a given paper"

    async def execute(
        self,
        paper_id: str,
        direction: str = "both",
    ) -> str:
        """Get citation network."""
        return f"Citation network for {paper_id} ({direction})"
