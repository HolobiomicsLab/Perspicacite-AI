"""Tools for RAG modes."""

from typing import Any, Protocol

from perspicacite.logging import get_logger
from perspicacite.rag.tools.lotus import LotusSearchTool

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
    """Live academic web search across user-selected databases.

    Wraps the shared web aggregator (``run_web_aggregator_search``) so RAG
    modes (currently Profound, but reusable elsewhere) can invoke a single
    "web_search" tool whose results come back as a small JSON-style summary
    that ``profound._parse_web_tool_results`` already knows how to parse.

    The tool deliberately produces a compact string payload (one entry per
    line: ``Title|Authors|Year|DOI|URL|Snippet``) rather than full JSON so
    profound's existing parser keeps working with no behaviour changes.
    """

    name = "web_search"
    description = (
        "Search live academic literature databases (Semantic Scholar, "
        "OpenAlex, Europe PMC, etc.) when the knowledge base is missing or "
        "insufficient. Returns title / authors / year / DOI / abstract per hit."
    )

    def __init__(
        self,
        *,
        app_state: Any = None,
        databases: list[str] | None = None,
        max_results: int = 5,
    ) -> None:
        self._app_state = app_state
        # ``databases`` is best-effort: callers may also pass it per-call.
        self._default_databases = databases
        self._max_results = max_results

    async def execute(
        self,
        query: str,
        max_results: int | None = None,
        databases: list[str] | None = None,
        context: str | None = None,
        telemetry: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> str:
        """Run a live web aggregator search and return a JSON-encoded payload.

        Returns a JSON list of ``{title, authors, year, doi, url, content,
        citation, source}`` records. Profound's ``_parse_web_tool_results``
        handles both raw lists and JSON strings, so this structured form
        lets each paper appear as its own document instead of one giant
        concatenated blob.
        """
        from perspicacite.rag.web_search import run_web_aggregator_search

        n = max_results or self._max_results
        dbs = databases or self._default_databases
        try:
            papers = await run_web_aggregator_search(
                keyword_query=query,
                context=context,
                optimize_enabled=None,
                databases=dbs,
                max_docs=n,
                app_state=self._app_state,
                telemetry=telemetry,
            )
        except Exception as exc:
            logger.warning("web_search_tool_failed", error=str(exc))
            return "[]"

        if not papers:
            return "[]"

        import json as _json

        out: list[dict[str, Any]] = []
        for p in papers[:n]:
            title = getattr(p, "title", "") or "Untitled"
            authors_list = getattr(p, "authors", None) or []
            authors = ", ".join(a.name for a in authors_list[:3])
            year = getattr(p, "year", "") or ""
            doi = getattr(p, "doi", "") or ""
            url = getattr(p, "pdf_url", "") or getattr(p, "url", "") or ""
            abstract = (getattr(p, "abstract", "") or "").strip()
            # ``content`` is what profound's analyzer reads. Pack a short
            # citation line + abstract so the LLM sees both the paper
            # identity and the substance.
            citation = f"{title} ({year})" if year else title
            content = abstract or citation
            src_obj = getattr(p, "source", None)
            src_str = getattr(src_obj, "value", None) or (
                str(src_obj).replace("PaperSource.", "").lower() if src_obj else ""
            )
            out.append(
                {
                    "title": title,
                    "authors": authors,
                    "year": str(year) if year else "",
                    "doi": doi,
                    "url": url,
                    "content": content,
                    "citation": citation,
                    "source": src_str or "web_search",
                }
            )
        return _json.dumps(out)


__all__ = [
    "KBSearchTool",
    "LotusSearchTool",
    "Tool",
    "ToolRegistry",
    "WebSearchTool",
]
