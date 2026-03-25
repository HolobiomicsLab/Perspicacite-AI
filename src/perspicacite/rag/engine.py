"""RAG engine - main entry point for RAG operations."""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.llm.client import AsyncLLMClient
from perspicacite.llm.embeddings import EmbeddingProvider
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, StreamEvent
from perspicacite.retrieval.chroma_store import ChromaVectorStore
from perspicacite.rag.modes.agentic import AgenticRAGMode
from perspicacite.rag.tools import ToolRegistry

logger = get_logger("perspicacite.rag.engine")


class RAGEngine:
    """
    Main entry point for RAG operations.

    Routes requests to the appropriate mode handler.
    """

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        tool_registry: ToolRegistry,
        config: Config,
    ):
        """
        Initialize RAG engine.

        Args:
            llm_client: LLM client
            vector_store: Vector store
            embedding_provider: Embedding provider
            tool_registry: Tool registry
            config: Configuration
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.tool_registry = tool_registry
        self.config = config

        # Initialize agentic mode handler (only mode supported)
        self._handler = AgenticRAGMode(config)

    async def query(self, request: RAGRequest) -> RAGResponse:
        """
        Execute a RAG query (non-streaming).

        Args:
            request: RAG request

        Returns:
            RAG response
        """
        logger.info(
            "rag_query_start",
            mode=request.mode.value,
            kb=request.kb_name,
        )

        handler = self._get_handler(request.mode)

        try:
            response = await handler.execute(
                request=request,
                llm=self.llm_client,
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                tools=self.tool_registry,
            )

            logger.info(
                "rag_query_complete",
                mode=request.mode.value,
                sources=len(response.sources),
                iterations=response.iterations,
            )

            return response

        except Exception as e:
            logger.error(
                "rag_query_error",
                mode=request.mode.value,
                error=str(e),
            )
            raise

    async def query_stream(
        self,
        request: RAGRequest,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute a RAG query with streaming.

        Args:
            request: RAG request

        Yields:
            Stream events
        """
        logger.info(
            "rag_stream_start",
            mode=request.mode.value,
            kb=request.kb_name,
        )

        handler = self._get_handler(request.mode)

        try:
            async for event in handler.execute_stream(
                request=request,
                llm=self.llm_client,
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                tools=self.tool_registry,
            ):
                yield event

            logger.info("rag_stream_complete", mode=request.mode.value)

        except Exception as e:
            logger.error(
                "rag_stream_error",
                mode=request.mode.value,
                error=str(e),
            )
            # Yield error event
            yield StreamEvent(
                event="error",
                data=f'{{"message": "{str(e)}"}}',
            )

    def _get_handler(self, mode: RAGMode) -> AgenticRAGMode:
        """Get handler - always returns AgenticRAGMode."""
        return self._handler
