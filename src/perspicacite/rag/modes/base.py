"""Base class for RAG modes."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.models.rag import RAGRequest, RAGResponse, StreamEvent


class BaseRAGMode(ABC):
    """Base class for RAG modes."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    async def execute(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> RAGResponse:
        """Execute RAG query."""
        pass

    @abstractmethod
    async def execute_stream(
        self,
        request: RAGRequest,
        llm: Any,
        vector_store: Any,
        embedding_provider: Any,
        tools: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Execute RAG query with streaming."""
        pass

    def _build_messages(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build message list for LLM."""
        system = system_prompt or "Answer based on the provided documents."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {query}"},
        ]
