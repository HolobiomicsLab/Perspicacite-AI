"""RAG engine - main entry point for RAG operations."""

from collections.abc import AsyncIterator
from typing import Any

from perspicacite.config.schema import Config
from perspicacite.llm.client import AsyncLLMClient
from perspicacite.llm.embeddings import EmbeddingProvider
from perspicacite.logging import get_logger
from perspicacite.models.rag import RAGMode, RAGRequest, RAGResponse, StreamEvent
from perspicacite.provenance.collector import ProvenanceCollector
from perspicacite.provenance.context import collecting
from perspicacite.rag.modes import (
    AdvancedRAGMode,
    AgenticRAGMode,
    BasicRAGMode,
    ContradictionRAGMode,
    DeepResearchRAGMode,
    LiteratureSurveyRAGMode,
    ReasoningRAGMode,
)
from perspicacite.rag.tools import ToolRegistry
from perspicacite.retrieval.chroma_store import ChromaVectorStore

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
        session_store: Any = None,
        app_state: Any = None,
    ):
        """
        Initialize RAG engine.

        Args:
            llm_client: LLM client
            vector_store: Vector store
            embedding_provider: Embedding provider
            tool_registry: Tool registry
            config: Configuration
            session_store: Optional SessionStore for KB reference storage
        """
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.tool_registry = tool_registry
        self.config = config
        self.app_state = app_state
        self.provenance_store: Any | None = None

        # Build survey mode separately so we can inject the session_store
        # (needed by _store_references_to_all_kbs to write SQLite reference rows)
        survey_mode = LiteratureSurveyRAGMode(config)
        survey_mode.session_store = session_store

        reasoning_mode = ReasoningRAGMode(config)
        reasoning_mode.session_store = session_store

        # Initialize mode handlers for all supported modes.
        # DeepResearchRAGMode is the canonical handler; both DEEP_RESEARCH and
        # PROFOUND keys point to the same instance for backward compat.
        _deep_research_mode = DeepResearchRAGMode(config)
        self._modes: dict[RAGMode, Any] = {
            RAGMode.BASIC: BasicRAGMode(config),
            RAGMode.ADVANCED: AdvancedRAGMode(config),
            RAGMode.DEEP_RESEARCH: _deep_research_mode,
            RAGMode.PROFOUND: _deep_research_mode,   # backward-compat alias
            RAGMode.AGENTIC: AgenticRAGMode(config),
            RAGMode.LITERATURE_SURVEY: survey_mode,
            RAGMode.CONTRADICTION: ContradictionRAGMode(config),
            RAGMode.REASONING: reasoning_mode,
        }

    async def query(
        self,
        request: RAGRequest,
        *,
        message_id: str | None = None,
        conversation_id: str | None = None,
    ) -> RAGResponse:
        """
        Execute a RAG query (non-streaming).

        Args:
            request: RAG request
            message_id: Optional message ID for provenance tracking
            conversation_id: Optional conversation ID for provenance tracking

        Returns:
            RAG response
        """
        if getattr(request, "app_state", None) is None:
            request.app_state = getattr(self, "app_state", None)

        logger.info(
            "rag_query_start",
            mode=request.mode.value,
            kb=request.kb_name,
        )

        handler = self._get_mode_handler(request.mode)
        collector = self._build_collector(request, message_id, conversation_id)

        try:
            with collecting(collector):
                response = await handler.execute(
                    request=request,
                    llm=self.llm_client,
                    vector_store=self.vector_store,
                    embedding_provider=self.embedding_provider,
                    tools=self.tool_registry,
                )

            await self._save_provenance(collector, message_id)

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
        *,
        message_id: str | None = None,
        conversation_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute a RAG query with streaming.

        Args:
            request: RAG request
            message_id: Optional message ID for provenance tracking
            conversation_id: Optional conversation ID for provenance tracking

        Yields:
            Stream events
        """
        if getattr(request, "app_state", None) is None:
            request.app_state = getattr(self, "app_state", None)

        logger.info(
            "rag_stream_start",
            mode=request.mode.value,
            kb=request.kb_name,
        )

        handler = self._get_mode_handler(request.mode)
        collector = self._build_collector(request, message_id, conversation_id)

        # NOTE: don't use `with collecting(collector)` here.
        #
        # When this generator yields under Starlette's StreamingResponse, the
        # awaitable can resume in a different asyncio Context than the one
        # that set the ContextVar's token. Then `ContextVar.reset(token)`
        # raises `ValueError: Token was created in a different Context`,
        # which surfaces as `rag_stream_error` and silently skips the
        # `_save_provenance` call below — so no provenance row is written.
        #
        # Setting the ContextVar without trying to reset it is safe: each
        # streaming request runs in its own task; the var goes out of scope
        # with the task. The agentic save path uses the same pattern.
        from perspicacite.provenance.context import set_collector
        set_collector(collector)

        saved = False
        try:
            async for event in handler.execute_stream(
                request=request,
                llm=self.llm_client,
                vector_store=self.vector_store,
                embedding_provider=self.embedding_provider,
                tools=self.tool_registry,
            ):
                yield event

            await self._save_provenance(collector, message_id)
            saved = True

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
                data=f'{{"message": "{e!s}"}}',
            )
        finally:
            # Async generators can be closed early (client disconnect,
            # GeneratorExit), which bypasses the `except Exception` branch
            # and skips the save above. Make a best-effort save here so
            # provenance is still recorded for partially-consumed streams.
            if not saved:
                try:
                    await self._save_provenance(collector, message_id)
                except Exception as exc:
                    logger.warning("provenance_save_finally_failed", error=str(exc))

    def _build_collector(
        self,
        request: RAGRequest,
        message_id: str | None,
        conversation_id: str | None,
    ) -> ProvenanceCollector:
        """Build a ProvenanceCollector for the given request."""
        mode_val = request.mode.value if hasattr(request.mode, "value") else str(request.mode)
        return ProvenanceCollector(
            conversation_id=conversation_id,
            message_id=message_id,
            rag_mode=mode_val,
            request_params={
                "kb_name": request.kb_name,
                "kb_names": getattr(request, "kb_names", None),
                "top_k": getattr(request, "top_k", None),
                "recency_weight": getattr(request, "recency_weight", None),
                "recency_half_life_years": getattr(request, "recency_half_life_years", None),
                "bm25_weight": getattr(request, "bm25_weight", None),
                "vector_weight": getattr(request, "vector_weight", None),
            },
        )

    async def _save_provenance(
        self,
        collector: ProvenanceCollector,
        message_id: str | None,
    ) -> None:
        """Persist provenance record (best-effort — never raises)."""
        if self.provenance_store is None or not message_id:
            return
        try:
            await self.provenance_store.save(collector.finalize())
        except Exception as exc:  # best-effort: never break the stream
            logger.warning("provenance_save_failed", error=str(exc))

    def _get_mode_handler(self, mode: RAGMode) -> Any:
        """Get handler for the given mode."""
        if mode not in self._modes:
            raise ValueError(f"Unknown RAG mode: {mode}")
        return self._modes[mode]
