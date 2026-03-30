"""Chat endpoints with real RAG processing."""

import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from perspicacite.logging import get_logger
from perspicacite.models.api import ChatRequest, ChatResponse
from perspicacite.models.rag import RAGRequest, StreamEvent, RAGMode
from perspicacite.rag.engine import RAGEngine
from perspicacite.config.loader import load_config
from perspicacite.llm.client import AsyncLLMClient
from perspicacite.llm.embeddings import EmbeddingProvider
from perspicacite.retrieval.chroma_store import ChromaVectorStore
from perspicacite.rag.tools import ToolRegistry

logger = get_logger("perspicacite.api.chat")

router = APIRouter()

# Singleton instances for the API
_rag_engine: RAGEngine | None = None


async def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        config = load_config()
        llm_client = AsyncLLMClient(config.llm)
        vector_store = ChromaVectorStore(config.database.chroma_path)
        embedding_provider = EmbeddingProvider(config.knowledge_base.embedding_model)
        tool_registry = ToolRegistry()

        _rag_engine = RAGEngine(
            llm_client=llm_client,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            tool_registry=tool_registry,
            config=config,
        )
    return _rag_engine


def _get_rag_mode(mode_str: str) -> RAGMode:
    """Convert string mode to RAGMode enum."""
    mode_map = {
        "basic": RAGMode.BASIC,
        "advanced": RAGMode.ADVANCED,
        "profound": RAGMode.PROFOUND,
        "agentic": RAGMode.AGENTIC,
    }
    return mode_map.get(mode_str.lower(), RAGMode.AGENTIC)


@router.post("/chat")
async def chat(
    request: ChatRequest,
    rag_engine: RAGEngine = Depends(get_rag_engine),
):
    """
    Main chat endpoint with real RAG processing.

    If stream=False: returns ChatResponse
    If stream=True: returns SSE stream of StreamEvent
    """
    # Extract the last user message as the query
    query = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            query = msg.content
            break

    if not query:
        raise HTTPException(status_code=400, detail="No user message found in conversation")

    rag_mode = request.mode if isinstance(request.mode, RAGMode) else _get_rag_mode(request.mode)

    logger.info(
        "chat_request",
        query=query[:100],
        mode=rag_mode.value,
        stream=request.stream,
        max_iterations=request.max_iterations,
    )

    # Create RAG request
    rag_request = RAGRequest(
        query=query,
        kb_name=request.kb_name,
        mode=rag_mode,
        provider=request.provider,
        model=request.model,
        max_iterations=request.max_iterations,
        use_web_search=request.use_web_search,
        conversation_id=request.conversation_id,
    )

    if request.stream:
        # Streaming response with true RAG processing
        async def event_generator() -> AsyncIterator[dict]:
            full_answer = ""
            sources = []

            try:
                async for event in rag_engine.query_stream(rag_request):
                    if event.event == "status":
                        data = json.loads(event.data)
                        yield {
                            "event": "status",
                            "data": json.dumps({"message": data.get("message", "")}),
                        }

                    elif event.event == "source":
                        source_data = json.loads(event.data)
                        sources.append(source_data)
                        yield {"event": "source", "data": event.data}

                    elif event.event == "content":
                        data = json.loads(event.data)
                        delta = data.get("delta", "")
                        full_answer += delta
                        yield {"event": "content", "data": event.data}

                    elif event.event == "error":
                        yield {"event": "error", "data": event.data}
                        return

                # Send done event
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "conversation_id": request.conversation_id or "new",
                            "mode": rag_mode.value,
                            "sources": sources,
                        }
                    ),
                }

            except Exception as e:
                logger.error("chat_streaming_error", error=str(e))
                yield {
                    "event": "error",
                    "data": json.dumps({"message": f"Error processing request: {str(e)}"}),
                }

        return EventSourceResponse(event_generator())

    # Non-streaming response
    try:
        response = await rag_engine.query(rag_request)

        from perspicacite.models.messages import Message

        return ChatResponse(
            message=Message(
                id="msg-1",
                role="assistant",
                content=response.answer,
            ),
            sources=response.sources,
            conversation_id=request.conversation_id or "new",
            mode=rag_mode,
        )
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
