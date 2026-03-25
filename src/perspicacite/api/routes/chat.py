"""Chat endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from perspicacite.logging import get_logger
from perspicacite.models.api import ChatRequest, ChatResponse
from perspicacite.models.rag import RAGRequest, StreamEvent, RAGMode

logger = get_logger("perspicacite.api.chat")

router = APIRouter()


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
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    If stream=False: returns ChatResponse
    If stream=True: returns SSE stream of StreamEvent
    """
    # Log the mode being used
    rag_mode = _get_rag_mode(request.mode)
    logger.info(
        "chat_request",
        query=request.messages[-1].content if request.messages else "",
        mode=rag_mode.value,
        stream=request.stream,
        max_iterations=request.max_iterations,
    )
    
    if request.stream:
        # Streaming response
        async def event_generator():
            yield {"event": "status", "data": f'{{"message": "Using {rag_mode.value} mode..."}'}
            yield {"event": "content", "data": '{"delta": "Processing with ' + rag_mode.value + ' mode... "}'}
            yield {"event": "content", "data": '{"delta": "(Full implementation pending)"}'}
            yield {"event": "done", "data": f'{{"conversation_id": "{request.conversation_id or "new"}", "mode": "{rag_mode.value}"}}'}

        return EventSourceResponse(event_generator())

    # Non-streaming response
    return ChatResponse(
        message={
            "id": "msg-1",
            "role": "assistant",
            "content": f"Processed with {rag_mode.value} mode. (Full implementation pending)",
            "sources": [],
        },
        sources=[],
        conversation_id=request.conversation_id or "new",
        mode=rag_mode,
    )
