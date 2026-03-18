"""Chat endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from perspicacite.models.api import ChatRequest, ChatResponse
from perspicacite.models.rag import RAGRequest, StreamEvent

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    If stream=False: returns ChatResponse
    If stream=True: returns SSE stream of StreamEvent
    """
    if request.stream:
        # Streaming response
        async def event_generator():
            # Placeholder - would use actual RAG engine
            yield {"event": "status", "data": '{"message": "Processing..."}'}
            yield {"event": "content", "data": '{"delta": "This is a "}'}
            yield {"event": "content", "data": '{"delta": "streaming response."}'}
            yield {"event": "done", "data": '{"conversation_id": "test-123", "tokens_used": 100}'}

        return EventSourceResponse(event_generator())

    # Non-streaming response
    return ChatResponse(
        message={
            "id": "msg-1",
            "role": "assistant",
            "content": "This is a test response.",
            "sources": [],
        },
        sources=[],
        conversation_id="test-123",
        mode=request.mode,
    )
