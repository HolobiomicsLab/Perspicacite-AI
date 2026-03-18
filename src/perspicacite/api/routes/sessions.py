"""Session management endpoints."""

from fastapi import APIRouter

from perspicacite.models.messages import Conversation

router = APIRouter()


@router.get("/conversations")
async def list_conversations() -> list[Conversation]:
    """List all conversations."""
    return []


@router.get("/conversations/{id}")
async def get_conversation(id: str) -> Conversation:
    """Get conversation by ID."""
    return Conversation(id=id, title="Test Conversation")


@router.delete("/conversations/{id}")
async def delete_conversation(id: str):
    """Delete conversation."""
    return {"deleted": id}


@router.get("/providers")
async def list_providers():
    """List available LLM providers."""
    return {
        "providers": [
            {"name": "anthropic", "models": ["claude-3-5-sonnet-20241022"]},
            {"name": "openai", "models": ["gpt-4o"]},
        ]
    }
