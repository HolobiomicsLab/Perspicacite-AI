"""Message and session models."""

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from perspicacite.models.rag import SourceReference


class Message(BaseModel):
    """A chat message."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: list[SourceReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        content_preview = self.content[:50].replace("\n", " ")
        return f"Message(role='{self.role}', content='{content_preview}...')"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API serialization."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sources": [s.model_dump() for s in self.sources],
            "metadata": self.metadata,
        }


class Conversation(BaseModel):
    """A conversation (chat thread)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = None
    kb_name: str = "default"
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"Conversation(id='{self.id}', "
            f"title='{self.title}', messages={len(self.messages)})"
        )

    def add_message(self, message: Message) -> None:
        """Add a message and update timestamp."""
        self.messages.append(message)
        self.updated_at = datetime.now()

    def get_last_messages(self, n: int = 10) -> list[Message]:
        """Get last n messages."""
        return self.messages[-n:] if len(self.messages) > n else self.messages


class Session(BaseModel):
    """A user session."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "default"
    conversations: list[str] = Field(default_factory=list)  # Conversation IDs
    preferences: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return f"Session(id='{self.id}', user='{self.user_id}', conversations={len(self.conversations)})"
