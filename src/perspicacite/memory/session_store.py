"""SQLite-backed session and conversation persistence."""

import json
from pathlib import Path
from typing import Any

import aiosqlite

from perspicacite.logging import get_logger
from perspicacite.models.kb import KnowledgeBase
from perspicacite.models.messages import Conversation, Message

logger = get_logger("perspicacite.memory.session_store")

# SQL Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    title TEXT,
    kb_name TEXT DEFAULT 'default',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kb_metadata (
    name TEXT PRIMARY KEY,
    description TEXT,
    collection_name TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_config TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    paper_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
"""


class SessionStore:
    """SQLite-backed session and conversation persistence."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Initialize database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()
        logger.info("database_initialized", path=str(self.db_path))

    async def create_conversation(
        self,
        session_id: str,
        kb_name: str = "default",
        title: str | None = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            title=title or "New Conversation",
            kb_name=kb_name,
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO conversations (id, session_id, title, kb_name)
                VALUES (?, ?, ?, ?)
                """,
                (conversation.id, session_id, conversation.title, kb_name),
            )
            await db.commit()

        return conversation

    async def get_conversation(self, conv_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get conversation
            row = await db.execute_fetchall(
                "SELECT * FROM conversations WHERE id = ?",
                (conv_id,),
            )

            if not row:
                return None

            conv_data = row[0]

            # Get messages
            messages = await self.get_messages(conv_id)

            return Conversation(
                id=conv_data["id"],
                title=conv_data["title"],
                kb_name=conv_data["kb_name"],
                messages=messages,
            )

    async def list_conversations(self, session_id: str) -> list[Conversation]:
        """List all conversations for a session."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            rows = await db.execute_fetchall(
                "SELECT * FROM conversations WHERE session_id = ? ORDER BY updated_at DESC",
                (session_id,),
            )

            return [
                Conversation(
                    id=r["id"],
                    title=r["title"],
                    kb_name=r["kb_name"],
                )
                for r in rows
            ]

    async def add_message(self, conv_id: str, message: Message) -> None:
        """Add a message to a conversation."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, sources, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    conv_id,
                    message.role,
                    message.content,
                    json.dumps([s.model_dump() for s in message.sources]),
                    json.dumps(message.metadata),
                ),
            )

            # Update conversation timestamp
            await db.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conv_id,),
            )

            await db.commit()

    async def get_messages(self, conv_id: str, limit: int = 100) -> list[Message]:
        """Get messages for a conversation."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            rows = await db.execute_fetchall(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (conv_id, limit),
            )

            messages = []
            for r in rows:
                from perspicacite.models.rag import SourceReference

                sources_data = json.loads(r["sources"])
                sources = [SourceReference(**s) for s in sources_data]

                messages.append(
                    Message(
                        id=r["id"],
                        role=r["role"],
                        content=r["content"],
                        sources=sources,
                        metadata=json.loads(r["metadata"]),
                    )
                )

            return messages

    async def save_kb_metadata(self, kb: KnowledgeBase) -> None:
        """Save KB metadata."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO kb_metadata
                (name, description, collection_name, embedding_model, chunk_config, paper_count, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kb.name,
                    kb.description,
                    kb.collection_name,
                    kb.embedding_model,
                    kb.chunk_config.model_dump_json(),
                    kb.paper_count,
                    kb.chunk_count,
                ),
            )
            await db.commit()

    async def get_kb_metadata(self, name: str) -> KnowledgeBase | None:
        """Get KB metadata."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            rows = await db.execute_fetchall(
                "SELECT * FROM kb_metadata WHERE name = ?",
                (name,),
            )

            if not rows:
                return None

            r = rows[0]
            from perspicacite.models.kb import ChunkConfig

            return KnowledgeBase(
                name=r["name"],
                description=r["description"],
                collection_name=r["collection_name"],
                embedding_model=r["embedding_model"],
                chunk_config=ChunkConfig(**json.loads(r["chunk_config"])),
                paper_count=r["paper_count"],
                chunk_count=r["chunk_count"],
            )

    async def list_kbs(self) -> list[KnowledgeBase]:
        """List all knowledge bases."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            rows = await db.execute_fetchall(
                "SELECT * FROM kb_metadata ORDER BY updated_at DESC"
            )

            from perspicacite.models.kb import ChunkConfig

            return [
                KnowledgeBase(
                    name=r["name"],
                    description=r["description"],
                    collection_name=r["collection_name"],
                    embedding_model=r["embedding_model"],
                    chunk_config=ChunkConfig(**json.loads(r["chunk_config"])),
                    paper_count=r["paper_count"],
                    chunk_count=r["chunk_count"],
                )
                for r in rows
            ]

    async def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages.
        
        Returns True if conversation was found and deleted, False otherwise.
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Check if conversation exists
            row = await db.execute_fetchall(
                "SELECT id FROM conversations WHERE id = ?",
                (conv_id,),
            )
            if not row:
                return False

            # Delete conversation (messages will be cascade deleted)
            await db.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conv_id,),
            )
            await db.commit()

        logger.info("conversation_deleted", conversation_id=conv_id)
        return True

    async def delete_all_conversations(self) -> int:
        """Delete all conversations and their messages.
        
        Returns the number of conversations deleted.
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get count before deletion
            row = await db.execute_fetchall(
                "SELECT COUNT(*) as count FROM conversations"
            )
            count = row[0]["count"] if row else 0

            # Delete all conversations (messages will be cascade deleted)
            await db.execute("DELETE FROM conversations")
            await db.commit()

        logger.info("all_conversations_deleted", count=count)
        return count
