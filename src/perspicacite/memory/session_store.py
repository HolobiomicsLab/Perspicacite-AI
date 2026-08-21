"""SQLite-backed session and conversation persistence."""

import contextlib
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

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    total INTEGER DEFAULT 0,
    done_count INTEGER DEFAULT 0,
    result TEXT,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS kb_paper_references (
    id           TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(8)))),
    kb_name      TEXT NOT NULL,
    doi          TEXT,
    title        TEXT NOT NULL,
    authors_json TEXT DEFAULT '[]',
    year         INTEGER,
    abstract     TEXT,
    survey_query TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(kb_name, doi)
);
CREATE INDEX IF NOT EXISTS idx_kb_paper_refs_kb ON kb_paper_references(kb_name);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
"""

from perspicacite.provenance.schema import PROVENANCE_TABLE_SQL

SCHEMA = SCHEMA + "\n" + PROVENANCE_TABLE_SQL


class SessionStore:
    """SQLite-backed session and conversation persistence."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_available: bool = False

    async def init_db(self) -> None:
        """Initialize database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            try:
                await db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts "
                    "USING fts5(content, conversation_id UNINDEXED)"
                )
                self._fts_available = True
            except Exception:
                self._fts_available = False
            if self._fts_available:
                cur = await db.execute("SELECT count(*) FROM messages_fts")
                row = await cur.fetchone()
                if row and (row[0] or 0) == 0:
                    await db.execute(
                        "INSERT INTO messages_fts(content, conversation_id) "
                        "SELECT content, conversation_id FROM messages"
                    )
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

    async def list_conversations_by_kb(self, kb_name: str) -> list[Conversation]:
        """Return all conversations associated with a KB, newest first.

        Each returned :class:`Conversation` has its ``messages`` populated.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT id, session_id, title, kb_name, created_at, updated_at "
                "FROM conversations WHERE kb_name = ? ORDER BY updated_at DESC",
                (kb_name,),
            )

        convs: list[Conversation] = []
        for r in rows:
            messages = await self.get_messages(r["id"])
            convs.append(
                Conversation(
                    id=r["id"],
                    title=r["title"] or "",
                    kb_name=r["kb_name"] or kb_name,
                    messages=messages,
                )
            )
        return convs

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

            # Keep FTS index in sync
            if self._fts_available:
                with contextlib.suppress(Exception):
                    await db.execute(
                        "INSERT INTO messages_fts(content, conversation_id) VALUES (?, ?)",
                        (message.content, conv_id),
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

    async def search_conversations(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search over message content.

        Returns matching conversations with a snippet.
        Uses FTS5 if available, else falls back to a LIKE scan.
        """
        if not query or not query.strip():
            return []
        q = query.strip()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            try:
                if self._fts_available:
                    cur = await db.execute(
                        "SELECT conversation_id, "
                        "snippet(messages_fts, 0, '[', ']', '…', 12) AS snippet "
                        "FROM messages_fts WHERE messages_fts MATCH ? LIMIT ?",
                        (q, limit * 4),
                    )
                else:
                    raise RuntimeError("fts unavailable")
                rows = await cur.fetchall()
            except Exception:
                like = f"%{q}%"
                cur = await db.execute(
                    "SELECT conversation_id, substr(content, 1, 200) AS snippet "
                    "FROM messages WHERE content LIKE ? LIMIT ?",
                    (like, limit * 4),
                )
                rows = await cur.fetchall()
            seen: set[str] = set()
            out: list[dict[str, Any]] = []
            for r in rows:
                cid = r["conversation_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                c2 = await db.execute("SELECT title FROM conversations WHERE id = ?", (cid,))
                trow = await c2.fetchone()
                out.append(
                    {
                        "id": cid,
                        "title": trow["title"] if trow else None,
                        "snippet": r["snippet"],
                    }
                )
                if len(out) >= limit:
                    break
            return out

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

            rows = await db.execute_fetchall("SELECT * FROM kb_metadata ORDER BY updated_at DESC")

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

    async def store_paper_reference(
        self,
        kb_name: str,
        doi: str | None,
        title: str,
        authors: list[str],
        year: int | None,
        abstract: str | None,
        survey_query: str | None = None,
    ) -> bool:
        """Store a paper reference in the kb_paper_references table.

        Returns True if the row was inserted (new), False if it was a duplicate
        (same kb_name + doi already exists). Papers without a DOI are stored but
        do NOT participate in the UNIQUE constraint — callers that want to avoid
        duplicates for DOI-less papers must check before calling.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "INSERT OR IGNORE INTO kb_paper_references "
                "(kb_name, doi, title, authors_json, year, abstract, survey_query) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (kb_name, doi, title, json.dumps(authors), year, abstract, survey_query),
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def get_paper_references(self, kb_name: str) -> list[dict[str, Any]]:
        """Return all paper references stored for a KB, newest first.

        Each dict contains all columns plus an ``authors`` key (list[str]) decoded
        from ``authors_json``.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM kb_paper_references WHERE kb_name = ? ORDER BY created_at DESC",
                (kb_name,),
            )
            rows = await cur.fetchall()
        return [
            {**dict(r), "authors": json.loads(r["authors_json"] or "[]")}
            for r in rows
        ]

    async def delete_paper_references(self, kb_name: str) -> int:
        """Delete all paper reference rows for a KB.

        Returns the number of rows deleted.  Called automatically by
        ``delete_kb_metadata`` so that KB teardown leaves no orphaned rows.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "DELETE FROM kb_paper_references WHERE kb_name = ?", (kb_name,)
            )
            await db.commit()
            return cur.rowcount or 0

    async def delete_kb_metadata(self, name: str) -> bool:
        """Delete a KB row from ``kb_metadata``.

        Doesn't touch the underlying Chroma collection — the caller
        is responsible for ``vector_store.delete_collection`` if they
        also want the embeddings gone. Returns True when a row was
        deleted, False when no such KB existed.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "DELETE FROM kb_metadata WHERE name = ?", (name,)
            )
            await db.execute(
                "DELETE FROM kb_paper_references WHERE kb_name = ?", (name,)
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    async def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation and all its messages.

        Does NOT touch provenance: pair this with
        ``ProvenanceStore.purge_conversation`` or the verbatim prompt sidecar
        survives the deletion.

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

            # Purge FTS index rows for this conversation before deleting
            if getattr(self, "_fts_available", False):
                with contextlib.suppress(Exception):
                    await db.execute(
                        "DELETE FROM messages_fts WHERE conversation_id = ?",
                        (conv_id,),
                    )

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
            row = await db.execute_fetchall("SELECT COUNT(*) as count FROM conversations")
            count = row[0]["count"] if row else 0

            # Purge the entire FTS index
            if getattr(self, "_fts_available", False):
                with contextlib.suppress(Exception):
                    await db.execute("DELETE FROM messages_fts")

            # Delete all conversations (messages will be cascade deleted)
            await db.execute("DELETE FROM conversations")
            await db.commit()

        logger.info("all_conversations_deleted", count=count)
        return count
