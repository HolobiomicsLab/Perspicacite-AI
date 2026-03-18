"""Data models for Perspicacité v2."""

from perspicacite.models.papers import Paper, Author, PaperSource
from perspicacite.models.documents import DocumentChunk, ChunkMetadata
from perspicacite.models.kb import KnowledgeBase, ChunkConfig, KBStats
from perspicacite.models.search import SearchFilters, RetrievedChunk, SearchQuery
from perspicacite.models.rag import (
    RAGMode,
    SourceReference,
    RAGRequest,
    RAGResponse,
    StreamEvent,
)
from perspicacite.models.messages import Message, Conversation, Session

__all__ = [
    "Paper",
    "Author",
    "PaperSource",
    "DocumentChunk",
    "ChunkMetadata",
    "KnowledgeBase",
    "ChunkConfig",
    "KBStats",
    "SearchFilters",
    "RetrievedChunk",
    "SearchQuery",
    "RAGMode",
    "SourceReference",
    "RAGRequest",
    "RAGResponse",
    "StreamEvent",
    "Message",
    "Conversation",
    "Session",
]
