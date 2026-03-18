"""Protocol definitions for retrieval components."""

from typing import Any, Protocol

from perspicacite.models.documents import DocumentChunk
from perspicacite.models.search import RetrievedChunk, SearchFilters


class VectorStore(Protocol):
    """Protocol for vector store implementations."""

    async def create_collection(self, name: str, embedding_dim: int) -> None:
        """Create a new collection."""
        ...

    async def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        ...

    async def list_collections(self) -> list[str]:
        """List all collections."""
        ...

    async def add_documents(
        self,
        collection: str,
        chunks: list[DocumentChunk],
    ) -> int:
        """Add documents to a collection."""
        ...

    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Search for similar documents."""
        ...

    async def get_collection_stats(self, collection: str) -> dict[str, Any]:
        """Get collection statistics."""
        ...

    async def delete_documents(self, collection: str, ids: list[str]) -> int:
        """Delete documents by ID."""
        ...
