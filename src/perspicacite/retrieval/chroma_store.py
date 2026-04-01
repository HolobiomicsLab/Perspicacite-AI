"""ChromaDB vector store implementation."""

from typing import Any

import chromadb
# Note: IncludeEnum was removed in ChromaDB 0.6.0+, use Include type instead
try:
    from chromadb.api.types import IncludeEnum
except ImportError:
    from chromadb.api.types import Include
    IncludeEnum = None  # Will use literal values instead

from perspicacite.llm.embeddings import EmbeddingProvider
from perspicacite.logging import get_logger
from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.search import RetrievedChunk, SearchFilters

logger = get_logger("perspicacite.retrieval.chroma")


class ChromaVectorStore:
    """
    ChromaDB-backed vector store.

    Uses PersistentClient for disk-backed storage.
    Supports metadata filtering and hybrid search.
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_provider: EmbeddingProvider,
    ):
        """
        Initialize Chroma vector store.

        Args:
            persist_dir: Directory for persistent storage
            embedding_provider: Provider for generating embeddings
        """
        self.persist_dir = persist_dir
        self.embedding_provider = embedding_provider
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info(
            "chroma_store_initialized",
            persist_dir=persist_dir,
        )

    async def create_collection(
        self,
        name: str,
        embedding_dim: int | None = None,
    ) -> None:
        """
        Create a new collection.

        Args:
            name: Collection name
            embedding_dim: Embedding dimension (uses provider's dimension if None)
        """
        dim = embedding_dim or self.embedding_provider.dimension

        # Check if collection already exists
        try:
            existing = self.client.get_collection(name=name)
            if existing:
                logger.info("collection_already_exists", name=name)
                return
        except Exception:
            # Collection doesn't exist, proceed to create
            pass

        try:
            # Cosine space matches typical embedding APIs; L2 + (1-distance) breaks min_score filters.
            self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "dimension": dim},
            )
            logger.info(
                "collection_created",
                name=name,
                dimension=dim,
            )
        except Exception as e:
            # If already exists, just log and continue
            if "already exists" in str(e).lower():
                logger.info("collection_already_exists", name=name)
                return
            logger.error(
                "collection_create_failed",
                name=name,
                error=str(e),
            )
            raise

    async def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name=name)
            logger.info("collection_deleted", name=name)
        except Exception as e:
            logger.error(
                "collection_delete_failed",
                name=name,
                error=str(e),
            )
            raise

    async def list_collections(self) -> list[str]:
        """List all collections."""
        collections = self.client.list_collections()
        return [c.name for c in collections]

    async def add_documents(
        self,
        collection: str,
        chunks: list[DocumentChunk],
    ) -> int:
        """
        Add documents to a collection.

        Args:
            collection: Collection name
            chunks: Document chunks to add

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Get or create collection
        try:
            coll = self.client.get_collection(name=collection)
        except Exception:
            await self.create_collection(collection)
            coll = self.client.get_collection(name=collection)

        # Generate embeddings for chunks that don't have them
        texts_to_embed = []
        indices_to_embed = []

        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                texts_to_embed.append(chunk.text)
                indices_to_embed.append(i)

        if texts_to_embed:
            logger.debug(
                "generating_embeddings",
                count=len(texts_to_embed),
                collection=collection,
            )
            embeddings = await self.embedding_provider.embed(texts_to_embed)
            for idx, embedding in zip(indices_to_embed, embeddings):
                chunks[idx].embedding = embedding

        # Prepare data for Chroma (lists must stay aligned — never filter embeddings)
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        if any(e is None for e in embeddings):
            raise ValueError("All chunks must have embeddings before add_documents")
        metadatas = [_chunk_to_metadata(chunk.metadata) for chunk in chunks]

        # Add to Chroma
        try:
            coll.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(
                "documents_added",
                collection=collection,
                count=len(chunks),
            )
            return len(chunks)
        except Exception as e:
            logger.error(
                "add_documents_failed",
                collection=collection,
                error=str(e),
            )
            raise

    async def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search for similar documents.

        Args:
            collection: Collection name
            query_embedding: Query embedding vector
            top_k: Number of results
            filters: Optional metadata filters

        Returns:
            List of retrieved chunks with scores
        """
        try:
            coll = self.client.get_collection(name=collection)
        except Exception as e:
            logger.error(
                "collection_not_found",
                collection=collection,
                error=str(e),
            )
            return []

        # Convert filters to Chroma where clause
        where_clause = _filters_to_where(filters) if filters else None

        try:
            results = coll.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            # Convert to RetrievedChunk
            retrieved = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = float(results["distances"][0][i])
                    document = results["documents"][0][i]

                    # Chroma returns a distance (lower = more similar). Map to (0,1] so
                    # downstream min_score filters work for both cosine and legacy L2 collections.
                    score = 1.0 / (1.0 + max(0.0, distance))

                    chunk = DocumentChunk(
                        id=doc_id,
                        text=document,
                        metadata=_metadata_to_chunk(metadata),
                    )

                    retrieved.append(
                        RetrievedChunk(
                            chunk=chunk,
                            score=score,
                            retrieval_method="vector",
                        )
                    )

            logger.debug(
                "search_complete",
                collection=collection,
                query_hits=len(retrieved),
            )
            return retrieved

        except Exception as e:
            logger.error(
                "search_failed",
                collection=collection,
                error=str(e),
            )
            raise

    async def get_collection_stats(self, collection: str) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            coll = self.client.get_collection(name=collection)
            count = coll.count()
            return {
                "name": collection,
                "count": count,
            }
        except Exception:
            return {"name": collection, "count": 0}

    async def paper_exists(self, collection: str, paper_id: str) -> bool:
        """Check if a paper already exists in the collection by paper_id."""
        try:
            coll = self.client.get_collection(name=collection)
            # Query for any document with matching paper_id
            results = coll.query(
                query_texts=[""],  # Empty query
                n_results=1,
                where={"paper_id": paper_id},
                include=[],
            )
            # If any IDs returned, the paper exists
            return bool(results["ids"] and len(results["ids"][0]) > 0)
        except Exception as e:
            logger.error(
                "paper_exists_check_failed",
                collection=collection,
                paper_id=paper_id,
                error=str(e),
            )
            return False

    async def delete_documents(self, collection: str, ids: list[str]) -> int:
        """Delete documents by ID."""
        try:
            coll = self.client.get_collection(name=collection)
            coll.delete(ids=ids)
            logger.info(
                "documents_deleted",
                collection=collection,
                count=len(ids),
            )
            return len(ids)
        except Exception as e:
            logger.error(
                "delete_documents_failed",
                collection=collection,
                error=str(e),
            )
            raise


def _chunk_to_metadata(metadata: ChunkMetadata) -> dict[str, Any]:
    """Convert ChunkMetadata to Chroma metadata dict.
    
    ChromaDB only accepts simple types: str, int, float, bool.
    None values are not allowed.
    """
    result: dict[str, Any] = {
        "paper_id": metadata.paper_id,
        "chunk_index": metadata.chunk_index,
        "source": metadata.source.value if metadata.source else "bibtex",
    }
    
    # Only add non-None values
    if metadata.section is not None:
        result["section"] = metadata.section
    if metadata.page_number is not None:
        result["page_number"] = metadata.page_number
    if metadata.title is not None:
        result["title"] = metadata.title
    if metadata.authors is not None:
        result["authors"] = metadata.authors
    if metadata.year is not None:
        result["year"] = metadata.year
    if metadata.doi is not None:
        result["doi"] = metadata.doi
    if metadata.url is not None:
        result["url"] = metadata.url
        
    return result


def _metadata_to_chunk(metadata: dict[str, Any]) -> ChunkMetadata:
    """Convert Chroma metadata dict to ChunkMetadata."""
    from perspicacite.models.papers import PaperSource

    return ChunkMetadata(
        paper_id=metadata.get("paper_id", ""),
        chunk_index=metadata.get("chunk_index", 0),
        section=metadata.get("section"),
        page_number=metadata.get("page_number"),
        source=PaperSource(metadata.get("source", "bibtex")),
        title=metadata.get("title"),
        authors=metadata.get("authors"),
        year=metadata.get("year"),
        doi=metadata.get("doi"),
        url=metadata.get("url"),
    )


def _filters_to_where(filters: SearchFilters) -> dict[str, Any] | None:
    """Convert SearchFilters to Chroma where clause."""
    conditions = []

    if filters.year_min is not None:
        conditions.append({"year": {"$gte": filters.year_min}})
    if filters.year_max is not None:
        conditions.append({"year": {"$lte": filters.year_max}})
    if filters.authors:
        # Chroma doesn't support array contains directly
        # Use $in for each author
        conditions.append({"authors": {"$in": filters.authors}})
    if filters.journals:
        conditions.append({"journal": {"$in": filters.journals}})
    if filters.sources:
        conditions.append(
            {"source": {"$in": [s.value for s in filters.sources]}}
        )

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}
