"""Knowledge base building orchestrator."""

from pathlib import Path
from typing import Any

from perspicacite.llm.embeddings import EmbeddingProvider
from perspicacite.logging import get_logger
from perspicacite.models.documents import DocumentChunk
from perspicacite.models.kb import ChunkConfig, KnowledgeBase
from perspicacite.models.papers import Paper
from perspicacite.pipeline.chunking import chunk_text
from perspicacite.retrieval.chroma_store import ChromaVectorStore

logger = get_logger("perspicacite.pipeline.kb_builder")


class KBBuilder:
    """
    Orchestrates knowledge base creation.

    Pipeline: Source → Parse → Curate → Chunk → Embed → Store
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        chunk_config: ChunkConfig | None = None,
    ):
        """
        Initialize KB builder.

        Args:
            vector_store: Chroma vector store
            embedding_provider: For generating embeddings
            chunk_config: Chunking configuration
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.chunk_config = chunk_config or ChunkConfig()

    async def create_from_papers(
        self,
        name: str,
        papers: list[Paper],
        description: str | None = None,
    ) -> KnowledgeBase:
        """
        Create KB from list of papers.

        Args:
            name: KB name
            papers: List of papers with full_text
            description: Optional description

        Returns:
            Created knowledge base
        """
        logger.info(
            "kb_build_start",
            name=name,
            papers=len(papers),
        )

        # Create collection
        await self.vector_store.create_collection(
            name=name,
            embedding_dim=self.embedding_provider.dimension,
        )

        # Process papers
        total_chunks = 0
        for paper in papers:
            if not paper.full_text:
                logger.warning(
                    "paper_no_text",
                    paper_id=paper.id,
                    title=paper.title,
                )
                continue

            # Chunk paper
            chunks = await chunk_text(
                text=paper.full_text,
                paper=paper,
                config=self.chunk_config,
            )

            # Add to vector store
            if chunks:
                await self.vector_store.add_documents(name, chunks)
                total_chunks += len(chunks)

            logger.debug(
                "paper_processed",
                paper_id=paper.id,
                chunks=len(chunks),
            )

        # Create KB metadata
        kb = KnowledgeBase(
            name=name,
            description=description or f"Knowledge base with {len(papers)} papers",
            collection_name=name,
            embedding_model=self.embedding_provider.model_name,
            chunk_config=self.chunk_config,
            paper_count=len(papers),
            chunk_count=total_chunks,
        )

        logger.info(
            "kb_build_complete",
            name=name,
            papers=len(papers),
            chunks=total_chunks,
        )

        return kb

    async def add_papers(
        self,
        kb_name: str,
        papers: list[Paper],
    ) -> int:
        """
        Dynamically add papers to existing KB.

        Args:
            kb_name: Name of existing KB
            papers: Papers to add

        Returns:
            Number of chunks added
        """
        logger.info(
            "kb_add_papers",
            kb=kb_name,
            papers=len(papers),
        )

        total_chunks = 0
        for paper in papers:
            if not paper.full_text:
                continue

            chunks = await chunk_text(
                text=paper.full_text,
                paper=paper,
                config=self.chunk_config,
            )

            if chunks:
                await self.vector_store.add_documents(kb_name, chunks)
                total_chunks += len(chunks)

        logger.info(
            "kb_add_papers_complete",
            kb=kb_name,
            chunks_added=total_chunks,
        )

        return total_chunks

    async def add_from_urls(
        self,
        kb_name: str,
        urls: list[str],
        download_func: Any,
    ) -> int:
        """
        Download, parse, chunk, and add papers from URLs.

        Args:
            kb_name: KB name
            urls: URLs to process
            download_func: Async function to download URL content

        Returns:
            Number of chunks added
        """
        logger.info(
            "kb_add_urls",
            kb=kb_name,
            urls=len(urls),
        )

        papers = []
        for url in urls:
            try:
                content = await download_func(url)
                if content:
                    paper = Paper(
                        id=url,
                        title=f"Document from {url}",
                        url=url,
                        full_text=content,
                        source="web_search",
                    )
                    papers.append(paper)
            except Exception as e:
                logger.error(
                    "url_download_failed",
                    url=url,
                    error=str(e),
                )

        return await self.add_papers(kb_name, papers)
