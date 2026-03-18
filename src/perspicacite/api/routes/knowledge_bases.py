"""Knowledge base endpoints."""

from fastapi import APIRouter, HTTPException

from perspicacite.models.api import KBCreateRequest, KBAddPapersRequest
from perspicacite.models.kb import KnowledgeBase, KBStats

router = APIRouter()


@router.get("/kb")
async def list_knowledge_bases() -> list[KBStats]:
    """List all knowledge bases."""
    # Placeholder
    return [
        KBStats(
            name="default",
            description="Default knowledge base",
            paper_count=0,
            chunk_count=0,
            embedding_model="text-embedding-3-small",
        )
    ]


@router.post("/kb")
async def create_knowledge_base(request: KBCreateRequest) -> KnowledgeBase:
    """Create a new knowledge base."""
    # Placeholder
    return KnowledgeBase(
        name=request.name,
        description=request.description,
        collection_name=request.name,
        embedding_model=request.embedding_model,
        chunk_config=request.chunk_config,
    )


@router.get("/kb/{name}")
async def get_knowledge_base(name: str) -> KBStats:
    """Get knowledge base details."""
    return KBStats(
        name=name,
        paper_count=0,
        chunk_count=0,
        embedding_model="text-embedding-3-small",
    )


@router.delete("/kb/{name}")
async def delete_knowledge_base(name: str):
    """Delete a knowledge base."""
    return {"deleted": name}


@router.post("/kb/{name}/papers")
async def add_papers(name: str, request: KBAddPapersRequest):
    """Add papers to a knowledge base."""
    return {"added": len(request.papers), "kb": name}
