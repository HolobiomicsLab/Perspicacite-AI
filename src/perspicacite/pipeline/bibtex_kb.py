"""Import BibTeX files into a persisted Chroma knowledge base."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import bibtexparser
import httpx

from perspicacite.logging import get_logger
from perspicacite.models.kb import ChunkConfig, KnowledgeBase, chroma_collection_name_for_kb
from perspicacite.models.papers import Paper
from perspicacite.pipeline.download import get_pdf_with_fallback
from perspicacite.pipeline.parsers.pdf import PDFParser
from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig

logger = get_logger("perspicacite.pipeline.bibtex_kb")

ALLOWED_ENTRY_TYPES = frozenset(
    {
        "article",
        "inproceedings",
        "misc",
        "book",
        "inbook",
        "phdthesis",
        "mastersthesis",
        "techreport",
    }
)


def sanitize_kb_display_name(name: str) -> str:
    """Match KnowledgeBase name pattern ^[a-zA-Z0-9_-]+$."""
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("Knowledge base name is empty after sanitization")
    return s


def normalize_bibtex_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().strip("{}")
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/"):
        if d.lower().startswith(prefix.lower()):
            d = d[len(prefix) :].strip()
    if d.lower().startswith("doi:"):
        d = d[4:].strip()
    return d or None


def bibtexparser_entry_to_paper_dict(entry: dict[str, Any]) -> dict[str, Any]:
    """Map a bibtexparser entry to the flat dict expected by Paper.from_bibtex."""
    flat: dict[str, Any] = {}
    for k, v in entry.items():
        if k in ("ID", "ENTRYTYPE"):
            continue
        if v is None:
            continue
        flat[k.lower()] = v if isinstance(v, str) else str(v)

    if "journaltitle" in flat and "journal" not in flat:
        flat["journal"] = flat["journaltitle"]
    if "date" in flat and "year" not in flat:
        y = flat["date"][:4]
        if y.isdigit():
            flat["year"] = y
    if doi := flat.get("doi"):
        flat["doi"] = normalize_bibtex_doi(doi)
    return flat


def load_bibtex_entries(path: Path) -> list[dict[str, Any]]:
    """Parse a .bib file and return raw bibtexparser entries."""
    text = path.read_text(encoding="utf-8", errors="replace")
    db = bibtexparser.loads(text)
    return db.entries


def entries_to_papers(entries: list[dict[str, Any]]) -> list[Paper]:
    """Convert parsed BibTeX entries to Paper models (metadata only)."""
    papers: list[Paper] = []
    for e in entries:
        et = (e.get("ENTRYTYPE") or "").lower()
        if et not in ALLOWED_ENTRY_TYPES:
            continue
        flat = bibtexparser_entry_to_paper_dict(e)
        title = (flat.get("title") or "").strip()
        if not title:
            logger.warning("bibtex_skip_no_title", entry_key=e.get("ID"))
            continue
        try:
            papers.append(Paper.from_bibtex(flat))
        except Exception as ex:
            logger.warning(
                "bibtex_skip_entry",
                entry_key=e.get("ID"),
                error=str(ex),
            )
    return papers


async def enrich_papers_with_pdf(
    papers: list[Paper],
    *,
    http_client: httpx.AsyncClient,
    pdf_parser: PDFParser,
    unpaywall_email: str | None,
    alternative_endpoint: str | None = None,
    wiley_tdm_token: str | None = None,
    aaas_api_key: str | None = None,
    rsc_api_key: str | None = None,
    springer_api_key: str | None = None,
) -> dict[str, int]:
    """Download PDFs and set paper.full_text where possible."""
    stats = {"attempted": 0, "success": 0, "failed": 0, "skipped_no_doi": 0}
    for paper in papers:
        if not paper.doi:
            stats["skipped_no_doi"] += 1
            continue
        stats["attempted"] += 1
        try:
            pdf_bytes = await get_pdf_with_fallback(
                paper.doi,
                url=paper.url,
                alternative_endpoint=alternative_endpoint,
                http_client=http_client,
                unpaywall_email=unpaywall_email,
                wiley_tdm_token=wiley_tdm_token,
                aaas_api_key=aaas_api_key,
                rsc_api_key=rsc_api_key,
                springer_api_key=springer_api_key,
            )
            if pdf_bytes and len(pdf_bytes) > 1000:
                if pdf_bytes[:4] == b"%PDF":
                    parsed = await pdf_parser.parse(pdf_bytes)
                    text = parsed.text if parsed else None
                else:
                    text = pdf_bytes.decode("utf-8", errors="replace")
                if text and len(text.strip()) > 200:
                    paper.full_text = text
                    stats["success"] += 1
                    continue
            stats["failed"] += 1
        except Exception as ex:
            logger.warning("bibtex_pdf_failed", doi=paper.doi, error=str(ex))
            stats["failed"] += 1
    return stats


async def create_kb_from_bibtex(
    config: Any,
    *,
    kb_name: str,
    bib_path: Path,
    description: str | None,
    session_db: Path,
    chroma_dir: Path,
) -> dict[str, Any]:
    """
    Create a new KB, parse BibTeX, download PDFs, chunk+embed into Chroma, save metadata.

    Raises FileExistsError if a KB with the same sanitized name already exists.
    """
    from perspicacite.config.schema import Config
    from perspicacite.llm import LiteLLMEmbeddingProvider
    from perspicacite.memory.session_store import SessionStore
    from perspicacite.retrieval.chroma_store import ChromaVectorStore

    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")

    safe_name = sanitize_kb_display_name(kb_name)
    collection_name = chroma_collection_name_for_kb(safe_name)

    session_store = SessionStore(session_db)
    await session_store.init_db()

    if await session_store.get_kb_metadata(safe_name):
        raise FileExistsError(
            f"Knowledge base '{safe_name}' already exists. Delete it in the UI or DB first."
        )

    entries = load_bibtex_entries(bib_path)
    papers = entries_to_papers(entries)
    if not papers:
        raise ValueError(
            f"No importable entries in {bib_path} (need title + supported @entry types)."
        )

    embedding_provider = LiteLLMEmbeddingProvider(
        model=config.knowledge_base.embedding_model,
    )
    chroma_dir.mkdir(parents=True, exist_ok=True)
    session_db.parent.mkdir(parents=True, exist_ok=True)

    vector_store = ChromaVectorStore(
        persist_dir=str(chroma_dir.expanduser().resolve()),
        embedding_provider=embedding_provider,
    )

    await vector_store.create_collection(collection_name)

    kb = KnowledgeBase(
        name=safe_name,
        description=description or f"Imported from {bib_path.name}",
        collection_name=collection_name,
        embedding_model=embedding_provider.model_name,
        chunk_config=ChunkConfig(
            chunk_size=config.knowledge_base.chunk_size,
            chunk_overlap=config.knowledge_base.chunk_overlap,
        ),
    )
    await session_store.save_kb_metadata(kb)

    pdf_cfg = config.pdf_download
    pdf_parser = PDFParser()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http_client:
        pdf_stats = await enrich_papers_with_pdf(
            papers,
            http_client=http_client,
            pdf_parser=pdf_parser,
            unpaywall_email=pdf_cfg.unpaywall_email,
            alternative_endpoint=pdf_cfg.alternative_endpoint,
            wiley_tdm_token=pdf_cfg.wiley_tdm_token,
            aaas_api_key=pdf_cfg.aaas_api_key,
            rsc_api_key=pdf_cfg.rsc_api_key,
            springer_api_key=pdf_cfg.springer_api_key,
        )

    dkb_config = KnowledgeBaseConfig(
        vector_size=embedding_provider.dimension,
        chunk_size=config.knowledge_base.chunk_size,
        chunk_overlap=config.knowledge_base.chunk_overlap,
    )
    dkb = DynamicKnowledgeBase(
        vector_store,
        embedding_provider,
        config=dkb_config,
    )
    dkb.collection_name = collection_name
    dkb._initialized = True

    try:
        chunks_added = await dkb.add_papers(papers, include_full_text=True)
    except Exception:
        logger.exception("bibtex_kb_embed_failed", collection=collection_name)
        try:
            await vector_store.delete_collection(collection_name)
        except Exception:
            pass
        import aiosqlite

        async with aiosqlite.connect(session_store.db_path) as db:
            await db.execute("DELETE FROM kb_metadata WHERE name = ?", (safe_name,))
            await db.commit()
        raise

    kb.paper_count = len(papers)
    kb.chunk_count = chunks_added
    await session_store.save_kb_metadata(kb)

    logger.info(
        "bibtex_kb_created",
        name=safe_name,
        collection=collection_name,
        papers=len(papers),
        chunks=chunks_added,
        pdf=pdf_stats,
    )

    return {
        "name": safe_name,
        "collection_name": collection_name,
        "papers": len(papers),
        "chunks_added": chunks_added,
        "pdf_stats": pdf_stats,
        "bib_path": str(bib_path),
    }
