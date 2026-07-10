"""ChromaDB vector store implementation."""

from typing import Any

import chromadb

# Note: IncludeEnum was removed in ChromaDB 0.6.0+, use Include type instead
try:
    from chromadb.api.types import IncludeEnum
except ImportError:
    IncludeEnum = None  # Will use literal values instead

from perspicacite.llm.embeddings import EmbeddingFailedError, EmbeddingProvider, is_zero_vector
from perspicacite.logging import get_logger
from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.search import RetrievedChunk, SearchFilters

logger = get_logger("perspicacite.retrieval.chroma")


def _reject_degenerate_embeddings(
    texts: list[str], embeddings: list[list[float]], collection: str
) -> None:
    """Refuse embeddings that would poison the collection.

    Chroma accepts a zero vector and then reports cosine distance 1.0 against
    everything, so a poisoned collection answers every query with the same
    passages at the same score. No zero vector may be stored, whatever produced
    it: for a text with content it means the embedder failed, and for an empty
    text it means the chunk has nothing to retrieve.
    """
    if len(embeddings) != len(texts):
        raise EmbeddingFailedError(
            f"embedding provider returned {len(embeddings)} vector(s) for "
            f"{len(texts)} text(s) in collection {collection!r}; refusing to "
            "store misaligned embeddings"
        )
    degenerate = [
        index
        for index, vector in enumerate(embeddings)
        if is_zero_vector(vector)
    ]
    if not degenerate:
        return
    first = degenerate[0]
    reason = (
        "the embedding provider returned a zero vector"
        if texts[first].strip()
        else "the chunk has no text to embed"
    )
    raise EmbeddingFailedError(
        f"refusing to store {len(degenerate)} zero vector(s) in collection "
        f"{collection!r} (first at index {first}): {reason}"
    )


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
        *,
        per_chunk_context: list[str] | None = None,
    ) -> int:
        """
        Add documents to a collection.

        Args:
            collection: Collection name
            chunks: Document chunks to add
            per_chunk_context: Optional list of LLM-generated context
                strings (one per chunk, same order). When supplied, each
                string is prepended to the corresponding chunk's
                embedding text only (chunk.text is not modified). This is
                the integration point for Anthropic-style contextual
                retrieval — the caller generates contexts via
                ``retrieval.contextual.generate_chunk_contexts_bulk``
                and passes the result here. ``None`` or empty entries
                are treated as no-extra-context.

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

        # Generate embeddings for chunks that don't have them.
        # We prefix the chunk text with title + section/heading_path
        # (always, free) plus an optional LLM-generated context
        # (Anthropic-style contextual retrieval, when caller supplies
        # ``per_chunk_context``). Stored ``documents`` and downstream
        # ``chunk.text`` are unchanged so display/synthesis aren't
        # affected — only the embedding sees the prefixes.
        texts_to_embed = []
        indices_to_embed = []

        def _ctx_for(i: int) -> str | None:
            if not per_chunk_context:
                return None
            if i >= len(per_chunk_context):
                return None
            return per_chunk_context[i] or None

        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                texts_to_embed.append(
                    _compose_embedding_text(chunk, extra_context=_ctx_for(i))
                )
                indices_to_embed.append(i)

        if texts_to_embed:
            logger.debug(
                "generating_embeddings",
                count=len(texts_to_embed),
                collection=collection,
            )
            try:
                embeddings = await self.embedding_provider.embed(texts_to_embed)
            except Exception as exc:
                raise EmbeddingFailedError(
                    f"embedding {len(texts_to_embed)} chunk(s) for collection "
                    f"{collection!r} failed: {exc}"
                ) from exc
            _reject_degenerate_embeddings(texts_to_embed, embeddings, collection)
            for idx, embedding in zip(indices_to_embed, embeddings, strict=True):
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

        Raises:
            EmbeddingFailedError: the query vector has zero norm, which Chroma
                would answer with distance 1.0 for every document — a uniform
                score over an arbitrary ordering rather than a ranking.
        """
        if is_zero_vector(query_embedding):
            raise EmbeddingFailedError(
                f"refusing to search collection {collection!r} with a zero-norm query "
                "embedding: every document would score identically. The embedding "
                "provider most likely failed (check the API key and its quota)."
            )

        try:
            coll = self.client.get_collection(name=collection)
        except Exception as e:
            # Benign on first-run / no-KB queries — callers fall back to web
            # search. Emit at WARNING so it doesn't pollute error dashboards.
            logger.warning(
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

                    # Guard: some legacy collections (e.g. QADS-extracted embeddings) store
                    # None metadata. Reconstruct a minimal metadata dict from the doc_id
                    # so paper_id is available for downstream Source construction.
                    # doc_id format: "scifact:12345678_metadata" → paper_id "scifact:12345678"
                    if metadata is None:
                        paper_id_from_id = doc_id.removesuffix("_metadata") if "_metadata" in doc_id else doc_id
                        metadata = {"paper_id": paper_id_from_id}

                    chunk = DocumentChunk(
                        id=doc_id,
                        text=document or "",
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
            # Count unique papers from chunk metadata
            result = coll.get(include=["metadatas"])
            unique_papers = len({m["paper_id"] for m in result["metadatas"]}) if result["metadatas"] else 0
            return {
                "name": collection,
                "count": count,
                "unique_papers": unique_papers,
            }
        except Exception:
            return {"name": collection, "count": 0}

    async def paper_exists(self, collection: str, paper_id: str) -> bool:
        """Check if a paper already exists in the collection by paper_id."""
        try:
            coll = self.client.get_collection(name=collection)
            # Use get() with metadata filter instead of query() to avoid
            # ChromaDB's default embedding function (384-dim all-MiniLM-L6-v2)
            # which conflicts with the OpenAI embeddings used in the collection.
            results = coll.get(
                where={"paper_id": paper_id},
                include=[],
                limit=1,
            )
            return bool(results["ids"])
        except Exception as e:
            logger.error(
                "paper_exists_check_failed",
                collection=collection,
                paper_id=paper_id,
                error=str(e),
            )
            return False

    async def peek_paper_metadata_row(
        self, collection: str, paper_id: str
    ) -> dict[str, Any] | None:
        """Return one raw metadata dict for a paper_id (any chunk), or None."""
        try:
            coll = self.client.get_collection(name=collection)
            r = coll.get(
                where={"paper_id": paper_id},
                limit=1,
                include=["metadatas"],
            )
            metas = r.get("metadatas") or []
            return metas[0] if metas else None
        except Exception as e:
            logger.error(
                "peek_paper_metadata_failed",
                collection=collection,
                paper_id=paper_id,
                error=str(e),
            )
            return None

    async def list_paper_metadata(self, collection: str) -> list[dict[str, Any]]:
        """One merged row per ``paper_id`` for title/DOI resolution (RAG scope)."""
        try:
            coll = self.client.get_collection(name=collection)
            result = coll.get(include=["metadatas"])
        except Exception as e:
            logger.error(
                "list_paper_metadata_failed",
                collection=collection,
                error=str(e),
            )
            return []

        metas = result.get("metadatas") or []
        by_pid: dict[str, dict[str, Any]] = {}
        for m in metas:
            if not m:
                continue
            pid = m.get("paper_id")
            if not pid:
                continue
            cur = by_pid.get(pid)
            if cur is None:
                by_pid[pid] = {
                    "paper_id": pid,
                    "title": m.get("title"),
                    "authors": m.get("authors"),
                    "year": m.get("year"),
                    "doi": m.get("doi"),
                    "abstract": m.get("abstract"),
                }
            else:
                for k in ("title", "authors", "doi", "abstract"):
                    if m.get(k) and not cur.get(k):
                        cur[k] = m[k]
                if m.get("year") is not None and cur.get("year") is None:
                    cur["year"] = m["year"]
        return list(by_pid.values())

    async def list_chunk_texts(self, collection: str, limit: int = 2000) -> list[str]:
        """Return up to ``limit`` chunk documents from a collection.

        Fallback lexical (BM25) reference corpus for similarity screening when
        a KB has no stored abstracts. Empty docs dropped; missing collection
        / errors -> empty list.
        """
        try:
            coll = self.client.get_collection(name=collection)
            got = coll.get(limit=limit, include=["documents"])
        except Exception as e:
            logger.warning("list_chunk_texts_failed", collection=collection, error=str(e))
            return []
        return [d for d in (got.get("documents") or []) if d]

    async def list_paper_chunks(
        self, collection: str, max_per_paper: int = 20
    ) -> dict[str, list[str]]:
        """Per-``paper_id`` list of chunk texts (at most ``max_per_paper`` each).

        Used to represent papers that have no stored abstract: their chunk texts
        become the per-paper reference list for similarity scoring.
        """
        try:
            coll = self.client.get_collection(name=collection)
            result = coll.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error("list_paper_chunks_failed", collection=collection, error=str(e))
            return {}
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        by_paper: dict[str, list[str]] = {}
        for doc, meta in zip(docs, metas, strict=False):
            if not doc:
                continue
            pid = (meta or {}).get("paper_id") or ""
            if not pid:
                continue
            bucket = by_paper.setdefault(pid, [])
            if len(bucket) < max_per_paper:
                bucket.append(doc)
        return by_paper

    async def list_paper_ids_in_collection(
        self, collection_name: str
    ) -> list[tuple[str, str, int]]:
        """Return distinct ``(paper_id, title, chunk_count)`` for a collection.

        Backs the ``perspicacite://kb/{name}/papers`` MCP resource when the
        per-KB event log (Wave 4.3) is missing or empty (older KBs).
        Returns ``[]`` if the collection doesn't exist or none of the
        chunks carry a ``paper_id`` metadata key.
        """
        try:
            coll = self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.warning(
                "list_paper_ids_collection_missing",
                collection=collection_name,
                error=str(e),
            )
            return []
        try:
            data = coll.get(include=["metadatas"])
        except Exception as e:
            logger.error(
                "list_paper_ids_get_failed",
                collection=collection_name,
                error=str(e),
            )
            return []
        counts: dict[str, dict[str, Any]] = {}
        for meta in data.get("metadatas") or []:
            if not meta:
                continue
            pid = meta.get("paper_id")
            if not pid:
                continue
            entry = counts.setdefault(pid, {"title": meta.get("title", "") or "", "n": 0})
            entry["n"] += 1
            # Backfill title from any later chunk that has one.
            if not entry["title"] and meta.get("title"):
                entry["title"] = meta["title"]
        return [(pid, info["title"], info["n"]) for pid, info in counts.items()]

    async def get_chunks_by_paper_ids(
        self,
        collection: str,
        paper_ids: list[str],
    ) -> list[DocumentChunk]:
        """Fetch all chunks for given paper IDs using metadata filter.

        Uses coll.get() with ``$in`` filter — no embedding needed.
        Results are sorted by (paper_id, chunk_index) for contiguous reading.
        """
        if not paper_ids:
            return []
        try:
            coll = self.client.get_collection(name=collection)
            all_chunks: list[DocumentChunk] = []
            batch_size = 400
            for i in range(0, len(paper_ids), batch_size):
                batch = paper_ids[i : i + batch_size]
                result = coll.get(
                    where={"paper_id": {"$in": batch}},
                    include=["documents", "metadatas"],
                )
                for j, doc in enumerate(result["documents"]):
                    meta = result["metadatas"][j] if result["metadatas"] else {}
                    chunk = DocumentChunk(
                        id=result["ids"][j],
                        text=doc or "",
                        metadata=_metadata_to_chunk(meta),
                    )
                    all_chunks.append(chunk)
            all_chunks.sort(
                key=lambda c: (c.metadata.paper_id or "", c.metadata.chunk_index or 0)
            )
            logger.info(
                "get_chunks_by_paper_ids",
                collection=collection,
                paper_count=len(paper_ids),
                chunk_count=len(all_chunks),
            )
            return all_chunks
        except Exception as e:
            logger.error(
                "get_chunks_by_paper_ids_failed",
                collection=collection,
                error=str(e),
            )
            return []

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


def _compose_embedding_text(
    chunk: DocumentChunk, *, extra_context: str | None = None,
) -> str:
    """Build the text that goes into the embedding model for a chunk.

    Two stacked prefixes, both embedding-only (stored chunk.text and
    Chroma's `documents` field are unchanged):

    1. **Structural** — paper title, section, heading_path. Free, no
       LLM calls.
    2. **LLM-generated context** — Anthropic-style contextual retrieval,
       passed via ``extra_context``. Caller is responsible for
       generating it (see ``retrieval.contextual.generate_chunk_context``).

    Layout:
        [<extra_context (LLM-generated)>]

        [<title>] · [<section> | <heading_path>] · [<source_section>]
        <chunk.text>

    Empty/missing fields are skipped. Each prefix is capped so it
    can't drown short chunks in metadata.
    """
    md = chunk.metadata
    parts: list[str] = []
    if md.title:
        parts.append(md.title.strip())
    # Prefer explicit `section`; fall back to joined `heading_path`.
    section = md.section
    if not section and md.heading_path:
        section = " > ".join(str(h) for h in md.heading_path if h)
    if section:
        parts.append(section.strip())
    if md.source_section and md.source_section != section:
        parts.append(str(md.source_section).strip())
    structural = " · ".join(parts) if parts else ""
    if len(structural) > 280:
        structural = structural[:277] + "..."

    extra = (extra_context or "").strip()
    # Cap LLM prefix separately at 500 chars (slightly above the
    # configured contextual_retrieval_max_chars default of 400).
    if len(extra) > 500:
        extra = extra[:497] + "..."

    if not extra and not structural:
        return chunk.text
    blocks: list[str] = []
    if extra:
        blocks.append(extra)
    if structural:
        blocks.append(structural)
    blocks.append(chunk.text)
    return "\n\n".join(blocks)


def _chunk_to_metadata(metadata: ChunkMetadata) -> dict[str, Any]:
    """Convert ChunkMetadata to Chroma metadata dict.

    ChromaDB only accepts simple types: str, int, float, bool.
    None values are not allowed. List values are stored as JSON strings
    under the ``<field>_json`` key so we can round-trip them via
    ``_metadata_to_chunk``.
    """
    result: dict[str, Any] = {
        "paper_id": metadata.paper_id,
        "chunk_index": metadata.chunk_index,
        "source": metadata.source.value if metadata.source else "bibtex",
    }

    # Scalar identity / metadata fields
    scalar_fields = (
        "section", "page_number", "title", "authors", "year", "doi", "url",
        "abstract", "license_id",
        "content_type", "language", "source_file_path",
        "source_section", "page", "parent_paper_id",
        "symbol_name", "symbol_kind", "parent_class",
        "start_line", "end_line", "docstring",
        "embedding_model", "source_via", "cited_tool", "discovery_score",
    )
    for field in scalar_fields:
        val = getattr(metadata, field, None)
        if val is not None:
            result[field] = val

    # Bool fields (False is meaningful, not "missing")
    result["is_external"] = bool(getattr(metadata, "is_external", False))

    # Tuple-encoded char span -> "start,end" string (Chroma can't store tuples)
    cs = getattr(metadata, "char_span", None)
    if cs is not None and len(cs) == 2:
        result["char_span"] = f"{int(cs[0])},{int(cs[1])}"

    # List fields -> JSON string ("[]" omitted)
    import json as _json
    for field in ("heading_path", "figure_refs", "table_refs", "resource_refs", "imports"):
        val = getattr(metadata, field, None)
        if val:
            result[f"{field}_json"] = _json.dumps(list(val))

    # ASB paper metadata JSON (round-trip through chroma)
    if getattr(metadata, "paper_metadata_json", None) is not None:
        result["paper_metadata_json"] = metadata.paper_metadata_json

    return result


def _metadata_to_chunk(metadata: dict[str, Any]) -> ChunkMetadata:
    """Convert Chroma metadata dict to ChunkMetadata."""
    import json as _json

    from perspicacite.models.papers import PaperSource

    def _list(key: str) -> list[Any]:
        raw = metadata.get(f"{key}_json")
        if not raw:
            return []
        try:
            parsed = _json.loads(raw)
            return list(parsed) if isinstance(parsed, list) else []
        except Exception:
            return []

    cs_raw = metadata.get("char_span")
    char_span: tuple[int, int] | None = None
    if isinstance(cs_raw, str) and "," in cs_raw:
        try:
            a, b = cs_raw.split(",", 1)
            char_span = (int(a), int(b))
        except Exception:
            char_span = None

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
        abstract=metadata.get("abstract"),
        license_id=metadata.get("license_id"),
        content_type=metadata.get("content_type"),
        language=metadata.get("language"),
        heading_path=_list("heading_path") or None,
        source_file_path=metadata.get("source_file_path"),
        source_section=metadata.get("source_section"),
        page=metadata.get("page"),
        char_span=char_span,
        figure_refs=_list("figure_refs"),
        table_refs=_list("table_refs"),
        resource_refs=_list("resource_refs"),
        parent_paper_id=metadata.get("parent_paper_id"),
        is_external=bool(metadata.get("is_external", False)),
        symbol_name=metadata.get("symbol_name"),
        symbol_kind=metadata.get("symbol_kind"),
        parent_class=metadata.get("parent_class"),
        start_line=metadata.get("start_line"),
        end_line=metadata.get("end_line"),
        docstring=metadata.get("docstring"),
        imports=_list("imports"),
        embedding_model=metadata.get("embedding_model"),
        source_via=metadata.get("source_via"),
        cited_tool=metadata.get("cited_tool"),
        discovery_score=metadata.get("discovery_score"),
        paper_metadata_json=metadata.get("paper_metadata_json"),
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
    if filters.source_skill is not None:
        # 2026-05-15: composite skill-bundle KBs stamp each chunk with
        # `source_skill=<bundle.yml:name>` so queries can filter to one
        # skill inside a composite KB.
        conditions.append({"source_skill": filters.source_skill})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}
