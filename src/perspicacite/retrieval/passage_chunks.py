"""Strict chunk retrieval shared by single and multi-KB passage searches.

Model checks establish configured/local provider identity, not remote serving
attestation or the provenance of historical KB vectors. Ambiguous metadata and
wrappers without query producer identity cannot establish compatibility here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from perspicacite.llm.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingFailedError,
    TypedEmbeddingProvider,
    is_zero_vector,
)


@dataclass(frozen=True)
class PassageScope:
    collection_name: str
    kb_name: str | None
    embedding_model: str | None


@dataclass(frozen=True)
class PassageQuery:
    text: str
    top_k: int
    min_score: float
    filters: Any = None


class PassageCollectionError(RuntimeError):
    """A passage query failed in one or more requested collections."""

    def __init__(self, collection_errors: list[dict[str, Any]]) -> None:
        self.collection_errors = collection_errors
        super().__init__("Passage retrieval failed in requested collections")


def _single_model(name: Any) -> str:
    if not isinstance(name, str) or not name.strip() or any(s in name for s in "|+"):
        raise ValueError("Passage retrieval requires an unambiguous embedding model identity")
    return name.strip()


def _check_provider(provider: Any, scopes: list[PassageScope]) -> str:
    if not scopes or any(not s.collection_name for s in scopes):
        raise ValueError("Passage retrieval requires named collections and embedding metadata")
    if isinstance(provider, (CachedEmbeddingProvider, TypedEmbeddingProvider)):
        raise ValueError(
            "This embedding provider wrapper has no reliable query producer identity; "
            "use an explicitly configured unwrapped provider for passage retrieval"
        )
    expected = {_single_model(s.embedding_model) for s in scopes}
    configured = _single_model(getattr(provider, "model_name", None))
    if expected != {configured}:
        raise ValueError("Query embedding model does not match the requested KB embedding model")
    return configured


async def _embed_query(provider: Any, text: str, expected: str) -> list[float]:
    vectors = await provider.embed_query([text])
    producer = _single_model(getattr(provider, "last_used_model", provider.model_name))
    if producer != expected or _single_model(provider.model_name) != expected:
        raise ValueError("Query embedding model changed during embedding; vector search refused")
    if not isinstance(vectors, (list, tuple)) or len(vectors) != 1:
        raise EmbeddingFailedError("Query embedding must return exactly one vector")
    vector = vectors[0]
    if not vector or len(vector) != provider.dimension:
        raise EmbeddingFailedError("Query embedding vector has an invalid dimension")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
        raise EmbeddingFailedError("Query embedding vector has nonfinite or nonnumeric components")
    if is_zero_vector(vector):
        raise EmbeddingFailedError("Query embedding vector has zero norm")
    return vector


def _chunk_record(result: Any, scope: PassageScope) -> dict[str, Any]:
    chunk = getattr(result, "chunk", None)
    chunk_id = getattr(chunk, "id", None)
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Passage result has no actual chunk identity")
    text = getattr(chunk, "text", None)
    if not isinstance(text, str):
        raise ValueError("Passage result has no chunk text")
    score = float(result.score)
    if not math.isfinite(score):
        raise ValueError("Passage result score must be finite")
    metadata = getattr(chunk, "metadata", None)
    paper_id = (
        metadata.get("paper_id")
        if isinstance(metadata, dict)
        else getattr(metadata, "paper_id", None)
    )
    return dict(
        chunk_id=chunk_id,
        text=text,
        score=score,
        paper_id=paper_id,
        metadata=metadata,
        kb_name=scope.kb_name,
        collection_name=scope.collection_name,
    )


def _rank_chunks(records: list[dict[str, Any]], query: PassageQuery) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["collection_name"], record["chunk_id"])
        previous = unique.get(key)
        if previous is not None and previous["text"] != record["text"]:
            raise ValueError("Conflicting text for the same collection and chunk identity")
        if previous is None or record["score"] > previous["score"]:
            unique[key] = record
    ranked = [r for r in unique.values() if r["score"] >= query.min_score]
    ranked.sort(key=lambda r: (-r["score"], r["collection_name"], r["chunk_id"]))
    return ranked[: query.top_k]


async def retrieve_passage_chunks(
    retriever: Any, scopes: list[PassageScope], query: PassageQuery
) -> list[dict[str, Any]]:
    """Query declared scopes once, returning ranked chunks or explicit failure.

    Stores must raise on failure. Stores with a legacy tolerant ``search`` expose
    ``search_strict`` for this boundary. No partial collection result is returned
    as complete success; embedding failures abort the whole query.
    """
    expected = _check_provider(retriever.embedding_service, scopes)
    vector = await _embed_query(retriever.embedding_service, query.text, expected)
    search = getattr(retriever.vector_store, "search_strict", retriever.vector_store.search)
    records, errors = [], []
    for scope in scopes:
        try:
            results = await search(
                collection=scope.collection_name,
                query_embedding=vector,
                top_k=query.top_k * 2,
                filters=query.filters,
            )
        except EmbeddingFailedError:
            raise
        except Exception as error:
            errors.append(
                dict(collection_name=scope.collection_name, kb_name=scope.kb_name, error=str(error))
            )
            continue
        records.extend(_chunk_record(result, scope) for result in results)
    if errors:
        raise PassageCollectionError(errors)
    return _rank_chunks(records, query)
