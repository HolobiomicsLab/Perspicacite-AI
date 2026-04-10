"""WRRF helpers matching core/core.py retrieve_documents (v1), for Chroma search results."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, List, Tuple

from perspicacite.rag.utils import get_doc_citation


def doc_page_content(doc: Any) -> str:
    """Same role as LangChain Document.page_content in v1."""
    if hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
        return doc.chunk.text
    if isinstance(doc, dict) and "full_text" in doc:
        return doc["full_text"]
    if hasattr(doc, "content"):
        return str(doc.content)
    return str(doc)


def merge_three_chunks(before: Any, current: Any, after: Any) -> Tuple[str, Any]:
    """Port of core/core.py merge_documents for chunk-level results."""
    merged_text = ""
    if before:
        merged_text += doc_page_content(before) + "\n"
    merged_text += doc_page_content(current) + "\n"
    if after:
        merged_text += doc_page_content(after)

    if hasattr(current, "chunk") and hasattr(current.chunk, "metadata"):
        meta = current.chunk.metadata
        merged_metadata = copy.copy(meta) if meta is not None else {}
    else:
        merged_metadata = {}

    essential_fields = ["citation", "source_type", "url", "chunk", "id"]
    if isinstance(merged_metadata, dict):
        for field in essential_fields:
            if field not in merged_metadata:
                merged_metadata[field] = ""

    return merged_text, merged_metadata


def build_merged_search_result(
    merged_text: str,
    metadata: Any,
    wrrf_score: float,
    score: float = 0.5,
) -> Any:
    """Wrap merged text so format_documents_for_prompt / get_doc_citation still work."""
    from types import SimpleNamespace

    if hasattr(metadata, "title") or hasattr(metadata, "citation"):
        chunk_meta = metadata
    elif isinstance(metadata, dict):
        chunk_meta = SimpleNamespace(**{k: v for k, v in metadata.items()})
    else:
        chunk_meta = metadata

    chunk = SimpleNamespace(text=merged_text, metadata=chunk_meta)
    out = SimpleNamespace(chunk=chunk, score=score, wrrf_score=wrrf_score)
    return out


def select_wrrf_merged_documents(
    sorted_docs: List[Tuple[Any, float]],
    documents_info: dict[Any, Any],
    final_max_docs: int,
    max_docs_per_source: int,
) -> List[Any]:
    """
    v1: walk WRRF-sorted unique doc_ids, merge with neighbors in that list,
    enforce max_docs_per_source.
    """
    sorted_doc_ids = [doc_id for doc_id, _ in sorted_docs]
    sorted_documents = [documents_info[doc_id] for doc_id in sorted_doc_ids]
    selected: List[Any] = []
    source_counter: Counter[str] = Counter()

    for idx, doc_id in enumerate(sorted_doc_ids):
        wrrf_score = sorted_docs[idx][1]
        if len(selected) >= final_max_docs:
            break

        doc = documents_info[doc_id]
        before_doc = sorted_documents[idx - 1] if idx > 0 else None
        after_doc = sorted_documents[idx + 1] if idx < len(sorted_documents) - 1 else None

        merged_text, merged_meta = merge_three_chunks(before_doc, doc, after_doc)

        merged_obj = build_merged_search_result(
            merged_text,
            merged_meta,
            wrrf_score=wrrf_score,
            score=getattr(doc, "score", 0.5),
        )
        source = get_doc_citation(merged_obj)
        if source_counter[source] >= max_docs_per_source:
            continue
        selected.append(merged_obj)
        source_counter[source] += 1

    return selected
