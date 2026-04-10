"""Relevancy helpers ported from Perspicacite-AI-release/core/llm_utils.py (v1)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, List, Tuple


def extract_key_terms(query: str, min_length: int = 3) -> List[str]:
    """Extract important terms from a query for context relevance scoring (v1)."""
    stopwords = {
        "the", "and", "or", "of", "to", "a", "in", "that", "it",
        "is", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "do", "does", "did", "can", "could", "will",
        "would", "should", "what", "which", "who", "when", "where",
        "why", "how", "many", "much", "this", "these", "those", "some",
        "any", "all", "with", "for", "from", "about", "as",
    }

    tokens = query.lower().split()
    key_terms = []
    for token in tokens:
        token = token.strip(".,?!:;()[]{}\"'")
        if (
            token not in stopwords
            and len(token) >= min_length
            and not token.isdigit()
        ):
            key_terms.append(token)
    return key_terms


def assess_query_complexity(query: str) -> float:
    """Assess query complexity 0–1 for temperature tuning (v1)."""
    words = query.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    question_marks = query.count("?")
    complex_markers = sum(
        1
        for marker in [
            "compare",
            "difference",
            "explain",
            "analyze",
            "how",
            "why",
        ]
        if marker in query.lower()
    )

    complexity = (
        min(1.0, word_count / 25) * 0.4
        + min(1.0, avg_word_length / 8) * 0.2
        + min(1.0, question_marks / 2) * 0.1
        + min(1.0, complex_markers / 2) * 0.3
    )
    return complexity


def _get_meta_dict(doc: Any) -> dict:
    if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
        m = doc.chunk.metadata
        if hasattr(m, "model_dump"):
            return m.model_dump()
        if isinstance(m, dict):
            return m
        return {
            "citation": getattr(m, "citation", None) or getattr(m, "title", ""),
            "title": getattr(m, "title", None),
            "date": getattr(m, "year", None),
        }
    if isinstance(doc, dict) and "full_text" in doc:
        return {
            "citation": doc.get("title") or doc.get("doi") or "Unknown",
            "title": doc.get("title"),
            "date": doc.get("year"),
        }
    return {"citation": "Unknown"}


def reorder_documents_by_relevance(query: str, documents: List[Any]) -> List[Any]:
    """Reorder documents by term frequency + position + metadata (profonde.py v1)."""
    if not documents:
        return documents

    key_terms = extract_key_terms(query)
    scored_chunks: List[Tuple[Any, float]] = []

    for i, doc in enumerate(documents):
        if hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
            text = doc.chunk.text
        elif isinstance(doc, dict) and doc.get("full_text"):
            text = doc["full_text"]
        else:
            text = str(doc)

        term_freq_score = sum(text.lower().count(term.lower()) for term in key_terms)
        position_score = 1.0 / (i + 1)
        metadata_boost = 1.0
        meta = _get_meta_dict(doc)
        title = meta.get("title") or ""
        if title:
            title_match = sum(title.lower().count(term.lower()) for term in key_terms)
            if title_match > 0:
                metadata_boost += 0.5
        date_val = meta.get("date")
        if date_val:
            try:
                year = int(date_val)
                current_year = datetime.now().year
                age = max(0, current_year - year)
                metadata_boost += 0.5 * max(0, 1 - (age / 10))
            except (ValueError, TypeError):
                pass

        final_score = (
            term_freq_score * 0.6 + position_score * 0.2 + metadata_boost * 0.2
        )
        scored_chunks.append((doc, final_score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_chunks]
