"""Shared utilities for RAG modes.

This module contains common functions used across all RAG modes to reduce code duplication.
"""

from typing import Any, List

from perspicacite.models.rag import SourceReference


def format_references(sources: List[SourceReference]) -> str:
    """Format sources as a references section.

    Args:
        sources: List of source references

    Returns:
        Formatted references string in markdown
    """
    if not sources:
        return ""

    lines = ["---", "## References"]
    for i, src in enumerate(sources, 1):
        ref = f"[{i}] {src.title}"
        if src.authors:
            ref += f" - {src.authors}"
        if src.year:
            ref += f" ({src.year})"
        if src.doi:
            ref += f" - DOI: {src.doi}"
        lines.append(ref)

    return "\n".join(lines)


def prepare_sources(
    documents: List[Any],
    max_docs: int = 10,
    dedupe_by: str = "title",
) -> List[SourceReference]:
    """Prepare source references from documents with deduplication.

    Args:
        documents: List of document objects
        max_docs: Maximum number of sources to return
        dedupe_by: Field to use for deduplication ('title' or 'doi')

    Returns:
        List of SourceReference objects
    """
    seen = set()
    sources = []

    for doc in documents:
        # Extract metadata
        if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
            meta = doc.chunk.metadata
            title = getattr(meta, "title", "Untitled")
            authors = getattr(meta, "authors", [])
            year = getattr(meta, "year", None)
            doi = getattr(meta, "doi", None)
        elif isinstance(doc, dict):
            title = doc.get("title", doc.get("source", "Unknown"))
            authors = doc.get("authors", [])
            year = doc.get("year")
            doi = doc.get("doi")
        else:
            continue

        # Deduplicate
        dedupe_key = doi if (dedupe_by == "doi" and doi) else title
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        # Format authors
        authors_str = None
        if authors:
            if isinstance(authors, list):
                authors_str = ", ".join(str(a) for a in authors[:3])
                if len(authors) > 3:
                    authors_str += " et al."
            else:
                authors_str = str(authors)

        # Get relevance score
        relevance_score = getattr(doc, "score", 0.0)
        if hasattr(doc, "wrrf_score"):
            relevance_score = doc.wrrf_score

        sources.append(
            SourceReference(
                title=title,
                authors=authors_str,
                year=year,
                doi=doi,
                relevance_score=relevance_score,
            )
        )

        if len(sources) >= max_docs:
            break

    return sources


def get_doc_citation(doc: Any) -> str:
    """Extract citation from document.

    Args:
        doc: Document object

    Returns:
        Citation string
    """
    if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
        meta = doc.chunk.metadata
        if hasattr(meta, "citation"):
            return meta.citation
        if hasattr(meta, "title"):
            return meta.title
    if isinstance(doc, dict):
        return doc.get("citation", doc.get("source", "Unknown"))
    return "Unknown"


def format_documents_for_prompt(documents: List[Any]) -> str:
    """Format documents for inclusion in LLM prompt.

    Args:
        documents: List of document objects

    Returns:
        Formatted document string
    """
    formatted = []

    for i, doc in enumerate(documents, 1):
        # Extract text content
        if hasattr(doc, "chunk") and hasattr(doc.chunk, "text"):
            text = doc.chunk.text
        elif hasattr(doc, "content"):
            text = str(doc.content)
        else:
            text = str(doc)

        # Extract citation
        citation = get_doc_citation(doc)

        formatted.append(f"[{i}] Source: {citation}\n{text}")

    return "\n\n---\n\n".join(formatted)


def get_system_prompt() -> str:
    """Get the standard system prompt for response generation.

    Returns:
        System prompt string
    """
    return """You are a scientific AI assistant. Provide clear, well-structured answers using markdown formatting.

FORMAT REQUIREMENTS:
1. Start with a brief overview/introduction (2-3 sentences)
2. Use ## for main section headings (e.g., ## Overview, ## Key Points)
3. Use ### for subsections if needed
4. Use bullet points (- item) for lists
5. Use **bold** SPARINGLY - only for the most important 2-3 key terms
6. Use *italics* for emphasis on specific words or phrases
7. Separate paragraphs with blank lines
8. Include relevant examples if helpful

IMPORTANT: Do not put entire paragraphs in bold. Only individual important words or short phrases.

Your response should be easy to read with clear visual structure."""


def format_references_academic(papers: List[dict]) -> str:
    """Format papers as academic references with markdown links.

    Uses markdown link format: [Author et al., Year](url "full citation")

    Args:
        papers: List of paper dictionaries with title, authors, year, doi

    Returns:
        Formatted references section
    """
    if not papers:
        return ""

    ref_lines = ["## References\n"]

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        year = paper.get("year", "n.d.")
        doi = paper.get("doi", "")
        
        # Normalize DOI - remove existing URL prefix if present
        if doi:
            doi = doi.strip()
            for prefix in ("https://doi.org/", "http://dx.doi.org/", "doi:"):
                if doi.lower().startswith(prefix.lower()):
                    doi = doi[len(prefix):].strip()
        
        url = f"https://doi.org/{doi}" if doi else ""

        # Normalize authors to list (handle both list and comma-separated string)
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        elif not isinstance(authors, list):
            authors = []

        # Format authors: "FirstAuthor et al." if >2 authors
        if len(authors) == 0:
            author_str = "Unknown"
        elif len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        else:
            author_str = f"{authors[0]} et al."

        # Format full citation (for title attribute of the link)
        if authors:
            full_citation = f"{', '.join(authors)}. {year}. {title}."
        else:
            full_citation = f"{title}. {year}."

        # Use markdown link format
        if url:
            ref_lines.append(f'{i}. [{author_str}, {year}]({url} "{full_citation}")')
        else:
            ref_lines.append(f"{i}. {author_str}, {year}. {title}.")

    return "\n".join(ref_lines)


def deduplicate_chunk_overlaps(
    chunks: list[Any],
    overlap_words: int = 200,
) -> list[dict[str, Any]]:
    """Remove overlapping text between consecutive chunks of the same paper.

    The chunker uses a sliding window where the last ``overlap_words`` of chunk N
    are identical to the first ``overlap_words`` of chunk N+1.  This function
    detects and trims that overlap.

    Args:
        chunks: list of DocumentChunk objects sorted by (paper_id, chunk_index)
        overlap_words: max overlap in words (default 200 = chunk_overlap default)

    Returns:
        list of dicts with keys: text, paper_id, chunk_index, metadata
    """
    if not chunks:
        return []

    # Group by paper_id
    from collections import OrderedDict

    groups: OrderedDict[str, list[Any]] = OrderedDict()
    for chunk in chunks:
        pid = getattr(chunk.metadata, "paper_id", "") if hasattr(chunk, "metadata") else ""
        groups.setdefault(pid, []).append(chunk)

    results: list[dict[str, Any]] = []
    for pid, paper_chunks in groups.items():
        # Already sorted by chunk_index from the caller
        for i, chunk in enumerate(paper_chunks):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            meta = chunk.metadata if hasattr(chunk, "metadata") else None

            # Check overlap with previous chunk
            if i > 0 and overlap_words > 0:
                prev_text = paper_chunks[i - 1].text if hasattr(paper_chunks[i - 1], "text") else ""
                prev_words = prev_text.split()
                # Take the tail of previous chunk
                tail = prev_words[-overlap_words:]
                if tail:
                    tail_str = " ".join(tail)
                    # Check if this tail appears at the start of current text
                    curr_words = text.split()
                    for match_len in range(len(tail), 0, -1):
                        candidate = " ".join(tail[:match_len])
                        curr_prefix = " ".join(curr_words[:match_len])
                        if candidate and candidate == curr_prefix:
                            # Trim the overlap from current chunk
                            text = " ".join(curr_words[match_len:])
                            break

            results.append({
                "text": text,
                "paper_id": pid,
                "chunk_index": getattr(meta, "chunk_index", i) if meta else i,
                "metadata": meta,
            })

    return results


def format_paper_results_for_prompt(
    papers: list[dict[str, Any]],
    max_chars_per_paper: int = 4000,
) -> str:
    """Format paper-level results for LLM prompt.

    Each paper is a delimited section with metadata + truncated full text.

    Args:
        papers: list of dicts from search_two_pass()
        max_chars_per_paper: cap per paper in characters

    Returns:
        Formatted string for inclusion in LLM prompt
    """
    if not papers:
        return "No relevant papers found."

    sections: list[str] = []
    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown")
        authors = paper.get("authors", "")
        year = paper.get("year", "")
        doi = paper.get("doi", "")
        score = paper.get("paper_score", 0)
        full_text = paper.get("full_text", "")

        header = f"[Paper {i}]"
        if title:
            header += f" {title}"
        if authors:
            header += f" — {authors}"
        if year:
            header += f" ({year})"
        if doi:
            header += f" | DOI: {doi}"
        header += f" | relevance: {score:.2f}"

        # Truncate full text
        if len(full_text) > max_chars_per_paper:
            full_text = full_text[:max_chars_per_paper] + "\n[...truncated]"

        sections.append(f"{header}\n\n{full_text}")

    return "\n\n---\n\n".join(sections)
