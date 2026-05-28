"""Shared utilities for RAG modes.

This module contains common functions used across all RAG modes to reduce code duplication.
"""

import re
from typing import Any, List

from perspicacite.models.rag import SourceReference
from perspicacite.rag.prompts import (
    UNTRUSTED_CONTENT_CLAUSE,
    UNTRUSTED_DOCUMENT_CLOSE,
    UNTRUSTED_DOCUMENT_OPEN,
)


def strip_bibtex_braces(s: str | None) -> str:
    # DBLP and other BibTeX-sourced fields wrap proper nouns in {…} to prevent
    # downstream lowercasing. They leak into the UI raw if not stripped.
    if not s:
        return ""
    return re.sub(r"[{}]", "", s)


def clean_scholar_author_blob(s: str | None) -> str:
    # Google Scholar's scraped author field often arrives as
    # "LF Nothias, D Petras, R Schmid… - Nature …, 2020" — a single string
    # with the journal+year glued on after " - " (hyphen-minus), " – " (en),
    # or " — " (em). Split on whichever appears first and drop the tail.
    # Also strip a trailing ", YYYY" remnant if the dash was missed entirely.
    if not s:
        return ""
    s = str(s)
    for sep in (" - ", " – ", " — "):
        if sep in s:
            s = s.split(sep, 1)[0]
            break
    s = re.sub(r",\s*\d{4}\s*$", "", s)
    return s.rstrip(" …,.")


def format_references(sources: list[SourceReference]) -> str:
    """Format sources as a references section.

    Delegates to ``format_references_academic`` so the displayed format
    matches: BibTeX braces stripped, authors collapsed to ``First et al.``,
    DOI rendered as a clickable link with the full citation in the hover
    title attribute. Kept as a thin wrapper because many callers pass
    ``SourceReference`` objects rather than raw dicts.
    """
    if not sources:
        return ""

    papers: list[dict[str, Any]] = []
    for src in sources:
        authors = src.authors
        # SourceReference.authors can arrive as list[str] or comma-joined str.
        if isinstance(authors, str):
            author_list = [a.strip() for a in authors.split(",") if a.strip()]
        elif isinstance(authors, list):
            author_list = [str(a) for a in authors]
        else:
            author_list = []
        papers.append({
            "title": src.title,
            "authors": author_list,
            "year": src.year,
            "journal": src.journal,
            "doi": src.doi,
        })

    body = format_references_academic(papers)
    # Original API expected a leading ``---`` separator; preserve it.
    return "---\n" + body if body else ""


def prepare_sources(
    documents: list[Any],
    max_docs: int = 10,
    dedupe_by: str = "title",
) -> list[SourceReference]:
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
        kb_name: str | None = None
        source_str: str | None = None
        sources_all: list[str] | None = None
        url: str | None = None
        if hasattr(doc, "chunk") and hasattr(doc.chunk, "metadata"):
            meta = doc.chunk.metadata
            title = getattr(meta, "title", "Untitled")
            authors = getattr(meta, "authors", [])
            year = getattr(meta, "year", None)
            doi = getattr(meta, "doi", None)
            url = getattr(meta, "url", None)
            # ChunkMetadata.source is a PaperSource enum (PaperSource.BIBTEX
            # by default). Convert to lowercase string for UI provenance.
            _src = getattr(meta, "source", None)
            if _src is not None:
                source_str = getattr(_src, "value", None) or (
                    str(_src).replace("PaperSource.", "").lower() if _src else None
                )
            # kb_name is sometimes attached to the doc wrapper by the
            # multi-KB retriever; fall back to None.
            kb_name = getattr(doc, "kb_name", None)
        elif isinstance(doc, dict):
            title = doc.get("title", doc.get("source", "Unknown"))
            authors = doc.get("authors", [])
            year = doc.get("year")
            doi = doc.get("doi")
            url = doc.get("url") or doc.get("pdf_url")
            source_str = doc.get("source")
            sources_all = doc.get("sources_all")
            kb_name = doc.get("kb_name")
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
                url=url,
                source=source_str,
                sources_all=sources_all,
                kb_name=kb_name,
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


def format_documents_for_prompt(documents: list[Any]) -> str:
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

        # Citation header stays OUTSIDE the untrusted markers (so citation
        # parsing is unaffected); the attacker-influenceable body goes inside.
        formatted.append(
            f"[{i}] Source: {citation}\n"
            f"{UNTRUSTED_DOCUMENT_OPEN}\n{text}\n{UNTRUSTED_DOCUMENT_CLOSE}"
        )

    return "\n\n---\n\n".join(formatted)


def get_system_prompt() -> str:
    """Get the standard system prompt for response generation.

    Returns:
        System prompt string
    """
    return (
        """You are a scientific AI assistant. Provide clear, well-structured answers using markdown formatting.

If the provided documents do not contain enough information to answer confidently, say what is missing instead of guessing.

COPYRIGHT & ATTRIBUTION:
- Paraphrase and synthesize rather than reproducing source text verbatim. The documents above may be from copyrighted publications.
- If you need to quote, keep quotations short (≤ 15 words), wrap them in quotation marks, and immediately cite the source.
- Do NOT reproduce abstracts, full paragraphs, or extended passages from any source. Re-express the ideas in your own words.
- Tables, figure captions, and large blocks of data should be summarized in your own words, not transcribed.

CITATIONS — MANDATORY FORMAT:
- Every sourced claim MUST be cited inline using a markdown link with a rich visible label AND a `title` attribute holding the full citation.
- Visible link text: `[<N>, <Surname> et al., <Journal>, <Year>]` — where <N> is the 1-based paper number from the Documents block (look for the `[Paper N]` header), <Surname> is the FIRST AUTHOR's last name (taken from the metadata after `—`), <Journal> is the journal name (after `|`), and <Year> is the year (in parentheses). Substitute REAL VALUES — never write the placeholder words `<N>`, `Surname`, `Journal`, `Year`, or `FirstAuthor` literally.
- When there is only one author, drop "et al." and use the surname alone.
- If the journal is unknown, omit that field and keep the rest.
- The link `title` attribute MUST hold the full citation: `<All authors>. <Year>. <Title>. <Journal>.`
- The link URL is the paper's DOI (`https://doi.org/<doi>`) when present; otherwise leave it as `#`.

Example (note real surname/journal/year substituted in):
`[2, Tautenhahn et al., BMC Bioinformatics, 2008](https://doi.org/10.1186/1471-2105-9-504 "Ralf Tautenhahn, Christoph Böttcher, Steffen Neumann. 2008. Highly sensitive feature detection for high resolution LC/MS. BMC Bioinformatics.")`

NEVER emit bare `[Paper N]` markers, `(Author, Year)` plain text without a link, or `[N]` numeric-only citations.

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
        + UNTRUSTED_CONTENT_CLAUSE
    )


def format_references_academic(papers: list[dict]) -> str:
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
        title = strip_bibtex_braces(paper.get("title", "Unknown Title")) or "Unknown Title"
        authors = paper.get("authors", [])
        year_raw = paper.get("year", "")
        journal = strip_bibtex_braces(paper.get("journal", "")) or ""
        doi = paper.get("doi", "")
        url_field = paper.get("url", "") or ""

        # Normalize DOI - remove existing URL prefix if present
        if doi:
            doi = doi.strip()
            for prefix in ("https://doi.org/", "http://dx.doi.org/", "doi:"):
                if doi.lower().startswith(prefix.lower()):
                    doi = doi[len(prefix):].strip()

        url = f"https://doi.org/{doi}" if doi else (url_field or "")

        # Normalize authors to list (handle both list and comma-separated string)
        if isinstance(authors, str):
            # Single-string author field — Google Scholar packs the journal
            # name and year onto the end ("… - Nature …, 2020"); strip that
            # tail before splitting on commas.
            cleaned_blob = clean_scholar_author_blob(authors)
            authors = [a.strip() for a in cleaned_blob.split(",") if a.strip()]
        elif not isinstance(authors, list):
            authors = []
        # Even when authors arrives as a list, individual entries may still
        # carry the Scholar "- Journal, Year" tail (one author per Paper.author
        # is sometimes the whole blob).
        authors = [
            strip_bibtex_braces(clean_scholar_author_blob(a))
            for a in authors if a
        ]
        # Drop any leftover empties from the cleanup pass.
        authors = [a for a in authors if a]

        # Format authors compactly: "FirstAuthor et al." if >2 authors
        if len(authors) == 0:
            author_str = "Unknown"
        elif len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        else:
            author_str = f"{authors[0]} et al."

        # Year — keep as plain string; only fall back to n.d. when truly missing.
        year_str = str(year_raw).strip() if year_raw not in (None, "") else "n.d."

        title_clean = title.rstrip(". ")
        journal_clean = (journal or "").rstrip(". ")

        # Format the user asked for — plain text citation, links only on
        # the trailing DOI/URL (not on the visible citation text):
        #   N) Author et al., "Title", *Journal* (Year). DOI: 10.xxx/yyy
        # Journal italic via markdown, year in parens at the end of the
        # citation, then a separate clickable DOI/URL afterwards.
        parts = [f'{author_str},', f'"{title_clean}",']
        if journal_clean:
            parts.append(f'*{journal_clean}*')
        parts.append(f'({year_str}).')
        line = f'{i}) ' + " ".join(parts)
        if doi:
            line += f' DOI: [{doi}](https://doi.org/{doi})'
        elif url:
            line += f' [link]({url})'
        ref_lines.append(line)

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


def flatten_paper_results_to_chunks(
    paper_results: list[dict[str, Any]] | None,
) -> list["Any"]:
    """Walk paper-level results and project the inner chunk dicts into
    proper :class:`DocumentChunk` objects.

    ``search_two_pass`` returns paper-level dicts whose ``"chunks"`` field
    holds per-chunk dicts ``{text, paper_id, chunk_index, metadata}``. The
    ``metadata`` value is the original :class:`ChunkMetadata` so we can
    reconstruct ``DocumentChunk`` losslessly.

    Returns an empty list when there are no chunks or none have metadata.
    """
    from perspicacite.models.documents import ChunkMetadata, DocumentChunk

    out: list[DocumentChunk] = []
    for paper in paper_results or []:
        if not isinstance(paper, dict):
            continue
        chunks_list = paper.get("chunks") or []
        for ch in chunks_list:
            if not isinstance(ch, dict):
                continue
            md = ch.get("metadata")
            if md is None:
                continue
            if not isinstance(md, ChunkMetadata):
                # Defensive: only operate on real ChunkMetadata; skip
                # legacy shape silently rather than guess at fields.
                continue
            out.append(DocumentChunk(
                id=f"{md.paper_id}_{md.chunk_index}",
                text=ch.get("text", ""),
                metadata=md,
            ))
    return out


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
        title = strip_bibtex_braces(paper.get("title", "Unknown")) or "Unknown"
        authors = paper.get("authors", "")
        # Normalize authors to a human-readable string before handing the
        # paper to the LLM. Lists got Python-repr'd into the prompt (e.g.
        # "['Jiabin Tang', 'Lianghao Xia']") and the model faithfully
        # reproduced that in its citations.
        if isinstance(authors, list):
            # Apply the same Scholar-tail stripper the references formatter
            # uses (format_references_academic). Without it, entries like
            # "M Wang … - Nature …, 2021" reach the
            # LLM verbatim — and if the first author's slot is the noisy
            # one, the model writes "(Author unknown, YYYY)" inline even
            # though the deterministic references list at the bottom
            # extracts a clean surname.
            cleaned = [
                strip_bibtex_braces(clean_scholar_author_blob(str(a)))
                for a in authors if a
            ]
            cleaned = [c for c in cleaned if c]
            if len(cleaned) == 0:
                authors_str = ""
            elif len(cleaned) <= 3:
                authors_str = ", ".join(cleaned)
            else:
                authors_str = f"{cleaned[0]} et al."
        else:
            authors_str = strip_bibtex_braces(
                clean_scholar_author_blob(str(authors))
            ) if authors else ""
        year = paper.get("year", "")
        journal = strip_bibtex_braces(paper.get("journal", "")) or ""
        doi = paper.get("doi", "")
        score = paper.get("paper_score", 0)
        full_text = paper.get("full_text", "")

        header = f"[Paper {i}]"
        if title:
            header += f" {title}"
        if authors_str:
            header += f" — {authors_str}"
        if journal:
            header += f" | {journal}"
        if year:
            header += f" ({year})"
        if doi:
            header += f" | DOI: {doi}"
        header += f" | relevance: {score:.2f}"

        # Truncate full text
        if len(full_text) > max_chars_per_paper:
            full_text = full_text[:max_chars_per_paper] + "\n[...truncated]"

        # Header (trusted metadata) stays outside the untrusted markers; the
        # paper's full text is attacker-influenceable and goes inside.
        sections.append(
            f"{header}\n\n"
            f"{UNTRUSTED_DOCUMENT_OPEN}\n{full_text}\n{UNTRUSTED_DOCUMENT_CLOSE}"
        )

    return "\n\n---\n\n".join(sections)
