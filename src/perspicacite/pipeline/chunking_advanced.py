"""Advanced text chunking strategies for document processing.

This module provides sophisticated chunking strategies ported from Perspicacite v1:
- Token-based chunking with proper tokenizers (tiktoken, HuggingFace)
- Semantic chunking using sentence embeddings
- Agentic chunking using LLM for intelligent partitioning
- Section-aware chunking with scientific document structure detection

Usage:
    from perspicacite.pipeline.chunking_advanced import AdvancedChunker
    
    chunker = AdvancedChunker(
        method="semantic",
        max_tokens=800,
        overlap_tokens=120,
        semantic_threshold=0.65,
    )
    chunks = await chunker.chunk_text(text, paper)
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple, Any, Dict

from perspicacite.logging import get_logger
from perspicacite.models.documents import ChunkMetadata, DocumentChunk
from perspicacite.models.kb import ChunkConfig
from perspicacite.models.papers import Paper

logger = get_logger("perspicacite.pipeline.chunking_advanced")

# Optional imports for advanced features
try:
    import numpy as np
except Exception:
    np = None

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# =============================================================================
# Section Detection Patterns
# =============================================================================

# Match common scientific section headers
SECTION_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:\d+|[IVXLCDM]+)(?:\.\d+)*\s*[-.)]?\s*)?"  # optional numbering
    r"(abstract|introduction|background|"
    r"method(?:s|ology)?|materials\s*(?:&|and)\s*methods|"
    r"results?(?:\s+and\s+discussion)?|discussion|related\s+work|conclusions?|experimental|"
    r"references?|bibliograph(?:y|ies)|works\s+cited|literature\s+cited|"
    r"acknowledg(e)?ments?|funding|author\s+contributions?|conflicts?\s+of\s+interest|ethics|"
    r"supplementary(?:\s+material)?|appendix)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)

# Reference sections to filter out
REFS_HEADER_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:\d+|[IVXLCDM]+)(?:\.\d+)*\s*[-.)]?\s*)?"
    r"(references?|bibliograph(?:y|ies)|works\s+cited|literature\s+cited)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)


# =============================================================================
# Tokenizer Functions
# =============================================================================

def _whitespace_tokenize(text: str) -> List[str]:
    """Fallback whitespace tokenizer."""
    return text.split()


def _candidate_tokenizer_ids(model_name: str) -> List[str]:
    """Return candidate HF model ids for tokenizer loading."""
    candidates = [model_name]
    if model_name.startswith("sentence-transformers/bge-"):
        candidates.append(model_name.replace("sentence-transformers/", "BAAI/"))
    if model_name.startswith("sentence-transformers/specter2"):
        candidates.append(model_name.replace("sentence-transformers/", "allenai/"))
    return candidates


def get_tokenizer(
    provider: Optional[str] = None,
    model_name: Optional[str] = None
) -> Callable[[str], List[int]]:
    """
    Get a tokenizer function that maps text -> list of token ids.
    
    Preference order: tiktoken (OpenAI) -> HF AutoTokenizer -> whitespace fallback.
    
    Args:
        provider: Provider name (e.g., 'openai')
        model_name: Model name for HuggingFace tokenizer
        
    Returns:
        Tokenizer encode function
    """
    # Try tiktoken for OpenAI-like tokenization
    if provider and provider.lower() == "openai" and tiktoken is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        logger.debug("chunking: using tiktoken cl100k_base")
        return lambda s: enc.encode(s)

    # Try HuggingFace tokenizer
    if model_name and AutoTokenizer is not None:
        last_err = None
        for cand in _candidate_tokenizer_ids(model_name):
            try:
                logger.info(f"chunking: loading HF tokenizer {cand}")
                tok = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
                return lambda s: tok.encode(s, add_special_tokens=False)
            except Exception as e:
                last_err = e
                logger.warning(f"chunking: failed to load tokenizer for {cand}: {e}")
        if last_err is not None:
            logger.warning(f"chunking: all tokenizer candidates failed, using whitespace fallback")

    # Fallback to whitespace
    logger.debug("chunking: using whitespace fallback tokenizer")
    return lambda s: [i for i, _ in enumerate(_whitespace_tokenize(s))]


# =============================================================================
# Section Detection
# =============================================================================

def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split text into sections based on scientific document headers.
    
    Returns list of (section_name, section_text) tuples.
    Drops reference sections automatically.
    
    Args:
        text: Document text
        
    Returns:
        List of (section_name, section_content) tuples
    """
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_name = "_full"
    current_buf: List[str] = []

    for line in lines:
        candidate = line.strip()
        if SECTION_PATTERN.match(candidate):
            if current_buf:
                sections.append((current_name, current_buf))
            current_name = candidate
            current_buf = []
        else:
            current_buf.append(line)

    if current_buf:
        sections.append((current_name, current_buf))

    # Convert to output format and filter references
    out = [(name, "\n".join(buf).strip()) for name, buf in sections if "\n".join(buf).strip()]
    filtered: List[Tuple[str, str]] = []
    
    for name, sect_text in out:
        header = (name or "").strip()
        if REFS_HEADER_PATTERN.match(header):
            logger.info(f"chunking: dropping reference section '{header[:80]}'")
            continue
        filtered.append((name, sect_text))
    
    logger.info(f"chunking: produced {len(filtered)} sections after reference filtering")
    return filtered


# =============================================================================
# Sentence Splitting
# =============================================================================

def _split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter tolerant to scientific text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    sentences: List[str] = []
    buffer = []
    
    for part in parts:
        chunks = re.split(r"(?<=[.!?])\s+", part)
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            # Accumulate very short fragments
            if len(ch) < 40 and buffer:
                buffer.append(ch)
                merged = " ".join(buffer)
                sentences.append(merged)
                buffer = []
            elif len(ch) < 40:
                buffer.append(ch)
            else:
                if buffer:
                    sentences.append(" ".join(buffer))
                    buffer = []
                sentences.append(ch)
    
    if buffer:
        sentences.append(" ".join(buffer))
    
    return sentences


# =============================================================================
# Chunking Strategies
# =============================================================================

def chunk_by_tokens(
    text: str,
    encode: Callable[[str], List[int]],
    max_tokens: int,
    overlap_tokens: int = 0,
) -> List[str]:
    """
    Chunk text by token counts with fixed overlap.
    
    Args:
        text: Input text
        encode: Tokenizer function
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if max_tokens <= 0:
        return [text]

    tokens = encode(text)
    if not tokens:
        return [text]

    chunks: List[str] = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        
        # Map token span back to text
        ratio_start = start / max(1, len(tokens))
        ratio_end = end / max(1, len(tokens))
        cs = int(ratio_start * len(text))
        ce = int(ratio_end * len(text))
        sub = text[cs:ce]
        
        chunk_text = sub.strip()
        if chunk_text:
            chunks.append(chunk_text)
        
        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)

    return [c for c in chunks if c]


def _get_sentence_embedder(model_name: Optional[str]) -> Optional[Any]:
    """Load sentence transformer model for semantic chunking."""
    if SentenceTransformer is None or model_name is None:
        return None
    try:
        logger.info(f"chunking: loading sentence embedder {model_name}")
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.warning(f"chunking: failed to load embedder {model_name}: {e}")
        return None


def chunk_by_semantics(
    text: str,
    encode: Callable[[str], List[int]],
    embed_model_name: Optional[str],
    threshold: float,
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Chunk text using semantic cohesion with hard token caps.
    
    Splits into sentences, embeds them, grows segments until similarity
    drops below threshold or token cap is reached.
    
    Args:
        text: Input text
        encode: Tokenizer function
        embed_model_name: Sentence transformer model name
        threshold: Similarity threshold for splitting
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk
        overlap_tokens: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if np is None:
        logger.warning("chunking: numpy not available, falling back to token chunking")
        return chunk_by_tokens(text, encode, max_tokens, overlap_tokens)

    embedder = _get_sentence_embedder(embed_model_name)
    if embedder is None:
        logger.warning("chunking: embedder not available, falling back to token chunking")
        return chunk_by_tokens(text, encode, max_tokens, overlap_tokens)

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    # Precompute embeddings
    try:
        sent_vecs = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"chunking: embedding failed, fallback to token chunking: {e}")
        return chunk_by_tokens(text, encode, max_tokens, overlap_tokens)

    def cos_sim(a, b) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    chunks: List[str] = []
    cur_sent_indices: List[int] = []
    cur_tokens = 0
    cur_centroid = None

    def append_chunk(end_index: int):
        nonlocal cur_sent_indices, cur_tokens, cur_centroid
        if not cur_sent_indices:
            return
        
        chunk_text = " ".join(sentences[i] for i in cur_sent_indices).strip()
        if chunk_text:
            chunks.append(chunk_text)
        
        # Prepare overlap
        if overlap_tokens > 0 and cur_sent_indices:
            overlap = []
            tok_budget = 0
            for i in reversed(cur_sent_indices):
                tok_budget += len(encode(sentences[i]))
                overlap.append(i)
                if tok_budget >= overlap_tokens:
                    break
            cur_sent_indices = list(reversed(overlap))
            cur_tokens = sum(len(encode(sentences[i])) for i in cur_sent_indices)
            cur_centroid = np.mean(sent_vecs[cur_sent_indices, :], axis=0) if cur_sent_indices else None
        else:
            cur_sent_indices = []
            cur_tokens = 0
            cur_centroid = None

    for idx, (s, v) in enumerate(zip(sentences, sent_vecs)):
        s_tokens = len(encode(s))
        
        if not cur_sent_indices:
            cur_sent_indices = [idx]
            cur_tokens = s_tokens
            cur_centroid = v.astype(np.float32)
            continue

        # Check token cap
        if cur_tokens + s_tokens > max_tokens and cur_tokens >= min_tokens:
            append_chunk(idx)
            cur_sent_indices = [idx]
            cur_tokens = s_tokens
            cur_centroid = v.astype(np.float32)
            continue

        # Check cohesion
        sim = cos_sim(cur_centroid, v) if cur_centroid is not None else 1.0
        if sim < threshold and cur_tokens >= min_tokens:
            append_chunk(idx)
            cur_sent_indices = [idx]
            cur_tokens = s_tokens
            cur_centroid = v.astype(np.float32)
            continue

        # Add to current segment
        cur_sent_indices.append(idx)
        cur_tokens += s_tokens
        if cur_centroid is not None:
            cur_centroid = (cur_centroid * (len(cur_sent_indices) - 1) + v) / len(cur_sent_indices)

        # Hard cap guard
        if cur_tokens > max_tokens:
            append_chunk(idx)

    # Flush remainder
    append_chunk(len(sentences))
    return chunks


# =============================================================================
# Advanced Chunker Class
# =============================================================================

class AdvancedChunker:
    """
    Advanced document chunker with multiple strategies.
    
    Supports:
    - token: Simple token-based chunking
    - semantic: Embedding-based semantic chunking
    - agentic: LLM-based intelligent chunking (requires llm_client)
    - section_aware: Respects document section boundaries
    """
    
    def __init__(
        self,
        method: str = "token",
        max_tokens: int = 800,
        overlap_tokens: int = 120,
        min_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        semantic_threshold: float = 0.65,
        embed_model_name: Optional[str] = None,
        section_aware: bool = False,
    ):
        """
        Initialize advanced chunker.
        
        Args:
            method: Chunking method (token, semantic, agentic)
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
            min_tokens: Minimum tokens per chunk (defaults to 40% of max)
            provider: LLM provider for agentic chunking
            model_name: Model name for tokenizer
            semantic_threshold: Similarity threshold for semantic chunking
            embed_model_name: Sentence transformer model for semantic chunking
            section_aware: Whether to respect section boundaries
        """
        self.method = method
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens or max(int(0.4 * max_tokens), 200)
        self.provider = provider
        self.model_name = model_name
        self.semantic_threshold = semantic_threshold
        self.embed_model_name = embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.section_aware = section_aware
        
        self.encode = get_tokenizer(provider=provider, model_name=model_name)
    
    async def chunk_text(
        self,
        text: str,
        paper: Paper,
        llm_client: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk text into DocumentChunks.
        
        Args:
            text: Text to chunk
            paper: Source paper for metadata
            llm_client: LLM client for agentic chunking (required if method="agentic")
            
        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []
        
        if self.section_aware:
            return await self._chunk_section_aware(text, paper, llm_client)
        
        # Get raw chunks based on method
        if self.method == "semantic":
            raw_chunks = chunk_by_semantics(
                text, self.encode, self.embed_model_name,
                self.semantic_threshold, self.max_tokens,
                self.min_tokens, self.overlap_tokens
            )
        elif self.method == "agentic":
            if llm_client is None:
                logger.warning("chunking: agentic method requires llm_client, falling back to token")
                raw_chunks = chunk_by_tokens(text, self.encode, self.max_tokens, self.overlap_tokens)
            else:
                raw_chunks = await self._chunk_agentic(text, llm_client)
        else:  # token
            raw_chunks = chunk_by_tokens(text, self.encode, self.max_tokens, self.overlap_tokens)
        
        # Convert to DocumentChunks
        return self._to_document_chunks(raw_chunks, paper)
    
    async def _chunk_section_aware(
        self,
        text: str,
        paper: Paper,
        llm_client: Optional[Any],
    ) -> List[DocumentChunk]:
        """Chunk respecting section boundaries."""
        sections = split_into_sections(text)
        all_chunks: List[DocumentChunk] = []
        chunk_index = 0
        
        for section_name, section_text in sections:
            if not section_text.strip():
                continue
            
            # Create temporary chunker for this section
            section_chunker = AdvancedChunker(
                method=self.method if self.method != "agentic" else "token",
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens,
                min_tokens=self.min_tokens,
                provider=self.provider,
                model_name=self.model_name,
                semantic_threshold=self.semantic_threshold,
                embed_model_name=self.embed_model_name,
                section_aware=False,  # Don't recurse
            )
            
            try:
                section_chunks = await section_chunker.chunk_text(section_text, paper, llm_client)

                # Update section metadata (ChunkMetadata is frozen, so create new)
                for chunk in section_chunks:
                    chunk.id = f"{paper.id}_{chunk_index}"
                    chunk.metadata = ChunkMetadata(
                        paper_id=paper.id,
                        chunk_index=chunk_index,
                        source=paper.source,
                        title=paper.title,
                        authors=_format_authors(paper.authors),
                        year=paper.year,
                        doi=paper.doi,
                        url=paper.url,
                        section=section_name,
                    )
                    chunk_index += 1
                
                all_chunks.extend(section_chunks)
            except Exception as e:
                logger.warning(f"chunking: error processing section '{section_name}': {e}")
                # Add as single chunk on error
                chunk = DocumentChunk(
                    id=f"{paper.id}_{chunk_index}",
                    text=section_text,
                    metadata=ChunkMetadata(
                        paper_id=paper.id,
                        chunk_index=chunk_index,
                        source=paper.source,
                        title=paper.title,
                        authors=_format_authors(paper.authors),
                        year=paper.year,
                        doi=paper.doi,
                        url=paper.url,
                        section=section_name,
                    ),
                )
                all_chunks.append(chunk)
                chunk_index += 1
        
        return all_chunks
    
    async def _chunk_agentic(
        self,
        text: str,
        llm_client: Any,
    ) -> List[str]:
        """Agentic chunking using LLM span partitioning.

        Slides over the text in character windows, asks the LLM to return
        JSON spans [{"start": int, "end": int}] covering each window, then
        validates and normalizes spans into contiguous non-overlapping chunks
        that respect token caps.
        """
        if not text.strip():
            return []

        if llm_client is None:
            logger.warning("chunking: agentic requires llm_client, falling back to token")
            return chunk_by_tokens(text, self.encode, self.max_tokens, self.overlap_tokens)

        window_chars = 8000
        min_tokens = self.min_tokens

        # --- helpers ---
        def _split_span_by_tokens(span_text: str) -> List[str]:
            if not span_text:
                return []
            if len(self.encode(span_text)) <= self.max_tokens:
                return [span_text]
            parts = re.split(r"(?<=[.!?;:])\s+", span_text)
            if len(parts) > 1:
                out: List[str] = []
                buf: List[str] = []
                buf_tok = 0
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    t = len(self.encode(p))
                    if buf and buf_tok + t > self.max_tokens:
                        out.append(" ".join(buf).strip())
                        buf = [p]
                        buf_tok = t
                    else:
                        buf.append(p)
                        buf_tok += t
                if buf:
                    out.append(" ".join(buf).strip())
                final: List[str] = []
                for piece in out:
                    if len(self.encode(piece)) <= self.max_tokens:
                        final.append(piece)
                    else:
                        words = piece.split()
                        if len(words) <= 1:
                            final.append(piece)
                        else:
                            mid = max(1, len(words) // 2)
                            final.extend([" ".join(words[:mid]), " ".join(words[mid:])])
                return final
            words = span_text.split()
            if len(words) <= 1:
                return [span_text]
            mid = max(1, len(words) // 2)
            return [" ".join(words[:mid]), " ".join(words[mid:])]

        # --- main loop ---
        chunks: List[str] = []
        pending_small = ""
        text_len = len(text)
        pos = 0

        while pos < text_len:
            win_end = min(pos + max(1, window_chars), text_len)
            win_text = text[pos:win_end]
            wlen = len(win_text)

            # Ask LLM to partition this window
            prompt = (
                "Partition the following text window into coherent retrieval chunks.\n"
                "Return STRICT JSON: {\"spans\": [{\"start\": <int>, \"end\": <int>}, ...]}\n"
                "Constraints:\n"
                "- spans must be sorted by start, contiguous (no gaps), non-overlapping\n"
                f"- fully cover 0..{wlen}\n"
                f"- each chunk roughly {self.max_tokens} tokens, minimum {min_tokens}\n"
                "- prefer boundaries at paragraph/sentence breaks\n"
                "- avoid breaking mid-sentence unless necessary\n\n"
                f"text:\n{win_text}\n\n"
                "Reply ONLY with the JSON object."
            )

            try:
                response = await llm_client.complete(prompt, temperature=0.0, max_tokens=1024)
            except Exception as e:
                logger.warning(f"chunking: agentic LLM call failed: {e}")
                # Fallback: token chunk this window
                window_chunks = chunk_by_tokens(win_text, self.encode, self.max_tokens, 0)
                chunks.extend(window_chunks)
                pos = win_end
                continue

            # Parse spans
            spans: List[Tuple[int, int]] = []
            if response:
                try:
                    import json as _json
                    first_brace = response.find("{")
                    last_brace = response.rfind("}")
                    payload = response[first_brace:last_brace + 1] if first_brace != -1 and last_brace != -1 else response
                    data = _json.loads(payload)
                    raw_spans = data.get("spans", [])
                    for sp in raw_spans:
                        if not isinstance(sp, dict):
                            continue
                        s = sp.get("start")
                        e = sp.get("end")
                        if isinstance(s, int) and isinstance(e, int):
                            spans.append((s, e))
                except Exception as e:
                    logger.warning(f"chunking: failed to parse agentic spans: {e}")

            # Normalize spans → sorted, clipped, contiguous
            norm_spans: List[Tuple[int, int]] = []
            if spans:
                spans.sort(key=lambda x: (x[0], x[1]))
                cur = 0
                for s, e in spans:
                    s = max(0, min(s, wlen))
                    e = max(0, min(e, wlen))
                    if e < s:
                        s, e = e, s
                    if s > cur:
                        norm_spans.append((cur, s))
                        cur = s
                    if e > cur:
                        norm_spans.append((cur, e))
                        cur = e
                if cur < wlen:
                    norm_spans.append((cur, wlen))
            else:
                norm_spans = [(0, wlen)]

            # Enforce token constraints: merge small, split large
            window_chunks: List[str] = []
            buffer = pending_small if pending_small else ""
            pending_small = ""

            for s, e in norm_spans:
                span_txt = win_text[s:e]
                candidate = (buffer + (" " if buffer and span_txt else "") + span_txt).strip()
                if not candidate:
                    continue
                cand_tokens = len(self.encode(candidate))
                if cand_tokens < min_tokens:
                    buffer = candidate
                    continue
                if cand_tokens > self.max_tokens:
                    pieces = _split_span_by_tokens(candidate)
                    for p in pieces:
                        window_chunks.append(p)
                    buffer = ""
                else:
                    window_chunks.append(candidate)
                    buffer = ""

            pending_small = buffer
            chunks.extend(window_chunks)
            pos = win_end

        # Flush leftover
        if pending_small.strip():
            chunks.append(pending_small.strip())

        # Apply token-based overlap
        if self.overlap_tokens > 0 and chunks:
            overlapped: List[str] = []
            for j, ch in enumerate(chunks):
                if j == 0:
                    overlapped.append(ch)
                    continue
                prev = overlapped[-1]
                prev_words = prev.split()
                tail_words = prev_words[-min(len(prev_words), self.overlap_tokens):]
                merged = (" ".join(tail_words) + " " + ch).strip()
                overlapped.append(merged)
            chunks = overlapped

        logger.info(f"chunking: agentic produced {len(chunks)} chunks")
        return chunks
    
    def _to_document_chunks(
        self,
        raw_chunks: List[str],
        paper: Paper,
    ) -> List[DocumentChunk]:
        """Convert raw text chunks to DocumentChunk objects."""
        chunks: List[DocumentChunk] = []
        
        for idx, chunk_text in enumerate(raw_chunks):
            chunk = DocumentChunk(
                id=f"{paper.id}_{idx}",
                text=chunk_text,
                metadata=ChunkMetadata(
                    paper_id=paper.id,
                    chunk_index=idx,
                    source=paper.source,
                    title=paper.title,
                    authors=_format_authors(paper.authors),
                    year=paper.year,
                    doi=paper.doi,
                    url=paper.url,
                ),
            )
            chunks.append(chunk)
        
        return chunks


def _format_authors(authors: List[Any]) -> Optional[str]:
    """Format authors list to string."""
    if not authors:
        return None
    
    names = []
    for author in authors:
        if hasattr(author, "family") and author.family:
            names.append(author.family)
        elif hasattr(author, "name"):
            names.append(author.name.split()[-1])
    
    if not names:
        return None
    
    if len(names) == 1:
        return names[0]
    elif len(names) == 2:
        return f"{names[0]} & {names[1]}"
    else:
        return f"{names[0]} et al."
