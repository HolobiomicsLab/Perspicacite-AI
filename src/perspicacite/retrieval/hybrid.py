"""Hybrid retrieval combining vector similarity and BM25.

This module implements hybrid retrieval as described in the release package:
- Vector similarity for semantic matching
- BM25 for lexical/keyword matching
- Optional LLM-based weight determination
- Score normalization and combination
"""

from typing import Any, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.retrieval.hybrid")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-1 range using min-max normalization."""
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.ones_like(scores) * 0.5  # All equal, return middle value
    
    return (scores - min_score) / (max_score - min_score)


def combine_scores(
    vector_scores: np.ndarray,
    bm25_scores: np.ndarray,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> np.ndarray:
    """
    Combine vector and BM25 scores with given weights.
    
    Args:
        vector_scores: Vector similarity scores
        bm25_scores: BM25 scores
        vector_weight: Weight for vector scores (0-1)
        bm25_weight: Weight for BM25 scores (0-1)
        
    Returns:
        Combined scores array
    """
    # Normalize both score arrays
    norm_vector = normalize_scores(vector_scores)
    norm_bm25 = normalize_scores(bm25_scores)
    
    # Combine with weights
    combined = vector_weight * norm_vector + bm25_weight * norm_bm25
    
    return combined


async def determine_weights_with_llm(
    query: str,
    llm: Any,
) -> Tuple[float, float]:
    """
    Use LLM to determine optimal weights for vector and BM25 retrieval.
    
    Ported from: core/hybrid_retrieval.py::determine_weights_with_llm()
    
    Args:
        query: The search query
        llm: LLM client
        
    Returns:
        Tuple of (vector_weight, bm25_weight)
    """
    system_prompt = """You are a weight determination system for hybrid document retrieval.
Your task is to analyze a query and output ONLY two numbers separated by a comma, representing the optimal weights for vector and BM25 retrieval.
The numbers must sum to 1.0.

Consider these factors in your analysis:
1. Higher vector weight (e.g., 0.7) if the query requires semantic understanding
2. Higher BM25 weight (e.g., 0.7) if the query contains specific named entities or exact terms
3. Balanced weights (0.5,0.5) if both aspects are equally important

IMPORTANT: Your response must be EXACTLY in this format: number,number
Example responses:
0.7,0.3
0.3,0.7
0.5,0.5

DO NOT include any explanation or additional text in your response.
"""
    
    try:
        response = await llm.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query to analyze: {query}"},
            ],
            temperature=0.3,
            max_tokens=20,
        )
        
        # Clean the response to ensure it only contains numbers and comma
        cleaned = ''.join(c for c in response.strip() if c.isdigit() or c == ',' or c == '.')
        parts = cleaned.split(',')
        
        if len(parts) >= 2:
            vector_weight = float(parts[0])
            bm25_weight = float(parts[1])
            
            # Ensure weights sum to 1.0
            total = vector_weight + bm25_weight
            if total == 0:
                logger.warning("LLM returned zero weights, using defaults")
                return 0.5, 0.5
            
            vector_weight /= total
            bm25_weight /= total
            
            logger.info(
                "hybrid_weights_determined",
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
            )
            return vector_weight, bm25_weight
        else:
            logger.warning("Could not parse LLM weights, using defaults")
            return 0.5, 0.5
            
    except Exception as e:
        logger.warning("hybrid_weight_determination_error", error=str(e))
        return 0.5, 0.5


def compute_bm25_scores(
    documents: List[str],
    query: str,
) -> np.ndarray:
    """
    Compute BM25 scores for documents given a query.
    
    Args:
        documents: List of document texts
        query: Search query
        
    Returns:
        Array of BM25 scores
    """
    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    return np.array(scores)


async def hybrid_retrieval(
    query: str,
    documents: List[Any],
    vector_scores: List[float],
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    use_llm_weights: bool = False,
    llm: Any = None,
) -> List[Tuple[Any, float]]:
    """
    Perform hybrid retrieval combining vector similarity and BM25.
    
    Ported from: core/hybrid_retrieval.py::hybrid_retrieval()
    
    Args:
        query: Search query
        documents: List of documents (with page_content attribute)
        vector_scores: Vector similarity scores for documents
        vector_weight: Weight for vector scores
        bm25_weight: Weight for BM25 scores
        use_llm_weights: Whether to use LLM to determine weights
        llm: LLM client (required if use_llm_weights is True)
        
    Returns:
        List of (document, combined_score) tuples sorted by score
    """
    logger.info("hybrid_retrieval_start", query=query[:100], num_docs=len(documents))
    
    # Determine weights using LLM if requested
    if use_llm_weights and llm is not None:
        vector_weight, bm25_weight = await determine_weights_with_llm(query, llm)
    
    # Extract document texts
    doc_texts = []
    for doc in documents:
        if hasattr(doc, 'chunk') and hasattr(doc.chunk, 'text'):
            text = doc.chunk.text
        elif hasattr(doc, 'page_content'):
            text = doc.page_content
        elif hasattr(doc, 'content'):
            text = str(doc.content)
        else:
            text = str(doc)
        doc_texts.append(text)
    
    # Compute BM25 scores
    bm25_scores = compute_bm25_scores(doc_texts, query)
    vector_scores_arr = np.array(vector_scores)
    
    logger.info(
        "hybrid_scores_computed",
        vector_scores_range=(vector_scores_arr.min(), vector_scores_arr.max()),
        bm25_scores_range=(bm25_scores.min(), bm25_scores.max()),
    )
    
    # Combine scores
    combined_scores = combine_scores(
        vector_scores_arr,
        bm25_scores,
        vector_weight,
        bm25_weight,
    )
    
    # Create result tuples
    results = list(zip(documents, combined_scores))
    
    # Sort by combined score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(
        "hybrid_retrieval_complete",
        top_score=results[0][1] if results else 0,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
    
    return results
