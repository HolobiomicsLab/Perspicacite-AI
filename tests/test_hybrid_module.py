"""Tests for the hybrid retrieval module."""

import pytest
import numpy as np
from dataclasses import dataclass
import sys
import os

# Add src to path and import directly to avoid __init__.py dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import hybrid module functions directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hybrid", 
    os.path.join(os.path.dirname(__file__), "..", "src", "perspicacite", "retrieval", "hybrid.py")
)
hybrid_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hybrid_module)

normalize_scores = hybrid_module.normalize_scores
combine_scores = hybrid_module.combine_scores
compute_bm25_scores = hybrid_module.compute_bm25_scores
hybrid_retrieval = hybrid_module.hybrid_retrieval
determine_weights_with_llm = hybrid_module.determine_weights_with_llm


def test_normalize_scores_basic():
    """Test basic score normalization."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize_scores(scores)
    
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert len(normalized) == len(scores)
    
    print("✅ Normalize scores basic test passed")


def test_combine_scores():
    """Test combining scores."""
    vector_scores = np.array([0.9, 0.7, 0.5])
    bm25_scores = np.array([0.3, 0.9, 0.6])
    
    combined = combine_scores(vector_scores, bm25_scores, 0.5, 0.5)
    
    assert 0 <= combined.min() <= combined.max() <= 1
    print("✅ Combine scores test passed")


def test_compute_bm25_scores():
    """Test BM25 score computation."""
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
    ]
    
    scores = compute_bm25_scores(documents, "neural networks")
    
    assert len(scores) == len(documents)
    assert scores[2] > scores[0]  # Doc 2 has the keywords
    
    print("✅ Compute BM25 scores test passed")


if __name__ == "__main__":
    print("\n=== Hybrid Module Tests ===\n")
    test_normalize_scores_basic()
    test_combine_scores()
    test_compute_bm25_scores()
    print("\n=== All tests passed! ===\n")
