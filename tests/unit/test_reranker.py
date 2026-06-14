"""Tests for CrossEncoderReranker."""

from perspicacite.retrieval.reranker import CrossEncoderReranker


def test_reranker_uses_explicit_model_name():
    r = CrossEncoderReranker(model_name="my-org/custom-reranker")
    assert r.model_name == "my-org/custom-reranker"


def test_reranker_default_model_name():
    r = CrossEncoderReranker()
    assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-12-v2"
