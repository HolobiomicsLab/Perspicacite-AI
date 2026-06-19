"""Regression: BibTeX-import KBs must embed local models via the factory.

Bug: ``pipeline/bibtex_kb.py`` hardcoded ``LiteLLMEmbeddingProvider`` for the embedding
provider, so a local sentence-transformers model (e.g. the default ``all-MiniLM-L6-v2``)
was sent to LiteLLM, which raised ``LLM Provider NOT provided`` and produced KBs with 0
chunks. The fix routes through :func:`create_embedding_provider`, which selects the local
``SentenceTransformerEmbeddingProvider`` for such models (lazy-loaded, no network).
"""

from __future__ import annotations

import inspect

from perspicacite.llm.embeddings import (
    FallbackEmbeddingProvider,
    LiteLLMEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    create_embedding_provider,
)


def test_local_minilm_routes_to_sentence_transformer():
    # the model SentenceTransformerEmbeddingProvider is lazy, so this does not download.
    provider = create_embedding_provider("all-MiniLM-L6-v2", use_local_fallback=False)
    assert isinstance(provider, SentenceTransformerEmbeddingProvider)
    assert provider.model_name == "all-MiniLM-L6-v2"


def test_api_model_routes_to_litellm():
    provider = create_embedding_provider("text-embedding-3-small", use_local_fallback=False)
    assert isinstance(provider, LiteLLMEmbeddingProvider)


def test_api_model_with_local_fallback_is_wrapped():
    provider = create_embedding_provider("text-embedding-3-small", use_local_fallback=True)
    assert isinstance(provider, FallbackEmbeddingProvider)


def test_bibtex_kb_uses_factory_not_hardcoded_litellm():
    # Lock the fix: both bibtex-KB builders must use the factory, not LiteLLM directly.
    from perspicacite.pipeline import bibtex_kb

    for fn in (bibtex_kb.create_kb_from_bibtex, bibtex_kb.add_bibtex_to_existing_kb):
        src = inspect.getsource(fn)
        assert "create_embedding_provider(" in src, f"{fn.__name__} should use the factory"
        assert "LiteLLMEmbeddingProvider(" not in src, (
            f"{fn.__name__} must not hardcode LiteLLMEmbeddingProvider"
        )
