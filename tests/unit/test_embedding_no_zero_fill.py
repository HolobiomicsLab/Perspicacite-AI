"""Regression test for the per-item embedding fallback.

The batch -> per-item retry exists so one over-long input cannot drop a whole
paper. It must not, however, substitute a zero vector for an input that keeps
failing: a zero vector is stored happily by Chroma and then sits at cosine
distance 1.0 from everything, so every query returns the same passages at a
constant score while the ingest reports success.
"""

import pytest

from perspicacite.llm.embeddings import LiteLLMEmbeddingProvider


class _FakeLiteLLM:
    """Fails on any multi-input batch, and on the single text named as poison."""

    def __init__(self, poison: str, dimension: int = 4):
        self.poison = poison
        self.dimension = dimension
        self.single_calls: list[str] = []

    async def aembedding(self, model: str, input: list[str]):  # noqa: A002
        if len(input) > 1:
            raise RuntimeError("batch too large")
        text = input[0]
        self.single_calls.append(text)
        if text == self.poison:
            raise RuntimeError("input exceeds the model token cap")
        return {"data": [{"embedding": [1.0] + [0.0] * (self.dimension - 1)}]}


@pytest.mark.asyncio
async def test_persistent_item_failure_raises_instead_of_zero_filling(monkeypatch):
    """A text that keeps failing must abort the embed, not become a zero vector."""
    provider = LiteLLMEmbeddingProvider()
    fake = _FakeLiteLLM(poison="poison")
    monkeypatch.setattr(provider, "_get_litellm", lambda: fake)

    with pytest.raises(RuntimeError, match="token cap"):
        await provider.embed(["fine", "poison"])

    assert "poison" in fake.single_calls, "the per-item retry should still run"


@pytest.mark.asyncio
async def test_batch_fallback_still_recovers_when_every_item_succeeds(monkeypatch):
    """The retry keeps its original purpose: a failed batch is embedded item by item."""
    provider = LiteLLMEmbeddingProvider()
    fake = _FakeLiteLLM(poison="__never__")
    monkeypatch.setattr(provider, "_get_litellm", lambda: fake)

    result = await provider.embed(["one", "two"])

    assert len(result) == 2
    assert all(vector[0] == 1.0 for vector in result)
    assert fake.single_calls == ["one", "two"]
