"""Ollama num_ctx forwarding (local-model context window fix).

Ollama silently defaults num_ctx to 2048, which truncates long RAG synthesis
prompts and makes the model emit empty output. The client must forward the
configured num_ctx for the Ollama provider (and nothing for others).
"""
import unittest

from perspicacite.config.schema import LLMConfig
from perspicacite.llm.client import AsyncLLMClient


class TestOllamaNumCtx(unittest.TestCase):
    def test_config_default(self):
        assert LLMConfig().ollama_num_ctx == 8192

    def test_config_override(self):
        assert LLMConfig(ollama_num_ctx=16384).ollama_num_ctx == 16384

    def test_ollama_provider_gets_num_ctx(self):
        client = AsyncLLMClient(LLMConfig())
        assert client._provider_extra_params("ollama") == {"num_ctx": 8192}

    def test_ollama_respects_config_value(self):
        client = AsyncLLMClient(LLMConfig(ollama_num_ctx=4096))
        assert client._provider_extra_params("ollama") == {"num_ctx": 4096}

    def test_non_ollama_providers_unchanged(self):
        client = AsyncLLMClient(LLMConfig())
        for provider in ("openai", "anthropic", "deepseek", "minimax"):
            assert client._provider_extra_params(provider) == {}


if __name__ == "__main__":
    unittest.main()
