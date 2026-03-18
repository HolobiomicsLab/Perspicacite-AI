"""Tests for LLM client."""

import pytest

from perspicacite.config.schema import LLMConfig, LLMProviderConfig
from perspicacite.llm.client import AsyncLLMClient


class TestAsyncLLMClient:
    """Tests for AsyncLLMClient."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        config = LLMConfig(
            default_provider="anthropic",
            providers={
                "anthropic": LLMProviderConfig(
                    base_url="https://api.anthropic.com",
                    timeout=60,
                    max_retries=3,
                ),
                "openai": LLMProviderConfig(
                    base_url="https://api.openai.com/v1",
                    timeout=60,
                    max_retries=3,
                ),
            },
        )
        return AsyncLLMClient(config)

    def test_init(self, client):
        """Test client initialization."""
        assert client.config.default_provider == "anthropic"

    def test_build_model_string(self, client):
        """Test model string building."""
        assert client._build_model_string("anthropic", "claude-3") == "anthropic/claude-3"
        assert client._build_model_string("openai", "gpt-4") == "openai/gpt-4"

    def test_get_provider_config(self, client):
        """Test getting provider config."""
        config = client._get_provider_config("anthropic")
        assert config.timeout == 60
        assert config.max_retries == 3

    def test_get_provider_config_unknown(self, client):
        """Test getting unknown provider config."""
        with pytest.raises(ValueError, match="Unknown provider"):
            client._get_provider_config("unknown")

    @pytest.mark.asyncio
    async def test_complete_mock(self, client, monkeypatch):
        """Test completion with mocked LiteLLM."""
        # Mock the litellm response
        class MockResponse:
            class Choice:
                class Message:
                    content = "Test response"
                message = Message()
            choices = [Choice()]
            usage = {"prompt_tokens": 10, "completion_tokens": 5}

        async def mock_acompletion(*args, **kwargs):
            return MockResponse()

        # Mock litellm module
        import litellm
        original = litellm.acompletion
        litellm.acompletion = mock_acompletion

        try:
            messages = [{"role": "user", "content": "Hello"}]
            response = await client.complete(
                messages=messages,
                model="claude-3-5-sonnet-20241022",
                provider="anthropic",
            )
            assert response == "Test response"
        finally:
            litellm.acompletion = original

    @pytest.mark.asyncio
    async def test_stream_mock(self, client, monkeypatch):
        """Test streaming with mocked LiteLLM."""
        class MockChunk:
            class Choice:
                class Delta:
                    content = "Hello"
                delta = Delta()
            choices = [Choice()]

        async def mock_acompletion(*args, **kwargs):
            async def generator():
                yield MockChunk()
            return generator()

        import litellm
        original = litellm.acompletion
        litellm.acompletion = mock_acompletion

        try:
            messages = [{"role": "user", "content": "Hello"}]
            chunks = []
            async for chunk in client.stream(
                messages=messages,
                model="claude-3-5-sonnet-20241022",
                provider="anthropic",
            ):
                chunks.append(chunk)
            assert chunks == ["Hello"]
        finally:
            litellm.acompletion = original


class TestLLMClientFallback:
    """Tests for fallback functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        config = LLMConfig(
            default_provider="anthropic",
            providers={
                "anthropic": LLMProviderConfig(
                    base_url="https://api.anthropic.com",
                    timeout=60,
                    max_retries=1,
                ),
            },
        )
        return AsyncLLMClient(config)

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, client, monkeypatch):
        """Test fallback when primary fails."""
        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            model = kwargs.get("model")
            if model == "primary-model":
                raise Exception("Primary failed")
            return "Fallback response"

        monkeypatch.setattr(client, "complete", mock_complete)

        messages = [{"role": "user", "content": "Hello"}]
        response = await client.complete_with_fallback(
            messages=messages,
            primary_model="primary-model",
            fallback_model="fallback-model",
        )

        assert response == "Fallback response"
        assert call_count == 2  # Primary + fallback
