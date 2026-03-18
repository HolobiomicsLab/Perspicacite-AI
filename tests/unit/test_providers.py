"""Tests for LLM provider registry."""

import os

import pytest

from perspicacite.llm.providers import (
    check_all_providers,
    get_available_providers,
    get_default_model_for_provider,
    get_max_tokens,
    get_models_for_provider,
    get_provider_info,
    supports_streaming,
    supports_tools,
    validate_provider_config,
)


class TestProviderRegistry:
    """Tests for provider registry functions."""

    def test_get_models_for_provider(self):
        """Test getting models."""
        models = get_models_for_provider("anthropic")
        assert "claude-3-5-sonnet-20241022" in models
        assert len(models) > 0

    def test_get_models_for_unknown_provider(self):
        """Test getting models for unknown provider."""
        models = get_models_for_provider("unknown")
        assert models == []

    def test_get_provider_info(self):
        """Test getting provider info."""
        info = get_provider_info("openai")
        assert info["supports_streaming"] is True
        assert info["supports_tools"] is True
        assert "gpt-4o" in info["models"]

    def test_supports_streaming(self):
        """Test streaming support check."""
        assert supports_streaming("anthropic") is True
        assert supports_streaming("unknown") is False

    def test_supports_tools(self):
        """Test tools support check."""
        assert supports_tools("openai") is True
        assert supports_tools("deepseek") is False

    def test_get_max_tokens(self):
        """Test getting max tokens."""
        assert get_max_tokens("anthropic") == 4096
        assert get_max_tokens("unknown") == 4096  # Default

    def test_get_default_model(self):
        """Test getting default model."""
        model = get_default_model_for_provider("anthropic")
        assert model == "claude-3-5-sonnet-20241022"

    def test_get_default_model_unknown(self):
        """Test getting default model for unknown provider."""
        model = get_default_model_for_provider("unknown")
        assert model is None


class TestProviderValidation:
    """Tests for provider validation."""

    def test_validate_unknown_provider(self):
        """Test validating unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            validate_provider_config("unknown", "model")

    def test_validate_unknown_model(self):
        """Test validating unknown model."""
        with pytest.raises(ValueError, match="Unknown model"):
            validate_provider_config("anthropic", "unknown-model")

    def test_validate_missing_api_key(self, monkeypatch):
        """Test validating without API key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key not configured"):
            validate_provider_config("anthropic", "claude-3-5-sonnet-20241022")

    def test_validate_success(self, monkeypatch):
        """Test successful validation."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Should not raise
        validate_provider_config("anthropic", "claude-3-5-sonnet-20241022")


class TestAvailableProviders:
    """Tests for available providers."""

    def test_get_available_providers(self, monkeypatch):
        """Test getting available providers."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        providers = get_available_providers()
        assert "anthropic" in providers
        assert "openai" not in providers
        assert "ollama" in providers  # No key needed

    def test_check_all_providers(self, monkeypatch):
        """Test checking all providers."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        results = check_all_providers()
        assert "anthropic" in results
        assert results["anthropic"]["configured"] is True
        assert results["anthropic"]["api_key_set"] is True

        assert "ollama" in results
        assert results["ollama"]["configured"] is True  # No key needed
