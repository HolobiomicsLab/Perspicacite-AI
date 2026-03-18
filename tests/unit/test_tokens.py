"""Tests for token management utilities."""

import pytest

from perspicacite.llm.tokens import (
    calculate_available_tokens,
    count_message_tokens,
    count_tokens,
    estimate_cost,
    get_model_token_limit,
    truncate_messages,
    truncate_to_tokens,
)


class TestCountTokens:
    """Tests for token counting."""

    def test_count_empty(self):
        """Test counting empty text."""
        assert count_tokens("") == 0
        assert count_tokens(None or "") == 0

    def test_count_simple(self):
        """Test counting simple text."""
        # Roughly 4 chars per token
        text = "Hello world"
        count = count_tokens(text)
        assert count > 0
        assert count < len(text)  # Should be less than char count

    def test_count_long(self):
        """Test counting long text."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        count = count_tokens(text)
        assert count > 100

    def test_count_with_model(self):
        """Test counting with specific model."""
        text = "Hello world"
        # Should use tiktoken for GPT models
        count = count_tokens(text, model="gpt-4")
        assert count > 0


class TestCountMessageTokens:
    """Tests for counting message tokens."""

    def test_count_empty_messages(self):
        """Test counting empty message list."""
        assert count_message_tokens([]) == 2  # Just overhead

    def test_count_simple_messages(self):
        """Test counting simple messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        count = count_message_tokens(messages)
        assert count > 8  # 4 per message overhead + content

    def test_count_with_system(self):
        """Test counting with system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        count = count_message_tokens(messages)
        assert count > 8


class TestTruncateToTokens:
    """Tests for text truncation."""

    def test_no_truncation_needed(self):
        """Test when no truncation is needed."""
        text = "Hello world"
        result = truncate_to_tokens(text, max_tokens=100)
        assert result == text

    def test_truncation(self):
        """Test truncation."""
        text = "The quick brown fox jumps over the lazy dog. " * 100
        result = truncate_to_tokens(text, max_tokens=10)
        assert len(result) < len(text)
        assert result.endswith("...")

    def test_truncation_empty(self):
        """Test truncating empty text."""
        result = truncate_to_tokens("", max_tokens=10)
        assert result == ""


class TestTruncateMessages:
    """Tests for message truncation."""

    def test_no_truncation_needed(self):
        """Test when no truncation is needed."""
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = truncate_messages(messages, max_tokens=100)
        assert len(result) == 1

    def test_keep_system_messages(self):
        """Test that system messages are preserved."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = truncate_messages(messages, max_tokens=20)
        # Should keep system message
        assert any(m["role"] == "system" for m in result)

    def test_keep_recent_messages(self):
        """Test that recent messages are preserved."""
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        result = truncate_messages(messages, max_tokens=15)
        # Should keep more recent messages
        assert len(result) >= 1


class TestTokenLimits:
    """Tests for token limit functions."""

    def test_get_model_token_limit(self):
        """Test getting token limits."""
        assert get_model_token_limit("claude-3-5-sonnet-20241022") == 200000
        assert get_model_token_limit("gpt-4o") == 128000
        assert get_model_token_limit("unknown") == 4096  # Default

    def test_calculate_available_tokens(self):
        """Test calculating available tokens."""
        available = calculate_available_tokens(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            max_output_tokens=4096,
        )
        assert available == 4096  # Capped at max_output

    def test_calculate_available_tokens_limited(self):
        """Test calculating when model limit is the constraint."""
        available = calculate_available_tokens(
            model="gpt-4",  # 8k limit
            input_tokens=7000,
            max_output_tokens=4096,
        )
        assert available < 4096  # Limited by model


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_estimate_cost_openai(self):
        """Test cost estimation for OpenAI."""
        cost = estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4o",
        )
        assert cost > 0
        # ~$0.005 for input + $0.0075 for output
        assert 0.01 < cost < 0.02

    def test_estimate_cost_anthropic(self):
        """Test cost estimation for Anthropic."""
        cost = estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="claude-3-5-sonnet-20241022",
        )
        assert cost > 0

    def test_estimate_cost_unknown(self):
        """Test cost estimation for unknown model."""
        cost = estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="unknown-model",
        )
        assert cost > 0  # Uses default pricing
