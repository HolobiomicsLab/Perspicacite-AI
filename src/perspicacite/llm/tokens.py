"""Token counting and management utilities."""

from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.llm.tokens")

# Approximate tokens per character for different languages
TOKENS_PER_CHAR = {
    "en": 0.25,  # English
    "default": 0.25,
}

# Model-specific token limits
MODEL_TOKEN_LIMITS = {
    # Anthropic
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
    # DeepSeek
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
    # Gemini
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-1.0-pro": 32000,
}

# Default output limits
DEFAULT_MAX_OUTPUT_TOKENS = 4096


def count_tokens(text: str, model: str | None = None) -> int:
    """
    Estimate token count for text.

    This is a rough approximation. For accurate counts, use the model's
    tokenizer, but this is sufficient for context window management.

    Args:
        text: Text to count
        model: Model name (for model-specific estimation)

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Try to use tiktoken for OpenAI models
    if model and ("gpt" in model or model.startswith("text-")):
        try:
            import tiktoken

            encoder = tiktoken.encoding_for_model(model)
            return len(encoder.encode(text))
        except Exception:
            pass

    # Fall back to character-based estimation
    # English text is roughly 4 characters per token
    estimated = len(text) // 4

    # Add overhead for special characters
    special_chars = sum(1 for c in text if ord(c) > 127)
    estimated += special_chars // 2

    return estimated


def count_message_tokens(messages: list[dict[str, str]], model: str | None = None) -> int:
    """
    Estimate token count for a list of messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name

    Returns:
        Estimated token count
    """
    total = 0

    # Base overhead per message
    per_message_overhead = 4

    for message in messages:
        total += per_message_overhead
        total += count_tokens(message.get("role", ""), model)
        total += count_tokens(message.get("content", ""), model)

    # Add overhead for the message list format
    total += 2

    return total


def truncate_to_tokens(text: str, max_tokens: int, model: str | None = None) -> str:
    """
    Truncate text to fit within max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        model: Model name

    Returns:
        Truncated text
    """
    if not text:
        return text

    estimated_chars = max_tokens * 4  # Rough: 4 chars per token

    if len(text) <= estimated_chars:
        return text

    # Truncate and add ellipsis
    truncated = text[:estimated_chars].rsplit(" ", 1)[0]
    return truncated + "..."


def truncate_messages(
    messages: list[dict[str, str]],
    max_tokens: int,
    model: str | None = None,
) -> list[dict[str, str]]:
    """
    Truncate messages to fit within max_tokens.

    Keeps system messages and most recent messages, truncates middle.

    Args:
        messages: List of messages
        max_tokens: Maximum tokens allowed
        model: Model name

    Returns:
        Truncated message list
    """
    if not messages:
        return messages

    current_tokens = count_message_tokens(messages, model)
    if current_tokens <= max_tokens:
        return messages

    # Separate system messages (always keep)
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    system_tokens = count_message_tokens(system_messages, model)
    available_for_others = max_tokens - system_tokens

    if available_for_others <= 0:
        # Only keep system messages, truncated
        return system_messages

    # Keep most recent messages that fit
    kept_messages = []
    current_count = 0

    for message in reversed(other_messages):
        msg_tokens = count_message_tokens([message], model)
        if current_count + msg_tokens <= available_for_others:
            kept_messages.insert(0, message)
            current_count += msg_tokens
        else:
            break

    return system_messages + kept_messages


def get_model_token_limit(model: str) -> int:
    """Get token limit for a model."""
    return MODEL_TOKEN_LIMITS.get(model, 4096)


def calculate_available_tokens(
    model: str,
    input_tokens: int,
    max_output_tokens: int | None = None,
) -> int:
    """
    Calculate available tokens for output.

    Args:
        model: Model name
        input_tokens: Tokens in input
        max_output_tokens: Desired output tokens

    Returns:
        Available tokens for output
    """
    model_limit = get_model_token_limit(model)
    max_output = max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS

    available = model_limit - input_tokens
    return min(available, max_output)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """
    Estimate API cost for a request.

    Args:
        input_tokens: Input token count
        output_tokens: Output token count
        model: Model name

    Returns:
        Estimated cost in USD
    """
    # Pricing per 1K tokens (approximate, update as needed)
    pricing = {
        # Anthropic
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        # OpenAI
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        # DeepSeek
        "deepseek-chat": {"input": 0.00014, "output": 0.00028},
    }

    model_pricing = pricing.get(model, {"input": 0.01, "output": 0.03})

    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]

    return input_cost + output_cost
