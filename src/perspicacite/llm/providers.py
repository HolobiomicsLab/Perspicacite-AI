"""LLM provider registry and utilities."""

import os
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.llm.providers")

# Provider registry with metadata
PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "anthropic": {
        "models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "env_key": "ANTHROPIC_API_KEY",
        "supports_streaming": True,
        "supports_tools": True,
        "max_tokens": 4096,
    },
    "openai": {
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
        "env_key": "OPENAI_API_KEY",
        "supports_streaming": True,
        "supports_tools": True,
        "max_tokens": 4096,
    },
    "deepseek": {
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "env_key": "DEEPSEEK_API_KEY",
        "supports_streaming": True,
        "supports_tools": False,
        "max_tokens": 4096,
    },
    "gemini": {
        "models": [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ],
        "env_key": "GOOGLE_API_KEY",
        "supports_streaming": True,
        "supports_tools": True,
        "max_tokens": 4096,
    },
    "ollama": {
        "models": [
            "llama3.1",
            "llama3",
            "mistral",
            "phi3",
        ],
        "env_key": None,  # Ollama doesn't require API key
        "supports_streaming": True,
        "supports_tools": False,
        "max_tokens": 4096,
        "requires_base_url": True,
    },
}


def get_available_providers() -> list[str]:
    """
    Get list of providers that have API keys configured.

    Returns:
        List of provider names
    """
    available = []
    for provider, config in PROVIDER_REGISTRY.items():
        env_key = config.get("env_key")
        if env_key is None or os.environ.get(env_key):
            available.append(provider)
    return available


def get_models_for_provider(provider: str) -> list[str]:
    """
    Get available models for a provider.

    Args:
        provider: Provider name

    Returns:
        List of model names
    """
    if provider not in PROVIDER_REGISTRY:
        return []
    return PROVIDER_REGISTRY[provider]["models"]


def validate_provider_config(provider: str, model: str) -> None:
    """
    Validate provider and model configuration.

    Args:
        provider: Provider name
        model: Model name

    Raises:
        ValueError: If provider or model is invalid
    """
    if provider not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider: {provider}. Available: {available}")

    provider_config = PROVIDER_REGISTRY[provider]

    # Check API key
    env_key = provider_config.get("env_key")
    if env_key and not os.environ.get(env_key):
        raise ValueError(
            f"API key not configured for {provider}. "
            f"Set {env_key} environment variable."
        )

    # Check model
    if model not in provider_config["models"]:
        available = ", ".join(provider_config["models"])
        raise ValueError(
            f"Unknown model: {model} for {provider}. Available: {available}"
        )


def get_provider_info(provider: str) -> dict[str, Any]:
    """
    Get information about a provider.

    Args:
        provider: Provider name

    Returns:
        Provider configuration dict
    """
    return PROVIDER_REGISTRY.get(provider, {})


def supports_streaming(provider: str) -> bool:
    """Check if provider supports streaming."""
    info = get_provider_info(provider)
    return info.get("supports_streaming", False)


def supports_tools(provider: str) -> bool:
    """Check if provider supports tool calling."""
    info = get_provider_info(provider)
    return info.get("supports_tools", False)


def get_max_tokens(provider: str) -> int:
    """Get max tokens for a provider."""
    info = get_provider_info(provider)
    return info.get("max_tokens", 4096)


def format_model_display_name(provider: str, model: str) -> str:
    """Format a nice display name for a model."""
    return f"{provider.title()}: {model}"


def get_default_model_for_provider(provider: str) -> str | None:
    """Get the default model for a provider."""
    models = get_models_for_provider(provider)
    return models[0] if models else None


def check_all_providers() -> dict[str, dict[str, Any]]:
    """
    Check status of all providers.

    Returns:
        Dict mapping provider name to status info
    """
    results = {}
    for provider, config in PROVIDER_REGISTRY.items():
        env_key = config.get("env_key")
        api_key_set = env_key is not None and bool(os.environ.get(env_key))
        no_key_needed = env_key is None

        results[provider] = {
            "configured": api_key_set or no_key_needed,
            "api_key_env": env_key,
            "api_key_set": api_key_set,
            "models": config["models"],
            "supports_streaming": config["supports_streaming"],
            "supports_tools": config["supports_tools"],
        }

    return results
