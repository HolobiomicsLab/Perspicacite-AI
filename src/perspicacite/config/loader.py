"""Configuration loader with layered loading."""

import os
from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import ValidationError

from perspicacite.config.schema import Config


def get_config_search_paths() -> list[Path]:
    """Get list of paths to search for config files."""
    paths = []

    # 1. Explicit env variable
    if env_path := os.environ.get("PERSPICACITE_CONFIG_PATH"):
        paths.append(Path(env_path).expanduser())

    # 2. Current directory
    paths.append(Path.cwd() / "config.yml")

    # 3. User config directory (XDG)
    if xdg_config := os.environ.get("XDG_CONFIG_HOME"):
        paths.append(Path(xdg_config) / "perspicacite" / "config.yml")
    else:
        paths.append(Path.home() / ".config" / "perspicacite" / "config.yml")

    return paths


def load_yaml_file(path: Path) -> Optional[dict[str, Any]]:
    """Load YAML file if it exists."""
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    except IOError as e:
        raise ValueError(f"Cannot read config file {path}: {e}") from e


def load_from_env() -> dict[str, Any]:
    """Load configuration overrides from environment variables."""
    overrides = {}

    # Map of env var -> config path
    env_mappings = {
        "PERSPICACITE_SERVER_HOST": ("server", "host"),
        "PERSPICACITE_SERVER_PORT": ("server", "port"),
        "PERSPICACITE_MCP_ENABLED": ("mcp", "enabled"),
        "PERSPICACITE_MCP_PORT": ("mcp", "port"),
        "PERSPICACITE_DB_PATH": ("database", "path"),
        "PERSPICACITE_DB_CHROMA_PATH": ("database", "chroma_path"),
        "PERSPICACITE_LLM_DEFAULT_PROVIDER": ("llm", "default_provider"),
        "PERSPICACITE_LLM_DEFAULT_MODEL": ("llm", "default_model"),
        "PERSPICACITE_LOG_LEVEL": ("logging", "level"),
        "PERSPICACITE_AUTH_ENABLED": ("auth", "enabled"),
        "PERSPICACITE_AUTH_TOKEN": ("auth", "token"),
    }

    for env_var, (section, key) in env_mappings.items():
        if value := os.environ.get(env_var):
            # Convert types
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)

            if section not in overrides:
                overrides[section] = {}
            overrides[section][key] = value

    return overrides


def load_config(path: Optional[str] = None) -> Config:
    """
    Load configuration from layered sources.

    Loading order (later overrides earlier):
    1. Built-in defaults (from Config model)
    2. Config file(s) from search paths
    3. Environment variables

    Args:
        path: Optional explicit config file path

    Returns:
        Validated Config instance

    Raises:
        ValueError: If config file is invalid
        ValidationError: If config fails Pydantic validation
    """
    # Start with defaults (from Config model defaults)
    config_dict: dict[str, Any] = {}

    # Load from config file(s)
    if path:
        # Explicit path
        file_config = load_yaml_file(Path(path).expanduser())
        if file_config:
            config_dict.update(file_config)
    else:
        # Search paths
        for search_path in get_config_search_paths():
            file_config = load_yaml_file(search_path)
            if file_config:
                config_dict.update(file_config)
                break  # Use first found

    # Apply environment overrides
    env_overrides = load_from_env()
    _deep_merge(config_dict, env_overrides)

    # Create and validate config
    try:
        return Config(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}") from e


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Deep merge override into base dict."""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_config(config: Config, path: Optional[Path] = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Config to save
        path: Path to save to (default: ~/.config/perspicacite/config.yml)
    """
    if path is None:
        config_dir = Path.home() / ".config" / "perspicacite"
        config_dir.mkdir(parents=True, exist_ok=True)
        path = config_dir / "config.yml"

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and save
    config_dict = config.model_dump()
    # Remove defaults to keep file minimal
    config_dict = _remove_defaults(config_dict)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _remove_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """Remove default values to keep config file minimal."""
    # For now, keep all values
    # TODO: Implement comparison with default Config to remove defaults
    return data
