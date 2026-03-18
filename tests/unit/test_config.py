"""Tests for configuration system."""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from perspicacite.config import load_config
from perspicacite.config.schema import Config


class TestConfigSchema:
    """Tests for Config schema validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.version == "2.0.0"
        assert config.server.port == 5468
        assert config.knowledge_base.chunk_size == 1000

    def test_config_validation(self):
        """Test config validation."""
        # Valid
        Config(version="2.0.0")

        # Invalid version
        with pytest.raises(ValidationError):
            Config(version="1.0.0")

    def test_chunk_config_validation(self):
        """Test chunk config validation."""
        # Valid
        Config(knowledge_base={"chunk_size": 500, "chunk_overlap": 100})

        # Invalid (overlap >= size)
        with pytest.raises(ValidationError):
            Config(knowledge_base={"chunk_size": 500, "chunk_overlap": 500})

    def test_path_expansion(self):
        """Test path expansion."""
        config = Config(database={"path": "~/test.db"})
        assert "~" not in str(config.database.path)
        assert config.database.path.is_absolute()

    def test_to_dict_masks_secrets(self):
        """Test that secrets are masked in to_dict."""
        config = Config(auth={"enabled": True, "token": "secret123"})
        d = config.to_dict()
        assert d["auth"]["token"] == "***"


class TestConfigLoader:
    """Tests for config loader."""

    def test_load_default(self):
        """Test loading default config."""
        config = load_config()
        assert isinstance(config, Config)
        assert config.version == "2.0.0"

    def test_load_from_explicit_path(self, temp_dir: Path):
        """Test loading from explicit path."""
        config_path = temp_dir / "config.yml"
        with open(config_path, "w") as f:
            yaml.dump({"server": {"port": 9999}}, f)

        config = load_config(str(config_path))
        assert config.server.port == 9999

    def test_load_from_env_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("PERSPICACITE_SERVER_PORT", "7777")

        config = load_config()
        assert config.server.port == 7777

    def test_load_invalid_yaml(self, temp_dir: Path):
        """Test loading invalid YAML."""
        config_path = temp_dir / "bad_config.yml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: [")

        with pytest.raises(ValueError):
            load_config(str(config_path))

    def test_load_invalid_config(self, temp_dir: Path):
        """Test loading invalid config values."""
        config_path = temp_dir / "config.yml"
        with open(config_path, "w") as f:
            yaml.dump({"version": "1.0.0"}, f)  # Invalid version

        with pytest.raises(ValueError):
            load_config(str(config_path))


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_server_host(self, monkeypatch):
        """Test SERVER_HOST override."""
        monkeypatch.setenv("PERSPICACITE_SERVER_HOST", "127.0.0.1")
        config = load_config()
        assert config.server.host == "127.0.0.1"

    def test_log_level(self, monkeypatch):
        """Test LOG_LEVEL override."""
        monkeypatch.setenv("PERSPICACITE_LOG_LEVEL", "DEBUG")
        config = load_config()
        assert config.logging.level == "DEBUG"

    def test_auth_token(self, monkeypatch):
        """Test AUTH_TOKEN override."""
        monkeypatch.setenv("PERSPICACITE_AUTH_TOKEN", "mytoken")
        config = load_config()
        assert config.auth.token == "mytoken"

    def test_boolean_parsing(self, monkeypatch):
        """Test boolean env var parsing."""
        monkeypatch.setenv("PERSPICACITE_MCP_ENABLED", "false")
        config = load_config()
        assert config.mcp.enabled is False

        monkeypatch.setenv("PERSPICACITE_MCP_ENABLED", "True")
        config = load_config()
        assert config.mcp.enabled is True

    def test_integer_parsing(self, monkeypatch):
        """Test integer env var parsing."""
        monkeypatch.setenv("PERSPICACITE_SERVER_PORT", "8080")
        config = load_config()
        assert config.server.port == 8080
