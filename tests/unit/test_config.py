"""Tests for configuration system."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from perspicacite.config import load_config
from perspicacite.config.schema import Config, PDFDownloadConfig


def test_max_pdf_attach_bytes_default():
    """Default cap is 30 MB so Zotero free-tier (300 MB) users don't blow
    18% of their quota on a single review article."""
    cfg = PDFDownloadConfig()
    assert cfg.max_pdf_attach_bytes == 30 * 1024 * 1024


def test_max_pdf_attach_bytes_override():
    cfg = PDFDownloadConfig(max_pdf_attach_bytes=10 * 1024 * 1024)
    assert cfg.max_pdf_attach_bytes == 10 * 1024 * 1024


def test_max_pdf_attach_bytes_zero_disables_cap():
    """0 is a sentinel meaning 'no cap' — must validate cleanly."""
    cfg = PDFDownloadConfig(max_pdf_attach_bytes=0)
    assert cfg.max_pdf_attach_bytes == 0


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


def test_config_reranker_model_default(tmp_path):
    from perspicacite.config.loader import load_config

    cfg_path = tmp_path / "c.yml"
    cfg_path.write_text("server:\n  port: 5468\n")
    cfg = load_config(str(cfg_path))
    assert cfg.rag_modes.reranker_model == "cross-encoder/ms-marco-MiniLM-L-12-v2"


def test_config_map_reduce_max_papers_default(tmp_path):
    from perspicacite.config.loader import load_config

    cfg_path = tmp_path / "c.yml"
    cfg_path.write_text("server:\n  port: 5468\n")
    cfg = load_config(str(cfg_path))
    assert cfg.rag_modes.agentic.map_reduce_max_papers == 8


def test_config_has_contradiction_settings():
    from perspicacite.config.schema import Config

    cfg = Config()
    assert cfg.rag_modes.contradiction is not None
    # mirrors `advanced`: hybrid on, rerank on, no planning
    assert cfg.rag_modes.contradiction.use_hybrid is True
    assert cfg.rag_modes.contradiction.rerank is True
    assert cfg.rag_modes.contradiction.enable_planning is False


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


def test_zotero_config_defaults() -> None:
    from perspicacite.config.schema import Config
    cfg = Config()
    assert cfg.zotero.enabled is False
    assert cfg.zotero.api_key == ""
    assert cfg.zotero.library_id == ""
    assert cfg.zotero.library_type == "user"
    assert cfg.zotero.collection_key == ""


def test_zotero_config_override() -> None:
    from perspicacite.config.schema import Config, ZoteroConfig
    cfg = Config(zotero=ZoteroConfig(
        enabled=True, api_key="k", library_id="123",
        library_type="group", collection_key="ABC",
    ))
    assert cfg.zotero.enabled is True
    assert cfg.zotero.library_type == "group"


def test_anchor_config_defaults():
    from perspicacite.config.schema import Config

    cfg = Config()
    assert cfg.anchor.strict is False
    assert cfg.anchor.near_threshold == 0.9
    assert cfg.anchor.audit_dir is None


def test_anchor_config_near_threshold_bounds():
    import pytest
    from pydantic import ValidationError

    from perspicacite.config.schema import AnchorConfig

    AnchorConfig(near_threshold=0.0)
    AnchorConfig(near_threshold=1.0)
    with pytest.raises(ValidationError):
        AnchorConfig(near_threshold=1.5)


def test_docling_extras_config_defaults():
    from perspicacite.config.schema import KnowledgeBaseConfig
    kb = KnowledgeBaseConfig()
    assert kb.docling_extract_tables_figures is False
    assert kb.docling_max_pages == 40
    assert kb.docling_timeout_s == 600
    kb2 = KnowledgeBaseConfig(docling_extract_tables_figures=True)
    assert kb2.docling_extract_tables_figures is True
