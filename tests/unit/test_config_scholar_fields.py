# tests/unit/test_config_scholar_fields.py
"""Config field tests for Google Scholar provider + abstract-only KB mode."""

import pytest

from perspicacite.config.schema import Config, GoogleScholarConfig, KnowledgeBaseConfig


def test_google_scholar_config_defaults():
    cfg = Config()
    assert cfg.google_scholar.enabled is False
    assert cfg.google_scholar.headless is True
    assert cfg.google_scholar.delay_seconds == 2.0
    assert cfg.google_scholar.max_results == 20


def test_google_scholar_can_be_enabled():
    cfg = Config(google_scholar=GoogleScholarConfig(enabled=True))
    assert cfg.google_scholar.enabled is True


def test_knowledge_base_ingest_mode_default():
    kb = KnowledgeBaseConfig()
    assert kb.ingest_mode == "auto"


def test_knowledge_base_ingest_mode_abstract_only():
    kb = KnowledgeBaseConfig(ingest_mode="abstract_only")
    assert kb.ingest_mode == "abstract_only"


def test_knowledge_base_ingest_mode_rejects_invalid():
    with pytest.raises(Exception):
        KnowledgeBaseConfig(ingest_mode="nonsense")
