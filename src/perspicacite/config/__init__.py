"""Configuration system for Perspicacité v2."""

from perspicacite.config.schema import Config
from perspicacite.config.loader import load_config

__all__ = ["Config", "load_config"]
