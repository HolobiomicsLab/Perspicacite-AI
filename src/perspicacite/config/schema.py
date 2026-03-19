"""Pydantic models for configuration."""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=5468, ge=1024, le=65535)
    reload: bool = False

    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5468"
    ])


class MCPConfig(BaseModel):
    """MCP server configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=5500, ge=1024, le=65535)
    transport: Literal["stdio", "sse"] = "stdio"


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: Path = Field(default=Path("~/.local/share/perspicacite/memory.db"))
    chroma_path: Path = Field(default=Path("~/.local/share/perspicacite/chroma"))

    @field_validator("path", "chroma_path")
    @classmethod
    def expand_home(cls, v: Path) -> Path:
        """Expand ~ to home directory."""
        return v.expanduser()


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base defaults."""

    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    preserve_sections: bool = True
    default_top_k: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    base_url: str
    timeout: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)


class LLMConfig(BaseModel):
    """LLM configuration."""

    default_provider: str = "deepseek"
    default_model: str = "deepseek-chat"  # DeepSeek V3

    providers: dict[str, LLMProviderConfig] = Field(default_factory=lambda: {
        "anthropic": LLMProviderConfig(
            base_url="https://api.anthropic.com",
            timeout=120,
            max_retries=3,
        ),
        "openai": LLMProviderConfig(
            base_url="https://api.openai.com/v1",
            timeout=60,
            max_retries=3,
        ),
        "deepseek": LLMProviderConfig(
            base_url="https://api.deepseek.com",
            timeout=60,
            max_retries=3,
        ),
        "minimax": LLMProviderConfig(
            base_url="https://api.minimaxi.com/anthropic",  # Anthropic-compatible API for Chinese users
            timeout=120,
            max_retries=3,
        ),
    })

    context: dict[str, Any] = Field(default_factory=lambda: {
        "max_tokens": 8000,
        "chat_history_turns": 10,
        "summarize_threshold": 20,
    })


class RAGModeSettings(BaseModel):
    """Settings for a RAG mode."""

    max_iterations: int = Field(default=1, ge=1)
    tools: list[str] = Field(default_factory=list)
    rerank: bool = True
    query_expansion: bool = False
    enable_planning: bool = False
    enable_reflection: bool = False
    build_citation_graph: bool = False


class RAGModesConfig(BaseModel):
    """RAG mode configurations."""

    quick: RAGModeSettings = Field(default_factory=lambda: RAGModeSettings(
        max_iterations=1,
        tools=["kb_search"],
        rerank=False,
    ))
    standard: RAGModeSettings = Field(default_factory=lambda: RAGModeSettings(
        max_iterations=1,
        tools=["kb_search"],
        rerank=True,
    ))
    advanced: RAGModeSettings = Field(default_factory=lambda: RAGModeSettings(
        max_iterations=2,
        tools=["kb_search", "web_search"],
        rerank=True,
        query_expansion=True,
    ))
    deep: RAGModeSettings = Field(default_factory=lambda: RAGModeSettings(
        max_iterations=5,
        tools=["kb_search", "web_search", "fetch_pdf", "citation_network"],
        rerank=True,
        enable_planning=True,
        enable_reflection=True,
    ))
    citation: RAGModeSettings = Field(default_factory=lambda: RAGModeSettings(
        max_iterations=3,
        tools=["kb_search", "web_search", "citation_network"],
        rerank=True,
        build_citation_graph=True,
    ))


class SciLexAPIConfig(BaseModel):
    """Configuration for a SciLEx API."""

    enabled: bool = True
    rate_limit: int = Field(default=100, ge=1)


class SciLexConfig(BaseModel):
    """SciLEx integration configuration."""

    enabled: bool = True
    config_path: Optional[Path] = None

    apis: dict[str, SciLexAPIConfig] = Field(default_factory=lambda: {
        "semantic_scholar": SciLexAPIConfig(enabled=True, rate_limit=100),
        "openalex": SciLexAPIConfig(enabled=True, rate_limit=100),
        "pubmed": SciLexAPIConfig(enabled=True, rate_limit=10),
        "arxiv": SciLexAPIConfig(enabled=True, rate_limit=100),
        "ieee": SciLexAPIConfig(enabled=False, rate_limit=100),
        "springer": SciLexAPIConfig(enabled=False, rate_limit=100),
        "elsevier": SciLexAPIConfig(enabled=False, rate_limit=100),
        "hal": SciLexAPIConfig(enabled=True, rate_limit=100),
        "dblp": SciLexAPIConfig(enabled=True, rate_limit=100),
        "istex": SciLexAPIConfig(enabled=True, rate_limit=100),
    })

    collection: dict[str, Any] = Field(default_factory=lambda: {
        "default_max_papers": 100,
        "quality_threshold": 0.7,
        "deduplicate": True,
        "download_pdfs": True,
    })


class WebSearchConfig(BaseModel):
    """Web search configuration."""

    providers: list[str] = Field(default_factory=lambda: [
        "google_scholar", "semantic_scholar"
    ])
    cache_ttl: int = Field(default=3600, ge=0)  # seconds


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = True
    token: Optional[str] = None  # Set via env: PERSPICACITE_AUTH_TOKEN


class UIConfig(BaseModel):
    """UI configuration."""

    theme: Literal["light", "dark", "system"] = "system"
    citation_format: Literal["nature", "apa", "mla", "ieee"] = "nature"


class Config(BaseModel):
    """Main configuration for Perspicacité v2."""

    version: str = "2.0.0"
    config_name: str = "default"

    server: ServerConfig = Field(default_factory=ServerConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag_modes: RAGModesConfig = Field(default_factory=RAGModesConfig)
    scilex: SciLexConfig = Field(default_factory=SciLexConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        """Validate configuration consistency."""
        # Ensure chunk_overlap < chunk_size
        if self.knowledge_base.chunk_overlap >= self.knowledge_base.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version string."""
        if not v.startswith("2."):
            raise ValueError("Config version must be 2.x")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, masking secrets."""
        d = self.model_dump()
        # Mask auth token if present
        if self.auth.token:
            d["auth"]["token"] = "***"
        return d
