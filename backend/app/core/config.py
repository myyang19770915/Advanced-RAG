from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from env or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "sqlite+aiosqlite:///./data/app.db"

    # Vector store
    vector_store: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # LLM
    llm_provider: Literal["mock", "openai"] = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # Embedding
    embedding_provider: Literal["mock", "openai"] = "mock"
    embedding_base_url: str = ""   # if empty, falls back to llm_base_url
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 64

    # Docling
    docling_provider: Literal["mock", "local", "http"] = "mock"
    docling_http_url: str = ""

    # Misc
    data_dir: Path = Path("./data")
    admin_token: str = "change-me"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def uploads_dir(self) -> Path:
        p = self.data_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def markdown_dir(self) -> Path:
        p = self.data_dir / "markdown"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chunks_dir(self) -> Path:
        p = self.data_dir / "chunks"
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
