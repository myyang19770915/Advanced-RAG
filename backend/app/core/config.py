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

    # VLM (Vision-Language Model — for image captioning during PDF parsing)
    vlm_provider: Literal["mock", "openai", "off"] = "off"
    vlm_base_url: str = ""    # if empty, falls back to llm_base_url
    vlm_api_key: str = ""     # if empty, falls back to llm_api_key
    vlm_model: str = ""       # if empty, falls back to llm_model
    vlm_max_image_side: int = 1024  # downscale images longer than this (px) before sending

    # Embedding
    embedding_provider: Literal["mock", "openai"] = "mock"
    embedding_base_url: str = ""   # if empty, falls back to llm_base_url
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 64

    # Docling
    docling_provider: Literal["mock", "local", "http"] = "mock"
    docling_http_url: str = ""

    # ── Retrieval ──
    # Hybrid search (dense + BM25 sparse, fused with RRF on the Qdrant side).
    hybrid_search_enabled: bool = True
    sparse_provider: Literal["bm25", "off"] = "bm25"   # "off" disables sparse even if hybrid_search_enabled
    sparse_model: str = "Qdrant/bm25"                  # FastEmbed model name

    # Retrieve more candidates when reranking is on, so the reranker has signal.
    retrieve_top_k: int = 30        # candidates fetched from Qdrant
    final_top_n: int = 5            # final results sent to LLM (after rerank/truncate)

    # Reranker (cross-encoder service, Cohere/Jina-compatible HTTP API).
    rerank_enabled: bool = False
    rerank_provider: Literal["off", "cohere_compat", "mock"] = "off"
    rerank_base_url: str = ""       # e.g. http://192.168.x.x:8002 (no trailing /rerank)
    rerank_api_key: str = ""
    rerank_model: str = ""          # e.g. "bge-reranker-v2-m3"
    rerank_timeout_s: int = 30

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
