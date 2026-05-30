"""Provider factory: returns instances based on settings.

Tests can override these via monkeypatching or by setting env vars.
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.providers.docling import (
    DoclingProvider,
    HttpDocling,
    LocalDocling,
    MockDocling,
)
from app.providers.embedding import (
    EmbeddingProvider,
    MockEmbedding,
    OpenAICompatibleEmbedding,
)
from app.providers.llm import LLMProvider, MockLLM, OpenAICompatibleLLM
from app.providers.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreProvider,
)


@lru_cache
def get_llm() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "openai":
        return OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, s.llm_model)
    return MockLLM()


@lru_cache
def get_embedding() -> EmbeddingProvider:
    s = get_settings()
    if s.embedding_provider == "openai":
        # Use dedicated embedding base URL if set, otherwise fall back to LLM URL
        base_url = s.embedding_base_url or s.llm_base_url
        return OpenAICompatibleEmbedding(
            base_url, s.llm_api_key, s.embedding_model, s.embedding_dim
        )
    return MockEmbedding(dim=s.embedding_dim)


@lru_cache
def get_vector_store() -> VectorStoreProvider:
    s = get_settings()
    if s.vector_store == "qdrant" and s.qdrant_url:
        return QdrantVectorStore(s.qdrant_url, s.qdrant_api_key)
    return InMemoryVectorStore()


@lru_cache
def get_docling() -> DoclingProvider:
    s = get_settings()
    if s.docling_provider == "local":
        return LocalDocling()
    if s.docling_provider == "http" and s.docling_http_url:
        return HttpDocling(s.docling_http_url)
    return MockDocling()


def reset_provider_cache() -> None:
    """Used in tests to re-evaluate settings."""
    get_llm.cache_clear()
    get_embedding.cache_clear()
    get_vector_store.cache_clear()
    get_docling.cache_clear()
