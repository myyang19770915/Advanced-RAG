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
from app.providers.vlm import (
    MockVLM,
    OffVLM,
    OpenAICompatibleVLM,
    VLMProvider,
)
from app.providers.sparse import BM25Sparse, OffSparse, SparseEmbedder
from app.providers.rerank import (
    CohereCompatReranker,
    MockReranker,
    OffReranker,
    Reranker,
)


@lru_cache
def get_llm() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "openai":
        return OpenAICompatibleLLM(s.llm_base_url, s.llm_api_key, s.llm_model)
    return MockLLM()


@lru_cache
def get_vlm() -> VLMProvider:
    s = get_settings()
    if s.vlm_provider == "openai":
        base_url = s.vlm_base_url or s.llm_base_url
        api_key = s.vlm_api_key or s.llm_api_key
        model = s.vlm_model or s.llm_model
        return OpenAICompatibleVLM(base_url, api_key, model, s.vlm_max_image_side)
    if s.vlm_provider == "mock":
        return MockVLM()
    return OffVLM()


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


@lru_cache
def get_sparse_embedder() -> SparseEmbedder:
    """BM25 (lexical) encoder for hybrid search; off when disabled."""
    s = get_settings()
    if s.hybrid_search_enabled and s.sparse_provider == "bm25":
        return BM25Sparse(model=s.sparse_model)
    return OffSparse()


@lru_cache
def get_reranker() -> Reranker:
    """Cross-encoder reranker; OffReranker is a no-op pass-through."""
    s = get_settings()
    if not s.rerank_enabled or s.rerank_provider == "off":
        return OffReranker()
    if s.rerank_provider == "mock":
        return MockReranker()
    if s.rerank_provider == "cohere_compat" and s.rerank_base_url and s.rerank_model:
        return CohereCompatReranker(
            base_url=s.rerank_base_url,
            model=s.rerank_model,
            api_key=s.rerank_api_key,
            timeout_s=s.rerank_timeout_s,
        )
    # Misconfigured → safer to no-op than to crash queries.
    return OffReranker()


def reset_provider_cache() -> None:
    """Used in tests to re-evaluate settings."""
    get_llm.cache_clear()
    get_embedding.cache_clear()
    get_vector_store.cache_clear()
    get_docling.cache_clear()
    get_sparse_embedder.cache_clear()
    get_reranker.cache_clear()
