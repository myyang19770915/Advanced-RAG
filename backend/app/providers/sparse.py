"""Sparse (lexical) embedder for hybrid search.

We expose a tiny Protocol-style API so callers can stay agnostic of the actual
backend (BM25 today, SPLADE / mini-coil tomorrow).  A "sparse vector" is just
two aligned lists: ``indices`` (int term ids) and ``values`` (their weights).

BM25 implementation uses FastEmbed's ``Qdrant/bm25`` model which is purely
local (only a small tokenizer config is downloaded) and produces IDF-weighted
term frequencies — Qdrant computes the final BM25 score server-side when the
sparse vector collection is configured with ``Modifier.IDF``.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

    @property
    def is_empty(self) -> bool:
        return not self.indices


class SparseEmbedder(Protocol):
    async def embed(self, texts: list[str]) -> list[SparseVector]: ...
    async def embed_query(self, text: str) -> SparseVector: ...
    @property
    def model_name(self) -> str: ...


class OffSparse:
    """No-op sparse encoder: returns empty vectors.  Used when sparse_provider='off'."""

    @property
    def model_name(self) -> str:  # noqa: D401
        return "off"

    async def embed(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[], values=[]) for _ in texts]

    async def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[], values=[])


class BM25Sparse:
    """FastEmbed BM25 sparse encoder.

    The model is loaded lazily on first use to avoid paying the import/IO cost
    when sparse search is disabled.  Encoding itself is sync (CPU-bound), so we
    wrap it in ``asyncio.to_thread`` to stay non-blocking.
    """

    def __init__(self, model: str = "Qdrant/bm25") -> None:
        self._model_name = model
        self._doc_encoder = None   # type: ignore[var-annotated]
        self._query_encoder = None  # type: ignore[var-annotated]

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._doc_encoder is None:
            from fastembed import SparseTextEmbedding  # type: ignore
            self._doc_encoder = SparseTextEmbedding(model_name=self._model_name)
            # BM25 uses the same model for query/doc encoding (purely lexical).
            self._query_encoder = self._doc_encoder

    @staticmethod
    def _to_sparse(e) -> SparseVector:
        # fastembed returns ``SparseEmbedding`` with .indices / .values (numpy arrays)
        return SparseVector(
            indices=[int(i) for i in e.indices],
            values=[float(v) for v in e.values],
        )

    async def embed(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        def _run() -> list[SparseVector]:
            self._ensure_loaded()
            return [self._to_sparse(e) for e in self._doc_encoder.embed(texts)]
        return await asyncio.to_thread(_run)

    async def embed_query(self, text: str) -> SparseVector:
        def _run() -> SparseVector:
            self._ensure_loaded()
            # ``query_embed`` produces a single vector; fall back to embed() if missing.
            try:
                gen = self._query_encoder.query_embed(text)
            except AttributeError:
                gen = self._query_encoder.embed([text])
            return self._to_sparse(next(iter(gen)))
        return await asyncio.to_thread(_run)
