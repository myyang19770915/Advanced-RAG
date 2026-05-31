"""Reranker provider.

A reranker takes (query, [candidate_texts]) and returns a new ordering based
on a cross-encoder model.  We keep the wire format compatible with
Cohere / Jina / TEI / Infinity-server / xinference (the de-facto standard):

    POST {base_url}/rerank
    {
        "model":     "<model-name>",
        "query":     "...",
        "documents": ["text1", "text2", ...],
        "top_n":     5,
        "return_documents": false
    }
    →
    {
        "results": [
            {"index": 3, "relevance_score": 0.91},
            {"index": 0, "relevance_score": 0.78},
            ...
        ]
    }

If the user has no reranker yet (``RERANK_PROVIDER=off``) the factory returns
``OffReranker`` which is a no-op pass-through — so the call site doesn't need
to branch on the toggle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RerankResult:
    index: int                 # position in the original ``documents`` list
    relevance_score: float


class Reranker(Protocol):
    @property
    def enabled(self) -> bool: ...
    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]: ...


class OffReranker:
    """Pass-through: returns the original order, preserving caller scores."""

    @property
    def enabled(self) -> bool:
        return False

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        n = len(documents) if top_n is None else min(top_n, len(documents))
        # Score = 1.0 - i/N keeps the original order; downstream code only uses
        # the index, not this synthetic score.
        return [
            RerankResult(index=i, relevance_score=1.0 - i / max(1, len(documents)))
            for i in range(n)
        ]


class MockReranker:
    """Deterministic mock used in tests: scores by descending document length."""

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        scored = sorted(
            ((i, len(d)) for i, d in enumerate(documents)),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_n is not None:
            scored = scored[:top_n]
        max_len = max((s for _, s in scored), default=1)
        return [RerankResult(index=i, relevance_score=s / max_len) for i, s in scored]


class CohereCompatReranker:
    """HTTP reranker speaking the Cohere/Jina-compatible ``/rerank`` schema.

    Works out-of-the-box with: Cohere API, Jina AI Reranker API, TEI rerank
    server, infinity (michaelfeil/infinity), xinference, and most other modern
    rerank servers.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: int = 30,
    ) -> None:
        # Accept both ``http://host`` and ``http://host/v1`` — strip trailing slashes.
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s

    @property
    def enabled(self) -> bool:
        return True

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not documents:
            return []
        import httpx  # local import keeps cold start lighter

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if top_n is not None:
            payload["top_n"] = int(top_n)

        url = f"{self._base_url}/rerank"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("results") or data.get("data") or []
        out: list[RerankResult] = []
        for item in items:
            try:
                idx = int(item["index"])
                # Some servers use "score" instead of "relevance_score".
                score = float(item.get("relevance_score", item.get("score", 0.0)))
                out.append(RerankResult(index=idx, relevance_score=score))
            except (KeyError, TypeError, ValueError):
                continue
        return out
