from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx


class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbedding:
    """Deterministic embeddings via hashing.

    Produces vectors with stable values per text so identical inputs ->
    identical vectors. Useful for tests and offline demos.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _embed_one(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand hash to required dim
        out: list[float] = []
        i = 0
        while len(out) < self._dim:
            # 4 bytes -> int -> [-1, 1]
            chunk = h[i % len(h) : i % len(h) + 4]
            if len(chunk) < 4:
                chunk = (chunk + h)[:4]
            val = int.from_bytes(chunk, "big", signed=False) / 0xFFFFFFFF
            out.append((val * 2.0) - 1.0)
            i += 4
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAICompatibleEmbedding:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model, "input": texts}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings", headers=headers, json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        return [d["embedding"] for d in data["data"]]
