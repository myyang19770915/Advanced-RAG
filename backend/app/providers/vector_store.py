from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStoreProvider(Protocol):
    async def ensure_collection(self, name: str, dim: int) -> None: ...

    async def upsert(self, collection: str, points: list[VectorPoint]) -> int: ...

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    async def count(
        self, collection: str, filters: dict[str, Any] | None = None
    ) -> int: ...

    async def delete_by_filter(
        self, collection: str, filters: dict[str, Any]
    ) -> int: ...

    async def delete_collection(self, collection: str) -> None: ...

    async def get_unique_field_values(
        self, collection: str, field: str, limit: int = 2000
    ) -> list[str]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _payload_matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    for k, v in filters.items():
        if isinstance(v, list):
            if payload.get(k) not in v:
                return False
        else:
            if payload.get(k) != v:
                return False
    return True


class InMemoryVectorStore:
    """In-memory vector store for tests and offline runs."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, VectorPoint]] = {}

    async def ensure_collection(self, name: str, dim: int) -> None:
        self._collections.setdefault(name, {})

    async def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        store = self._collections.setdefault(collection, {})
        for p in points:
            store[p.id] = p
        return len(points)

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        store = self._collections.get(collection, {})
        scored: list[SearchHit] = []
        for p in store.values():
            if filters and not _payload_matches(p.payload, filters):
                continue
            scored.append(
                SearchHit(id=p.id, score=_cosine(vector, p.vector), payload=p.payload)
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    async def count(
        self, collection: str, filters: dict[str, Any] | None = None
    ) -> int:
        store = self._collections.get(collection, {})
        if not filters:
            return len(store)
        return sum(1 for p in store.values() if _payload_matches(p.payload, filters))

    async def delete_by_filter(
        self, collection: str, filters: dict[str, Any]
    ) -> int:
        store = self._collections.get(collection, {})
        to_del = [pid for pid, p in store.items() if _payload_matches(p.payload, filters)]
        for pid in to_del:
            del store[pid]
        return len(to_del)

    async def delete_collection(self, collection: str) -> None:
        self._collections.pop(collection, None)

    async def get_unique_field_values(
        self, collection: str, field: str, limit: int = 2000
    ) -> list[str]:
        store = self._collections.get(collection, {})
        seen: set[str] = set()
        for p in store.values():
            v = p.payload.get(field)
            if v and isinstance(v, str):
                seen.add(v)
        return sorted(seen)


class QdrantVectorStore:
    """Real Qdrant via qdrant-client (async)."""

    def __init__(self, url: str, api_key: str = "") -> None:
        # Import lazily so tests don't need the client installed at import time.
        from qdrant_client import AsyncQdrantClient

        self._client = AsyncQdrantClient(url=url, api_key=api_key or None)

    async def ensure_collection(self, name: str, dim: int) -> None:
        from qdrant_client.http import models as qm

        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if name not in names:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    async def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        from qdrant_client.http import models as qm

        qpoints = [
            qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
        ]
        await self._client.upsert(collection_name=collection, points=qpoints)
        return len(points)

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        from qdrant_client.http import models as qm

        qfilter = None
        if filters:
            must = []
            for k, v in filters.items():
                if isinstance(v, list):
                    must.append(
                        qm.FieldCondition(key=k, match=qm.MatchAny(any=v))
                    )
                else:
                    must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
            qfilter = qm.Filter(must=must)
        result = await self._client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(id=str(h.id), score=float(h.score), payload=h.payload or {})
            for h in result.points
        ]

    async def count(
        self, collection: str, filters: dict[str, Any] | None = None
    ) -> int:
        from qdrant_client.http import models as qm

        qfilter = None
        if filters:
            must = [
                qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                for k, v in filters.items()
                if not isinstance(v, list)
            ]
            qfilter = qm.Filter(must=must)
        res = await self._client.count(
            collection_name=collection, count_filter=qfilter, exact=True
        )
        return int(res.count)

    async def delete_by_filter(
        self, collection: str, filters: dict[str, Any]
    ) -> int:
        from qdrant_client.http import models as qm

        must = []
        for k, v in filters.items():
            if isinstance(v, list):
                must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
            else:
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        qfilter = qm.Filter(must=must)
        count = await self.count(collection, filters)
        await self._client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(filter=qfilter),
        )
        return count

    async def delete_collection(self, collection: str) -> None:
        await self._client.delete_collection(collection_name=collection)

    async def get_unique_field_values(
        self, collection: str, field: str, limit: int = 2000
    ) -> list[str]:
        """Scroll all points and return sorted unique non-empty values of `field`."""
        seen: set[str] = set()
        offset = None
        while len(seen) < limit:
            result = await self._client.scroll(
                collection_name=collection,
                limit=100,
                offset=offset,
                with_payload=[field],
                with_vectors=False,
            )
            points, next_offset = result
            for p in points:
                if p.payload:
                    v = p.payload.get(field)
                    if v and isinstance(v, str):
                        seen.add(v)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(seen)
