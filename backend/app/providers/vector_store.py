from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol


# ── Named-vector constants ──
# We always store the dense vector under the name "dense", and (optionally)
# the BM25 sparse vector under "bm25".  Keeping these as named vectors lets
# the same collection support both pure-dense and hybrid queries.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


@dataclass
class VectorPoint:
    id: str
    vector: list[float]                       # dense
    payload: dict[str, Any] = field(default_factory=dict)
    sparse_indices: list[int] | None = None   # BM25 term ids
    sparse_values: list[float] | None = None  # BM25 weights


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStoreProvider(Protocol):
    async def ensure_collection(
        self, name: str, dim: int, *, sparse: bool = False
    ) -> None: ...

    async def upsert(self, collection: str, points: list[VectorPoint]) -> int: ...

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 30,
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

    async def ensure_collection(
        self, name: str, dim: int, *, sparse: bool = False
    ) -> None:
        # ``sparse`` is recorded but the in-memory store has no schema concept.
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

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """RRF-fused dense + sparse for the in-memory store (test parity)."""
        store = self._collections.get(collection, {})
        candidates = [
            p for p in store.values()
            if not filters or _payload_matches(p.payload, filters)
        ]
        if not candidates:
            return []

        # Dense ranking
        dense_scored = sorted(
            candidates,
            key=lambda p: _cosine(dense_vector, p.vector),
            reverse=True,
        )[: top_k * 2]
        dense_rank = {p.id: r for r, p in enumerate(dense_scored)}

        # Sparse ranking via simple dot product on term ids
        q = dict(zip(sparse_indices, sparse_values))
        def _sparse_score(p: VectorPoint) -> float:
            if not p.sparse_indices:
                return 0.0
            return sum(
                q.get(i, 0.0) * v for i, v in zip(p.sparse_indices, p.sparse_values or [])
            )
        sparse_scored = sorted(
            (p for p in candidates if p.sparse_indices),
            key=_sparse_score,
            reverse=True,
        )[: top_k * 2]
        sparse_rank = {p.id: r for r, p in enumerate(sparse_scored)}

        # RRF fusion (k=60 is the standard constant)
        k = 60.0
        fused: dict[str, float] = {}
        for pid, r in dense_rank.items():
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + r)
        for pid, r in sparse_rank.items():
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + r)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            SearchHit(id=pid, score=score, payload=store[pid].payload)
            for pid, score in ranked
            if pid in store
        ]

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
    """Real Qdrant via qdrant-client (async).

    Supports two collection layouts:
      * Legacy: single un-named dense vector (older collections).
      * Hybrid: named ``dense`` vector + optional ``bm25`` sparse vector.

    A new collection is created in the hybrid layout when ``sparse=True``.
    Queries auto-detect the layout at runtime via ``get_collection``.
    """

    def __init__(self, url: str, api_key: str = "") -> None:
        # Import lazily so tests don't need the client installed at import time.
        from qdrant_client import AsyncQdrantClient

        self._client = AsyncQdrantClient(url=url, api_key=api_key or None)
        # Cache "is this collection hybrid?" lookups (cleared on ensure/delete).
        self._hybrid_cache: dict[str, bool] = {}

    async def _is_hybrid(self, name: str) -> bool:
        """Return True if collection uses named ``dense`` + sparse layout."""
        cached = self._hybrid_cache.get(name)
        if cached is not None:
            return cached
        try:
            info = await self._client.get_collection(name)
            params = info.config.params
            # Named vectors -> vectors_config is a dict; legacy -> VectorParams
            vectors = params.vectors
            is_named = isinstance(vectors, dict) and DENSE_VECTOR_NAME in vectors
            has_sparse = bool(getattr(params, "sparse_vectors", None)) and \
                SPARSE_VECTOR_NAME in (params.sparse_vectors or {})
            hybrid = bool(is_named and has_sparse)
        except Exception:  # noqa: BLE001
            hybrid = False
        self._hybrid_cache[name] = hybrid
        return hybrid

    async def ensure_collection(
        self, name: str, dim: int, *, sparse: bool = False
    ) -> None:
        from qdrant_client.http import models as qm

        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if name in names:
            return  # don't reshape existing schema; delete_collection+ensure to migrate

        if sparse:
            await self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR_NAME: qm.VectorParams(
                        size=dim, distance=qm.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: qm.SparseVectorParams(
                        modifier=qm.Modifier.IDF,  # let Qdrant compute IDF for BM25
                    ),
                },
            )
            self._hybrid_cache[name] = True
        else:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            self._hybrid_cache[name] = False

    async def upsert(self, collection: str, points: list[VectorPoint]) -> int:
        from qdrant_client.http import models as qm

        hybrid = await self._is_hybrid(collection)

        qpoints: list[qm.PointStruct] = []
        for p in points:
            if hybrid:
                vec: dict[str, Any] = {DENSE_VECTOR_NAME: p.vector}
                if p.sparse_indices:
                    vec[SPARSE_VECTOR_NAME] = qm.SparseVector(
                        indices=p.sparse_indices,
                        values=p.sparse_values or [],
                    )
                qpoints.append(qm.PointStruct(id=p.id, vector=vec, payload=p.payload))
            else:
                qpoints.append(qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload))
        await self._client.upsert(collection_name=collection, points=qpoints)
        return len(points)

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None):
        from qdrant_client.http import models as qm

        if not filters:
            return None
        must = []
        for k, v in filters.items():
            if isinstance(v, list):
                must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
            else:
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
        return qm.Filter(must=must)

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Pure dense search.  Works against both legacy and hybrid collections."""
        from qdrant_client.http import models as qm  # noqa: F401  (kept for parity)

        hybrid = await self._is_hybrid(collection)
        query_arg: Any
        using: str | None
        if hybrid:
            query_arg = vector
            using = DENSE_VECTOR_NAME
        else:
            query_arg = vector
            using = None

        kwargs: dict[str, Any] = dict(
            collection_name=collection,
            query=query_arg,
            query_filter=self._build_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        if using:
            kwargs["using"] = using
        result = await self._client.query_points(**kwargs)
        return [
            SearchHit(id=str(h.id), score=float(h.score), payload=h.payload or {})
            for h in result.points
        ]

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """RRF-fused dense + BM25 sparse query (Qdrant server-side).

        Falls back to pure-dense if the collection isn't hybrid (legacy schema)
        or if the sparse vector is empty (e.g. query has no recognised terms).
        """
        from qdrant_client.http import models as qm

        hybrid = await self._is_hybrid(collection)
        has_sparse = hybrid and bool(sparse_indices)
        if not has_sparse:
            return await self.search(collection, dense_vector, top_k=top_k, filters=filters)

        qfilter = self._build_filter(filters)
        # Prefetch a wider pool for each modality so RRF has room to re-order.
        prefetch_limit = max(top_k * 2, 50)
        prefetch = [
            qm.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=qfilter,
            ),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse_indices, values=sparse_values),
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=qfilter,
            ),
        ]
        result = await self._client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
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
        res = await self._client.count(
            collection_name=collection,
            count_filter=self._build_filter(filters),
            exact=True,
        )
        return int(res.count)

    async def delete_by_filter(
        self, collection: str, filters: dict[str, Any]
    ) -> int:
        from qdrant_client.http import models as qm

        qfilter = self._build_filter(filters)
        count = await self.count(collection, filters)
        await self._client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(filter=qfilter),
        )
        return count

    async def delete_collection(self, collection: str) -> None:
        self._hybrid_cache.pop(collection, None)
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
