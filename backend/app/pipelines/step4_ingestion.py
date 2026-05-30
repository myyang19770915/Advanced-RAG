"""Step 4: Embed chunks and upsert to vector store.

Important: when document_id is provided, point ID is
`uuid5(NAMESPACE_URL, f"{document_id}#{chunk_index}")` so that different
versions of the same file get independent point spaces and do not collide.
Legacy fallback (no document_id): `uuid5(NAMESPACE_URL, f"{original_file}#{chunk_index}")`.
"""

from __future__ import annotations

import uuid

from app.providers.embedding import EmbeddingProvider
from app.providers.vector_store import VectorPoint, VectorStoreProvider
from app.schemas import ChunkPayload


def point_id_for(original_file: str, chunk_index: int, document_id: str = "") -> str:
    key = f"{document_id}#{chunk_index}" if document_id else f"{original_file}#{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


async def embed_and_upsert(
    chunks: list[ChunkPayload],
    *,
    collection: str,
    embedding: EmbeddingProvider,
    vector_store: VectorStoreProvider,
    document_id: str = "",
) -> int:
    if not chunks:
        return 0
    await vector_store.ensure_collection(collection, embedding.dim)
    vectors = await embedding.embed([c.text for c in chunks])
    points: list[VectorPoint] = []
    for c, vec in zip(chunks, vectors):
        pid = point_id_for(c.original_file, c.chunk_index, document_id=document_id)
        payload = c.model_dump()
        if document_id:
            payload["document_id"] = document_id
        points.append(VectorPoint(id=pid, vector=vec, payload=payload))
    await vector_store.upsert(collection, points)
    return len(points)
