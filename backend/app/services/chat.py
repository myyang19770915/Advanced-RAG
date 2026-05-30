"""Chat / RAG query service."""

from __future__ import annotations

import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatMessage, ChatSession, Project
from app.pipelines import step5_query
from app.providers.factory import get_embedding, get_llm, get_vector_store
from app.providers.vector_store import VectorStoreProvider
from app.schemas import Citation

# Per-collection taxonomy cache: collection -> (fetched_at, {l1: [...], l2: [...], l3: [...]})
# Invalidated when TTL expires or when training completes (see invalidate_taxonomy_cache).
_taxonomy_cache: dict[str, tuple[float, dict[str, list[str]]]] = {}
_TAXONOMY_TTL = 300.0  # seconds (5 min)


async def _get_taxonomy(
    collection: str, vector_store: VectorStoreProvider
) -> dict[str, list[str]]:
    """Return cached l1/l2/l3 unique values for a collection, refreshing on TTL."""
    now = time.monotonic()
    cached = _taxonomy_cache.get(collection)
    if cached and now - cached[0] < _TAXONOMY_TTL:
        return cached[1]
    taxonomy: dict[str, list[str]] = {}
    for level in ("l1", "l2", "l3"):
        taxonomy[level] = await vector_store.get_unique_field_values(collection, level)
    _taxonomy_cache[collection] = (now, taxonomy)
    return taxonomy


def invalidate_taxonomy_cache(collection: str) -> None:
    """Call this after training completes so the next query re-fetches taxonomy."""
    _taxonomy_cache.pop(collection, None)


def _hit_to_citation(hit) -> Citation:  # type: ignore[no-untyped-def]
    payload = hit.payload or {}
    return Citation(
        original_file=payload.get("original_file", "?"),
        chunk_index=int(payload.get("chunk_index", 0)),
        score=float(hit.score),
        text_preview=(payload.get("text", "") or "")[:240],
    )


async def chat(
    *,
    db: AsyncSession,
    project: Project,
    question: str,
    top_k: int = 5,
    session_id: str | None = None,
) -> tuple[str, list[Citation], str]:
    llm = get_llm()
    embedding = get_embedding()
    vector_store = get_vector_store()

    answer, hits = await step5_query.run_query(
        question=question,
        collection=project.collection_name,
        project_name=project.name,
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        top_k=top_k,
        available_taxonomy=await _get_taxonomy(project.collection_name, vector_store),
    )
    citations = [_hit_to_citation(h) for h in hits]

    # Persist session/message
    if not session_id:
        session = ChatSession(
            id=str(uuid.uuid4()),
            project_id=project.id,
            title=question[:80],
        )
        db.add(session)
        await db.flush()
        session_id = session.id
    db.add(
        ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=question,
        )
    )
    db.add(
        ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=answer,
            citations=[c.model_dump() for c in citations],
        )
    )
    await db.commit()
    return answer, citations, session_id
