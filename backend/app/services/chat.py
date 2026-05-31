"""Chat / RAG query service."""

from __future__ import annotations

import time
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ChatMessage, ChatSession, Project
from app.pipelines import step5_query
from app.providers.factory import (
    get_embedding,
    get_llm,
    get_reranker,
    get_sparse_embedder,
    get_vector_store,
)
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
    doc_id = payload.get("document_id") or None
    page = payload.get("page")
    bbox = payload.get("bbox")
    raw_types = payload.get("content_types") or []
    content_types = [str(t) for t in raw_types if isinstance(t, str)] if isinstance(raw_types, list) else []
    primary_type = str(payload.get("primary_type") or "text")
    return Citation(
        document_id=doc_id,
        original_file=payload.get("original_file", "?"),
        chunk_index=int(payload.get("chunk_index", 0)),
        score=float(hit.score),
        text_preview=(payload.get("text", "") or ""),
        download_url=(f"/api/files/{doc_id}/download" if doc_id else None),
        page=int(page) if isinstance(page, (int, float)) else None,
        bbox=[float(x) for x in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None,
        page_width=payload.get("page_width"),
        page_height=payload.get("page_height"),
        content_types=content_types,
        primary_type=primary_type,
    )


async def load_history(
    db: AsyncSession, session_id: str, max_turns: int = 6
) -> list[dict]:
    """Return the most recent messages for a session as plain dicts.

    Returned in oldest-first order, limited to the last ``max_turns * 2``
    rows (user+assistant pairs). Each item is ``{role, content}``.
    """
    if not session_id:
        return []
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(max_turns * 2)
    )
    rows = (await db.execute(stmt)).scalars().all()
    # reverse to oldest-first
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


async def ensure_session(
    db: AsyncSession, project_id: str, session_id: str | None, first_question: str
) -> str:
    """Return an existing session id or create a new one (without commit)."""
    if session_id:
        existing = await db.get(ChatSession, session_id)
        if existing and existing.project_id == project_id:
            return session_id
    session = ChatSession(
        id=str(uuid.uuid4()),
        project_id=project_id,
        title=first_question[:80],
    )
    db.add(session)
    await db.flush()
    return session.id


async def persist_turn(
    db: AsyncSession,
    session_id: str,
    question: str,
    answer: str,
    citations: list[Citation],
) -> None:
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
    sparse_embedder = get_sparse_embedder()
    reranker = get_reranker()
    from app.core.config import get_settings
    settings = get_settings()

    history = await load_history(db, session_id) if session_id else []

    answer, hits = await step5_query.run_query(
        question=question,
        collection=project.collection_name,
        project_name=project.name,
        llm=llm,
        embedding=embedding,
        vector_store=vector_store,
        top_k=top_k or settings.final_top_n,
        available_taxonomy=await _get_taxonomy(project.collection_name, vector_store),
        history=history,
        sparse_embedder=sparse_embedder,
        reranker=reranker,
        retrieve_top_k=settings.retrieve_top_k,
    )
    citations = [_hit_to_citation(h) for h in hits]

    session_id = await ensure_session(db, project.id, session_id, question)
    await persist_turn(db, session_id, question, answer, citations)
    return answer, citations, session_id
