from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.db.session import SessionDep, SessionLocal
from app.models import ChatMessage, ChatSession, Project
from app.pipelines import step5_query
from app.providers.factory import get_embedding, get_llm, get_vector_store
from app.schemas import (
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionSummary,
    Citation,
)
from app.services.chat import (
    chat as chat_service,
    _hit_to_citation,
    _get_taxonomy,
    ensure_session,
    load_history,
    persist_turn,
)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: SessionDep) -> ChatResponse:
    project = await db.get(Project, payload.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    answer, citations, session_id = await chat_service(
        db=db,
        project=project,
        question=payload.question,
        top_k=payload.top_k,
        session_id=payload.session_id,
    )
    return ChatResponse(answer=answer, citations=citations, session_id=session_id)


@router.post("/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """Server-sent events streaming endpoint with multi-turn history."""

    async def event_gen():
        async with SessionLocal() as db:
            project = await db.get(Project, payload.project_id)
            if not project:
                yield "event: error\ndata: project not found\n\n"
                return

            session_id = await ensure_session(
                db, project.id, payload.session_id, payload.question
            )
            yield "event: session\ndata: " + json.dumps({"session_id": session_id}) + "\n\n"

            history = await load_history(db, session_id)
            llm = get_llm()
            embedding = get_embedding()
            vector_store = get_vector_store()
            taxonomy = await _get_taxonomy(project.collection_name, vector_store)
            plan = await step5_query.plan_query(
                question=payload.question, llm=llm, project_name=project.name,
                available_taxonomy=taxonomy, history=history,
            )
            hits = await step5_query.retrieve(
                plan=plan,
                collection=project.collection_name,
                embedding=embedding,
                vector_store=vector_store,
                top_k=payload.top_k,
            )
            low_conf = step5_query.is_low_confidence(hits)
            mode = "clarify" if low_conf else "answer"
            top_score = max((h.score for h in hits), default=0.0)
            yield "event: meta\ndata: " + json.dumps(
                {"mode": mode, "top_score": top_score, "threshold": step5_query.LOW_CONFIDENCE_THRESHOLD}
            ) + "\n\n"

            citation_objs = [_hit_to_citation(h) for h in hits]
            # In clarify mode, suppress citations from the UI (weak hits aren't sources).
            citations_payload = [] if low_conf else [c.model_dump() for c in citation_objs]
            yield "event: citations\ndata: " + json.dumps(citations_payload) + "\n\n"

            collected: list[str] = []
            async for token in step5_query.answer_stream(
                question=payload.question, hits=hits, llm=llm,
                history=history, mode=mode,
            ):
                collected.append(token)
                yield "event: token\ndata: " + json.dumps({"t": token}) + "\n\n"

            full_answer = "".join(collected)
            # Persist actual citations regardless of UI suppression, so the
            # next turn's history retains traceability.
            await persist_turn(
                db, session_id, payload.question, full_answer,
                [] if low_conf else citation_objs,
            )
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
async def get_session(session_id: str, db: SessionDep) -> ChatSessionDetail:
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    rows = (
        await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
    ).scalars().all()
    messages = [
        ChatMessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            citations=[Citation(**c) for c in (m.citations or [])],
            created_at=m.created_at.isoformat(),
        )
        for m in rows
    ]
    return ChatSessionDetail(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        messages=messages,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: SessionDep) -> dict:
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    await db.delete(session)
    await db.commit()
    return {"deleted": session_id}
