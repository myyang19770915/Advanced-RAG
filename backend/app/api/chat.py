from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.db.session import SessionDep, SessionLocal
from app.models import Project
from app.pipelines import step5_query
from app.providers.factory import get_embedding, get_llm, get_vector_store
from app.schemas import ChatRequest, ChatResponse, Citation
from app.services.chat import chat as chat_service, _hit_to_citation, _get_taxonomy

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
    """Server-sent events streaming endpoint."""

    async def event_gen():
        async with SessionLocal() as db:
            project = await db.get(Project, payload.project_id)
            if not project:
                yield "event: error\ndata: project not found\n\n"
                return
            llm = get_llm()
            embedding = get_embedding()
            vector_store = get_vector_store()
            taxonomy = await _get_taxonomy(project.collection_name, vector_store)
            plan = await step5_query.plan_query(
                question=payload.question, llm=llm, project_name=project.name,
                available_taxonomy=taxonomy,
            )
            hits = await step5_query.retrieve(
                plan=plan,
                collection=project.collection_name,
                embedding=embedding,
                vector_store=vector_store,
                top_k=payload.top_k,
            )
            citations = [_hit_to_citation(h).model_dump() for h in hits]
            yield "event: citations\ndata: " + json.dumps(citations) + "\n\n"
            async for token in step5_query.answer_stream(
                question=payload.question, hits=hits, llm=llm
            ):
                yield "event: token\ndata: " + json.dumps({"t": token}) + "\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
