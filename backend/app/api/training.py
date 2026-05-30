from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select

from app.db.session import SessionDep, SessionLocal
from app.models import Document, Project
from app.schemas import (
    DocumentStatus,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from app.services.training import process_document, get_chunk_progress

router = APIRouter(prefix="/projects/{project_id}/training", tags=["training"])


async def _run_job(project_id: str, document_ids: list[str]) -> None:
    """Background runner: opens its own DB session per document."""
    async with SessionLocal() as db:
        project = await db.get(Project, project_id)
        if not project:
            return
        for doc_id in document_ids:
            doc = await db.get(Document, doc_id)
            if not doc:
                continue
            await process_document(db, project=project, document=doc)


@router.post("/start", response_model=TrainingStartResponse)
async def start_training(
    project_id: str,
    db: SessionDep,
    background: BackgroundTasks,
) -> TrainingStartResponse:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    q = await db.execute(
        select(Document).where(
            Document.project_id == project_id,
            Document.status.in_(["uploaded", "failed"]),
        )
    )
    docs = list(q.scalars())
    if not docs:
        raise HTTPException(status_code=400, detail="no pending documents")
    doc_ids = [d.id for d in docs]
    for d in docs:
        d.status = "queued"
    project.status = "running"
    await db.commit()

    background.add_task(_run_job, project_id, doc_ids)
    return TrainingStartResponse(job_id=str(uuid.uuid4()), document_ids=doc_ids)


@router.post("/reset", response_model=TrainingStartResponse)
async def reset_training(
    project_id: str,
    db: SessionDep,
    background: BackgroundTasks,
) -> TrainingStartResponse:
    """Reset ALL documents back to 'uploaded' (deletes Qdrant collection) then starts training."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    q = await db.execute(
        select(Document).where(Document.project_id == project_id)
    )
    docs = list(q.scalars())
    if not docs:
        raise HTTPException(status_code=400, detail="no documents to reset")

    # Delete Qdrant collection so vectors are re-created with current dim
    from app.providers.factory import get_vector_store
    from app.services.training import collection_name_for
    try:
        vs = get_vector_store()
        await vs.delete_collection(collection_name_for(project.name))
    except Exception:  # noqa: BLE001  – missing collection is fine
        pass

    # Reset every document
    for d in docs:
        d.status = "uploaded"
        d.chunk_count = 0
        d.error_message = ""
    project.status = "running"
    await db.commit()

    doc_ids = [d.id for d in docs]
    background.add_task(_run_job, project_id, doc_ids)
    return TrainingStartResponse(job_id=str(uuid.uuid4()), document_ids=doc_ids)


@router.get("/status", response_model=TrainingStatusResponse)
async def training_status(project_id: str, db: SessionDep) -> TrainingStatusResponse:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    q = await db.execute(
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.asc())
    )
    docs = list(q.scalars())
    statuses = [
        DocumentStatus(
            document_id=d.id,
            filename=d.original_filename,
            status=d.status,
            chunk_count=d.chunk_count,
            error_message=d.error_message,
            **get_chunk_progress(d.id),
        )
        for d in docs
    ]
    if not docs:
        overall = "idle"
    elif any(d.status in {"queued", "converting", "chunking", "embedding"} for d in docs):
        overall = "running"
    elif any(d.status == "failed" for d in docs) and not any(
        d.status == "ready" for d in docs
    ):
        overall = "failed"
    elif all(d.status == "ready" for d in docs):
        overall = "done"
    else:
        overall = "partial"
    if overall in {"done", "failed"} and project.status != overall:
        project.status = overall
        await db.commit()
    return TrainingStatusResponse(
        project_id=project_id, overall_status=overall, documents=statuses
    )
