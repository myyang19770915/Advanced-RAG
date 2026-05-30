from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.db.session import SessionDep
from app.models import ChatMessage, ChatSession, Project
from app.schemas import ChatSessionSummary, ProjectCreate, ProjectOut
from app.services.training import collection_name_for

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
async def create_project(payload: ProjectCreate, db: SessionDep) -> Project:
    existing = await db.execute(select(Project).where(Project.name == payload.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="project name already exists")
    project = Project(
        name=payload.name,
        description=payload.description,
        collection_name=collection_name_for(payload.name),
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: SessionDep) -> list[Project]:
    q = await db.execute(select(Project).order_by(Project.created_at.desc()))
    return list(q.scalars())


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: str, db: SessionDep) -> Project:
    proj = await db.get(Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="project not found")
    return proj


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: SessionDep) -> None:
    proj = await db.get(Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="project not found")
    await db.delete(proj)
    await db.commit()


@router.get("/{project_id}/chat/sessions", response_model=list[ChatSessionSummary])
async def list_chat_sessions(
    project_id: str, db: SessionDep
) -> list[ChatSessionSummary]:
    proj = await db.get(Project, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="project not found")
    rows = (
        await db.execute(
            select(
                ChatSession.id,
                ChatSession.title,
                ChatSession.created_at,
                func.count(ChatMessage.id).label("message_count"),
            )
            .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
            .where(ChatSession.project_id == project_id)
            .group_by(ChatSession.id)
            .order_by(ChatSession.created_at.desc())
        )
    ).all()
    return [
        ChatSessionSummary(
            id=r.id,
            title=r.title,
            created_at=r.created_at.isoformat(),
            message_count=int(r.message_count or 0),
        )
        for r in rows
    ]
