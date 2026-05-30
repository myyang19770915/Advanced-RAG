from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.db.session import SessionDep
from app.models import Project
from app.schemas import ProjectCreate, ProjectOut
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
