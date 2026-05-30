from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db.session import SessionDep
from app.models import Project
from app.schemas import AuditReport, CoverageReport
from app.services import qc as qc_service

router = APIRouter(prefix="/projects/{project_id}/qc", tags=["qc"])


@router.get("/coverage", response_model=CoverageReport)
async def coverage(project_id: str, db: SessionDep) -> CoverageReport:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return await qc_service.coverage(db, project)


@router.post("/audit", response_model=AuditReport)
async def audit(project_id: str, db: SessionDep, limit: int = 10) -> AuditReport:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    return await qc_service.audit(db, project, limit=limit)
