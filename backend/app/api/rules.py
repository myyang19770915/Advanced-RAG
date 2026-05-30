from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db.session import SessionDep
from app.models import Document, Project, Rule
from app.providers.factory import get_llm
from app.schemas import (
    RuleBuildRequest,
    RuleBuildResponse,
    RuleOut,
    RuleSetActive,
)
from app.services.rules import build_candidates

router = APIRouter(prefix="/projects/{project_id}/rule", tags=["rules"])


@router.post("/build", response_model=RuleBuildResponse)
async def build(
    project_id: str, payload: RuleBuildRequest, db: SessionDep
) -> RuleBuildResponse:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")

    samples = []
    if payload.sample_document_ids:
        q = await db.execute(
            select(Document).where(Document.id.in_(payload.sample_document_ids))
        )
        for d in q.scalars():
            if d.markdown_path:
                from pathlib import Path

                samples.append(Path(d.markdown_path))
    llm = get_llm()
    candidates = await build_candidates(
        sample_markdowns=samples, domain_hint=payload.domain_hint, llm=llm
    )
    if not candidates:
        from app.schemas import RuleCandidate
        candidates = [
            RuleCandidate(label=l, name=n, system_prompt=p)
            for l, n, p in [
                ("A", "Strict", "Extract sections strictly."),
                ("B", "Balanced", "Extract sections with metadata."),
                ("C", "Loose", "Summarize and chunk freely."),
            ]
        ]
    return RuleBuildResponse(candidates=candidates)


@router.post("", response_model=RuleOut)
async def set_active(
    project_id: str, payload: RuleSetActive, db: SessionDep
) -> Rule:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    # deactivate existing
    q = await db.execute(select(Rule).where(Rule.project_id == project_id))
    for r in q.scalars():
        r.is_active = False
    rule = Rule(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name=payload.name,
        system_prompt=payload.system_prompt,
        is_active=True,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return rule


@router.get("", response_model=list[RuleOut])
async def list_rules(project_id: str, db: SessionDep) -> list[Rule]:
    q = await db.execute(select(Rule).where(Rule.project_id == project_id))
    return list(q.scalars())
