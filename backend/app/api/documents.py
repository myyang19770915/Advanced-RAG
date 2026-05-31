from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionDep, SessionLocal
from app.models import Document, Project
from app.schemas import DocumentOut, TrainingStartResponse

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])


# ── background job helper (shared with training.py) ──────────────────────────

async def _run_single(project_id: str, document_id: str) -> None:
    from app.services.training import process_document
    async with SessionLocal() as db:
        project = await db.get(Project, project_id)
        doc = await db.get(Document, document_id)
        if project and doc:
            await process_document(db, project=project, document=doc)


# ── upload ────────────────────────────────────────────────────────────────────

@router.post(
    "", response_model=list[DocumentOut], status_code=status.HTTP_201_CREATED
)
async def upload_documents(
    project_id: str,
    db: SessionDep,
    files: list[UploadFile] = File(...),
) -> list[Document]:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    settings = get_settings()
    proj_dir = settings.uploads_dir / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    created: list[Document] = []
    for f in files:
        if not f.filename:
            continue
        safe_name = Path(f.filename).name

        # Auto-increment version when a document with the same filename exists
        q = await db.execute(
            select(Document)
            .where(Document.project_id == project_id, Document.original_filename == safe_name)
            .order_by(Document.version.desc())
        )
        existing = q.scalars().first()
        version = (existing.version + 1) if existing else 1

        doc_id = str(uuid.uuid4())
        stored = proj_dir / f"{doc_id}__{safe_name}"
        content = await f.read()
        # write_bytes is blocking; offload large uploads to a worker thread.
        await asyncio.to_thread(stored.write_bytes, content)
        doc = Document(
            id=doc_id,
            project_id=project_id,
            original_filename=safe_name,
            stored_path=str(stored),
            status="uploaded",
            version=version,
        )
        db.add(doc)
        created.append(doc)
    await db.commit()
    for d in created:
        await db.refresh(d)
    return created


# ── list ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[DocumentOut])
async def list_documents(project_id: str, db: SessionDep) -> list[Document]:
    if not await db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="project not found")
    q = await db.execute(
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.desc())
    )
    return list(q.scalars())


# ── delete ────────────────────────────────────────────────────────────────────

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    project_id: str,
    document_id: str,
    db: SessionDep,
) -> None:
    """Delete a document: removes Qdrant vectors, disk files, and DB record."""
    doc = await db.get(Document, document_id)
    if not doc or doc.project_id != project_id:
        raise HTTPException(status_code=404, detail="document not found")
    project = await db.get(Project, project_id)

    # 1. Remove vectors from Qdrant (best-effort)
    from app.services.training import delete_document_vectors
    await delete_document_vectors(project=project, document=doc)

    # 2. Remove disk files
    settings = get_settings()
    for path_str in [doc.stored_path, doc.markdown_path]:
        if path_str:
            Path(path_str).unlink(missing_ok=True)
    # Chunks JSON: PDF/MD pipeline names it after the markdown stem;
    # QA pipeline (xlsx/csv) names it after the stored file stem.
    if doc.markdown_path:
        chunks_file = settings.chunks_dir / (Path(doc.markdown_path).stem + ".chunks.json")
        chunks_file.unlink(missing_ok=True)
    elif doc.stored_path:
        chunks_file = settings.chunks_dir / (Path(doc.stored_path).stem + ".chunks.json")
        chunks_file.unlink(missing_ok=True)

    # 3. Remove DB record (cascades to QaTest via Document.project relation)
    await db.delete(doc)
    await db.commit()


# ── per-document retrain ──────────────────────────────────────────────────────

@router.post("/{document_id}/retrain", response_model=TrainingStartResponse)
async def retrain_document(
    project_id: str,
    document_id: str,
    db: SessionDep,
    background: BackgroundTasks,
) -> TrainingStartResponse:
    """Delete this document's vectors and re-run the full training pipeline.

    Useful after changing the active Rule or fixing a processing error.
    """
    doc = await db.get(Document, document_id)
    if not doc or doc.project_id != project_id:
        raise HTTPException(status_code=404, detail="document not found")
    if doc.status in {"queued", "converting", "chunking", "embedding"}:
        raise HTTPException(status_code=409, detail="document is already being processed")
    project = await db.get(Project, project_id)

    # Delete old vectors so stale chunks don't linger if new run produces fewer
    from app.services.training import delete_document_vectors
    await delete_document_vectors(project=project, document=doc)

    doc.status = "uploaded"
    doc.chunk_count = 0
    doc.error_message = ""
    await db.commit()

    background.add_task(_run_single, project_id, document_id)
    return TrainingStartResponse(job_id=str(uuid.uuid4()), document_ids=[document_id])


# ── download ──────────────────────────────────────────────────────────────────

download_router = APIRouter(prefix="/files", tags=["files"])


@download_router.get("/{document_id}/download")
async def download(document_id: str, db: SessionDep) -> FileResponse:
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    path = Path(doc.stored_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="file missing on disk")
    return FileResponse(path, filename=doc.original_filename)


@download_router.get("/{document_id}/chunk/{chunk_index}")
async def chunk_location(document_id: str, chunk_index: int, db: SessionDep) -> dict:
    """Return page + bbox for a specific chunk so the frontend can render a PDF highlight."""
    doc = await db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    # PDF/MD docs locate chunks via markdown_path; QA docs (xlsx/csv) via stored_path.
    if doc.markdown_path:
        md_path = Path(doc.markdown_path)
        chunks_path = md_path.parent / (md_path.stem + ".chunks.json")
    elif doc.stored_path:
        from app.core.config import get_settings as _gs
        stored = Path(doc.stored_path)
        chunks_path = _gs().chunks_dir / (stored.stem + ".chunks.json")
    else:
        raise HTTPException(status_code=404, detail="document not yet processed")
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="chunks file not found")

    chunks_raw = await asyncio.to_thread(chunks_path.read_text, encoding="utf-8")
    chunks = json.loads(chunks_raw)
    for c in chunks:
        if int(c.get("chunk_index", -1)) == chunk_index:
            return {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "page": c.get("page"),
                "bbox": c.get("bbox"),
                "page_width": c.get("page_width"),
                "page_height": c.get("page_height"),
                "text_preview": (c.get("text") or "")[:200],
            }
    raise HTTPException(status_code=404, detail=f"chunk {chunk_index} not found")
