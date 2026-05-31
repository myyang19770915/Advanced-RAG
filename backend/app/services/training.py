"""Training orchestration: glue step1 -> step4 + DB updates."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import Document, Project, QaTest, Rule
from app.pipelines import (
    step1_pdf_to_md,
    step1b_qa_to_chunks,
    step2_md_to_json,
    step3_md_to_qa,
    step4_ingestion,
)
from app.providers.factory import (
    get_docling,
    get_embedding,
    get_llm,
    get_sparse_embedder,
    get_vector_store,
    get_vlm,
)
from app.services.chat import invalidate_taxonomy_cache

logger = logging.getLogger(__name__)

# In-memory store for chunk-level progress during the "chunking" phase.
# Key: document_id → {"done": int, "total": int}
_chunk_progress: dict[str, dict[str, int]] = {}


def get_chunk_progress(document_id: str) -> dict[str, int]:
    """Return {"chunks_done": N, "chunks_total": M} for a document currently being chunked."""
    p = _chunk_progress.get(document_id, {"done": 0, "total": 0})
    return {"chunks_done": p["done"], "chunks_total": p["total"]}


def collection_name_for(project_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", project_name).strip("_")
    return f"rag_{safe or 'default'}"


async def delete_document_vectors(
    *,
    project: Project,
    document: Document,
) -> int:
    """Delete all Qdrant vectors that belong to *document*.

    Uses the ``document_id`` payload field written by step4_ingestion.
    Silently ignores errors (e.g. collection doesn't exist yet).
    Returns number of deleted points.
    """
    vector_store = get_vector_store()
    try:
        deleted = await vector_store.delete_by_filter(
            project.collection_name,
            {"document_id": document.id},
        )
        if deleted:
            logger.info(
                "Deleted %d vectors for document %s (%s)",
                deleted, document.id, document.original_filename,
            )
        return deleted
    except Exception:  # noqa: BLE001
        logger.warning(
            "Could not delete vectors for document %s (collection may not exist)",
            document.id,
        )
        return 0


async def process_document(
    db: AsyncSession,
    *,
    project: Project,
    document: Document,
) -> None:
    settings = get_settings()
    docling = get_docling()
    llm = get_llm()
    vlm = get_vlm()
    embedding = get_embedding()
    vector_store = get_vector_store()

    try:
        document.status = "converting"
        await db.commit()

        # ── Branch: QA dataset (xlsx / csv) takes a direct path that skips
        # step1 (PDF→MD) and step2 (MD→chunks). Each row becomes one chunk.
        if step1b_qa_to_chunks.is_qa_file(document.stored_path):
            document.status = "chunking"
            await db.commit()
            _chunk_progress[document.id] = {"done": 0, "total": 0}
            chunks = await step1b_qa_to_chunks.qa_file_to_chunks(
                Path(document.stored_path),
                original_file=document.original_filename,
                project_name=project.name,
                output_dir=settings.chunks_dir,
            )
            _chunk_progress[document.id] = {"done": len(chunks), "total": len(chunks)}
            # Seed QaTest table from the dataset itself (Q/A pairs are ground truth).
            for c in chunks:
                if c.question and c.answer:
                    db.add(
                        QaTest(
                            id=str(uuid.uuid4()),
                            project_id=project.id,
                            document_id=document.id,
                            question=c.question,
                            expected_answer=c.answer,
                        )
                    )
            document.status = "embedding"
            await db.commit()
            count = await step4_ingestion.embed_and_upsert(
                chunks,
                collection=project.collection_name,
                embedding=embedding,
                vector_store=vector_store,
                document_id=document.id,
                sparse_embedder=get_sparse_embedder(),
            )
            _chunk_progress.pop(document.id, None)
            document.chunk_count = count
            document.status = "ready"
            invalidate_taxonomy_cache(project.collection_name)
            await db.commit()
            return

        # step1
        md_path = await step1_pdf_to_md.pdf_to_markdown(
            Path(document.stored_path), settings.markdown_dir, docling, vlm=vlm
        )
        document.markdown_path = str(md_path)
        document.status = "chunking"
        await db.commit()

        # Find active rule (optional)
        rule_q = await db.execute(
            select(Rule).where(Rule.project_id == project.id, Rule.is_active == True)  # noqa: E712
        )
        rule = rule_q.scalar_one_or_none()
        system_prompt = rule.system_prompt if rule else None

        # step2
        _chunk_progress[document.id] = {"done": 0, "total": 0}

        def _on_section_done(done: int, total: int) -> None:
            _chunk_progress[document.id] = {"done": done, "total": total}

        chunks = await step2_md_to_json.markdown_to_chunks(
            md_path,
            original_file=document.original_filename,
            project_name=project.name,
            llm=llm,
            system_prompt=system_prompt,
            output_dir=settings.chunks_dir,
            progress_callback=_on_section_done,
        )
        _chunk_progress.pop(document.id, None)

        # step3 (QA tests) - best effort, don't fail the whole job
        try:
            qas = await step3_md_to_qa.generate_qa(md_path, llm=llm)
            for q in qas:
                db.add(
                    QaTest(
                        id=str(uuid.uuid4()),
                        project_id=project.id,
                        document_id=document.id,
                        question=q["question"],
                        expected_answer=q.get("answer", ""),
                    )
                )
        except Exception:  # noqa: BLE001
            logger.exception("QA generation failed (continuing)")

        # step4
        document.status = "embedding"
        await db.commit()
        count = await step4_ingestion.embed_and_upsert(
            chunks,
            collection=project.collection_name,
            embedding=embedding,
            vector_store=vector_store,
            document_id=document.id,
            sparse_embedder=get_sparse_embedder(),
        )
        document.chunk_count = count
        document.status = "ready"
        invalidate_taxonomy_cache(project.collection_name)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training pipeline failed for document %s", document.id)
        document.status = "failed"
        document.error_message = str(exc)[:1500]
        await db.commit()
