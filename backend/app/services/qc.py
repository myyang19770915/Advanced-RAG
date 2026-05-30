"""Quality Control services: coverage check + answer audit."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, Project, QaTest
from app.pipelines import step3b_quality_gate, step5_query
from app.providers.factory import get_embedding, get_llm, get_vector_store
from app.schemas import AuditReport, AuditResult, CoverageReport


async def coverage(db: AsyncSession, project: Project) -> CoverageReport:
    vector_store = get_vector_store()

    docs_q = await db.execute(
        select(Document).where(Document.project_id == project.id)
    )
    docs = list(docs_q.scalars())
    documents_total = len(docs)
    documents_ready = sum(1 for d in docs if d.status == "ready")
    chunks_expected = sum(d.chunk_count for d in docs)
    chunks_in_db = await vector_store.count(
        project.collection_name, filters={"project_name": project.name}
    )
    ratio = (chunks_in_db / chunks_expected) if chunks_expected else 0.0
    return CoverageReport(
        project_id=project.id,
        documents_total=documents_total,
        documents_ready=documents_ready,
        chunks_in_db=chunks_in_db,
        chunks_expected=chunks_expected,
        coverage_ratio=ratio,
    )


async def audit(db: AsyncSession, project: Project, *, limit: int = 10) -> AuditReport:
    llm = get_llm()
    embedding = get_embedding()
    vector_store = get_vector_store()

    q = await db.execute(
        select(QaTest)
        .where(QaTest.project_id == project.id, QaTest.status == "pending")
        .limit(limit)
    )
    tests = list(q.scalars())

    results: list[AuditResult] = []
    passed = 0
    for t in tests:
        answer, hits = await step5_query.run_query(
            question=t.question,
            collection=project.collection_name,
            project_name=project.name,
            llm=llm,
            embedding=embedding,
            vector_store=vector_store,
            top_k=5,
        )
        verdict = await step3b_quality_gate.judge(
            question=t.question,
            answer=answer,
            context_chunks=[h.payload.get("text", "") for h in hits],
            llm=llm,
        )
        t.status = "passed" if verdict["passed"] else "failed"
        t.audit = {**verdict, "answer": answer}
        if verdict["passed"]:
            passed += 1
        results.append(
            AuditResult(
                qa_test_id=t.id,
                question=t.question,
                answer=answer,
                passed=verdict["passed"],
                reason=verdict["reason"],
            )
        )
    await db.commit()
    return AuditReport(
        project_id=project.id,
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )
