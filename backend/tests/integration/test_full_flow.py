"""End-to-end style test using mock providers: upload -> train -> chat -> qc."""

from __future__ import annotations

import asyncio
import io


async def _wait_for_done(client, project_id, max_iters=30):
    for _ in range(max_iters):
        r = await client.get(f"/api/projects/{project_id}/training/status")
        assert r.status_code == 200
        if r.json()["overall_status"] in {"done", "failed", "partial"}:
            return r.json()
        await asyncio.sleep(0.1)
    raise AssertionError("training did not finish")


async def test_full_smoke_flow(client):
    # 1. create project
    r = await client.post("/api/projects", json={"name": "smoke", "description": ""})
    assert r.status_code == 201
    project = r.json()
    pid = project["id"]

    # 2. upload a fake "PDF" (using .txt so MockDocling returns its content)
    fake = b"# Sample\n\nFirst paragraph.\n\nSecond paragraph mentioning overtime."
    files = {"files": ("sample.txt", io.BytesIO(fake), "text/plain")}
    r = await client.post(f"/api/projects/{pid}/documents", files=files)
    assert r.status_code == 201, r.text
    docs = r.json()
    assert len(docs) == 1

    # 3. start training
    r = await client.post(f"/api/projects/{pid}/training/start")
    assert r.status_code == 200, r.text

    # 4. poll until done
    status = await _wait_for_done(client, pid)
    assert status["overall_status"] == "done", status
    assert status["documents"][0]["chunk_count"] >= 1

    # 5. chat
    r = await client.post(
        "/api/chat",
        json={"project_id": pid, "question": "What about overtime?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]
    assert isinstance(body["citations"], list)

    # 6. qc coverage
    r = await client.get(f"/api/projects/{pid}/qc/coverage")
    assert r.status_code == 200
    cov = r.json()
    assert cov["documents_total"] == 1
    assert cov["chunks_in_db"] >= 1

    # 7. qc audit
    r = await client.post(f"/api/projects/{pid}/qc/audit?limit=3")
    assert r.status_code == 200
    audit = r.json()
    assert audit["total"] >= 0
