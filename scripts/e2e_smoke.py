"""Standalone smoke script.

Drives the backend through the full flow in-process (no docker required),
using mock providers. Useful in CI or for a quick offline sanity check.

Usage:
    cd backend && source .venv/bin/activate
    python ../scripts/e2e_smoke.py
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("DOCLING_PROVIDER", "mock")


async def main() -> int:
    from httpx import ASGITransport, AsyncClient

    from app.db.session import init_db
    from app.main import create_app

    app = create_app()
    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # 1. project
        r = await c.post("/api/projects", json={"name": "smoke-cli"})
        r.raise_for_status()
        pid = r.json()["id"]
        print(f"created project {pid}")

        # 2. upload
        content = b"# Manual\n\nSection A.\n\nSection B describes overtime pay.\n"
        files = {"files": ("manual.txt", io.BytesIO(content), "text/plain")}
        r = await c.post(f"/api/projects/{pid}/documents", files=files)
        r.raise_for_status()
        print(f"uploaded {len(r.json())} document(s)")

        # 3. train
        r = await c.post(f"/api/projects/{pid}/training/start")
        r.raise_for_status()
        print("training started")

        for _ in range(50):
            await asyncio.sleep(0.1)
            r = await c.get(f"/api/projects/{pid}/training/status")
            r.raise_for_status()
            s = r.json()
            if s["overall_status"] in {"done", "failed", "partial"}:
                break
        print(f"training status: {s['overall_status']}, docs={s['documents']}")

        # 4. chat
        r = await c.post("/api/chat", json={"project_id": pid, "question": "overtime?"})
        r.raise_for_status()
        body = r.json()
        print(f"chat answer: {body['answer'][:120]}")
        print(f"citations: {len(body['citations'])}")

        # 5. QC
        r = await c.get(f"/api/projects/{pid}/qc/coverage")
        r.raise_for_status()
        print(f"coverage: {r.json()}")

        r = await c.post(f"/api/projects/{pid}/qc/audit?limit=3")
        r.raise_for_status()
        a = r.json()
        print(f"audit: passed={a['passed']} failed={a['failed']} total={a['total']}")

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
