from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

_settings = get_settings()

engine = create_async_engine(_settings.database_url, future=True, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def init_db() -> None:
    """Create tables and run lightweight schema migrations."""
    import re
    from pathlib import Path

    url = _settings.database_url

    # SQLite only: ensure parent directory exists
    m = re.search(r"sqlite.*://+(.+)", url)
    if m:
        db_path = Path(m.group(1).lstrip("/"))
        if not db_path.name == ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)

    # Import models to register them with metadata
    from app import models  # noqa: F401  (side-effect import)
    from app.db.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Migrate: add columns that may be missing (database-agnostic via SQLAlchemy inspector)
    from sqlalchemy import inspect as sa_inspect

    async with engine.connect() as conn:
        existing_cols = await conn.run_sync(
            lambda sync_conn: {c["name"] for c in sa_inspect(sync_conn).get_columns("document")}
        )
        if "version" not in existing_cols:
            await conn.execute(
                text("ALTER TABLE document ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
            )
            await conn.commit()
