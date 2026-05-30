"""Pytest fixtures.

Each test gets a fresh in-memory SQLite DB and clean provider singletons.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("DOCLING_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_DIM", "32")
os.environ.setdefault("ADMIN_TOKEN", "test-token")


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR to a tmp folder per test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Reset config + provider caches
    from app.core.config import get_settings
    from app.providers.factory import reset_provider_cache

    get_settings.cache_clear()
    reset_provider_cache()
    yield


@pytest.fixture
async def app():
    from app.main import create_app
    application = create_app()
    # Run lifespan manually to init DB
    from app.db.session import init_db
    await init_db()
    return application


@pytest.fixture
async def client(app):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
