from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import chat, documents, projects, qc, rules, settings
from app.api import training as training_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("INFO")
    await init_db()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="Advanced RAG API",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    api_prefix = "/api"
    app.include_router(projects.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)
    app.include_router(documents.download_router, prefix=api_prefix)
    app.include_router(rules.router, prefix=api_prefix)
    app.include_router(training_router.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(qc.router, prefix=api_prefix)
    app.include_router(settings.router, prefix=api_prefix)

    return app


app = create_app()
