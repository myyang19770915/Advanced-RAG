from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas import SettingsOut

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsOut)
async def get_current_settings() -> SettingsOut:
    s = get_settings()
    return SettingsOut(
        llm_provider=s.llm_provider,
        llm_base_url=s.llm_base_url,
        llm_model=s.llm_model,
        embedding_provider=s.embedding_provider,
        embedding_model=s.embedding_model,
        embedding_dim=s.embedding_dim,
        vector_store=s.vector_store,
        qdrant_url=s.qdrant_url,
        docling_provider=s.docling_provider,
    )
