from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    collection_name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), default="created")
    description: Mapped[str] = mapped_column(String(1000), default="")
    keyword_dictionary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    documents = relationship(
        "Document", back_populates="project", cascade="all, delete-orphan"
    )
    rules = relationship(
        "Rule", back_populates="project", cascade="all, delete-orphan"
    )
    qa_tests = relationship(
        "QaTest", back_populates="project", cascade="all, delete-orphan"
    )
    chat_sessions = relationship(
        "ChatSession", back_populates="project", cascade="all, delete-orphan"
    )
