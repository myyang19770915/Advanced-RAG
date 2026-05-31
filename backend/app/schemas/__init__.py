from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- Project ----------
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""


class ProjectOut(ORMModel):
    id: str
    name: str
    collection_name: str
    status: str
    description: str
    keyword_dictionary: dict[str, Any]
    created_at: datetime


# ---------- Rule ----------
class RuleBuildRequest(BaseModel):
    sample_document_ids: list[str] = Field(default_factory=list)
    domain_hint: str = ""


class RuleCandidate(BaseModel):
    label: str  # A / B / C
    name: str
    system_prompt: str


class RuleBuildResponse(BaseModel):
    candidates: list[RuleCandidate]


class RuleSetActive(BaseModel):
    name: str
    system_prompt: str


class RuleOut(ORMModel):
    id: str
    project_id: str
    name: str
    system_prompt: str
    is_active: bool


# ---------- Document ----------
class DocumentOut(ORMModel):
    id: str
    project_id: str
    original_filename: str
    status: str
    chunk_count: int
    error_message: str
    created_at: datetime
    version: int = 1


# ---------- Training ----------
class TrainingStartResponse(BaseModel):
    job_id: str
    document_ids: list[str]


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    error_message: str = ""
    chunks_done: int = 0
    chunks_total: int = 0


class TrainingStatusResponse(BaseModel):
    project_id: str
    overall_status: str  # idle / running / done / failed
    documents: list[DocumentStatus]


# ---------- Chat / Query ----------
class ChatRequest(BaseModel):
    project_id: str
    question: str
    top_k: int = 5
    session_id: str | None = None


class Citation(BaseModel):
    document_id: str | None = None
    original_file: str
    chunk_index: int
    score: float
    text_preview: str
    download_url: str | None = None
    # PDF positional metadata for inline preview (None if non-PDF or unknown).
    page: int | None = None
    bbox: list[float] | None = None
    page_width: float | None = None
    page_height: float | None = None
    # Content type metadata: which kinds of layout elements feed this chunk.
    # primary_type is the most informative single label for UI badges.
    content_types: list[str] = Field(default_factory=list)
    primary_type: str = "text"


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str | None = None


class ChatSessionSummary(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[Citation] = []
    created_at: str


class ChatSessionDetail(BaseModel):
    id: str
    title: str
    created_at: str
    messages: list[ChatMessageOut]


# ---------- QC ----------
class CoverageReport(BaseModel):
    project_id: str
    documents_total: int
    documents_ready: int
    chunks_in_db: int
    chunks_expected: int
    coverage_ratio: float


class AuditResult(BaseModel):
    qa_test_id: str
    question: str
    answer: str
    passed: bool
    reason: str


class AuditReport(BaseModel):
    project_id: str
    total: int
    passed: int
    failed: int
    results: list[AuditResult]


# ---------- Settings ----------
class SettingsOut(BaseModel):
    llm_provider: str
    llm_base_url: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    vector_store: str
    qdrant_url: str
    docling_provider: str


# ---------- Internal pipeline DTOs ----------
class ChunkPayload(BaseModel):
    document_id: str = ""
    text: str
    original_file: str
    source_file: str
    chunk_index: int
    project_name: str
    document_title: str = ""
    article_id: str = ""
    section_id: str = ""
    tags: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)
    l1: str = ""
    l2: str = ""
    l3: str = ""
    # PDF positional metadata (None for non-PDF or unknown).
    # page: 1-based page number. bbox: [l, t, r, b] in PDF point coordinates,
    # origin BOTTOMLEFT (PDF native). page_width/height in points (1 pt = 1/72 in).
    page: int | None = None
    bbox: list[float] | None = None
    page_width: float | None = None
    page_height: float | None = None
    headings: list[str] = Field(default_factory=list)
    # Content type metadata derived from Docling doc_items[*].label.
    # primary_type ranks: picture > table > formula > code > heading > text.
    # For QA-dataset chunks (xlsx/csv) primary_type == "qa".
    content_types: list[str] = Field(default_factory=list)
    primary_type: str = "text"
    # QA-dataset specific fields (populated when source is xlsx/csv).
    question: str = ""
    answer: str = ""
    row_no: int | None = None
