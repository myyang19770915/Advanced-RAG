"""Step 1b: QA dataset (xlsx / csv) -> ChunkPayload list.

Replaces step1 + step2 for QA-pair source files. Each row becomes ONE chunk:

    text = "Q: {question}\nA: {answer}"

so dense / sparse retrieval can match against both the question phrasing and
the canonical answer. Category columns ("問題分類", "分類細項") are written into
Qdrant payload as l1 / l2 so the existing taxonomy filter UI works unchanged.

Supported column synonyms (case- & whitespace-insensitive):

    question:  Q, 問題, question, "Q (要測試的問題)", q
    answer:    A, 答案, answer, "A (標準答案)", 標準答案, a
    l1:        問題分類, category, category1, l1, 主分類
    l2:        分類細項, subcategory, category2, l2, 子分類
    row_id:    編號, id, no, 序號
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

from app.schemas import ChunkPayload

logger = logging.getLogger(__name__)


# Column name → canonical key mapping. Keys are lowercased + whitespace-stripped.
_COLUMN_ALIASES: dict[str, str] = {
    # question
    "q": "question",
    "question": "question",
    "問題": "question",
    "q(要測試的問題)": "question",
    "要測試的問題": "question",
    # answer
    "a": "answer",
    "answer": "answer",
    "答案": "answer",
    "標準答案": "answer",
    "a(標準答案)": "answer",
    # l1
    "問題分類": "l1",
    "category": "l1",
    "category1": "l1",
    "l1": "l1",
    "主分類": "l1",
    # l2
    "分類細項": "l2",
    "subcategory": "l2",
    "category2": "l2",
    "l2": "l2",
    "子分類": "l2",
    # row id
    "編號": "row_id",
    "id": "row_id",
    "no": "row_id",
    "序號": "row_id",
}


def _normalize_header(h: Any) -> str:
    return re.sub(r"\s+", "", str(h or "")).lower()


def _map_headers(raw_headers: list[Any]) -> dict[int, str]:
    """Return {column_index: canonical_key} for recognised columns."""
    mapping: dict[int, str] = {}
    for i, h in enumerate(raw_headers):
        key = _COLUMN_ALIASES.get(_normalize_header(h))
        if key:
            mapping[i] = key
    return mapping


def _row_to_record(row: list[Any], mapping: dict[int, str]) -> dict[str, str]:
    record: dict[str, str] = {}
    for idx, key in mapping.items():
        if idx < len(row):
            v = row[idx]
            record[key] = "" if v is None else str(v).strip()
    return record


def _read_xlsx_sync(path: Path) -> list[list[Any]]:
    import openpyxl  # type: ignore

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active  # first sheet
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return rows


def _read_csv_sync(path: Path) -> list[list[Any]]:
    raw = path.read_bytes()
    # Try utf-8-sig first (handles BOM from Excel exports), then big5, then latin-1.
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    return [list(r) for r in reader]


async def _read_rows(path: Path) -> list[list[Any]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return await asyncio.to_thread(_read_xlsx_sync, path)
    if suffix == ".csv":
        return await asyncio.to_thread(_read_csv_sync, path)
    raise ValueError(f"Unsupported QA file extension: {suffix}")


async def qa_file_to_chunks(
    source_path: Path,
    *,
    original_file: str,
    project_name: str,
    output_dir: Path | None = None,
) -> list[ChunkPayload]:
    """Parse a QA dataset file (xlsx/csv) into ChunkPayload list.

    Each row that has both a non-empty question and answer becomes one chunk.
    Rows missing either field are skipped (and logged).

    If ``output_dir`` is provided, also persists ``<stem>.chunks.json`` mirroring
    the PDF pipeline output for parity / debugging.
    """
    rows = await _read_rows(source_path)
    if not rows:
        logger.warning("step1b: %s is empty", source_path.name)
        return []

    header = rows[0]
    mapping = _map_headers(header)
    if "question" not in mapping.values() or "answer" not in mapping.values():
        raise ValueError(
            f"QA file '{source_path.name}' missing required Question/Answer columns. "
            f"Detected header: {header!r}. Expected one of: Q/問題/question + A/標準答案/answer."
        )

    doc_title = Path(original_file).stem
    chunks: list[ChunkPayload] = []
    skipped = 0
    for i, row in enumerate(rows[1:], start=1):
        rec = _row_to_record(list(row), mapping)
        q = rec.get("question", "").strip()
        a = rec.get("answer", "").strip()
        if not q or not a:
            skipped += 1
            continue
        l1 = rec.get("l1", "").strip() or "QA"
        l2 = rec.get("l2", "").strip() or "General"
        raw_row_id = rec.get("row_id", "").strip()
        try:
            row_no: int | None = int(float(raw_row_id)) if raw_row_id else i
        except ValueError:
            row_no = i
        # Embed Q only: user queries are question-shaped, so Q-only vectors
        # align better with the query embedding than a Q+A mixture.
        # The full answer is stored in the 'answer' payload field and returned
        # verbatim when confidence is high (see step5_query.qa_direct).
        text = q
        tags = [t for t in [l1, l2] if t and t.upper() != "N/A"]
        chunks.append(
            ChunkPayload(
                text=text,
                original_file=original_file,
                source_file=source_path.name,
                chunk_index=len(chunks),
                project_name=project_name,
                document_title=doc_title,
                article_id=l1,
                section_id=l2,
                tags=tags,
                suggested_questions=[q],
                l1=l1,
                l2=l2 if l2.upper() != "N/A" else "",
                l3="",
                content_types=["qa"],
                primary_type="qa",
                question=q,
                answer=a,
                row_no=row_no,
            )
        )

    logger.info(
        "step1b: %s → %d QA chunks (skipped %d empty rows)",
        source_path.name, len(chunks), skipped,
    )

    if output_dir and chunks:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (source_path.stem + ".chunks.json")
        await asyncio.to_thread(
            out_path.write_text,
            json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return chunks


# Extensions this pipeline accepts (used by training dispatcher).
QA_FILE_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".csv"}


def is_qa_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in QA_FILE_EXTENSIONS
