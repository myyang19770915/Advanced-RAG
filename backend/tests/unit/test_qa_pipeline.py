"""Unit tests for step1b QA dataset ingestion and qa_direct path."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.pipelines import step1b_qa_to_chunks
from app.pipelines.step5_query import qa_direct, QA_CONFIDENCE_THRESHOLD
from app.providers.vector_store import SearchHit


@pytest.mark.asyncio
async def test_csv_basic(tmp_path: Path) -> None:
    src = tmp_path / "qa.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["編號", "Q (要測試的問題)", "A (標準答案)", "問題分類", "分類細項"])
        w.writerow([1, "請假怎麼申請?", "至 HCP 系統申請。", "考勤管理", "請假"])
        w.writerow([2, "離職證明要去哪裡領?", "向人資處領取。", "離職留停", "N/A"])
        w.writerow([3, "", "缺問題", "X", "Y"])  # skipped
        w.writerow([4, "缺答案", "", "X", "Y"])  # skipped

    chunks = await step1b_qa_to_chunks.qa_file_to_chunks(
        src, original_file="qa.csv", project_name="demo"
    )
    assert len(chunks) == 2

    c0 = chunks[0]
    # text should be Q-only for embedding (not Q+A mixture)
    assert c0.text == "請假怎麼申請?"
    assert c0.l1 == "考勤管理"
    assert c0.l2 == "請假"
    assert c0.primary_type == "qa"
    assert c0.content_types == ["qa"]
    assert c0.question == "請假怎麼申請?"
    assert c0.answer == "至 HCP 系統申請。"
    assert c0.row_no == 1
    assert "考勤管理" in c0.tags and "請假" in c0.tags
    assert c0.suggested_questions == ["請假怎麼申請?"]
    assert c0.original_file == "qa.csv"
    assert c0.chunk_index == 0

    # "N/A" subcategory should be filtered out of l2 and tags.
    c1 = chunks[1]
    assert c1.l1 == "離職留停"
    assert c1.l2 == ""
    assert c1.tags == ["離職留停"]


@pytest.mark.asyncio
async def test_xlsx_basic(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    src = tmp_path / "qa.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["編號", "Q", "A", "問題分類", "分類細項"])
    ws.append([1, "Q1?", "A1.", "薪酬", "勞退"])
    ws.append([2, "Q2?", "A2.", "員工關係", "員工福利"])
    wb.save(src)

    chunks = await step1b_qa_to_chunks.qa_file_to_chunks(
        src, original_file="qa.xlsx", project_name="demo", output_dir=tmp_path / "out"
    )
    assert len(chunks) == 2
    assert chunks[0].l1 == "薪酬"
    assert chunks[1].l2 == "員工福利"
    # chunks.json should be persisted
    out_files = list((tmp_path / "out").glob("*.chunks.json"))
    assert len(out_files) == 1


@pytest.mark.asyncio
async def test_missing_required_columns_raises(tmp_path: Path) -> None:
    src = tmp_path / "bad.csv"
    src.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        await step1b_qa_to_chunks.qa_file_to_chunks(
            src, original_file="bad.csv", project_name="demo"
        )


def test_is_qa_file() -> None:
    assert step1b_qa_to_chunks.is_qa_file("a.xlsx")
    assert step1b_qa_to_chunks.is_qa_file("a.csv")
    assert step1b_qa_to_chunks.is_qa_file(Path("a.XLS"))
    assert not step1b_qa_to_chunks.is_qa_file("a.pdf")
    assert not step1b_qa_to_chunks.is_qa_file("a.md")


# ── qa_direct() ──────────────────────────────────────────────────────────────

def _make_hit(score: float, primary_type: str = "qa", answer: str = "raw answer") -> SearchHit:
    return SearchHit(
        id="abc",
        score=score,
        payload={
            "primary_type": primary_type,
            "question": "問題?",
            "answer": answer,
            "text": "問題?",
            "original_file": "qa.xlsx",
            "chunk_index": 0,
        },
    )


def test_qa_direct_high_confidence() -> None:
    hit = _make_hit(score=QA_CONFIDENCE_THRESHOLD + 0.01)
    is_direct, text, trimmed = qa_direct([hit])
    assert is_direct is True
    assert text == "raw answer"
    assert len(trimmed) == 1


def test_qa_direct_below_threshold() -> None:
    hit = _make_hit(score=QA_CONFIDENCE_THRESHOLD - 0.01)
    is_direct, text, trimmed = qa_direct([hit])
    assert is_direct is False
    assert text == ""
    assert len(trimmed) == 1  # original list unchanged


def test_qa_direct_non_qa_chunk() -> None:
    hit = _make_hit(score=0.99, primary_type="text")
    is_direct, _, _ = qa_direct([hit])
    assert is_direct is False


def test_qa_direct_empty_hits() -> None:
    is_direct, text, trimmed = qa_direct([])
    assert is_direct is False
    assert trimmed == []


def test_qa_direct_trims_to_single_citation() -> None:
    """When QA direct fires, only the top hit is returned regardless of list size."""
    hits = [
        _make_hit(score=0.95, answer="first answer"),
        _make_hit(score=0.82, answer="second answer"),
        _make_hit(score=0.70, answer="third answer"),
    ]
    is_direct, text, trimmed = qa_direct(hits)
    assert is_direct is True
    assert text == "first answer"
    assert len(trimmed) == 1
    assert trimmed[0].payload["answer"] == "first answer"
