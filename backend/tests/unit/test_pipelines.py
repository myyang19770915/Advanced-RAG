from pathlib import Path

from app.pipelines.step1_pdf_to_md import pdf_to_markdown
from app.pipelines.step2_md_to_json import markdown_to_chunks
from app.providers.docling import MockDocling
from app.providers.llm import MockLLM


async def test_step1_writes_markdown(tmp_path: Path):
    src = tmp_path / "sample.txt"
    src.write_text("# Hello\n\nworld", encoding="utf-8")
    out = await pdf_to_markdown(src, tmp_path / "md", MockDocling())
    assert out.exists()
    assert out.suffix == ".md"
    assert "Hello" in out.read_text(encoding="utf-8")


async def test_step2_produces_chunks_from_mock_llm(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# Title\n\nParagraph one with content.\n\nParagraph two with more content.\n",
        encoding="utf-8",
    )
    chunks = await markdown_to_chunks(
        md,
        original_file="doc.pdf",
        project_name="p1",
        llm=MockLLM(),
        output_dir=tmp_path / "chunks",
    )
    assert len(chunks) >= 1
    assert all(c.project_name == "p1" for c in chunks)
    assert all(c.original_file == "doc.pdf" for c in chunks)
    assert (tmp_path / "chunks" / "doc.chunks.json").exists()
