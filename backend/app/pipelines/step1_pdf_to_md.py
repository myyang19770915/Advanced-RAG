"""Step 1: PDF -> Markdown via Docling provider."""

from __future__ import annotations

from pathlib import Path

from app.providers.docling import DoclingProvider


async def pdf_to_markdown(
    pdf_path: Path, output_dir: Path, docling: DoclingProvider
) -> Path:
    """Convert PDF to Markdown and persist to output_dir.

    Returns the markdown file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    md_text = await docling.to_markdown(pdf_path)
    md_path = output_dir / (pdf_path.stem + ".md")
    md_path.write_text(md_text, encoding="utf-8")
    return md_path
