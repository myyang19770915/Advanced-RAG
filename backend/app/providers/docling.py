from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DoclingProvider(Protocol):
    async def to_markdown(self, pdf_path: Path) -> str: ...


class MockDocling:
    """Deterministic markdown for tests.

    If the file is a text file, returns its contents wrapped in Markdown.
    Otherwise emits a fake structured document derived from the filename.
    """

    async def to_markdown(self, pdf_path: Path) -> str:
        if pdf_path.suffix.lower() in {".txt", ".md"} and pdf_path.exists():
            return pdf_path.read_text(encoding="utf-8")
        name = pdf_path.stem
        return (
            f"# {name}\n\n"
            f"## Section 1\nThis is mock content for {name}.\n\n"
            f"## Section 2\nSecond paragraph for {name}, used for chunking tests.\n\n"
            f"## Section 3\nThird and final mock section.\n"
        )


class LocalDocling:
    """Calls the docling Python SDK (full ML pipeline with layout detection)."""

    async def to_markdown(self, pdf_path: Path) -> str:
        try:
            import torch  # type: ignore
            # KVM VMs without CPU feature passthrough lack SSE4.2/AVX2;
            # disabling mkldnn makes PyTorch fall back to pure C++ kernels.
            torch.backends.mkldnn.enabled = False
        except ImportError:
            pass

        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "docling not installed. pip install docling, or set DOCLING_PROVIDER=mock"
            ) from e

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()


class HttpDocling:
    """Calls a remote Docling-compatible HTTP service."""

    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")

    async def to_markdown(self, pdf_path: Path) -> str:
        import httpx

        with pdf_path.open("rb") as f:
            files = {"file": (pdf_path.name, f.read(), "application/pdf")}
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self._url}/convert", files=files)
            resp.raise_for_status()
            data = resp.json()
        return data.get("markdown", "")
