from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class ExtractedPicture:
    """A picture extracted from a PDF by Docling.

    `self_ref` is the DoclingDocument internal reference (e.g. ``#/pictures/0``)
    used to inject the VLM caption back into the corresponding picture item.
    """

    self_ref: str
    page: int | None
    bbox: list[float] | None              # [l, t, r, b] in PDF points
    page_width: float | None
    page_height: float | None
    image_bytes: bytes
    image_mime: str = "image/png"
    caption: str = ""                     # Docling's own caption text (if any)


class DoclingProvider(Protocol):
    async def to_markdown(self, pdf_path: Path) -> str: ...

    async def to_docling_json(self, pdf_path: Path) -> dict | None:
        """Return serialized DoclingDocument as dict, or None if unsupported (mock)."""
        ...

    async def extract_pictures(self, pdf_path: Path) -> list[ExtractedPicture]:
        """Return all pictures from the PDF. Empty list if unsupported."""
        ...

    async def inject_picture_descriptions(
        self,
        pdf_path: Path,
        descriptions: dict[str, str],
    ) -> tuple[str, dict | None]:
        """Attach VLM descriptions to picture items, then re-export markdown + json.

        `descriptions` is keyed by `self_ref` from ExtractedPicture.
        Returns (markdown, docling_json_dict_or_None).
        """
        ...


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

    async def to_docling_json(self, pdf_path: Path) -> dict | None:
        # Mock provider has no PDF positional info.
        return None

    async def extract_pictures(self, pdf_path: Path) -> list[ExtractedPicture]:
        return []

    async def inject_picture_descriptions(
        self,
        pdf_path: Path,
        descriptions: dict[str, str],
    ) -> tuple[str, dict | None]:
        return await self.to_markdown(pdf_path), None


class LocalDocling:
    """Calls the docling Python SDK (full ML pipeline with layout detection)."""

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    def _convert(self, pdf_path: Path):  # noqa: ANN202
        try:
            import torch  # type: ignore
            torch.backends.mkldnn.enabled = False
        except ImportError:
            pass
        from docling.datamodel.base_models import InputFormat  # type: ignore
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
        from docling.document_converter import (  # type: ignore
            DocumentConverter,
            PdfFormatOption,
        )

        key = str(pdf_path.resolve())
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Enable picture image extraction so we can later send them to a VLM.
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        self._cache[key] = result.document
        return result.document

    async def to_markdown(self, pdf_path: Path) -> str:
        # _convert() runs the docling ML pipeline (CPU heavy, seconds-to-minutes).
        # Offload to a worker thread so the FastAPI event loop stays responsive
        # (otherwise K8s liveness/readiness probes time out).
        doc = await asyncio.to_thread(self._convert, pdf_path)
        return await asyncio.to_thread(doc.export_to_markdown)

    async def to_docling_json(self, pdf_path: Path) -> dict | None:
        doc = await asyncio.to_thread(self._convert, pdf_path)
        try:
            return await asyncio.to_thread(doc.export_to_dict)
        except Exception:  # noqa: BLE001
            return None

    async def extract_pictures(self, pdf_path: Path) -> list[ExtractedPicture]:
        doc = await asyncio.to_thread(self._convert, pdf_path)
        return await asyncio.to_thread(self._extract_pictures_sync, doc)

    @staticmethod
    def _extract_pictures_sync(doc) -> list["ExtractedPicture"]:  # noqa: ANN001
        import io as _io

        pics: list[ExtractedPicture] = []
        for pic in getattr(doc, "pictures", []) or []:
            # page + bbox come from prov[0]
            prov = getattr(pic, "prov", None) or []
            page = None
            bbox = None
            pw = ph = None
            if prov:
                p0 = prov[0]
                page = int(getattr(p0, "page_no", 0)) or None
                bb = getattr(p0, "bbox", None)
                if bb is not None:
                    bbox = [
                        float(bb.l), float(bb.t), float(bb.r), float(bb.b),
                    ]
                # Page size from doc.pages[page_no]
                try:
                    pages_dict = getattr(doc, "pages", None) or {}
                    page_obj = pages_dict.get(page) if isinstance(pages_dict, dict) else None
                    if page_obj is not None:
                        sz = getattr(page_obj, "size", None)
                        if sz is not None:
                            pw = float(getattr(sz, "width", 0)) or None
                            ph = float(getattr(sz, "height", 0)) or None
                except Exception:  # noqa: BLE001
                    pw = ph = None

            # Get PIL image and serialize to PNG bytes
            try:
                pil = pic.get_image(doc)
            except Exception:  # noqa: BLE001
                pil = None
            if pil is None:
                continue
            buf = _io.BytesIO()
            try:
                pil.save(buf, format="PNG")
            except Exception:  # noqa: BLE001
                continue
            img_bytes = buf.getvalue()

            # Existing caption text (Docling's own short caption)
            cap_text = ""
            try:
                cap_text = (pic.caption_text(doc) or "").strip()
            except Exception:  # noqa: BLE001
                cap_text = ""

            self_ref = getattr(pic, "self_ref", "") or ""
            pics.append(
                ExtractedPicture(
                    self_ref=self_ref,
                    page=page,
                    bbox=bbox,
                    page_width=pw,
                    page_height=ph,
                    image_bytes=img_bytes,
                    image_mime="image/png",
                    caption=cap_text,
                )
            )
        return pics

    async def inject_picture_descriptions(
        self,
        pdf_path: Path,
        descriptions: dict[str, str],
    ) -> tuple[str, dict | None]:
        """Attach a PictureDescriptionData annotation to each picture, then
        re-export markdown (which will include the description inline) and json.
        """
        doc = await asyncio.to_thread(self._convert, pdf_path)
        return await asyncio.to_thread(
            self._inject_descriptions_sync, doc, descriptions
        )

    @staticmethod
    def _inject_descriptions_sync(
        doc,  # noqa: ANN001
        descriptions: dict[str, str],
    ) -> tuple[str, dict | None]:
        if descriptions:
            try:
                from docling_core.types.doc.document import (  # type: ignore
                    PictureDescriptionData,
                )
            except Exception:  # noqa: BLE001
                PictureDescriptionData = None  # type: ignore[assignment]

            for pic in getattr(doc, "pictures", []) or []:
                ref = getattr(pic, "self_ref", "") or ""
                desc = descriptions.get(ref, "").strip()
                if not desc:
                    continue
                if PictureDescriptionData is not None:
                    try:
                        ann = PictureDescriptionData(text=desc, provenance="vlm")
                        anns = getattr(pic, "annotations", None)
                        if anns is None:
                            pic.annotations = [ann]
                        else:
                            anns.append(ann)
                        continue
                    except Exception:  # noqa: BLE001
                        pass
                # Fallback: store on a private attribute (won't affect export)
                pic._vlm_description = desc

        md = doc.export_to_markdown()
        try:
            js = doc.export_to_dict()
        except Exception:  # noqa: BLE001
            js = None
        return md, js



class HttpDocling:
    """Calls a remote Docling-compatible HTTP service."""

    def __init__(self, url: str) -> None:
        self._url = url.rstrip("/")

    async def to_markdown(self, pdf_path: Path) -> str:
        import httpx

        # Offload sync file read to a thread; large PDFs can block event loop.
        content = await asyncio.to_thread(pdf_path.read_bytes)
        files = {"file": (pdf_path.name, content, "application/pdf")}
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self._url}/convert", files=files)
            resp.raise_for_status()
            data = resp.json()
        return data.get("markdown", "")

    async def to_docling_json(self, pdf_path: Path) -> dict | None:
        # Remote service may not expose the full DoclingDocument; not implemented yet.
        return None

    async def extract_pictures(self, pdf_path: Path) -> list[ExtractedPicture]:
        return []

    async def inject_picture_descriptions(
        self,
        pdf_path: Path,
        descriptions: dict[str, str],
    ) -> tuple[str, dict | None]:
        return await self.to_markdown(pdf_path), None
