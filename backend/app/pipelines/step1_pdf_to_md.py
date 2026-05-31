"""Step 1: PDF -> Markdown (+ DoclingDocument JSON) via Docling provider.

If a VLM provider is configured (vlm_provider != 'off'), every picture
extracted from the PDF is sent to the VLM, and the resulting description is
injected back into the DoclingDocument as a ``PictureDescriptionData``
annotation. The re-exported markdown then contains the picture descriptions
inline, so the downstream HybridChunker can index them as searchable text.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from app.providers.docling import DoclingProvider, ExtractedPicture
from app.providers.vlm import VLMProvider

logger = logging.getLogger(__name__)


async def _caption_pictures(
    pictures: list[ExtractedPicture],
    vlm: VLMProvider,
    concurrency: int = 2,
) -> dict[str, str]:
    """Caption every picture concurrently. Returns {self_ref: caption}."""
    if not pictures:
        return {}
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(p: ExtractedPicture) -> tuple[str, str]:
        async with sem:
            try:
                text = await vlm.describe_image(
                    image_bytes=p.image_bytes, image_mime=p.image_mime
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM describe failed for %s: %s", p.self_ref, exc)
                text = ""
            return p.self_ref, text

    results = await asyncio.gather(*[_one(p) for p in pictures])
    return {ref: txt for ref, txt in results if txt}


async def pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
    docling: DoclingProvider,
    vlm: VLMProvider | None = None,
) -> Path:
    """Convert PDF to Markdown and persist to output_dir.

    Also persists:
    - ``<stem>.doc.json`` — full DoclingDocument for layout-aware chunking
    - ``<stem>.images.json`` — list of pictures with page/bbox/caption (debug)

    Returns the markdown file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initial conversion (so the DoclingDocument is cached)
    md_text = await docling.to_markdown(pdf_path)

    # 2. Picture captioning (only if VLM available)
    pictures: list[ExtractedPicture] = []
    captions: dict[str, str] = {}
    if vlm is not None:
        try:
            pictures = await docling.extract_pictures(pdf_path)
            logger.info(
                "step1: extracted %d pictures from %s", len(pictures), pdf_path.name
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("step1: extract_pictures failed: %s", exc)
            pictures = []

        if pictures:
            captions = await _caption_pictures(pictures, vlm)
            logger.info(
                "step1: VLM captioned %d/%d pictures",
                len(captions),
                len(pictures),
            )

            # Re-export markdown with descriptions injected.
            try:
                md_text, _ = await docling.inject_picture_descriptions(
                    pdf_path, captions
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("step1: inject_picture_descriptions failed: %s", exc)

    md_path = output_dir / (pdf_path.stem + ".md")
    md_path.write_text(md_text, encoding="utf-8")

    # 3. Persist final DoclingDocument JSON (after caption injection)
    try:
        doc_json = await docling.to_docling_json(pdf_path)
    except Exception:  # noqa: BLE001
        doc_json = None
    if doc_json is not None:
        doc_json_path = output_dir / (pdf_path.stem + ".doc.json")
        doc_json_path.write_text(
            json.dumps(doc_json, ensure_ascii=False), encoding="utf-8"
        )

    # 4. Persist images.json for debug / future UI features.
    if pictures:
        images_meta = [
            {
                "self_ref": p.self_ref,
                "page": p.page,
                "bbox": p.bbox,
                "page_width": p.page_width,
                "page_height": p.page_height,
                "docling_caption": p.caption,
                "vlm_description": captions.get(p.self_ref, ""),
            }
            for p in pictures
        ]
        images_json_path = output_dir / (pdf_path.stem + ".images.json")
        images_json_path.write_text(
            json.dumps(images_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return md_path
