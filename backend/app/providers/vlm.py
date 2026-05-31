"""Vision-Language Model providers.

A VLM takes an image + a text prompt and returns a textual description.
We use OpenAI-compatible /chat/completions with `image_url` content parts
(base64 data URI). LM Studio with Qwen2.5-VL / Qwen3-VL supports this.
"""

from __future__ import annotations

import base64
import io
from typing import Protocol

import httpx


class VLMProvider(Protocol):
    async def describe_image(
        self,
        *,
        image_bytes: bytes,
        image_mime: str = "image/png",
        prompt: str = "",
        max_tokens: int = 512,
    ) -> str:
        """Return a textual description of the image. Empty string on failure."""
        ...


def _maybe_downscale(image_bytes: bytes, max_side: int) -> tuple[bytes, str]:
    """Best-effort downscale via Pillow; returns (bytes, mime)."""
    try:
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > max_side:
            ratio = max_side / float(m)
            img = img.resize((int(w * ratio), int(h * ratio)))
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue(), "image/png"
    except Exception:  # noqa: BLE001
        return image_bytes, "image/png"


class MockVLM:
    async def describe_image(
        self,
        *,
        image_bytes: bytes,
        image_mime: str = "image/png",
        prompt: str = "",
        max_tokens: int = 512,
    ) -> str:
        return f"[mock image description, {len(image_bytes)} bytes]"


class OffVLM:
    """A no-op VLM that returns empty string — disables image captioning."""

    async def describe_image(
        self,
        *,
        image_bytes: bytes,
        image_mime: str = "image/png",
        prompt: str = "",
        max_tokens: int = 512,
    ) -> str:
        return ""


DEFAULT_VLM_PROMPT = (
    "You are an expert at describing figures, diagrams, charts, and screenshots from "
    "academic and technical PDFs. Describe this image in detail in the same language "
    "as any text shown in the image (default to English if no text). Include:\n"
    "- What kind of figure it is (architecture diagram, plot, table, photo, screenshot, etc.)\n"
    "- Key labels, axis names, numeric values, and any visible text (transcribe accurately)\n"
    "- Structural relationships (arrows, groupings, flows)\n"
    "- The main insight or message conveyed.\n"
    "Be factual and concise (3-8 sentences). Do not invent details that are not visible."
)


class OpenAICompatibleVLM:
    """Calls an OpenAI-compatible chat endpoint with image_url content parts."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        max_image_side: int = 1024,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._max_image_side = max_image_side

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def describe_image(
        self,
        *,
        image_bytes: bytes,
        image_mime: str = "image/png",
        prompt: str = "",
        max_tokens: int = 512,
    ) -> str:
        if not image_bytes:
            return ""
        img_bytes, mime = _maybe_downscale(image_bytes, self._max_image_side)
        b64 = base64.b64encode(img_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        user_prompt = (prompt or DEFAULT_VLM_PROMPT).strip() + "\n/no_think"
        payload = {
            "model": self._default_model,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content", "") or ""
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            return f"[vlm error: {type(exc).__name__}]"
