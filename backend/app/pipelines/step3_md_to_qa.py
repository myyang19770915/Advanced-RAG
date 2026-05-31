"""Step 3: Generate validation QA pairs from a markdown document."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from app.providers.llm import LLMProvider

SYSTEM_PROMPT = (
    "You are a QA generator. Given a markdown document, generate 10 test "
    "questions (and short reference answers) that a knowledge base built "
    "from this document should be able to answer. "
    "Return ONLY a raw JSON object (no markdown, no code fences) in this exact shape: "
    "{\"questions\": [{\"question\": \"...\", \"answer\": \"...\"}]}. "
    "Output ONLY the JSON."
)


def _extract_json_str(raw: str) -> str:
    raw = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", raw, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return m.group(0)
    return raw


_QA_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer":   {"type": "string"},
                },
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


async def generate_qa(md_path: Path, *, llm: LLMProvider) -> list[dict]:
    markdown = await asyncio.to_thread(md_path.read_text, encoding="utf-8")
    # Send only first 4000 chars to avoid context overflow
    content = markdown[:4000]
    raw = await llm.complete(
        system=SYSTEM_PROMPT,
        user=content,
        temperature=0.2,
        response_format_json=False,
        json_schema=_QA_SCHEMA,
    )
    try:
        data = json.loads(_extract_json_str(raw))
        items = data.get("questions") or []
    except json.JSONDecodeError:
        items = []
    # normalise
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        q = (it.get("question") or "").strip()
        a = (it.get("answer") or "").strip()
        if q:
            out.append({"question": q, "answer": a})
    return out
