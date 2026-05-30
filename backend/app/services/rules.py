"""Rule builder: produce candidate system prompts from sample documents."""

from __future__ import annotations

import json
from pathlib import Path

from app.providers.llm import LLMProvider
from app.schemas import RuleCandidate

RULE_BUILDER_SYSTEM = (
    "You are a Rule Builder for a RAG system. Given sample markdown "
    "documents and an optional domain hint, propose 3 candidate system "
    "prompts (label A, B, C) for parsing this domain's documents into "
    "structured chunks with metadata. Each candidate must include label, "
    "name, system_prompt. Return strictly JSON: "
    "{\"candidates\": [{...}, {...}, {...}]}."
)

_RULE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label":         {"type": "string"},
                    "name":          {"type": "string"},
                    "system_prompt": {"type": "string"},
                },
                "required": ["label", "name", "system_prompt"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


async def build_candidates(
    *, sample_markdowns: list[Path], domain_hint: str, llm: LLMProvider
) -> list[RuleCandidate]:
    samples = []
    for p in sample_markdowns[:3]:
        if p.exists():
            samples.append(f"### {p.name}\n" + p.read_text(encoding="utf-8")[:2000])
    user = f"DOMAIN HINT: {domain_hint or '(none)'}\n\nSAMPLES:\n" + "\n\n".join(samples)
    raw = await llm.complete(
        system=RULE_BUILDER_SYSTEM,
        user=user,
        temperature=0.3,
        response_format_json=False,
        json_schema=_RULE_SCHEMA,
    )
    raw = raw.strip()
    try:
        data = json.loads(raw)
        items = data.get("candidates") or []
    except json.JSONDecodeError:
        # fallback: try to extract JSON from markdown fences or bare object
        import re
        fence = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', raw, re.DOTALL)
        if fence:
            raw = fence.group(1).strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            data = json.loads(raw)
            items = data.get("candidates") or []
        except json.JSONDecodeError:
            items = []
    out: list[RuleCandidate] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        out.append(
            RuleCandidate(
                label=str(it.get("label", "?"))[:4],
                name=str(it.get("name", "Candidate")),
                system_prompt=str(it.get("system_prompt", "")),
            )
        )
    return out
