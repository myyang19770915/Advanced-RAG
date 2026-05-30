"""Step 3b: AI Quality Gate."""

from __future__ import annotations

import json

from app.providers.llm import LLMProvider

JUDGE_SYSTEM_PROMPT = (
    "You are a strict QA judge / audit. Given a QUESTION, a CANDIDATE ANSWER, "
    "and the supporting CONTEXT chunks, decide whether the answer is "
    "correct, faithful to the context, and complete. "
    "Return a JSON object with keys: passed (boolean) and reason (string)."
)

_JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["passed", "reason"],
    "additionalProperties": False,
}


async def judge(
    *,
    question: str,
    answer: str,
    context_chunks: list[str],
    llm: LLMProvider,
) -> dict:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        f"CONTEXT:\n" + "\n---\n".join(context_chunks[:8])
    )
    raw = await llm.complete(
        system=JUDGE_SYSTEM_PROMPT,
        user=user,
        temperature=0.0,
        response_format_json=False,
        json_schema=_JUDGE_SCHEMA,
    )
    try:
        data = json.loads(raw)
        return {
            "passed": bool(data.get("passed", False)),
            "reason": str(data.get("reason", "")),
        }
    except json.JSONDecodeError:
        return {"passed": False, "reason": "judge returned invalid JSON"}
