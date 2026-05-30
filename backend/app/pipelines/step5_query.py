"""Step 5: Advanced RAG query (Agent1 intent + retrieval + Agent2 answer)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
import re

from app.providers.embedding import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.providers.vector_store import SearchHit, VectorStoreProvider

_AGENT1_BASE_PROMPT = (
    "You are Agent1, the query planner / intent analyzer. Given a user "
    "question, output a JSON object describing how to retrieve relevant "
    "chunks. Output keys: rewritten_query (string), keywords (list), "
    "filters (object).\n"
    "Return strictly JSON."
)

# Keep backward-compat constant (used only if no taxonomy is available)
AGENT1_SYSTEM_PROMPT = _AGENT1_BASE_PROMPT


def _build_agent1_prompt(available_taxonomy: dict[str, list[str]] | None) -> str:
    """Build Agent1 system prompt, optionally injecting the known taxonomy.

    When `available_taxonomy` is provided, Agent1 is told the exact l1/l2/l3
    values present in the collection. It must pick from that list or omit the
    filter entirely — preventing mismatches with indexed chunk payloads.
    """
    if not available_taxonomy or not any(available_taxonomy.values()):
        return _AGENT1_BASE_PROMPT

    lines = [_AGENT1_BASE_PROMPT, "\nKnown classification taxonomy for this project:"]
    for level in ("l1", "l2", "l3"):
        vals = available_taxonomy.get(level, [])
        if vals:
            lines.append(f"  {level} values: {vals}")
    lines.append(
        "\nFor filters: if the question clearly matches one of the taxonomy "
        "values above, add l1 (and optionally l2/l3) to filters using the "
        "EXACT string from the list. If no good match exists, omit l1/l2/l3 "
        "from filters entirely."
    )
    return "\n".join(lines)

_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rewritten_query": {"type": "string"},
        "keywords":        {"type": "array", "items": {"type": "string"}},
        "filters":         {"type": "object", "additionalProperties": True},
    },
    "required": ["rewritten_query", "keywords", "filters"],
    "additionalProperties": False,
}

AGENT2_SYSTEM_PROMPT = (
    "You are Agent2. Answer the user question using ONLY the provided "
    "context chunks. Always cite sources by their original_file and chunk "
    "index in the format [orig#idx]. If the answer is not in the context, "
    "reply that you don't know. keywords: answer, context."
)


@dataclass
class QueryPlan:
    rewritten_query: str
    keywords: list[str]
    filters: dict


def _parse_plan(raw: str, fallback_question: str) -> QueryPlan:
    raw = raw.strip()
    fence = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    return QueryPlan(
        rewritten_query=str(data.get("rewritten_query") or fallback_question),
        keywords=list(data.get("keywords") or []),
        filters=dict(data.get("filters") or {}),
    )


def _format_context(hits: list[SearchHit]) -> str:
    lines = []
    for h in hits:
        payload = h.payload or {}
        tag = f"[{payload.get('original_file','?')}#{payload.get('chunk_index',0)}]"
        lines.append(f"{tag} {payload.get('text','')}")
    return "\n---\n".join(lines)


async def plan_query(
    *,
    question: str,
    llm: LLMProvider,
    project_name: str | None = None,
    available_taxonomy: dict[str, list[str]] | None = None,
) -> QueryPlan:
    prompt = _build_agent1_prompt(available_taxonomy)
    raw = await llm.complete(
        system=prompt,
        user=question,
        temperature=0.0,
        response_format_json=False,
        json_schema=_PLAN_SCHEMA,
    )
    plan = _parse_plan(raw, question)
    if project_name:
        plan.filters.setdefault("project_name", project_name)

    # Validate l1/l2/l3 against known taxonomy (case-insensitive).
    # If no taxonomy is available, strip all three to avoid zero-result filters.
    if available_taxonomy and any(available_taxonomy.values()):
        for key in ("l1", "l2", "l3"):
            val = plan.filters.get(key)
            if not val:
                continue
            known = available_taxonomy.get(key, [])
            # Accept exact match or case-insensitive match (use canonical case)
            canonical = next((k for k in known if k.lower() == val.lower()), None)
            if canonical:
                plan.filters[key] = canonical
            else:
                plan.filters.pop(key, None)
    else:
        for key in ("l1", "l2", "l3"):
            plan.filters.pop(key, None)

    return plan


async def retrieve(
    *,
    plan: QueryPlan,
    collection: str,
    embedding: EmbeddingProvider,
    vector_store: VectorStoreProvider,
    top_k: int = 5,
) -> list[SearchHit]:
    vectors = await embedding.embed([plan.rewritten_query])
    return await vector_store.search(
        collection, vectors[0], top_k=top_k, filters=plan.filters or None
    )


async def answer(
    *, question: str, hits: list[SearchHit], llm: LLMProvider
) -> str:
    context = _format_context(hits) or "(no context found)"
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    return await llm.complete(
        system=AGENT2_SYSTEM_PROMPT, user=user, temperature=0.2
    )


async def answer_stream(
    *, question: str, hits: list[SearchHit], llm: LLMProvider
) -> AsyncIterator[str]:
    context = _format_context(hits) or "(no context found)"
    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    async for chunk in llm.stream(
        system=AGENT2_SYSTEM_PROMPT, user=user, temperature=0.2
    ):
        yield chunk


async def run_query(
    *,
    question: str,
    collection: str,
    project_name: str | None,
    llm: LLMProvider,
    embedding: EmbeddingProvider,
    vector_store: VectorStoreProvider,
    top_k: int = 5,
    available_taxonomy: dict[str, list[str]] | None = None,
) -> tuple[str, list[SearchHit]]:
    plan = await plan_query(
        question=question,
        llm=llm,
        project_name=project_name,
        available_taxonomy=available_taxonomy,
    )
    hits = await retrieve(
        plan=plan,
        collection=collection,
        embedding=embedding,
        vector_store=vector_store,
        top_k=top_k,
    )
    text = await answer(question=question, hits=hits, llm=llm)
    return text, hits
