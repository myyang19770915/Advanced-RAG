from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol

import httpx


class LLMProvider(Protocol):
    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.2,
        response_format_json: bool = False,
        json_schema: dict | None = None,
    ) -> str: ...

    async def stream(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        if False:  # pragma: no cover - protocol stub
            yield ""


class MockLLM:
    """Deterministic mock for tests and offline demos.

    Strategy: inspects the system prompt for cue words and returns
    realistic structured output. For arbitrary prompts it echoes a short
    canned answer.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.2,
        response_format_json: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        self.calls.append({"system": system, "user": user})

        s = system.lower()

        # Audit judge (must be checked first because the judge prompt mentions
        # "QA", "answer", and "context" which would otherwise match the QA /
        # Agent2 branches below).
        if "judge" in s or "audit" in s:
            return json.dumps({"passed": True, "reason": "mock judge ok"})

        # Rule builder (must come before chunk+json because RULE_BUILDER_SYSTEM
        # mentions both "chunks" and "JSON" in its text).
        if "rule builder" in s or ("candidates" in s and "system_prompt" in s):
            return json.dumps(
                {
                    "candidates": [
                        {"label": "A", "name": "Strict", "system_prompt": "Extract sections strictly."},
                        {"label": "B", "name": "Balanced", "system_prompt": "Extract sections with metadata."},
                        {"label": "C", "name": "Loose", "system_prompt": "Summarize and chunk freely."},
                    ]
                },
                ensure_ascii=False,
            )

        # Markdown -> JSON chunks
        if "chunk" in s and "json" in s:
            chunks = []
            paragraphs = [p.strip() for p in user.split("\n\n") if p.strip()]
            for i, p in enumerate(paragraphs[:20]):
                chunks.append(
                    {
                        "text": p,
                        "document_title": "Mock Document",
                        "article_id": f"sec-{i+1}",
                        "section_id": "",
                        "tags": ["mock", "demo"],
                        "suggested_questions": [f"Q about {p[:20]}?"],
                        "l1": "demo",
                        "l2": "mock",
                        "l3": "chunk",
                    }
                )
            return json.dumps({"chunks": chunks}, ensure_ascii=False)

        # Generate QA
        if "qa" in s or "test question" in s or "generate" in s and "question" in s:
            return json.dumps(
                {
                    "questions": [
                        {"question": f"Mock question {i+1}?", "answer": f"Mock answer {i+1}"}
                        for i in range(5)
                    ]
                },
                ensure_ascii=False,
            )

        # (Rule builder is handled above, before chunk+json)

        # Agent1: intent
        if "intent" in s or "query plan" in s:
            return json.dumps(
                {
                    "rewritten_query": user.strip(),
                    "keywords": [w for w in user.split() if len(w) > 2][:5],
                    "filters": {},
                },
                ensure_ascii=False,
            )

        # Agent2: answer
        if "answer" in s and "context" in s:
            return f"[mock answer] Based on the provided context, here is a stub answer for: {user[:120]}"

        return f"[mock] {user[:200]}"

    async def stream(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        full = await self.complete(system=system, user=user, model=model, temperature=temperature)
        # Yield word by word
        for token in full.split():
            yield token + " "


class OpenAICompatibleLLM:
    """Calls an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, api_key: str, default_model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.2,
        response_format_json: bool = False,
        json_schema: dict | None = None,
    ) -> str:
        # Append /no_think for Qwen3-style reasoning models so they skip
        # chain-of-thought and return the answer directly in `content`.
        system_prompt = system + "\n/no_think"
        payload: dict = {
            "model": model or self._default_model,
            "temperature": temperature,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
        }
        # LM Studio supports "json_schema" and "text" only (not "json_object").
        # When a schema is provided use json_schema enforcement (token-level).
        # When only response_format_json=True is set, fall back to prompt-only.
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        msg = data["choices"][0]["message"]
        # Qwen3 thinking models put the actual answer in `content`; if empty,
        # fall back to `reasoning_content` (the thinking trace).
        return msg.get("content") or msg.get("reasoning_content", "")

    async def stream(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        payload = {
            "model": model or self._default_model,
            "temperature": temperature,
            "stream": True,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:  # noqa: BLE001
                        continue
