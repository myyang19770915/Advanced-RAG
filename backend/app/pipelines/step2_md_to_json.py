"""Step 2: Markdown -> structured JSON chunks via LLM (with metadata).

Strategy:
- Split the markdown by top-level headings so each LLM call processes one
  logical section (avoids context-window overflow for long documents).
- Strip markdown code-block wrappers from LLM responses before JSON parsing
  (Qwen / LLaMA models habitually wrap output in ```json ... ```).
- Fall back to naive paragraph splitter only when JSON parsing fails after
  stripping, so the pipeline never blocks downstream steps.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path

from app.providers.llm import LLMProvider
from app.schemas import ChunkPayload

logger = logging.getLogger(__name__)

# Maximum markdown characters sent to LLM in one call.
# ~4 000 chars ≈ ~1 000 tokens, well within 32 k context.
_SECTION_MAX_CHARS = 4000

_CHUNK_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":                {"type": "string"},
                    "document_title":      {"type": "string"},
                    "article_id":          {"type": "string"},
                    "section_id":          {"type": "string"},
                    "tags":                {"type": "array", "items": {"type": "string"}},
                    "suggested_questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "document_title", "article_id", "section_id",
                             "tags", "suggested_questions"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chunks"],
    "additionalProperties": False,
}

# Dedicated schema for document-level topic classification (simple, isolated task).
_CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "l1": {"type": "string"},
        "l2": {"type": "string"},
        "l3": {"type": "string"},
    },
    "required": ["l1", "l2", "l3"],
    "additionalProperties": False,
}

_CLASSIFY_PROMPT = (
    "You are a topic classifier. Given a document title, classify it into three levels.\n"
    "Return ONLY raw JSON (no markdown): {\"l1\": \"...\", \"l2\": \"...\", \"l3\": \"...\"}\n"
    "Rules:\n"
    "- l1: broad category, 1-2 words. Examples: Technology, Science, Business, Health, Law\n"
    "- l2: medium category, 1-3 words. Examples: Artificial Intelligence, Software Engineering, "
    "Machine Learning, Data Science, Computer Vision\n"
    "- l3: specific topic, 1-4 words. Examples: Document Processing, PDF Parsing, "
    "Open Source Tools, Natural Language Processing\n"
    "Use simple English words only. Do NOT use section names, numbers, or JSON objects."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a document parsing engine. "
    "Split the provided Markdown section into semantic chunks.\n"
    "Return ONLY a raw JSON object (no markdown, no code fences) in this exact shape:\n"
    "{\"chunks\": [{\"text\": \"...\", \"document_title\": \"...\", "
    "\"article_id\": \"...\", \"section_id\": \"...\", "
    "\"tags\": [...], \"suggested_questions\": [...]}]}\n"
    "Rules:\n"
    "- Keep each chunk under 800 characters.\n"
    "- article_id: heading text of the section (e.g. '1 Introduction').\n"
    "- section_id: sub-heading if present, else empty string.\n"
    "- tags: 3-5 relevant keyword strings.\n"
    "- suggested_questions: 1-3 questions a reader might ask about this chunk.\n"
    "- document_title: infer from content or leave as empty string.\n"
    "Output ONLY the JSON. Do NOT wrap in ```json or any other markup."
)


def _sanitize_taxonomy_value(v: str) -> str:
    """Sanitize a l1/l2/l3 value — reject JSON blobs, section numbers, overly long strings."""
    v = (v or "").strip()
    if not v:
        return ""
    # Reject JSON blobs or code fences
    if v[0] in ('{', '[', '`', '>'):
        return ""
    # Reject section numbers like "5.1 Benchmark Dataset" or "3.3 Pipelines"
    if re.match(r"^\d", v):
        return ""
    # Reject values that look like sentences or are suspiciously long
    if len(v) > 60:
        return ""
    return v


async def _classify_document(
    title: str, llm: LLMProvider
) -> tuple[str, str, str]:
    """Single LLM call to classify a document by title into l1/l2/l3 topic labels."""
    try:
        raw = await llm.complete(
            system=_CLASSIFY_PROMPT,
            user=f"Document title: {title}",
            response_format_json=False,
            json_schema=_CLASSIFY_SCHEMA,
            temperature=0.0,
        )
        data = json.loads(_extract_json_str(raw))
        l1 = _sanitize_taxonomy_value(data.get("l1", ""))
        l2 = _sanitize_taxonomy_value(data.get("l2", ""))
        l3 = _sanitize_taxonomy_value(data.get("l3", ""))
        if l1 and l2 and l3:
            logger.info("step2: doc classification l1=%r l2=%r l3=%r", l1, l2, l3)
            return l1, l2, l3
    except Exception as exc:  # noqa: BLE001
        logger.warning("step2: doc classification failed: %s", exc)
    return "General", "Uncategorized", "Document"


def _extract_title(markdown: str) -> str:
    """Extract the first top-level heading from the markdown as the document title."""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return ""


def _naive_split(markdown: str, max_chars: int = 800) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for para in re.split(r"\n\s*\n", markdown):
        para = para.strip()
        if not para:
            continue
        if size + len(para) > max_chars and current:
            parts.append("\n\n".join(current))
            current, size = [para], len(para)
        else:
            current.append(para)
            size += len(para)
    if current:
        parts.append("\n\n".join(current))
    return parts


def _split_into_sections(markdown: str, max_chars: int = _SECTION_MAX_CHARS) -> list[str]:
    """Split markdown by top-level headings; keep each piece ≤ max_chars."""
    # Split on lines that start with one or more # characters
    raw_sections = re.split(r"(?m)^(?=#{1,3} )", markdown)
    sections: list[str] = []
    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
        # If a single section is too large, further split by paragraphs
        if len(sec) <= max_chars:
            sections.append(sec)
        else:
            sub = _naive_split(sec, max_chars)
            sections.extend(sub)
    return sections or [markdown]


def _extract_json_str(raw: str) -> str:
    """Strip markdown code-block wrappers and extract the JSON object/array."""
    # Remove ```json ... ``` or ``` ... ``` wrappers
    raw = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", raw, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    # Try to find the first { ... } span in case of extra prose
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        return m.group(0)
    return raw


def _coerce_chunks(
    data: dict,
    *,
    original_file: str,
    source_file: str,
    project_name: str,
    index_offset: int = 0,
    default_taxonomy: tuple[str, str, str] = ("", "", ""),
) -> list[ChunkPayload]:
    out: list[ChunkPayload] = []
    raw_chunks = data.get("chunks") or []
    for i, c in enumerate(raw_chunks):
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or "").strip()
        if not text:
            continue
        out.append(
            ChunkPayload(
                text=text,
                original_file=original_file,
                source_file=source_file,
                chunk_index=index_offset + i,
                project_name=project_name,
                document_title=c.get("document_title", ""),
                article_id=c.get("article_id", ""),
                section_id=c.get("section_id", ""),
                tags=c.get("tags") or [],
                suggested_questions=c.get("suggested_questions") or [],
                l1=default_taxonomy[0],
                l2=default_taxonomy[1],
                l3=default_taxonomy[2],
            )
        )
    return out


async def _chunks_for_section(
    section: str,
    *,
    original_file: str,
    source_file: str,
    project_name: str,
    llm: LLMProvider,
    prompt: str,
    index_offset: int,
    default_taxonomy: tuple[str, str, str] = ("", "", ""),
) -> list[ChunkPayload]:
    """Call LLM for one section and return parsed chunks."""
    raw = await llm.complete(
        system=prompt,
        user=section,
        response_format_json=False,
        json_schema=_CHUNK_SCHEMA,
        temperature=0.0,
    )
    try:
        data = json.loads(_extract_json_str(raw))
        chunks = _coerce_chunks(
            data,
            original_file=original_file,
            source_file=source_file,
            project_name=project_name,
            index_offset=index_offset,
            default_taxonomy=default_taxonomy,
        )
        if chunks:
            return chunks
        logger.warning("LLM returned empty chunks for section, using naive split. raw=%s", raw[:200])
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for section, using naive split. raw=%s", raw[:200])

    # Fallback for this section
    result = []
    for i, text in enumerate(_naive_split(section)):
        result.append(ChunkPayload(
            text=text,
            original_file=original_file,
            source_file=source_file,
            chunk_index=index_offset + i,
            project_name=project_name,
            document_title=Path(source_file).stem,
            l1=default_taxonomy[0],
            l2=default_taxonomy[1],
            l3=default_taxonomy[2],
        ))
    return result


async def markdown_to_chunks(
    md_path: Path,
    *,
    original_file: str,
    project_name: str,
    llm: LLMProvider,
    system_prompt: str | None = None,
    output_dir: Path | None = None,
    progress_callback: "Callable[[int, int], None] | None" = None,
) -> list[ChunkPayload]:
    markdown = md_path.read_text(encoding="utf-8")
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    sections = _split_into_sections(markdown)
    total = len(sections)
    logger.info("step2: %d section(s) to process for %s", total, original_file)

    # One dedicated classification call for the whole document (much more reliable
    # than asking the chunking LLM to classify each section inline).
    doc_title = _extract_title(markdown) or Path(original_file).stem
    default_taxonomy = await _classify_document(doc_title, llm)

    all_chunks: list[ChunkPayload] = []
    if progress_callback:
        progress_callback(0, total)  # signal total upfront
    for i, sec in enumerate(sections):
        chunks = await _chunks_for_section(
            sec,
            original_file=original_file,
            source_file=md_path.name,
            project_name=project_name,
            llm=llm,
            prompt=prompt,
            index_offset=len(all_chunks),
            default_taxonomy=default_taxonomy,
        )
        all_chunks.extend(chunks)
        if progress_callback:
            progress_callback(i + 1, total)

    logger.info("step2: %d total chunks for %s", len(all_chunks), original_file)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (md_path.stem + ".chunks.json")
        out_path.write_text(
            json.dumps([c.model_dump() for c in all_chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return all_chunks

