"""Simple PDF splitter abstraction.

For MVP we do not actually split a PDF (would need PyMuPDF). We treat each
uploaded file as a single document. The placeholder exists so the
upstream API surface matches the original TigerAI design and we can later
plug in real splitting (page ranges, page-per-file, etc.).
"""

from __future__ import annotations

from pathlib import Path


def split(file_path: Path, output_dir: Path) -> list[Path]:
    """Return list of (possibly split) document paths.

    Current behaviour: returns [file_path] unchanged.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return [file_path]
