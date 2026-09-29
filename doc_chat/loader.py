"""Get markdown out of whatever the user uploaded."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".yaml", ".yml"}


def load_markdown(path: str | Path, *, extract: Callable[[str], str] | None = None) -> str:
    """
    Plain-text files are read as-is — running them through the layout/OCR
    pipeline would mean rasterizing text that is already text. Everything else
    (PDFs, scans, photos) goes through `extract`, normally a call into
    `doc_pipeline.process_document`.
    """
    path = Path(path)
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace")
    if extract is None:
        raise ValueError(
            f"{path.suffix or 'This file type'} needs the extraction pipeline, "
            "but no extractor was provided."
        )
    return extract(str(path))
