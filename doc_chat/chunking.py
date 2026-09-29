"""Split extracted markdown into retrieval chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PAGE_RE = re.compile(r"^#+\s*page\s*(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of the document, with the context needed to cite it."""

    index: int
    text: str
    heading: str = ""
    page: int | None = None

    @property
    def label(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(f"page {self.page}")
        if self.heading:
            parts.append(self.heading)
        return " — ".join(parts) if parts else f"chunk {self.index + 1}"


def _blocks(markdown: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", markdown) if block.strip()]


def chunk_markdown(
    markdown: str, *, max_chars: int = 1400, overlap_chars: int = 200
) -> list[Chunk]:
    """
    Group markdown blocks into chunks of at most roughly `max_chars`, carrying
    the last `overlap_chars` of a chunk into the next one so passages that
    straddle a boundary stay retrievable. The nearest preceding heading and
    `## Page N` marker travel with each chunk so answers can cite a location.
    """
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    heading = ""
    page: int | None = None
    start_heading = ""
    start_page: int | None = None

    def emit() -> str:
        text = "\n\n".join(buffer).strip()
        if text:
            chunks.append(
                Chunk(index=len(chunks), text=text, heading=start_heading, page=start_page)
            )
        return text

    for block in _blocks(markdown):
        match = HEADING_RE.match(block.splitlines()[0])
        if match:
            heading = match.group(2).strip()
            page_match = PAGE_RE.match(block.splitlines()[0])
            if page_match:
                page = int(page_match.group(1))

        if buffer and buffer_len + len(block) > max_chars:
            text = emit()
            tail = text[-overlap_chars:] if overlap_chars else ""
            buffer = [tail] if tail else []
            buffer_len = len(tail)
            start_heading, start_page = heading, page

        if not buffer:
            start_heading, start_page = heading, page

        buffer.append(block)
        buffer_len += len(block) + 2

    emit()
    return chunks
