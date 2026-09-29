"""Turn retrieved chunks plus chat history into messages for the LLM."""

from __future__ import annotations

from typing import Sequence

from .chunking import Chunk

SYSTEM_PROMPT = (
    "You are a document assistant. Answer strictly from the document excerpts "
    "provided in the user message. Cite the excerpt labels (e.g. [page 3]) that "
    "support each claim. If the excerpts do not contain the answer, say so "
    "plainly instead of guessing."
)


def format_context(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{chunk.label}]\n{chunk.text}" for chunk in chunks)


def format_sources(chunks: Sequence[Chunk]) -> str:
    if not chunks:
        return ""
    labels = ", ".join(dict.fromkeys(chunk.label for chunk in chunks))
    return f"\n\n*Sources: {labels}*"


def build_chat_messages(
    question: str,
    chunks: Sequence[Chunk],
    history: Sequence[dict[str, str]] = (),
    *,
    document_name: str = "the document",
    max_history_turns: int = 6,
) -> list[dict[str, str]]:
    """
    History is passed through verbatim (trimmed to the last `max_history_turns`
    messages) so follow-ups like "and the second one?" still resolve, while the
    excerpts are attached to the current question only — re-attaching them to
    every past turn would blow up the context on long conversations.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(
        {"role": message["role"], "content": message["content"]}
        for message in list(history)[-max_history_turns:]
        if message.get("content")
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Excerpts from {document_name}:\n\n"
                f"{format_context(chunks)}\n\n"
                f"Question: {question}"
            ),
        }
    )
    return messages
