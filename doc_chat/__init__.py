"""
doc_chat
========

Retrieval-augmented chat on top of the extraction pipeline.

    document (.pdf/.txt/.md/image)
        -> doc_pipeline.process_document (or plain read for text files)
        -> markdown
        -> chunk_markdown  -> DocumentIndex (BM25)
        -> retrieve(question) -> build_chat_messages -> OpenAI-compatible LLM
"""

from .chunking import Chunk, chunk_markdown
from .llm import LLMError, chat_completion, stream_chat_completion
from .loader import TEXT_EXTENSIONS, load_markdown
from .rag import SYSTEM_PROMPT, build_chat_messages, format_sources
from .retrieval import DocumentIndex

__all__ = [
    "Chunk",
    "chunk_markdown",
    "DocumentIndex",
    "LLMError",
    "chat_completion",
    "stream_chat_completion",
    "TEXT_EXTENSIONS",
    "load_markdown",
    "SYSTEM_PROMPT",
    "build_chat_messages",
    "format_sources",
]
