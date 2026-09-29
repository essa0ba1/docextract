from __future__ import annotations

import json

import pytest

from doc_chat import (
    DocumentIndex,
    LLMError,
    build_chat_messages,
    chunk_markdown,
    format_sources,
    load_markdown,
)
from doc_chat import llm as llm_module

SAMPLE = """## Page 1

The annual revenue of Acme Corp reached 4.2 million euros in 2023.

## Page 2

The head office is located in Casablanca and employs 38 people.
"""


def test_chunks_carry_page_and_heading():
    chunks = chunk_markdown(SAMPLE, max_chars=80, overlap_chars=0)
    assert [chunk.page for chunk in chunks] == [1, 2]
    assert chunks[-1].label == "page 2 — Page 2"
    assert "Casablanca" in chunks[-1].text


def test_chunking_respects_max_chars_and_overlaps():
    markdown = "\n\n".join(f"Paragraph number {i} with filler text." for i in range(20))
    chunks = chunk_markdown(markdown, max_chars=120, overlap_chars=30)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert chunks[1].text.startswith(chunks[0].text[-30:])


def test_retrieval_ranks_the_relevant_chunk_first():
    index = DocumentIndex.build("acme.pdf", SAMPLE, max_chars=80, overlap_chars=0)
    chunks = index.retrieve("Where is the head office?", top_k=1)
    assert "Casablanca" in chunks[0].text


def test_retrieval_falls_back_to_opening_chunks_for_unmatched_queries():
    index = DocumentIndex.build("acme.pdf", SAMPLE, max_chars=80, overlap_chars=0)
    chunks = index.retrieve("zzzz", top_k=1)
    assert [chunk.index for chunk in chunks] == [0]


def test_build_chat_messages_includes_context_and_history():
    index = DocumentIndex.build("acme.pdf", SAMPLE)
    chunks = index.retrieve("revenue")
    messages = build_chat_messages(
        "What was the revenue?",
        chunks,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        document_name="acme.pdf",
    )
    assert messages[0]["role"] == "system"
    assert [m["content"] for m in messages[1:3]] == ["hi", "hello"]
    assert "4.2 million" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Question: What was the revenue?")


def test_format_sources_deduplicates_labels():
    chunks = chunk_markdown(SAMPLE, max_chars=80, overlap_chars=0)
    assert format_sources([chunks[0], chunks[0]]) == "\n\n*Sources: page 1 — Page 1*"
    assert format_sources([]) == ""


def test_load_markdown_reads_text_files_without_the_pipeline(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("plain contents", encoding="utf-8")
    assert load_markdown(path) == "plain contents"


def test_load_markdown_requires_an_extractor_for_pdfs(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError):
        load_markdown(path)
    assert load_markdown(path, extract=lambda _: "# extracted") == "# extracted"


class _FakeResponse:
    def __init__(self, *, ok=True, status_code=200, lines=(), payload=None, text=""):
        self.ok = ok
        self.status_code = status_code
        self._lines = lines
        self._payload = payload
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def test_stream_chat_completion_yields_deltas(monkeypatch):
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        "",
        "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]}),
        "data: [DONE]",
    ]
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=False):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _FakeResponse(lines=lines)

    monkeypatch.setattr(llm_module.requests, "post", fake_post)

    deltas = list(
        llm_module.stream_chat_completion(
            [{"role": "user", "content": "hi"}],
            base_url="http://localhost:1234/v1/",
            model="test-model",
            api_key="secret-key",
        )
    )

    assert "".join(deltas) == "Hello world"
    assert captured["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["body"]["stream"] is True


def test_chat_completion_raises_llm_error_with_provider_message(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            ok=False,
            status_code=401,
            payload={"error": {"message": "Incorrect API key"}},
            text="{}",
        )

    monkeypatch.setattr(llm_module.requests, "post", fake_post)

    with pytest.raises(LLMError, match="Incorrect API key"):
        llm_module.chat_completion([{"role": "user", "content": "hi"}])
