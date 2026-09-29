"""Minimal client for any OpenAI-compatible /chat/completions endpoint.

Works against OpenAI, Groq, Together, OpenRouter, vLLM, llama.cpp, Ollama
(`http://localhost:11434/v1`) — anything speaking the same wire format.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, Sequence

import requests

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT = 180

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the endpoint is unreachable or answers with an error."""


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _payload(
    messages: Sequence[Message], model: str, temperature: float, stream: bool
) -> dict:
    return {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "stream": stream,
    }


def _raise_for_error(response: requests.Response) -> None:
    if response.ok:
        return
    detail = response.text.strip()
    try:
        detail = response.json()["error"]["message"]
    except Exception:  # noqa: BLE001 — error bodies vary by provider
        pass
    raise LLMError(f"{response.status_code} from LLM endpoint: {detail}")


def stream_chat_completion(
    messages: Sequence[Message],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> Iterator[str]:
    """Yield answer fragments as the endpoint produces them."""
    try:
        response = requests.post(
            _endpoint(base_url),
            headers=_headers(api_key),
            json=_payload(messages, model, temperature, stream=True),
            timeout=timeout,
            stream=True,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach {base_url}: {exc}") from exc

    with response:
        _raise_for_error(response)
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:") :].strip()
            if data == "[DONE]":
                return
            try:
                choices = json.loads(data).get("choices") or []
            except json.JSONDecodeError:
                continue
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content


def chat_completion(
    messages: Sequence[Message],
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    temperature: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    try:
        response = requests.post(
            _endpoint(base_url),
            headers=_headers(api_key),
            json=_payload(messages, model, temperature, stream=False),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach {base_url}: {exc}") from exc

    _raise_for_error(response)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Unexpected response from LLM endpoint: {response.text[:500]}") from exc
