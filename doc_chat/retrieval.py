"""BM25 retrieval over document chunks — no extra runtime dependencies."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .chunking import Chunk, chunk_markdown

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class DocumentIndex:
    """Searchable index over one document's chunks."""

    name: str
    markdown: str
    chunks: list[Chunk]
    _doc_tokens: list[Counter] = field(default_factory=list, repr=False)
    _doc_lengths: list[int] = field(default_factory=list, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avg_len: float = 0.0

    @classmethod
    def build(
        cls, name: str, markdown: str, *, max_chars: int = 1400, overlap_chars: int = 200
    ) -> "DocumentIndex":
        chunks = chunk_markdown(markdown, max_chars=max_chars, overlap_chars=overlap_chars)
        index = cls(name=name, markdown=markdown, chunks=chunks)
        index._fit()
        return index

    def _fit(self) -> None:
        self._doc_tokens = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._doc_lengths = [sum(counts.values()) for counts in self._doc_tokens]
        total = len(self._doc_tokens)
        self._avg_len = (sum(self._doc_lengths) / total) if total else 0.0

        document_frequency: Counter = Counter()
        for counts in self._doc_tokens:
            document_frequency.update(counts.keys())
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = [0.0] * len(self.chunks)
        if not query_terms or not self._avg_len:
            return scores
        for position, counts in enumerate(self._doc_tokens):
            length = self._doc_lengths[position]
            total = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + K1 * (1 - B + B * length / self._avg_len)
                total += self._idf.get(term, 0.0) * frequency * (K1 + 1) / denominator
            scores[position] = total
        return scores

    def retrieve(self, query: str, *, top_k: int = 4) -> list[Chunk]:
        """
        Return the `top_k` best-matching chunks in document order. Falls back to
        the opening chunks when the query shares no vocabulary with the document
        (e.g. "summarize this"), which still gives the model something to work
        with instead of an empty context.
        """
        if not self.chunks:
            return []
        scores = self.score(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected = [i for i in ranked[:top_k] if scores[i] > 0]
        if not selected:
            selected = list(range(min(top_k, len(self.chunks))))
        return [self.chunks[i] for i in sorted(selected)]
