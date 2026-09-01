"""Pure BM25 Okapi ranking — no Hermes runtime imports.

Retrieval core for the skill-retrieval plugin. Standard Okapi BM25
(k1=1.5, b=0.75, Lucene-style clipped IDF) implemented with the stdlib only
so the module unit-tests without the runtime installed.
"""

from __future__ import annotations

import math
import re
from typing import Iterable

K1 = 1.5
B = 0.75

_WORD_RE = re.compile(r"[^\w\s]+")


def tokenize(text) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Defends against non-str ``user_message`` payloads (some platforms deliver
    the message as a list of text parts, a dict, or None) by flattening to a
    plain string first.
    """
    if isinstance(text, dict):
        text = text.get("text") or text.get("caption") or ""
    if not isinstance(text, str):
        if isinstance(text, (list, tuple)):
            parts: list[str] = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    value = part.get("text") or part.get("caption")
                    if isinstance(value, str):
                        parts.append(value)
            text = " ".join(parts)
        elif text is None:
            text = ""
        else:
            text = str(text)
    text = text.lower()
    text = _WORD_RE.sub(" ", text)
    return text.split()


class BM25Index:
    """Okapi BM25 retriever over a small corpus of (id, text) documents.

    Builds an inverted index with precomputed per-(term, doc) weights so a
    query only touches the posting lists of its terms, never the full corpus.
    """

    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self._built = False
        self._ids: list[str] = []
        self._vocab: dict[str, int] = {}
        self._postings: dict[int, list[tuple[int, float]]] = {}

    def build(self, ids: Iterable[str], texts: Iterable[str]) -> None:
        ids = list(ids)
        texts = list(texts)
        if len(ids) != len(texts):
            raise ValueError("ids and texts must be the same length")
        self._ids = ids

        vocab: dict[str, int] = {}
        tokenized: list[list[str]] = []
        doc_lens: list[int] = []
        for text in texts:
            tokens = tokenize(text)
            tokenized.append(tokens)
            doc_lens.append(len(tokens))
            for term in tokens:
                if term not in vocab:
                    vocab[term] = len(vocab)
        self._vocab = vocab

        n_docs = len(texts)
        avgdl = sum(doc_lens) / n_docs if n_docs else 1.0

        df: dict[int, int] = {}
        for tokens in tokenized:
            for tid in {vocab[t] for t in tokens}:
                df[tid] = df.get(tid, 0) + 1

        idf: dict[int, float] = {}
        for tid, count in df.items():
            idf[tid] = max(0.0, math.log((n_docs - count + 0.5) / (count + 0.5)))

        postings: dict[int, list[tuple[int, float]]] = {}
        k1, b = self.k1, self.b
        for i, tokens in enumerate(tokenized):
            if not tokens:
                continue
            counts: dict[int, int] = {}
            for term in tokens:
                tid = vocab[term]
                counts[tid] = counts.get(tid, 0) + 1
            dl = doc_lens[i]
            denom = k1 * (1.0 - b + b * dl / avgdl)
            for tid, tf in counts.items():
                sat = (tf * (k1 + 1.0)) / (tf + denom)
                postings.setdefault(tid, []).append((i, sat * idf[tid]))

        self._postings = postings
        self._built = True

    def retrieve(self, query: str, top_k: int = 6) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(id, score)`` pairs, highest score first."""
        if not self._built:
            return []
        scores: dict[int, float] = {}
        seen: set[int] = set()
        for term in tokenize(query):
            tid = self._vocab.get(term)
            if tid is None or tid in seen:
                continue
            seen.add(tid)
            for doc_idx, weight in self._postings.get(tid, ()):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + weight
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [
            (self._ids[idx], score)
            for idx, score in ranked[:top_k]
            if score > 0
        ]
