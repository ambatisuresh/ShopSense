"""BM25 exact-token retrieval -- the deterministic half of hybrid search
(Step 4). Uses rank_bm25 when installed; falls back to a compact
from-scratch BM25Okapi reimplementation if it isn't, so this module has no
hard dependency on that package.

One deliberate choice worth knowing about upfront: tokenization is
`re.findall(r"[a-z0-9]+", text.lower())`, not a plain `.lower().split()`.
Plain whitespace-split leaves trailing punctuation stuck to a word -- a
question ending "...for returning electronics?" tokenizes to
"electronics?", which will never exact-match the corpus's clean
"electronics" token, silently breaking keyword matching on any question
that isn't already punctuation-free (i.e. almost all real questions).
"""
from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _SimpleBM25:
    """Minimal BM25Okapi reimplementation, used only when rank_bm25 isn't importable."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / len(corpus_tokens)) if corpus_tokens else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in corpus_tokens:
            freqs: dict[str, int] = {}
            for tok in doc:
                freqs[tok] = freqs.get(tok, 0) + 1
            self.doc_freqs.append(freqs)
            for tok in freqs:
                df[tok] = df.get(tok, 0) + 1
        n = len(corpus_tokens)
        self.idf = {tok: math.log((n - freq + 0.5) / (freq + 0.5) + 1) for tok, freq in df.items()}

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.doc_freqs)
        for tok in query_tokens:
            idf = self.idf.get(tok)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(tok, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        self._corpus_tokens = [_tokenize(c["text"]) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus_tokens)
        except ImportError:
            self._bm25 = _SimpleBM25(self._corpus_tokens)

    def search(self, query: str, k: int = 10) -> list[int]:
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i]["cid"] for i in order]