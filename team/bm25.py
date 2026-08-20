"""A compact BM25Okapi reimplementation.

Standalone by design: this session doesn't have access to the real M4
rag/bm25_index.py source, only its documented behavior (rank_bm25 when
installed, otherwise this exact fallback — k1=1.5, b=0.75, tokenizer
`re.findall(r"[a-z0-9]+", text.lower())`). Replicating that documented
fallback here, rather than inventing something new, means this module and
your real rag/bm25_index.py should retrieve identically over the same
corpus. Swap this out for an import of the real one once this step is
merged into your project, if you'd rather not maintain two copies.
"""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    """BM25Okapi over a fixed list of documents (plain strings)."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.n_docs = len(documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        )
        self.doc_term_freqs = [Counter(doc) for doc in self.tokenized_docs]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_freq = Counter()
        for doc in self.tokenized_docs:
            for term in set(doc):
                doc_freq[term] += 1
        return {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in doc_freq.items()
        }

    def _score(self, query_terms: list[str], doc_index: int) -> float:
        freqs = self.doc_term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf.get(term, 0.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / (self.avg_doc_length or 1)
            )
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """Return up to `top_k` (doc_index, score) pairs, highest score
        first, excluding zero-score (no term overlap) results entirely."""
        query_terms = tokenize(query)
        scored = [(i, self._score(query_terms, i)) for i in range(self.n_docs)]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]