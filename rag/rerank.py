"""Cross-encoder reranking (Step 6).

A bi-encoder's dense score is coarse (query and document embedded
independently, then compared from a distance); a cross-encoder reads the
query and each candidate *together* and scores true relevance -- more
accurate, too slow to run over the whole corpus, so it only re-scores the
fused top-N pool from Step 5.

`FakeReranker` (lexical token overlap) is the offline/test stand-in --
useful for tests that shouldn't need to download model weights.
"""
from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def rerank(self, query: str, chunks_by_id: dict, cand_ids: list[int], k: int = 5) -> list[int]: ...


class CrossEncoderReranker:
    _MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._MODEL_NAME)

    def rerank(self, query: str, chunks_by_id: dict, cand_ids: list[int], k: int = 5) -> list[int]:
        pairs = [(query, chunks_by_id[cid]["text"]) for cid in cand_ids]
        scores = self._model.predict(pairs)
        order = sorted(range(len(cand_ids)), key=lambda i: scores[i], reverse=True)
        return [cand_ids[i] for i in order[:k]]


class FakeReranker:
    """Lexical token-overlap reranker -- no model download. Uses the same
    punctuation-stripping tokenizer as bm25_index, for the same reason."""

    def rerank(self, query: str, chunks_by_id: dict, cand_ids: list[int], k: int = 5) -> list[int]:
        from rag.bm25_index import _tokenize

        q_tokens = set(_tokenize(query))

        def score(cid: int) -> int:
            c_tokens = set(_tokenize(chunks_by_id[cid]["text"]))
            return len(q_tokens & c_tokens)

        ordered = sorted(cand_ids, key=score, reverse=True)
        return ordered[:k]