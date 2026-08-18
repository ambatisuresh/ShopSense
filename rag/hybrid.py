"""Reciprocal Rank Fusion -- combine dense + BM25 ranked lists (Step 5).

`sum(1 / (c + rank))` per chunk across every input ranking, so a chunk
ranked highly by *either* retriever floats to the top: semantic recall
from dense search plus a deterministic exact-token guarantee from BM25.
"""
from __future__ import annotations


def rrf_fuse(rankings: list[list[int]], k: int = 10, c: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0) + 1 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]