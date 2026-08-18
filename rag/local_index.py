"""Brute-force in-memory vector index -- offline stand-in for Qdrant
(Step 3). Mirrors qdrant_index.dense_search's call shape so tests don't
need a live QDRANT_URL. Not a production substitute: O(n) per query, fine
for a ~100-chunk corpus, not for a real production-size one.
"""
from __future__ import annotations

import math
from typing import Optional


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)


class LocalVectorIndex:
    def __init__(self):
        self._chunks: dict[int, dict] = {}
        self._vectors: dict[int, list[float]] = {}

    def upsert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        for c, v in zip(chunks, vectors):
            self._chunks[c["cid"]] = c
            self._vectors[c["cid"]] = v

    def count(self) -> int:
        return len(self._chunks)

    def dense_search(self, embed_query_fn, query: str, k: int = 10,
                      section: Optional[str] = None, doc_slug: Optional[str] = None) -> list[int]:
        qvec = embed_query_fn(query)
        candidates = list(self._chunks.values())
        if section:
            candidates = [c for c in candidates if c["section"] == section]
        if doc_slug:
            candidates = [c for c in candidates if c["doc_slug"] == doc_slug]
        scored = [(c["cid"], _cosine(qvec, self._vectors[c["cid"]])) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:k]]