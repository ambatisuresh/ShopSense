"""Retrieval evaluation against golden_set.json (Lab C).

`golden_set.json` (unlike the notebook's earthquake-id golden set) doesn't
name gold chunk ids directly — it names the *document titles* a good
answer must cite (`must_cite`). `gold_chunks()` resolves that to the set of
corpus chunk ids belonging to those documents, so precision@k / recall@k /
MRR are computed exactly the way the notebook does, just at document-level
relevance instead of single-row relevance.

Items with an empty `must_cite` — category `injection` or `unanswerable` —
aren't retrieval-scored here: there is no "right document" for a prompt-
injection ticket or a genuinely-unanswerable question, so scoring them
against an empty gold set would only ever be a division-by-zero or a
meaningless 0. `evaluate()` skips them; `eval_groundedness.py`'s
`must_not_contain` / refusal checks are what those categories are actually
graded on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def load_golden_set(path) -> list[dict]:
    return json.loads(Path(path).read_text())


def gold_chunks(item: dict, chunks: list[dict]) -> set[int]:
    titles = set(item.get("must_cite") or [])
    if not titles:
        return set()
    return {c["cid"] for c in chunks if c["doc_title"] in titles}


def precision_at_k(retrieved: list[int], gold: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    return sum(1 for cid in topk if cid in gold) / k


def recall_at_k(retrieved: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & gold) / len(gold)


def mrr(retrieved: list[int], gold: set[int]) -> float:
    for i, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def evaluate(retrieve_fn: Callable[..., list[int]], golden: list[dict], chunks: list[dict],
             k: int = 5) -> dict:
    """Aggregate precision@k / recall@k / MRR over every golden_set.json
    item that has a resolvable gold set. `retrieve_fn(question, k=k)` must
    return an ordered list of chunk ids."""
    rows = []
    scored_ids = []
    for item in golden:
        gold = gold_chunks(item, chunks)
        if not gold:
            continue
        got = retrieve_fn(item["question"], k=k)
        rows.append((precision_at_k(got, gold, k), recall_at_k(got, gold, k), mrr(got, gold)))
        scored_ids.append(item["id"])
    if not rows:
        return {"precision@k": 0.0, "recall@k": 0.0, "mrr": 0.0, "n_scored": 0, "scored_ids": []}
    p = sum(r[0] for r in rows) / len(rows)
    r_ = sum(r[1] for r in rows) / len(rows)
    m = sum(r[2] for r in rows) / len(rows)
    return {"precision@k": p, "recall@k": r_, "mrr": m, "n_scored": len(rows), "scored_ids": scored_ids}