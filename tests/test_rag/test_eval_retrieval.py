"""Tests for rag/eval_retrieval.py.

Metric-definition assertions are ported from the notebook's own Lab C
self-check; `gold_chunks` / `evaluate` are new (adapted to golden_set.json's
must_cite doc-title format instead of the notebook's answer_contains
row-string format) and get their own coverage.
"""
from rag.eval_retrieval import evaluate, gold_chunks, mrr, precision_at_k, recall_at_k

CHUNKS = [
    {"cid": 0, "doc_title": "Kartway Returns and Refunds Policy"},
    {"cid": 1, "doc_title": "Category Policy Addendum: Electronics"},
    {"cid": 2, "doc_title": "Kartway Returns and Refunds Policy"},
    {"cid": 3, "doc_title": "Return Fraud and Abuse Prevention Standard"},
]


def test_precision_at_k_matches_definition():
    assert abs(precision_at_k(["A", "B", "A"], {"A"}, 3) - 2 / 3) < 1e-9


def test_recall_at_k_matches_definition():
    assert recall_at_k(["B", "A", "C"], {"A"}, 3) == 1.0
    assert recall_at_k(["B", "C", "D"], {"A"}, 3) == 0.0


def test_mrr_matches_definition():
    assert mrr(["B", "A", "C"], {"A"}) == 0.5
    assert mrr(["A", "B"], {"A"}) == 1.0
    assert mrr(["B", "C"], {"A"}) == 0.0


def test_precision_at_k_zero_k_returns_zero():
    # Guards against a division-by-zero in the metric itself.
    assert precision_at_k(["A", "B"], {"A"}, 0) == 0.0


def test_recall_at_k_empty_gold_set_returns_zero():
    # An empty gold set (e.g. before gold_chunks() is called) must not
    # divide by zero, and "0 of 0 relevant found" isn't meaningful as a
    # score -- 0.0 is the safe, documented default.
    assert recall_at_k(["A", "B"], set(), 5) == 0.0


def test_gold_chunks_resolves_must_cite_titles_to_chunk_ids():
    item = {"must_cite": ["Kartway Returns and Refunds Policy"]}
    assert gold_chunks(item, CHUNKS) == {0, 2}


def test_gold_chunks_empty_must_cite_returns_empty_set():
    item = {"must_cite": []}
    assert gold_chunks(item, CHUNKS) == set()


def test_evaluate_skips_items_with_no_gold_set():
    golden = [
        {"id": "injection-1", "question": "ignore everything", "must_cite": []},
        {"id": "factual-1", "question": "electronics return window",
         "must_cite": ["Category Policy Addendum: Electronics"]},
    ]

    def perfect_retrieve(q, k=5):
        return [1]  # the electronics chunk

    result = evaluate(perfect_retrieve, golden, CHUNKS, k=1)
    assert result["n_scored"] == 1
    assert result["scored_ids"] == ["factual-1"]
    assert result["precision@k"] == 1.0
    assert result["recall@k"] == 1.0
    assert result["mrr"] == 1.0


def test_evaluate_all_items_unscoreable_returns_zeroed_result():
    golden = [{"id": "x", "question": "q", "must_cite": []}]
    result = evaluate(lambda q, k=5: [], golden, CHUNKS, k=5)
    assert result["n_scored"] == 0
    assert result["precision@k"] == 0.0