"""Tests for rag/local_index.py and rag/qdrant_index.py -- Step 3.

Only the offline stand-in (LocalVectorIndex) and qdrant_index.get_client()'s
guard clause are tested here -- no live Qdrant Cloud calls, same "no live
API calls in the test suite" rule as test_embeddings.py. The real Qdrant
path (create_collection/upsert_chunks/dense_search against an actual
cluster) is what you verified by hand with scripts/run_qdrant_index.py and
scripts/verify_qdrant.py.
"""
import pytest

from rag.local_index import LocalVectorIndex
from rag.qdrant_index import get_client

# Hand-picked vectors, not FakeEmbedder's hashing -- keeps this test's
# expectations exact and independent of how Step 2's embedder happens to
# behave, so a change to FakeEmbedder later can't silently break this file.
CHUNKS = [
    {"cid": 0, "text": "electronics return window", "section": "category_policy", "doc_slug": "category-electronics"},
    {"cid": 1, "text": "grocery items non returnable", "section": "category_policy", "doc_slug": "category-groceries"},
    {"cid": 2, "text": "refund approval tiers", "section": "policy", "doc_slug": "refund-authority"},
]
VECTORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


def _build_index():
    index = LocalVectorIndex()
    index.upsert(CHUNKS, VECTORS)
    return index


def test_count_matches_number_of_upserted_chunks():
    index = _build_index()
    assert index.count() == 3


def test_dense_search_returns_the_closest_vector_first():
    index = _build_index()
    # a query vector pointing exactly at chunk 0's direction should rank chunk 0 first
    results = index.dense_search(lambda q: [1.0, 0.0, 0.0], "electronics", k=1)
    assert results == [0]


def test_dense_search_orders_by_closeness_not_just_exact_match():
    index = _build_index()
    # closer to chunk 1's direction than chunk 2's -- chunk 1 should rank first
    results = index.dense_search(lambda q: [0.0, 0.9, 0.1], "query", k=3)
    assert results[0] == 1


def test_section_filter_restricts_results():
    index = _build_index()
    # query vector points at chunk 0, but filtering to section="policy" should force chunk 2
    results = index.dense_search(lambda q: [1.0, 0.0, 0.0], "anything", k=5, section="policy")
    assert results == [2]


def test_doc_slug_filter_restricts_results():
    index = _build_index()
    results = index.dense_search(lambda q: [1.0, 0.0, 0.0], "anything", k=5, doc_slug="category-groceries")
    assert results == [1]


def test_k_truncates_results():
    index = _build_index()
    results = index.dense_search(lambda q: [1.0, 0.0, 0.0], "anything", k=2)
    assert len(results) == 2


def test_get_client_fails_loudly_without_qdrant_url(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        get_client()