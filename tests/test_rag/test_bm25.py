"""Tests for rag/bm25_index.py -- Step 4.

Runs whether or not `rank_bm25` is installed -- BM25Index falls back to
`_SimpleBM25` when it isn't, and every assertion here only depends on the
public `.search()` contract.
"""
from rag.bm25_index import BM25Index, _tokenize

CHUNKS = [
    {"cid": 0, "text": "Electronics returns must be initiated within 15 days of receipt"},
    {"cid": 1, "text": "Grocery items are non-returnable and non-refundable"},
    {"cid": 2, "text": "Apparel returns may be initiated within 45 days of receipt"},
    {"cid": 3, "text": "Refunds up to 2000 rupees are auto-approved by an agent"},
]


def test_exact_keyword_match_ranks_first():
    idx = BM25Index(CHUNKS)
    results = idx.search("grocery non-returnable", k=2)
    assert results[0] == 1


def test_search_returns_chunk_ids_not_positions():
    idx = BM25Index([{"cid": 99, "text": "unique clause about warranty"}])
    results = idx.search("warranty", k=1)
    assert results == [99]


def test_disjoint_query_still_returns_k_results_deterministically():
    idx = BM25Index(CHUNKS)
    results_a = idx.search("nonexistent gibberish query", k=3)
    results_b = idx.search("nonexistent gibberish query", k=3)
    assert results_a == results_b
    assert len(results_a) == 3


def test_electronics_vs_apparel_disambiguated_by_exact_token():
    idx = BM25Index(CHUNKS)
    assert idx.search("electronics 15 days", k=1) == [0]
    assert idx.search("apparel 45 days", k=1) == [2]


def test_tokenizer_strips_trailing_punctuation():
    # The exact bug this module's tokenizer was written to avoid: a plain
    # .split() would leave "electronics?" as its own token, which would
    # never match the corpus's clean "electronics".
    assert _tokenize("returning electronics?") == ["returning", "electronics"]
    assert _tokenize("What is the cost: $2,000?") == ["what", "is", "the", "cost", "2", "000"]


def test_tokenizer_lowercases():
    assert _tokenize("ELECTRONICS Return") == ["electronics", "return"]


def test_question_with_trailing_punctuation_still_matches_clean_corpus_word():
    idx = BM25Index(CHUNKS)
    # Without the punctuation-aware tokenizer, "electronics?" would never
    # match the corpus's "Electronics" token and this chunk would score 0.
    results = idx.search("What about returning electronics?", k=1)
    assert results == [0]