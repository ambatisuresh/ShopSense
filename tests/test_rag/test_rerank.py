"""Tests for rag/rerank.py.

Everything here runs against FakeReranker -- the offline stand-in for
CrossEncoderReranker. That's deliberate, not a shortcut: CrossEncoderReranker
needs sentence-transformers + a downloaded model (`cross-encoder/ms-marco-
MiniLM-L-6-v2`), neither of which this environment (or your Intel Mac,
right now) can reliably get. FakeReranker implements the exact same
`rerank(query, chunks_by_id, cand_ids, k) -> list[int]` contract, so these
tests pin down that contract regardless of which implementation is plugged
in later.
"""
from rag.rerank import CrossEncoderReranker, FakeReranker

CHUNKS_BY_ID = {
    1: {"cid": 1, "text": "electronics returns must be initiated within 15 days of receipt"},
    2: {"cid": 2, "text": "grocery items are non-returnable and non-refundable"},
    3: {"cid": 3, "text": "apparel returns may be initiated within 45 days"},
    4: {"cid": 4, "text": "furniture returns must be initiated within 20 days of delivery"},
}


def test_rerank_returns_only_ids_from_the_candidate_pool():
    reranker = FakeReranker()
    cands = [1, 2, 3]
    ranked = reranker.rerank("electronics return window", CHUNKS_BY_ID, cands, k=2)
    assert set(ranked).issubset(set(cands))
    assert len(ranked) == 2


def test_lexically_closer_candidate_ranks_first():
    reranker = FakeReranker()
    ranked = reranker.rerank("electronics returns 15 days", CHUNKS_BY_ID, [1, 2, 3], k=3)
    assert ranked[0] == 1


def test_k_larger_than_pool_returns_whole_pool():
    reranker = FakeReranker()
    ranked = reranker.rerank("anything", CHUNKS_BY_ID, [1, 2], k=10)
    assert len(ranked) == 2


def test_empty_candidate_pool_returns_empty():
    # rerank() is always called on some upstream shortlist (step 5's fused
    # list) -- if that shortlist is empty, rerank must not error, just
    # pass the emptiness through.
    reranker = FakeReranker()
    assert reranker.rerank("electronics", CHUNKS_BY_ID, [], k=5) == []


def test_k_zero_returns_empty():
    reranker = FakeReranker()
    assert reranker.rerank("electronics", CHUNKS_BY_ID, [1, 2, 3], k=0) == []


def test_ties_keep_stable_order():
    # cid 2 and cid 3 both share zero tokens with a nonsense query, so
    # they tie at score 0. Python's sort is stable, so the original
    # candidate order should be preserved rather than shuffled.
    reranker = FakeReranker()
    ranked = reranker.rerank("zzz nonword", CHUNKS_BY_ID, [2, 3], k=2)
    assert ranked == [2, 3]


def test_tokenizer_strips_trailing_punctuation():
    # Same bug class test_bm25.py pins down: a query ending "...15 days?"
    # must still match the corpus's clean "days" token. FakeReranker
    # imports rag.bm25_index._tokenize specifically so this can't drift
    # out of sync with the BM25 fix.
    reranker = FakeReranker()
    ranked = reranker.rerank("electronics returns within 15 days?", CHUNKS_BY_ID, [1, 2, 3, 4], k=1)
    assert ranked == [1]


def test_importing_rerank_module_does_not_require_sentence_transformers():
    # CrossEncoderReranker.__init__ imports sentence_transformers lazily,
    # not at module load time -- so just importing rag.rerank (which the
    # line above already did) and referencing the class must succeed even
    # though sentence-transformers isn't installed/working in this
    # environment. Only *instantiating* CrossEncoderReranker would need
    # the real library -- and this test deliberately doesn't do that.
    assert CrossEncoderReranker._MODEL_NAME == "cross-encoder/ms-marco-MiniLM-L-6-v2"