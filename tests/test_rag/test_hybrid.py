"""Tests for rag/hybrid.py's Reciprocal Rank Fusion -- Step 5.

Pure logic, no embeddings/BM25/Qdrant involved -- these test the fusion
math itself on hand-built rankings, so they're fast and need nothing live.
"""
from rag.hybrid import rrf_fuse


def test_doc_favoured_by_both_lists_wins():
    fused = rrf_fuse([["a", "b", "c"], ["a", "c", "d"]], k=4)
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c", "d"}


def test_k_truncates_output():
    fused = rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], k=2)
    assert len(fused) == 2


def test_single_ranking_passthrough_order_preserved():
    fused = rrf_fuse([["x", "y", "z"]], k=3)
    assert fused == ["x", "y", "z"]


def test_empty_rankings_returns_empty():
    assert rrf_fuse([], k=5) == []


def test_item_in_only_one_list_is_still_included():
    # BM25 alone found "electronics"; dense search never returned it at all.
    # It should still show up in the fused result, just not necessarily first.
    fused = rrf_fuse([["a", "b"], ["a", "electronics"]], k=10)
    assert "electronics" in fused


def test_top_rank_in_both_lists_beats_top_rank_in_only_one():
    # "a" is #1 in both lists; "b" is #1 in only the second list.
    # Reciprocal-rank scoring should still put "a" ahead.
    fused = rrf_fuse([["a", "c"], ["a", "b"]], k=3)
    assert fused[0] == "a"


def test_three_rankings_fuse_together():
    fused = rrf_fuse([["a", "b"], ["b", "c"], ["b", "a"]], k=3)
    # "b" appears in all three lists (twice at rank 1) -- should come out on top.
    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}