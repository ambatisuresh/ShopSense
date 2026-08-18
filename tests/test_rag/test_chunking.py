"""Tests for rag/chunking.py -- Step 1's clause-aware splitter.

Runs against the real data/corpus/ (not a synthetic fixture), since the
whole point of this module is handling this corpus's specific mix of
bold-pseudo-headers, numeral-outside-bold headers, and warranty-policy.md's
real ATX headers correctly.
"""
from pathlib import Path

from rag.chunking import build_chunks, load_index

# tests/test_rag/test_chunking.py -> up 3 levels -> project root -> data/
CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _chunks():
    return build_chunks(CORPUS_DIR)


def test_produces_a_reasonable_number_of_chunks():
    chunks = _chunks()
    assert 60 <= len(chunks) <= 200, f"expected a few dozen clause chunks, got {len(chunks)}"


def test_cid_equals_position_for_qdrant_point_ids():
    chunks = _chunks()
    for i, c in enumerate(chunks):
        assert c["cid"] == i


def test_every_chunk_has_nonempty_text():
    chunks = _chunks()
    assert all(c["text"].strip() for c in chunks)


def test_doc_titles_match_index_json():
    chunks = _chunks()
    index = load_index(CORPUS_DIR)
    expected_titles = {e["title"] for e in index}
    got_titles = {c["doc_title"] for c in chunks}
    assert got_titles == expected_titles, f"missing/extra docs: {expected_titles ^ got_titles}"


def test_returns_policy_clause_2_1_is_its_own_chunk_with_correct_number():
    chunks = _chunks()
    hits = [c for c in chunks if c["doc_slug"] == "returns-policy" and c["clause_number"] == "2.1"]
    assert len(hits) == 1
    assert "Standard Return Window" in hits[0]["clause_title"]
    assert "30 days" in hits[0]["text"]


def test_warranty_policy_atx_headers_are_split_at_clause_level():
    # warranty-policy.md is the one doc using real Markdown '##'/'###' headers,
    # not the bold-pseudo-header style every other doc in the corpus uses.
    chunks = _chunks()
    numbers = {c["clause_number"] for c in chunks if c["doc_slug"] == "warranty-policy"}
    assert {"2.1", "2.2", "3.1", "4.1", "6.1", "6.2"}.issubset(numbers)


def test_escalation_tone_numeral_outside_bold_is_parsed():
    # "4.3.6 **Refund Requests Above Automated Cap**" -- number sits outside the bold span.
    chunks = _chunks()
    hits = [c for c in chunks if c["doc_slug"] == "escalation-tone" and c["clause_number"] == "4.3.6"]
    assert len(hits) == 1
    assert "automated cap" in hits[0]["text"]


def test_metadata_and_subtitle_lines_do_not_become_empty_chunks():
    # fraud-abuse.md opens with "**Policy #: ...**", "**Subject: ...**", "**Effective
    # Date: ...**" -- bold lines with no body before the next heading. They must not
    # surface as empty-text chunks.
    chunks = _chunks()
    fraud_chunks = [c for c in chunks if c["doc_slug"] == "fraud-abuse"]
    assert all(c["text"].strip() for c in fraud_chunks)
    assert not any(c["clause_title"] == "Subject" for c in fraud_chunks)


def test_long_clause_is_split_but_keeps_shared_metadata():
    chunks = _chunks()
    from collections import Counter

    counts = Counter((c["doc_slug"], c["clause_number"]) for c in chunks)
    split_keys = [key for key, n in counts.items() if n > 1]
    for doc_slug, clause_number in split_keys:
        pieces = [c for c in chunks if c["doc_slug"] == doc_slug and c["clause_number"] == clause_number]
        parts = sorted(c["part"] for c in pieces)
        assert parts == list(range(len(pieces))), f"non-contiguous parts for {doc_slug}/{clause_number}"
        assert len({c["doc_title"] for c in pieces}) == 1