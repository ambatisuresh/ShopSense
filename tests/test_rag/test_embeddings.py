"""Tests for rag/embeddings.py -- Step 2.

Only FakeEmbedder and the get_embedder() guard clause are tested here --
no real API calls, matching the same "no live API calls in the test
suite" convention your M1-M3 tests already use (fake LLM stubs, monkeypatched
fixtures). GeminiEmbedder itself needs a live GOOGLE_API_KEY and a network
call, so it's exercised by scripts/run_embeddings.py instead, by hand.
"""
import math

import pytest

from rag.embeddings import FakeEmbedder, get_embedder


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def test_fake_embedder_is_deterministic():
    embedder = FakeEmbedder(dim=16)
    assert embedder.embed_query("hello world") == embedder.embed_query("hello world")


def test_fake_embedder_different_text_gives_different_vector():
    embedder = FakeEmbedder(dim=16)
    assert embedder.embed_query("hello world") != embedder.embed_query("goodbye moon")


def test_fake_embedder_respects_requested_dimension():
    embedder = FakeEmbedder(dim=32)
    vec = embedder.embed_query("any text")
    assert len(vec) == 32


def test_fake_embedder_embed_documents_matches_embed_query_per_item():
    embedder = FakeEmbedder(dim=16)
    texts = ["return window", "refund policy"]
    docs_result = embedder.embed_documents(texts)
    assert docs_result[0] == embedder.embed_query(texts[0])
    assert docs_result[1] == embedder.embed_query(texts[1])


def test_fake_embedder_shared_words_nudge_similarity_higher():
    # Not a claim of real semantic understanding (see the module's docstring) --
    # just checking the lexical nudge exists at all: two texts sharing several
    # words should score more similar than two sharing none.
    embedder = FakeEmbedder(dim=64)
    a = embedder.embed_query("electronics return window is fifteen days")
    b = embedder.embed_query("electronics return window is short")
    c = embedder.embed_query("bananas spaceship unrelated nonsense")

    sim_related = _cosine(a, b)
    sim_unrelated = _cosine(a, c)
    assert sim_related > sim_unrelated


def test_get_embedder_fails_loudly_without_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_embedder()


def test_get_embedder_error_message_mentions_the_offline_alternative(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FakeEmbedder"):
        get_embedder()