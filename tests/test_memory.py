"""
tests/test_memory.py

Deterministic tests for memory/customer_memory.py - a fake Supermemory client
and a fake litellm.completion, no live API calls. Same conventions as M2's
tests/test_tools.py and tests/test_agent.py: monkeypatched fixtures, run via
`pytest tests/test_memory.py -v -s`.

Coverage:
  - write/read round-trip for both memory kinds (episodic, semantic)
  - namespace isolation across two different customer_refs
  - the two-layer recall() fallback (extracted memories -> raw documents)
  - get_context_block()'s formatting and its "" defaults (no history / recall failure)
  - _with_backoff() retry behavior in isolation
  - retry actually wired into remember_ticket/remember_preference/recall -
    regression tests pinned to the live Gemini 503 that motivated adding it
  - empty customer_ref rejected consistently across all entry points
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import memory.customer_memory as customer_memory
from memory.customer_memory import CustomerMemory, MemoryHit, _container_tag, _with_backoff


# --------------------------------------------------------------------------- #
# Fakes - just enough of the Supermemory SDK's shape to exercise our code,
# not a reimplementation of Supermemory itself.
# --------------------------------------------------------------------------- #

class _FakeChunk:
    def __init__(self, content):
        self.content = content


class _FakeMemoryHit:
    def __init__(self, memory, metadata, similarity=0.9):
        self.memory = memory
        self.metadata = metadata
        self.similarity = similarity


class _FakeDocHit:
    def __init__(self, content, metadata, score=0.5):
        self.chunks = [_FakeChunk(content)]
        self.metadata = metadata
        self.score = score


class _FakeSearchResult:
    def __init__(self, results):
        self.results = results


class _FakeSearch:
    """Mirrors client.search.memories()/.documents() - both read the same
    backing store by default, filtered by container_tag, so a plain
    remember_*() -> recall() round-trip works without extra setup. Individual
    tests override .memories or .documents directly when they need to
    simulate the async-indexing-lag fallback path."""

    def __init__(self, store):
        self._store = store

    def memories(self, q, container_tag, limit):
        items = [it for it in self._store if it["container_tag"] == container_tag]
        hits = [_FakeMemoryHit(it["content"], it["metadata"]) for it in items[:limit]]
        return _FakeSearchResult(hits)

    def documents(self, q, container_tags, limit):
        items = [it for it in self._store if it["container_tag"] in container_tags]
        hits = [_FakeDocHit(it["content"], it["metadata"]) for it in items[:limit]]
        return _FakeSearchResult(hits)


class FakeSupermemoryClient:
    """fail_times lets a test simulate N transient failures before .add()
    starts succeeding - used to prove _with_backoff is actually wired into
    the write path, not just defined and unused."""

    def __init__(self, fail_times: int = 0, fail_exc: Exception | None = None):
        self.store: list[dict] = []
        self.add_calls = 0
        self._fail_times = fail_times
        self._fail_exc = fail_exc or RuntimeError("transient supermemory error")
        self.search = _FakeSearch(self.store)

    def add(self, content, container_tag, metadata=None):
        self.add_calls += 1
        if self.add_calls <= self._fail_times:
            raise self._fail_exc
        self.store.append({"content": content, "container_tag": container_tag, "metadata": metadata or {}})
        return {"id": f"mem_{len(self.store)}"}


def _fake_completion(text: str):
    def _completion(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    return _completion


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_client():
    return FakeSupermemoryClient()


@pytest.fixture
def cm(monkeypatch, fake_client):
    """A CustomerMemory wired to the fake client. Also silences time.sleep so
    retry tests (which deliberately trigger backoff) don't actually wait."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr(customer_memory.time, "sleep", lambda *_: None)
    monkeypatch.setattr(customer_memory.supermemory, "Supermemory", lambda api_key=None: fake_client)
    instance = CustomerMemory()
    return instance


# --------------------------------------------------------------------------- #
# _container_tag / empty customer_ref handling
# --------------------------------------------------------------------------- #

def test_container_tag_namespaces_by_customer_ref():
    assert _container_tag("KW-C-00002") == "shopsense_customer_KW-C-00002"


@pytest.mark.parametrize("bad_ref", ["", "   ", None])
def test_container_tag_rejects_empty_customer_ref(bad_ref):
    with pytest.raises(ValueError):
        _container_tag(bad_ref)


@pytest.mark.parametrize("bad_ref", ["", "   "])
def test_remember_and_recall_reject_empty_customer_ref(cm, bad_ref):
    with pytest.raises(ValueError):
        cm.remember_ticket(bad_ref, "KW-T-1", "summary", "2026-08-16")
    with pytest.raises(ValueError):
        cm.remember_preference(bad_ref, "some fact")
    with pytest.raises(ValueError):
        cm.recall(bad_ref, "query")


# --------------------------------------------------------------------------- #
# write/read round-trip
# --------------------------------------------------------------------------- #

def test_remember_ticket_writes_episodic_memory(cm, fake_client):
    cm.remember_ticket("KW-C-00002", "KW-T-000002", "Wrong item received; asked for order ID.", "2026-08-16")

    assert len(fake_client.store) == 1
    entry = fake_client.store[0]
    assert entry["container_tag"] == "shopsense_customer_KW-C-00002"
    assert entry["metadata"]["type"] == "episodic"
    assert entry["metadata"]["ticket_id"] == "KW-T-000002"
    assert entry["metadata"]["date"] == "2026-08-16"


def test_remember_preference_writes_semantic_memory(cm, fake_client):
    cm.remember_preference("KW-C-00002", "Prefers replacement over refund.")

    assert len(fake_client.store) == 1
    entry = fake_client.store[0]
    assert entry["metadata"]["type"] == "semantic"
    assert entry["content"] == "Prefers replacement over refund."


def test_remember_ticket_passes_through_extra_metadata(cm, fake_client):
    cm.remember_ticket("KW-C-00002", "KW-T-000002", "summary", "2026-08-16", source="test")
    assert fake_client.store[0]["metadata"]["source"] == "test"


def test_recall_returns_memory_hits(cm):
    cm.remember_preference("KW-C-00002", "Prefers replacement over refund.")

    hits = cm.recall("KW-C-00002", "preference", k=3)

    assert len(hits) == 1
    assert isinstance(hits[0], MemoryHit)
    assert hits[0].kind == "semantic"
    assert hits[0].text == "Prefers replacement over refund."


def test_recall_returns_empty_list_when_nothing_found(cm):
    assert cm.recall("KW-C-99999", "anything") == []


# --------------------------------------------------------------------------- #
# namespace isolation
# --------------------------------------------------------------------------- #

def test_recall_never_crosses_customer_namespaces(cm):
    cm.remember_preference("KW-C-A", "Customer A's private fact.")
    cm.remember_preference("KW-C-B", "Customer B's private fact.")

    hits_a = cm.recall("KW-C-A", "fact")
    hits_b = cm.recall("KW-C-B", "fact")

    assert [h.text for h in hits_a] == ["Customer A's private fact."]
    assert [h.text for h in hits_b] == ["Customer B's private fact."]


# --------------------------------------------------------------------------- #
# two-layer recall fallback (extracted memories -> raw documents)
# --------------------------------------------------------------------------- #

def test_recall_falls_back_to_documents_when_memories_layer_is_empty(cm, fake_client, monkeypatch):
    # Simulate Supermemory's extraction lag: the memories layer hasn't
    # caught up yet, but the raw document is already searchable.
    fake_client.store.append({
        "container_tag": "shopsense_customer_KW-C-00002",
        "content": "Raw document content, not yet extracted into a memory.",
        "metadata": {"type": "episodic"},
    })
    monkeypatch.setattr(
        fake_client.search, "memories",
        lambda q, container_tag, limit: _FakeSearchResult([]),
    )

    hits = cm.recall("KW-C-00002", "query", k=3)

    assert len(hits) == 1
    assert hits[0].text == "Raw document content, not yet extracted into a memory."


# --------------------------------------------------------------------------- #
# get_context_block
# --------------------------------------------------------------------------- #

def test_get_context_block_empty_string_when_no_history(cm):
    assert cm.get_context_block("KW-C-00002", "query") == ""


def test_get_context_block_formats_hits(cm):
    cm.remember_preference("KW-C-00002", "Prefers replacement over refund.")

    block = cm.get_context_block("KW-C-00002", "preference")

    assert block.startswith("Known customer history:")
    assert "[semantic] Prefers replacement over refund." in block


def test_get_context_block_degrades_to_empty_string_on_recall_failure(cm, monkeypatch):
    """Regression test: get_context_block runs BEFORE the agent loop starts
    (agent/support_agent.py), so an uncaught exception here would crash the
    ticket entirely rather than just losing context. Pinned to the fix made
    after get_context_block was found to propagate recall() failures."""
    def _boom(*a, **k):
        raise RuntimeError("supermemory is down")
    monkeypatch.setattr(cm, "recall", _boom)

    assert cm.get_context_block("KW-C-00002", "query") == ""


# --------------------------------------------------------------------------- #
# summarize_ticket (LLM call)
# --------------------------------------------------------------------------- #

def test_summarize_ticket_returns_llm_text(cm, monkeypatch):
    monkeypatch.setattr(
        customer_memory.litellm, "completion",
        _fake_completion("Customer received the wrong item; asked for their order ID."),
    )

    summary = cm.summarize_ticket(
        raw_text="I ordered tools but got a cookbook instead.",
        issue_type="PRODUCT",
        resolution="Asked customer for order ID.",
    )

    assert summary == "Customer received the wrong item; asked for their order ID."


def test_summarize_ticket_retries_through_transient_llm_failure(cm, monkeypatch):
    """Regression test pinned to the live Gemini 503 seen during an actual
    batch run - summarize_ticket must not give up on the first failure."""
    calls = {"n": 0}

    def _flaky_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("503 UNAVAILABLE: model overloaded")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok summary"))])

    monkeypatch.setattr(customer_memory.litellm, "completion", _flaky_completion)

    summary = cm.summarize_ticket("raw", "PRODUCT", "resolution")

    assert summary == "ok summary"
    assert calls["n"] == 3


# --------------------------------------------------------------------------- #
# _with_backoff, tested in isolation
# --------------------------------------------------------------------------- #

def test_with_backoff_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(customer_memory.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert _with_backoff(flaky, max_retries=3) == "ok"
    assert calls["n"] == 2


def test_with_backoff_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(customer_memory.time, "sleep", lambda *_: None)

    def always_fails():
        raise RuntimeError("permanent failure")

    with pytest.raises(RuntimeError, match="permanent failure"):
        _with_backoff(always_fails, max_retries=3)


# --------------------------------------------------------------------------- #
# retry actually wired into the write path (not just defined and unused)
# --------------------------------------------------------------------------- #

def test_remember_ticket_survives_two_transient_write_failures(monkeypatch):
    flaky_client = FakeSupermemoryClient(fail_times=2)
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr(customer_memory.time, "sleep", lambda *_: None)
    monkeypatch.setattr(customer_memory.supermemory, "Supermemory", lambda api_key=None: flaky_client)
    cm = CustomerMemory()

    cm.remember_ticket("KW-C-00002", "KW-T-1", "summary", "2026-08-16")

    assert flaky_client.add_calls == 3  # 2 failures + 1 success
    assert len(flaky_client.store) == 1


def test_remember_ticket_raises_after_exhausting_retries(monkeypatch):
    flaky_client = FakeSupermemoryClient(fail_times=10)  # never succeeds within max_retries
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr(customer_memory.time, "sleep", lambda *_: None)
    monkeypatch.setattr(customer_memory.supermemory, "Supermemory", lambda api_key=None: flaky_client)
    cm = CustomerMemory()

    with pytest.raises(RuntimeError):
        cm.remember_ticket("KW-C-00002", "KW-T-1", "summary", "2026-08-16")

    assert len(flaky_client.store) == 0  # never wrote a partial/corrupt entry


# --------------------------------------------------------------------------- #
# wait_until_searchable
# --------------------------------------------------------------------------- #

def test_wait_until_searchable_returns_true_once_indexed(cm, fake_client, monkeypatch):
    monkeypatch.setattr(customer_memory.time, "sleep", lambda *_: None)
    cm.remember_ticket("KW-C-00002", "KW-T-1", "summary", "2026-08-16")

    assert cm.wait_until_searchable("KW-C-00002", "summary", timeout_s=6, poll_s=2) is True


def test_wait_until_searchable_times_out_when_never_indexed(cm, monkeypatch):
    monkeypatch.setattr(customer_memory.time, "sleep", lambda *_: None)
    assert cm.wait_until_searchable("KW-C-00002", "nothing written", timeout_s=6, poll_s=2) is False