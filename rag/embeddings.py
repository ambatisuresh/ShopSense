"""Embeddings for the RAG corpus (Step 2).

LangChain + Gemini (`gemini-embedding-001`), wrapped with a disk cache (so
re-running costs zero API calls the second time) and a throttle (<=100
req/min on the free tier), plus retry+backoff on 429s.

`FakeEmbedder` is a deterministic, no-network stand-in -- useful for
writing tests that don't need a real API key, or for a quick wiring check
before spending API quota. `get_embedder()` fails loudly if GOOGLE_API_KEY
is missing rather than silently falling back to it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

MODEL_NAME = "gemini-embedding-001"
DEFAULT_CACHE_DIR = ".embcache"
BATCH_SIZE = 90
THROTTLE_PAUSE_SECONDS = 60
MAX_RETRIES = 5


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _with_backoff(fn):
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise


class GeminiEmbedder:
    """Real embeddings via langchain-google-genai, disk-cached + throttled."""

    def __init__(self, model: str = MODEL_NAME, cache_dir: str | Path = DEFAULT_CACHE_DIR):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._base = GoogleGenerativeAIEmbeddings(model=model)  # reads GOOGLE_API_KEY
        self._model = model
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self._model}::{text}".encode("utf-8")).hexdigest()

    def _get(self, text: str):
        p = self._cache_dir / f"{self._key(text)}.json"
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _put(self, text: str, vec: list[float]) -> None:
        (self._cache_dir / f"{self._key(text)}.json").write_text(json.dumps(vec))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            v = self._get(t)
            out[i] = v
            if v is None:
                todo.append(i)
        for b in range(0, len(todo), BATCH_SIZE):
            idx = todo[b : b + BATCH_SIZE]
            vecs = _with_backoff(lambda idx=idx: self._base.embed_documents([texts[i] for i in idx]))
            for i, v in zip(idx, vecs):
                out[i] = v
                self._put(texts[i], v)
            if b + BATCH_SIZE < len(todo):
                time.sleep(THROTTLE_PAUSE_SECONDS)
        return out

    def embed_query(self, text: str) -> list[float]:
        v = self._get(text)
        if v is None:
            v = _with_backoff(lambda: self._base.embed_query(text))
            self._put(text, v)
        return v


class FakeEmbedder:
    """Deterministic hash-based embeddings -- no network, no API key.
    For tests and quick wiring checks only, never for judging real
    retrieval quality (it has no real sense of meaning)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        # Pure bag-of-hashed-tokens: each word deterministically nudges one
        # dimension in one direction. No whole-string base vector -- that
        # was the bug (see rag/embeddings.py history): a whole-string hash
        # is essentially unrelated noise between two different sentences,
        # and that noise could outweigh a shared-vocabulary signal, letting
        # two unrelated texts occasionally score "more similar" than two
        # related ones.
        tokens = text.lower().split()
        vals = [0.0] * self.dim
        if not tokens:
            seed = hashlib.sha256(text.encode("utf-8")).digest()
            vals = [(seed[i % len(seed)] / 255.0) * 2 - 1 for i in range(self.dim)]
        else:
            for tok in tokens:
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                vals[idx] += sign
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embedder() -> Embedder:
    """Fails loudly without GOOGLE_API_KEY rather than silently downgrading
    to FakeEmbedder -- a caller that wants the offline path constructs
    FakeEmbedder() directly instead."""
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Set it in your .env, or use FakeEmbedder() directly "
            "for a wiring smoke test."
        )
    return GeminiEmbedder()