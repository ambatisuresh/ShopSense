"""
memory/customer_memory.py

Persistent per-customer memory for ShopSense (Milestone 3).

Wraps the Supermemory SDK to store and recall two of the four memory kinds
introduced in Day 2 Session 1 (Memory Engineering):

  - episodic  -> "what happened"      - a distilled summary of a resolved ticket
  - semantic  -> "what is stably true" - a durable customer preference or pattern

Procedural memory is deliberately NOT built here: ShopSense's "how to do X"
already lives in the policy corpus (returns/shipping/warranty docs -> M3's
Qdrant half), so a separate procedural layer would just duplicate it.

Each customer's memories are isolated via Supermemory's container_tag, scoped
to that customer's `customer_ref` (e.g. "CUST00042") rather than one shared
namespace - so recall() for one customer can never surface another's history.
"""

from __future__ import annotations

import os
import random
import time
import logging
from dataclasses import dataclass
from typing import Optional

import litellm
import supermemory

logger = logging.getLogger(__name__)

_TAG_PREFIX = "shopsense_customer_"


def _with_backoff(fn, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry fn() with exponential backoff + jitter on transient failures - a
    503 from an overloaded model, a transient Supermemory API blip. Kept
    standalone here rather than reusing tools/reliability.py's
    make_robust_tool: that wrapper logs into TOOL_CALL_LOG and shares a
    circuit breaker with the agent's bound tools, and memory calls aren't
    agent tool calls - routing them through the same log/breaker would
    pollute tool-call counts and trip state that belongs to a different
    concern.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
                time.sleep(delay)
    raise last_exc


def _container_tag(customer_ref: str) -> str:
    """Namespace a customer's memories under their own container_tag."""
    if not customer_ref or not customer_ref.strip():
        raise ValueError("customer_ref is required to scope memory")
    return f"{_TAG_PREFIX}{customer_ref.strip()}"


@dataclass
class MemoryHit:
    kind: str      # "episodic" | "semantic" | "?"
    text: str
    score: float


class CustomerMemory:
    """Per-customer episodic + semantic memory, backed by Supermemory."""

    def __init__(self, api_key: Optional[str] = None, llm_model: Optional[str] = None):
        api_key = api_key or os.getenv("SUPERMEMORY_API_KEY")
        if not api_key:
            raise RuntimeError("SUPERMEMORY_API_KEY is not set")
        self._client = supermemory.Supermemory(api_key=api_key)
        # Same LLM used for the M1 intake parser / M2 agent by default - override only if needed.
        self._llm_model = llm_model or os.getenv("LLM_MODEL", "gemini/gemini-flash-latest")

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #

    def remember_ticket(
        self,
        customer_ref: str,
        ticket_id: str,
        summary: str,
        date: str,
        **extra,
    ) -> None:
        """
        Write one episodic memory: a distilled record of a resolved ticket.

        `summary` should already be a short, concrete sentence (issue + resolution).
        Callers should produce it via `summarize_ticket()` below rather than passing
        raw ticket text - raw text is noisy and re-embeds poorly for later recall.
        """
        _with_backoff(lambda: self._client.add(
            content=summary,
            container_tag=_container_tag(customer_ref),
            metadata={"type": "episodic", "ticket_id": ticket_id, "date": date, **extra},
        ))
        logger.info("remembered episodic ticket=%s customer=%s", ticket_id, customer_ref)

    def remember_preference(self, customer_ref: str, fact: str, **extra) -> None:
        """
        Write one semantic memory: a durable, stable fact about the customer
        (a stated preference, a repeat-behavior pattern - e.g. "prefers replacement
        over refund", "has disputed 3 deliveries in the last 30 days").

        Not every ticket produces one - only call this when something durable
        actually surfaced, not on every resolution.
        """
        _with_backoff(lambda: self._client.add(
            content=fact,
            container_tag=_container_tag(customer_ref),
            metadata={"type": "semantic", **extra},
        ))
        logger.info("remembered semantic fact for customer=%s", customer_ref)

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #

    def recall(self, customer_ref: str, query: str, k: int = 3) -> list[MemoryHit]:
        """
        Semantic search over one customer's memories.

        Queries the extracted-MEMORIES layer first (readable `.memory` text on
        each hit); falls back to the raw documents layer only if extraction
        hasn't caught up yet, since Supermemory indexes new writes asynchronously.
        Same two-layer pattern as the Day 2 Session 1 notebook's `recall()`.
        """
        tag = _container_tag(customer_ref)
        out: list[MemoryHit] = []

        mem_results = _with_backoff(
            lambda: self._client.search.memories(q=query, container_tag=tag, limit=k).results
        )
        for r in mem_results:
            if getattr(r, "memory", None):
                out.append(
                    MemoryHit(
                        kind=(r.metadata or {}).get("type", "?"),
                        text=r.memory,
                        score=float(r.similarity or 0.0),
                    )
                )

        if not out:
            doc_results = _with_backoff(
                lambda: self._client.search.documents(q=query, container_tags=[tag], limit=k).results
            )
            for r in doc_results:
                text = " ".join(c.content for c in (r.chunks or []) if c and c.content).strip()
                if text:
                    out.append(
                        MemoryHit(
                            kind=(r.metadata or {}).get("type", "?"),
                            text=text,
                            score=float(r.score or 0.0),
                        )
                    )
        return out

    def get_context_block(self, customer_ref: str, query: str, k: int = 3) -> str:
        """
        Convenience wrapper for agent/prompts.py: recall + format as a short
        text block ready to inject into the system prompt.

        Returns "" both when there's no history yet AND when recall fails
        even after retries - callers can always safely concatenate it into
        the prompt without a None-check, and a memory outage never blocks a
        ticket from being handled. This is called BEFORE the agent loop
        starts (see agent/support_agent.py), so an uncaught exception here
        would crash the entire ticket run over what is, at worst, a loss of
        context - not something that should take the ticket down with it.
        """
        try:
            hits = self.recall(customer_ref, query, k=k)
        except Exception as e:
            logger.warning("recall failed for customer=%s, proceeding with no history: %s", customer_ref, e)
            return ""
        if not hits:
            return ""
        lines = [f"- [{h.kind}] {h.text}" for h in hits]
        return "Known customer history:\n" + "\n".join(lines)

    # ------------------------------------------------------------------ #
    # ticket -> memory note (LLM distillation)
    # ------------------------------------------------------------------ #

    def summarize_ticket(self, raw_text: str, issue_type: str, resolution: str) -> str:
        """
        Distill a resolved ticket into one concrete episodic sentence via LLM -
        same pattern as Day 2 Session 1's `summarize_turns` (real LLM call,
        temperature=0 for reproducibility, short and fact-preserving).
        """
        system = (
            "Summarize a resolved customer-support ticket into ONE short, concrete "
            "sentence for long-term memory. Preserve specific facts (issue type, "
            "resolution, amounts) if present. Return only the sentence, under 40 words."
        )
        user = (
            f"Issue type: {issue_type}\n"
            f"Resolution: {resolution}\n"
            f"Ticket text: {raw_text}\n\n"
            "Summary:"
        )
        resp = _with_backoff(lambda: litellm.completion(
            model=self._llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ))
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # test / dev helper
    # ------------------------------------------------------------------ #

    def wait_until_searchable(
        self,
        customer_ref: str,
        probe_query: str,
        timeout_s: int = 60,
        poll_s: int = 3,
    ) -> bool:
        """
        Supermemory indexes new memories asynchronously. Poll until a probe query
        returns a hit rather than guessing a fixed delay - same pattern as the
        notebook's ingestion self-check.

        Intended for tests / interactive verification only. The production
        write-back path (remember_ticket at the end of an agent run) should NOT
        block on this - the next ticket for that customer will simply see it
        once indexing catches up.
        """
        tag = _container_tag(customer_ref)
        for _ in range(max(1, timeout_s // poll_s)):
            if self._client.search.memories(q=probe_query, container_tag=tag, limit=1).results:
                return True
            time.sleep(poll_s)
        return False