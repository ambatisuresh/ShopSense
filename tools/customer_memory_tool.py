"""
tools/customer_memory_tool.py

Agent-facing tool for writing semantic ("durable preference") memories, per
the M3 decision that preference-writing is agent-driven, not pipeline-driven.

`customer_ref` is NEVER an argument the model supplies. It is bound into the
tool via closure when the tool is built for a given ticket - the same trust
boundary M1/M2 already established (ticket_id/raw_text set from source data,
not model output; process_refund never trusts the claimed amount). Letting the
model choose which customer's namespace a memory lands in would let a wrong or
injected ref write into the wrong customer's memory. So the model only ever
sees `note_customer_preference(fact: str)` - no ref, no ticket id, nothing it
could get wrong or manipulate.

Episodic memory (remember_ticket) is NOT exposed here - it is not agent-driven,
it fires deterministically on every ticket resolution from scripts/run_agent.py.
Only the semantic "is this durable" judgment call is delegated to the agent.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from memory.customer_memory import CustomerMemory
from tools.reliability import make_robust_tool


def build_customer_memory_tool(
    customer_ref: str,
    memory: CustomerMemory,
    ticket_id: Optional[str] = None,
):
    """
    Factory: returns a LangChain tool scoped to one customer for one ticket run.

    Called once per ticket - alongside order_lookup/shipping_tracker/
    refund_calculator/refund_replace - in the same per-ticket tool-building
    step in agent/support_agent.py (`build_tools(customer_ref, memory, ...)`).
    This tool needs ticket-specific context (customer_ref, a live
    CustomerMemory instance) at bind time, unlike the other four, which are
    stateless and can stay module-level.

    `ticket_id`, if given, is stored as provenance metadata on the memory
    (which ticket surfaced this fact) - useful for later audit/eval, not
    something the model needs to know about or supply.
    """

    def _note_customer_preference(fact: str) -> str:
        extra = {"source_ticket_id": ticket_id} if ticket_id else {}
        memory.remember_preference(customer_ref, fact, **extra)
        return f"Noted for future tickets: {fact}"

    robust_note = make_robust_tool(_note_customer_preference, "note_customer_preference")

    @tool
    def note_customer_preference(fact: str) -> str:
        """
        Record ONE durable, stable fact about this customer for future tickets -
        a stated preference (e.g. "prefers replacement over refund") or an
        observed repeat pattern (e.g. "has disputed 3 deliveries in 30 days").

        Call this ONLY when something durable actually surfaced in this ticket,
        not for one-off details specific to this single ticket - those belong
        in this ticket's own resolution, not long-term memory. Most tickets
        should NOT result in a call to this tool.
        """
        return robust_note(fact)

    return note_customer_preference