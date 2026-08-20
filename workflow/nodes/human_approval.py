"""
ShopSense M5 - Part 6: the `human_approval` node - the real interrupt()
pause, replacing Step 5's placeholder.

Follows Lab B Part B6/B8 exactly:
    - `interrupt(payload)` is the ONLY thing this node does before reading
      its result. NOTHING else happens first - no logging side effects, no
      writes, nothing irreversible. Lab B Part B8's whole lesson is that on
      resume, a node RE-RUNS FROM THE TOP, so anything before `interrupt()`
      in the same node executes twice. This node has nothing before it but
      state reads used to build the payload, which are side-effect-free by
      construction.
    - The payload is what a real reviewer looks at - not "Approve?" with no
      context (which guarantees a rubber stamp per the notebook's own
      warning). It carries the redline draft, why the ticket was flagged
      (`concerns`), what it's based on (`citations`), and the policy
      numbers, so a reviewer can actually make the call.
    - `interrupt()` gives back a VALUE, not AUTHORIZATION (the notebook's
      explicit production note). This node records `approver_id` as
      whatever the resume call supplied - never inferred, never defaulted
      to something that looks like an identity. `workflow.routing.
      route_after_human` is where that gets ENFORCED (an "approved" status
      with no `approver_id` does not reach finalize) - decide in a node
      (record faithfully), enforce in an edge (refuse to proceed without
      it) - same split as everywhere else in this workflow.

LAZY IMPORT, same pattern M4's rag/rerank.py used for CrossEncoderReranker
("lazily imported so the module loads fine without the dependency
installed"): `interrupt_fn` defaults to `None`, and the REAL
`langgraph.types.interrupt` is only imported inside `build_human_approval_
node()` if no override is given - so this module has ZERO top-level
langgraph dependency, unlike workflow/graph.py. That is what makes this
node's own logic testable with a real, executed pytest run in an
environment with no langgraph installed at all - inject a fake
`interrupt_fn` and nothing here ever touches the langgraph package.

SCOPE NOTE: this node's outcomes are "approved" (optionally with an
`edited_draft` override - edit-then-approve) and "rejected" - matching Lab
B's own two REQUIRED outcomes. Lab B's third outcome, "changes_requested"
(loop back to `draft_redline` for a redraft, bounded by MAX_REVISIONS), is
a natural extension but is OUT OF SCOPE for this milestone - the brief asks
for "conditional route (standard = auto-approve, non-standard = human-
approval interrupt)", not a redraft loop. `route_after_human` fails closed
(routes to "end") on anything that isn't exactly "approved" with an
approver_id, so an unhandled "changes_requested" value degrades safely
rather than crashing - it just doesn't loop back and redraft. Wiring that
in later needs one more branch in the router and one more edge in
graph.py, nothing else changes.
"""

from typing import Callable, Optional

from workflow.state import TicketReviewState

InterruptFn = Callable[[dict], dict]


def _build_payload(state: TicketReviewState) -> dict:
    """Everything a real reviewer needs, and nothing more. Every value here
    is a JSON-serialisable primitive - required for interrupt() (payload
    must round-trip through the checkpointer) and consistent with state.py's
    own checkpoint-safety rule."""
    return {
        "question": "Approve this ticket resolution?",
        "ticket_id": state["ticket_id"],
        "customer_ref": state.get("customer_ref"),
        "order_id": state.get("order_id"),
        "issue_type": (state.get("parsed_ticket") or {}).get("issue_type"),
        "redline_draft": state["redline_draft"],
        "concerns": state["concerns"],
        "citations": state["citations"],
        "policy_eligible_amount": state.get("policy_eligible_amount"),
        "revision_count": state["revision_count"],
    }


def build_human_approval_node(
    interrupt_fn: Optional[InterruptFn] = None,
) -> Callable[[TicketReviewState], dict]:
    """Returns the `human_approval` node. `interrupt_fn(payload) -> dict`
    defaults to the REAL `langgraph.types.interrupt`, imported lazily on
    first call so this module stays importable/testable without langgraph
    installed. Pass a fake for tests (see
    tests/test_workflow/test_human_approval.py) - a callable that just
    returns a canned decision dict synchronously, standing in for a real
    pause/resume cycle the same way every other node's fixtures stand in
    for a live dependency.

    The expected decision dict shape (what a real `Command(resume=...)`
    call supplies): `{"action": "approved" | "rejected", "note": str,
    "approver_id": str, "edited_draft": str (optional)}`.
    """
    if interrupt_fn is None:
        from langgraph.types import interrupt as interrupt_fn  # noqa: F811 (lazy, deliberate)

    def human_approval(state: TicketReviewState) -> dict:
        decision = interrupt_fn(_build_payload(state))

        action = decision.get("action")
        return {
            "status": action,
            "approver_note": decision.get("note", ""),
            "approver_id": decision.get("approver_id"),
            "redline_draft": decision.get("edited_draft", state["redline_draft"]),
            "audit_log": [f"human_approval: reviewer action={action!r} approver_id={decision.get('approver_id')!r}"],
        }

    return human_approval