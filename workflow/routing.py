"""
ShopSense M5 - Parts 5 & 6: the routing functions, kept in their OWN module
with NO dependency on langgraph.

This split exists for a concrete reason, not just tidiness: workflow/graph.py
imports the real `langgraph` package at module load time, so anything that
lives in graph.py becomes untestable in an environment without langgraph
installed - even a function that has nothing to do with langgraph itself.
Both routers below are plain `dict -> str` functions; they deserve to stay
importable (and testable) on their own, exactly matching Lab B's own point
about route_after_checks: "no graph, no model, no tokens" needed to test a
router. workflow/graph.py imports these functions rather than defining them.
"""

from workflow.state import TicketReviewState


def route_after_review(state: TicketReviewState) -> str:
    """The only place this workflow decides auto_approve vs human_approval.
    Pure function of state - unit-testable with a plain dict, no graph
    needed, exactly like Lab B's route_after_checks self-check.

    Reads `classification` only - never re-derives a decision from
    `concerns` itself, because `classification` is the ONE field
    draft_redline (Step 4) already reconciled with `concerns` before this
    edge ever runs.

    FAIL-CLOSED, not fail-open: any value other than the exact string
    "standard" (including the empty-string default from seed_state, or a
    future classification value nobody's wired a branch for yet) routes to
    human_approval. A routing bug should never accidentally auto-approve a
    ticket it doesn't recognize.
    """
    if state["classification"] == "standard":
        return "auto_approve"
    return "human_approval"


def route_after_human(state: TicketReviewState) -> str:
    """Reads the reviewer's decision (`status`, written by human_approval
    right before this edge runs) and decides whether to finalize.

    FAIL-CLOSED, twice over:
      1. Any status other than the exact string "approved" (including
         "rejected", or a value from an unhandled outcome like
         "changes_requested" - see human_approval.py's scope note) routes
         to "end", never to finalize. A malformed or unrecognized decision
         payload must never accidentally commit a refund.
      2. Even "approved" does not reach finalize without an `approver_id`.
         This ENFORCES the notebook's own production warning that
         interrupt() gives back a value, not authorization - a workflow
         that records "approved" with no record of WHO approved it "will
         not survive its first audit". human_approval.py records whatever
         it was given faithfully; this function is where that gets
         enforced, not assumed.
    """
    if state["status"] == "approved" and state.get("approver_id"):
        return "finalize"
    return "end"