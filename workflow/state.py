"""
ShopSense M5 - Part 1: State schema for the ticket-review LangGraph workflow.

Mirrors the Day3 Session1 Lab B lesson (Part B1) applied to ShopSense's own
domain: extract -> compare-to-playbook -> draft redline -> conditional route
(auto-approve | human-approval interrupt).

THE ONE RULE THAT MATTERS HERE (Lab B, Part B1):
    - a CONTROL field describes the ticket's review AS IT IS RIGHT NOW.
      It must be overwritable, or the router that reads it will see stale
      data and the graph can loop forever.
    - an AUDIT field describes EVERYTHING THAT EVER HAPPENED during the
      review. It must accumulate (a reducer), or you lose the trail the
      moment a second node writes to it in the same run.

Concretely for ShopSense:
    - `concerns`   is the direct analogue of Lab B's `issues`     -> CONTROL
    - `audit_log`  is the direct analogue of Lab B's `issue_log`  -> AUDIT (reducer)
Get that one distinction wrong and either the router never routes correctly,
or the compliance trail silently loses steps.
"""

from operator import add
from typing import Annotated, Optional, TypedDict

# ---------------------------------------------------------------------------
# Tunable constants (loop guard - see Lab B Part B4 - "loops terminate
# because YOU made them terminate"). A human can send a ticket back for
# revision ("changes_requested") more than once; MAX_REVISIONS caps only the
# machine's own redraft loop, exactly like Lab B's MAX_REVISIONS did.
# ---------------------------------------------------------------------------
MAX_REVISIONS = 3


class TicketReviewState(TypedDict):
    # -- Identity / intake (written once by `extract`, then read-only) -----
    ticket_id: str
    raw_text: str
    customer_ref: Optional[str]          # resolved the same way M3 does it
    order_id: Optional[str]

    # -- Output of `extract` (M1's SupportTicket, as a plain dict so every
    #    field here stays a JSON-serialisable primitive - see the note on
    #    checkpoint safety at the bottom of this file) ----------------------
    parsed_ticket: dict                  # issue_type, sentiment, urgency,
                                          # claimed_refund_amount,
                                          # contains_suspicious_instructions,
                                          # confidence

    # -- Output of `compare_to_playbook` -------------------------------------
    # CONTROL fields: describe the CURRENT decision only. Overwritten every
    # time compare_to_playbook re-runs (e.g. after a human-requested revision
    # changes something worth re-checking).
    citations: list[dict]                # [{doc_slug, clause_number,
                                          #   clause_title, chunk_id}, ...]
    policy_eligible_amount: Optional[float]
    policy_action: Optional[str]         # "refund" | "replace" | "deny" | "escalate"
    concerns: list[str]                  # WHY this is non-standard, if it is.
                                          # Empty list = nothing standing in
                                          # the way of auto-approval.
                                          # <-- the router reads THIS field.
    classification: str                  # "" | "standard" | "non_standard"

    # -- Output of `draft_redline` (the one LLM-backed node) -----------------
    redline_draft: str                   # CONTROL: latest version only
    revision_count: int                  # CONTROL: loop guard counter

    # -- Human-approval gate + finalize ---------------------------------------
    status: str                          # "" | "pending_human" | "approved"
                                          # | "rejected" | "changes_requested"
                                          # | "auto_approved" | "finalized"
    approver_note: str                   # CONTROL: latest note only
    approver_id: Optional[str]           # WHO approved - never inferred,
                                          # always supplied by the resume call
    final_result: dict                   # CONTROL: the committed refund/
                                          # replace API response, written only
                                          # by `finalize` (downstream of any
                                          # pause - Lab B Part B8)

    # -- AUDIT field: the only field with a reducer in this whole schema.
    #    Every node that makes a decision appends ONE tagged string here.
    #    Never read by a router - it exists for the compliance trail alone.
    audit_log: Annotated[list[str], add]


def has_reducer(field_name: str) -> bool:
    """Same trick the notebook uses: Annotated[...] stashes its extras in
    __metadata__, which is how LangGraph (and we, here) can tell a reducer
    was registered for this channel."""
    return hasattr(TicketReviewState.__annotations__[field_name], "__metadata__")


def seed_state(
    ticket_id: str,
    raw_text: str,
    *,
    customer_ref: Optional[str] = None,
    order_id: Optional[str] = None,
) -> TicketReviewState:
    """The one place a fresh TicketReviewState gets constructed, so every
    caller (tests, the demo script, a real API handler) starts from the same
    shape. Mirrors Lab B's `seed` dict, just built by a function instead of
    typed out by hand at every call site (there are more fields here than
    Lab B's ReviewState had, so typing it out repeatedly is exactly the kind
    of copy-paste bug this function exists to prevent).

    `customer_ref` / `order_id` are optional INTAKE-METADATA hints, not
    anything this function invents. Per M3's decision #4, records.jsonl
    carries its own `customer_ref`/`order_ref` at the top level, sibling to
    `raw_text` - known before any parsing happens, and preferred over
    whatever the LLM parser guesses from free text (which M3 found is
    frequently None even on tickets clearly tied to a real order). The
    caller (Step 9's demo script, or a real intake handler) reads those
    straight off the source record and passes them here; the `extract` node
    (Step 2) then treats them as authoritative and only falls back to
    order_lookup when a hint is missing."""
    return TicketReviewState(
        ticket_id=ticket_id,
        raw_text=raw_text,
        customer_ref=customer_ref,
        order_id=order_id,
        parsed_ticket={},
        citations=[],
        policy_eligible_amount=None,
        policy_action=None,
        concerns=[],
        classification="",
        redline_draft="",
        revision_count=0,
        status="",
        approver_note="",
        approver_id=None,
        final_result={},
        audit_log=[],
    )


# ---------------------------------------------------------------------------
# Note on checkpoint safety (Day3 Session1, the production note right before
# Part B7's SqliteSaver cell): every field above is a str, int, float, bool,
# None, dict, or list of those - no Pydantic model, no custom class. That is
# deliberate. A checkpoint file is a deserialisation surface; keeping state
# to plain-old-data types means the eventual SqliteSaver (Step 8) needs no
# custom allow-listing to run safely, and `parsed_ticket`'s SupportTicket
# fields are stored as a plain dict for the same reason - the Pydantic
# object itself never touches the checkpoint.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Self-check, same shape as the notebook's Part B1 self-check cell.
    ann = TicketReviewState.__annotations__

    assert has_reducer("audit_log"), (
        "audit_log needs a reducer: Annotated[list[str], add]"
    )
    assert ann["audit_log"].__metadata__[0] is add, (
        "audit_log's reducer should be operator.add (list concatenation)"
    )

    control_fields = [
        "citations", "concerns", "classification", "redline_draft",
        "revision_count", "status", "approver_note", "final_result",
    ]
    for field in control_fields:
        assert not has_reducer(field), (
            f"{field} must NOT have a reducer - it is a control field the "
            f"router reads; if it accumulates it can never go back to "
            f"'nothing wrong', and the graph loops forever."
        )

    print("state fields:", list(ann))
    print("audit_log has reducer  :", has_reducer("audit_log"))
    print("concerns  has reducer  :", has_reducer("concerns"))
    print()

    seed = seed_state("SHOPSENSE-00042", "I want a refund for my broken headphones.")
    print("seed_state() ->")
    for k, v in seed.items():
        print(f"  {k:>22} : {v!r}")

    print("\nPASS - audit field accumulates, every control field overwrites.")