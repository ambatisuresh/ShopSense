"""
ShopSense M5 - Step 7: the `finalize` node.

This is the ONLY node in the whole graph that is allowed to take an
irreversible action - actually resolving the ticket (committing a refund,
or recording that no commit was needed). Lab B Part B8's rule, restated for
this milestone: irreversible work lives in its own node, strictly
downstream of every possible pause point (`human_approval`'s `interrupt()`),
never inside a node that might re-run before or during a pause.

Reached two ways, both already enforced upstream by the routers (Step 5/6),
not re-checked here:
    - auto_approve -> finalize   (standard ticket, no human needed)
    - human_approval -> finalize (non_standard ticket, reviewer approved
      AND route_after_human confirmed an approver_id was recorded)

finalize's job, in order:
    1. Decide whether this ticket needs a commit at all. A refund-type
       ticket (issue_type == "REFUND" or a claimed_refund_amount was
       present) needs one; a delivery-status reply or a warranty/product
       question does not - there is nothing to "commit" for those beyond
       sending the already-approved redline_draft.
    2. If a commit is needed, call the SAME `evaluate_refund` dependency
       compare_to_playbook (Step 3) used to classify the ticket - but this
       time with `commit=True`. See compare_to_playbook.py's module
       docstring ("STEP 7 ADDITION") for the full explanation of why this
       flag exists and what remains unconfirmed about M2's real
       `process_refund`.
    3. Populate `final_result` (a JSON-serializable record of what actually
       happened - the thing you'd show in a "ticket resolved" confirmation
       or write to an audit trail) and set `status = "finalized"`.
    4. Log every decision to `audit_log`.

finalize does NOT re-run classification, re-check the cap, or second-guess
the approval that got it here - by the time this node runs, either the
system (auto_approve) or a recorded human (human_approval) has already
signed off. finalize's only question is "does resolving this ticket require
an action, and if so, take it."
"""

from typing import Callable, Optional

from workflow.nodes.compare_to_playbook import EvaluateRefundFn, fixture_evaluate_refund
from workflow.state import TicketReviewState


def _needs_commit(parsed_ticket: dict) -> bool:
    if not parsed_ticket:
        return False
    return bool(
        parsed_ticket.get("issue_type") == "REFUND"
        or parsed_ticket.get("claimed_refund_amount") is not None
    )


def build_finalize_node(
    evaluate_refund: EvaluateRefundFn = fixture_evaluate_refund,
) -> Callable[[TicketReviewState], dict]:
    """Returns the `finalize` node: `TicketReviewState -> partial update`.

    `evaluate_refund` must be the SAME dependency (or an equivalent wired to
    the same underlying M2 adapter) passed to `build_compare_to_playbook_node`
    when the graph was built - see workflow/graph.py's `build_graph()`,
    which threads one `evaluate_refund` value to both nodes for exactly this
    reason. Passing two different instances would defeat the whole point of
    the `commit` flag.
    """

    def finalize(state: TicketReviewState) -> dict:
        parsed = state.get("parsed_ticket") or {}
        order_id = state.get("order_id")
        audit: list[str] = []

        final_result: dict = {
            "ticket_id": state["ticket_id"],
            "resolution": state.get("redline_draft"),
            "approver_id": state.get("approver_id"),
            "committed": False,
        }

        if _needs_commit(parsed):
            if not order_id:
                # Should be unreachable - compare_to_playbook already routes
                # a refund-with-no-order_id ticket to non_standard, and a
                # human reviewer approving it doesn't change that there's
                # still nothing to commit against. Fail closed: record the
                # gap instead of guessing an order_id or skipping silently.
                final_result["committed"] = False
                final_result["commit_error"] = "no order_id on record - cannot commit a refund"
                audit.append("finalize: refund needed but no order_id present - NOT committed")
            else:
                # commit=True: the one and only call site in the whole
                # workflow allowed to pass this. See compare_to_playbook.py
                # ("STEP 7 ADDITION") for the unresolved question of what
                # commit=True actually does against the real M2 adapter.
                result = evaluate_refund(
                    order_id=order_id, claimed_amount=parsed.get("claimed_refund_amount"), commit=True,
                )
                final_result["committed"] = True
                final_result["eligible_amount"] = result.get("eligible_amount")
                final_result["action"] = result.get("action")
                audit.append(
                    f"finalize: committed refund evaluation -> "
                    f"eligible=₹{result.get('eligible_amount')} action={result.get('action')}"
                )
        else:
            audit.append("finalize: no commit needed for this ticket (not a refund request)")

        audit.append(f"finalize: status=finalized approver_id={state.get('approver_id')!r}")

        return {
            "status": "finalized",
            "final_result": final_result,
            "audit_log": audit,
        }

    return finalize