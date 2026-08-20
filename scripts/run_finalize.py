"""
ShopSense M5 - Step 7 demo: extract -> compare_to_playbook -> draft_redline
-> (auto_approve | human_approval) -> finalize, standalone, calling the
node functions directly (no compiled graph, no langgraph needed).

Same shape as scripts/run_human_approval.py, extended one node further so
you can see finalize's final_result for both a standard (auto-approved,
refund) ticket and a non_standard (human-approved, refund) ticket, plus a
non-refund ticket to show the no-commit-needed path.

A SHARED evaluate_refund instance is passed to both compare_to_playbook and
finalize below - deliberately, to mirror exactly how build_graph() wires
them in workflow/graph.py. Watch the printed call log: each ticket shows
exactly one commit=False call (during compare_to_playbook) followed by, for
refund tickets only, exactly one commit=True call (during finalize).
"""

from workflow.nodes.compare_to_playbook import (
    build_compare_to_playbook_node,
    fixture_retrieve_citations,
)
from workflow.nodes.draft_redline import build_draft_redline_node
from workflow.nodes.extract import build_extract_node, fixture_parse_ticket, fixture_resolve_customer_ref
from workflow.nodes.finalize import build_finalize_node
from workflow.nodes.human_approval import build_human_approval_node
from workflow.routing import route_after_human, route_after_review
from workflow.state import seed_state


def make_logging_evaluate_refund():
    """Wraps fixture_evaluate_refund's logic but prints + records every
    call, so the demo output makes the commit=False -> commit=True sequence
    visible instead of just asserting it in a test."""
    calls = []

    def evaluate_refund(order_id, claimed_amount, commit=False):
        calls.append((order_id, claimed_amount, commit))
        print(f"    [evaluate_refund called] order_id={order_id} claimed_amount={claimed_amount} commit={commit}")
        return {
            "eligible_amount": claimed_amount,
            "action": "refund" if claimed_amount else None,
            "fraud_flag": False,
            "amount_mismatch": False,
        }

    evaluate_refund.calls = calls
    return evaluate_refund


def make_fake_reviewer(action: str, approver_id: str = "reviewer-42", note: str = ""):
    def interrupt_fn(payload):
        return {"action": action, "approver_id": approver_id, "note": note}
    return interrupt_fn


SCENARIOS = [
    dict(
        ticket_id="SHOPSENSE-00001",
        raw_text="My headphones arrived broken, I want a refund of Rs. 500.",
        customer_ref="CUST-500", order_id="ORD-1",
        note="standard (auto-approve) refund ticket",
    ),
    dict(
        ticket_id="SHOPSENSE-00002",
        raw_text="This laptop bag is defective, refund me Rs. 3500.",
        customer_ref="CUST-501", order_id="ORD-2",
        note="non_standard (human-approval) refund ticket - over the cap",
    ),
    dict(
        ticket_id="SHOPSENSE-00004",
        raw_text="Where is my order? It's three days late.",
        customer_ref="CUST-503", order_id="ORD-4",
        note="standard delivery ticket - no commit needed at all",
    ),
]

if __name__ == "__main__":
    for demo in SCENARIOS:
        print(f"=== {demo['ticket_id']} ({demo['note']}) ===")
        evaluate_refund = make_logging_evaluate_refund()  # one shared instance per ticket, like build_graph()

        extract = build_extract_node(fixture_parse_ticket, fixture_resolve_customer_ref)
        compare_to_playbook = build_compare_to_playbook_node(evaluate_refund, fixture_retrieve_citations)
        draft_redline = build_draft_redline_node()
        finalize = build_finalize_node(evaluate_refund)

        state = seed_state(
            demo["ticket_id"], demo["raw_text"],
            customer_ref=demo.get("customer_ref"), order_id=demo.get("order_id"),
        )
        state.update(extract(state))
        state.update(compare_to_playbook(state))
        state.update(draft_redline(state))
        print(f"  classification: {state['classification']}")

        route = route_after_review(state)
        if route == "auto_approve":
            state["status"] = "approved"
            state["approver_id"] = "system:auto-approval"
            state["approver_note"] = "auto-approved: classification=standard, no concerns raised"
            print("  auto_approve: approved without human review")
        else:
            human_approval = build_human_approval_node(make_fake_reviewer("approved", note="Confirmed with finance."))
            state.update(human_approval(state))
            print(f"  human_approval: reviewer approved (approver_id={state['approver_id']})")
            after_human = route_after_human(state)
            assert after_human == "finalize", f"expected to reach finalize, got {after_human!r}"

        state.update(finalize(state))
        print(f"  status        : {state['status']}")
        print(f"  final_result  : {state['final_result']}")
        print(f"  evaluate_refund call sequence: {[c[2] for c in evaluate_refund.calls]}")
        print()

"""
EXPECTED OUTPUT
---------------
SHOPSENSE-00001 (standard refund): commit sequence [False, True], final_result.committed=True
SHOPSENSE-00002 (non_standard refund, human-approved): commit sequence [False, True], final_result.committed=True
SHOPSENSE-00004 (standard delivery, no refund): commit sequence [False] only - finalize never
    calls evaluate_refund at all, so no commit=True entry appears; final_result.committed=False
"""