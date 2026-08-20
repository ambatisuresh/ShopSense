"""
ShopSense M5 - Part 6 demo: extract -> compare_to_playbook -> draft_redline
-> human_approval, standalone, calling the node functions directly (no
compiled graph, no langgraph needed - build_human_approval_node's lazy
import means this runs anywhere).

A fake interrupt_fn simulates a reviewer's decision synchronously - stands
in for a real pause/resume cycle the same way every prior demo's fixtures
stand in for a live dependency. For the REAL pause/resume behaviour
(interrupt() actually parking the graph, Command(resume=...) waking it back
up in a possibly-different process), see scripts/run_graph.py - that one
needs your real langgraph install.
"""

from workflow.nodes.compare_to_playbook import (
    build_compare_to_playbook_node,
    fixture_evaluate_refund,
    fixture_retrieve_citations,
)
from workflow.nodes.draft_redline import build_draft_redline_node
from workflow.nodes.extract import build_extract_node, fixture_parse_ticket, fixture_resolve_customer_ref
from workflow.nodes.human_approval import build_human_approval_node
from workflow.routing import route_after_human
from workflow.state import seed_state

extract = build_extract_node(fixture_parse_ticket, fixture_resolve_customer_ref)
compare_to_playbook = build_compare_to_playbook_node(fixture_evaluate_refund, fixture_retrieve_citations)
draft_redline = build_draft_redline_node()


def make_fake_reviewer(action: str, approver_id: str = "reviewer-42", note: str = "", edited_draft: str = None):
    """A fake interrupt_fn: instead of actually pausing, immediately
    returns the decision a reviewer would eventually supply via
    Command(resume=...). Prints the payload it was shown first, so you can
    see exactly what a real human reviewer would be looking at."""
    def interrupt_fn(payload):
        print("  [reviewer sees this payload]:")
        for k, v in payload.items():
            print(f"    {k}: {v}")
        decision = {"action": action, "approver_id": approver_id, "note": note}
        if edited_draft:
            decision["edited_draft"] = edited_draft
        print(f"  [reviewer responds]: {decision}")
        return decision
    return interrupt_fn


SCENARIOS = [
    dict(
        ticket_id="SHOPSENSE-00002",
        raw_text="This laptop bag is defective, refund me Rs. 3500.",
        customer_ref="CUST-501", order_id="ORD-2",
        reviewer_action="approved", note="Confirmed with finance, one-off exception.",
    ),
    dict(
        ticket_id="SHOPSENSE-00003",
        raw_text="Get me a human agent right now or I will sue.",
        customer_ref="CUST-502", order_id="ORD-3",
        reviewer_action="rejected", note="Escalating to legal separately, not a standard resolution.",
    ),
    dict(
        ticket_id="SHOPSENSE-00006",
        raw_text="This laptop bag is defective, refund me Rs. 2200.",
        customer_ref="CUST-504", order_id="ORD-6",
        reviewer_action="approved", note="",  # deliberately no approver_id below
        no_approver_id=True,
    ),
]

if __name__ == "__main__":
    for demo in SCENARIOS:
        state = seed_state(
            demo["ticket_id"], demo["raw_text"],
            customer_ref=demo.get("customer_ref"), order_id=demo.get("order_id"),
        )
        state.update(extract(state))
        state.update(compare_to_playbook(state))
        state.update(draft_redline(state))

        print(f"=== {demo['ticket_id']} (classification={state['classification']}) ===")

        approver_id = None if demo.get("no_approver_id") else "reviewer-42"
        interrupt_fn = make_fake_reviewer(demo["reviewer_action"], approver_id=approver_id, note=demo["note"])
        human_approval = build_human_approval_node(interrupt_fn)
        update = human_approval(state)
        state.update(update)

        route = route_after_human(state)
        print(f"  status              : {state['status']}")
        print(f"  approver_id         : {state['approver_id']}")
        print(f"  route_after_human   : {route!r}"
              + (" -> would finalize" if route == "finalize" else " -> stops here, no finalize"))
        for line in update["audit_log"]:
            print(f"  audit: {line}")
        print()

"""
EXPECTED OUTPUT
---------------
SHOPSENSE-00002: reviewer approves with an approver_id -> route_after_human = "finalize"
SHOPSENSE-00003: reviewer rejects -> route_after_human = "end"
SHOPSENSE-00006: reviewer "approves" but supplies NO approver_id -> status="approved"
    but route_after_human STILL = "end" (fails closed - no finalize without a recorded approver)
"""