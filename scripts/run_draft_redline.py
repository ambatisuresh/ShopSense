"""
ShopSense M5 - Part 4 demo: run `extract` -> `compare_to_playbook` ->
`draft_redline` end to end, standalone, no graph yet. Uses every offline
fixture built so far - no LLM key, no live order table, no corpus/index.
"""

from workflow.nodes.compare_to_playbook import (
    build_compare_to_playbook_node,
    fixture_evaluate_refund,
    fixture_retrieve_citations,
)
from workflow.nodes.draft_redline import build_draft_redline_node
from workflow.nodes.extract import (
    build_extract_node,
    fixture_parse_ticket,
    fixture_resolve_customer_ref,
)
from workflow.state import seed_state

extract = build_extract_node(fixture_parse_ticket, fixture_resolve_customer_ref)
compare_to_playbook = build_compare_to_playbook_node(fixture_evaluate_refund, fixture_retrieve_citations)
draft_redline = build_draft_redline_node()  # deterministic composer (the default)

DEMO_TICKETS = [
    dict(ticket_id="SHOPSENSE-00001",
         raw_text="My headphones arrived broken, I want a refund of Rs. 500.",
         customer_ref="CUST-500", order_id="ORD-1"),
    dict(ticket_id="SHOPSENSE-00002",
         raw_text="This laptop bag is defective, refund me Rs. 3500.",
         customer_ref="CUST-501", order_id="ORD-2"),
    dict(ticket_id="SHOPSENSE-00003",
         raw_text="Get me a human agent right now or I will sue.",
         customer_ref="CUST-502", order_id="ORD-3"),
]

if __name__ == "__main__":
    for demo in DEMO_TICKETS:
        state = seed_state(
            demo["ticket_id"], demo["raw_text"],
            customer_ref=demo.get("customer_ref"), order_id=demo.get("order_id"),
        )
        state.update(extract(state))
        state.update(compare_to_playbook(state))
        update = draft_redline(state)

        print(f"=== {demo['ticket_id']} ===")
        print(f"  raw_text       : {demo['raw_text']!r}")
        print(f"  classification : {update['classification']}")
        print(f"  concerns       : {update['concerns']}")
        print(f"  revision_count : {update['revision_count']}")
        print(f"  redline_draft  :\n    {update['redline_draft']}")
        for line in update["audit_log"]:
            print(f"    audit: {line}")
        print()