"""
ShopSense M5 - Part 3 demo: run `extract` -> `compare_to_playbook` back to
back, standalone, no graph yet. Uses the offline fixtures from both nodes -
no LLM key, no live order table, no corpus/index needed.
"""

from workflow.nodes.compare_to_playbook import (
    build_compare_to_playbook_node,
    fixture_evaluate_refund,
    fixture_retrieve_citations,
)
from workflow.nodes.extract import (
    build_extract_node,
    fixture_parse_ticket,
    fixture_resolve_customer_ref,
)
from workflow.state import seed_state

extract = build_extract_node(fixture_parse_ticket, fixture_resolve_customer_ref)
compare_to_playbook = build_compare_to_playbook_node(fixture_evaluate_refund, fixture_retrieve_citations)

DEMO_TICKETS = [
    # 1. Standard: small refund, known order, no triggers.
    dict(ticket_id="SHOPSENSE-00001",
         raw_text="My headphones arrived broken, I want a refund of Rs. 500.",
         customer_ref="CUST-500", order_id="ORD-1"),
    # 2. Non-standard: refund amount over the ₹2,000 auto-approval cap.
    dict(ticket_id="SHOPSENSE-00002",
         raw_text="This laptop bag is defective, refund me Rs. 3500.",
         customer_ref="CUST-501", order_id="ORD-2"),
    # 3. Non-standard: escalation-tone.md triggers (legal + human request).
    dict(ticket_id="SHOPSENSE-00003",
         raw_text="Get me a human agent right now or I will sue.",
         customer_ref="CUST-502", order_id="ORD-3"),
    # 4. Non-standard: refund with no resolvable order_id.
    dict(ticket_id="SHOPSENSE-00004",
         raw_text="I want a refund of Rs 800 for my last purchase."),
    # 5. Standard: plain delivery question, no refund involved.
    dict(ticket_id="SHOPSENSE-00005",
         raw_text="Where is my order? It's three days late.",
         customer_ref="CUST-503", order_id="ORD-4"),
]

if __name__ == "__main__":
    for demo in DEMO_TICKETS:
        state = seed_state(
            demo["ticket_id"], demo["raw_text"],
            customer_ref=demo.get("customer_ref"), order_id=demo.get("order_id"),
        )
        state.update(extract(state))
        update = compare_to_playbook(state)

        print(f"=== {demo['ticket_id']} ===")
        print(f"  raw_text            : {demo['raw_text']!r}")
        print(f"  issue_type          : {state['parsed_ticket'].get('issue_type')}")
        print(f"  classification      : {update['classification']}")
        print(f"  concerns            : {update['concerns']}")
        print(f"  policy_eligible_amt : {update['policy_eligible_amount']}")
        print(f"  citations           : {update['citations']}")
        for line in update["audit_log"]:
            print(f"    audit: {line}")
        print()

"""
EXPECTED OUTPUT
---------------
SHOPSENSE-00001: classification=standard, concerns=[]
SHOPSENSE-00002: classification=non_standard, concerns=[... '4.1/4.2' ...]
SHOPSENSE-00003: classification=non_standard, concerns=[... '4.3.1' ..., '4.3.4' ...]
SHOPSENSE-00004: classification=non_standard, concerns=[... 'no order_id' ...]
SHOPSENSE-00005: classification=standard, concerns=[]
"""