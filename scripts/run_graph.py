"""
ShopSense M5 - Parts 5 & 6 demo: run the COMPILED graph end to end, via
graph.invoke() / Command(resume=...), for both the standard (no pause) and
non_standard (real interrupt + resume) paths.

Uses the EPHEMERAL default checkpointer (InMemorySaver) - the pause and the
resume both happen inside this one process/script run, so an in-heap saver
is enough to prove the graph's routing and interrupt/resume wiring. For a
real cross-process durability demo (state surviving a genuine process
restart via a disk-backed SqliteSaver), see scripts/run_checkpointing_pause.
py / _resume.py (Step 8) instead - that's the pair to run as two separate
`python3` invocations.

Needs the real `langgraph` package - not installable in the sandbox this
was built in, so this hasn't been executed by me. Written directly against
the confirmed API the Day3 Session1 notebook itself uses throughout
(StateGraph/interrupt/Command/compile/invoke) - run it in your real
shopsensevenv and let me know what you see.
"""

from langgraph.types import Command

from workflow.checkpointing import thread_config
from workflow.graph import build_graph
from workflow.state import seed_state
from workflow.visualize import print_node_list, show_graph

graph = build_graph()  # every dependency defaults to an offline fixture; InMemorySaver checkpointer

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
    print_node_list(graph)
    print()
    show_graph(graph, "ShopSense M5 - the finished ticket-review workflow", png_path="shopsense_graph.png")
    print()

    for demo in DEMO_TICKETS:
        cfg = thread_config(demo["ticket_id"])
        seed = seed_state(
            demo["ticket_id"], demo["raw_text"],
            customer_ref=demo.get("customer_ref"), order_id=demo.get("order_id"),
        )
        result = graph.invoke(seed, cfg)

        print(f"=== {demo['ticket_id']} ===")
        print(f"  classification : {result['classification']}")

        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"  PAUSED for human review. Reviewer would see:")
            print(f"    concerns      : {payload['concerns']}")
            print(f"    redline_draft : {payload['redline_draft'][:80]}...")

            # Simulate a reviewer resuming the SAME thread - possibly in a
            # different process, hours later, once real checkpointing (an
            # InMemorySaver here; SqliteSaver from Step 8 onward) is wired.
            final = graph.invoke(
                Command(resume={"action": "approved", "note": "Reviewed and approved.", "approver_id": "reviewer-42"}),
                cfg,
            )
            print(f"  RESUMED -> status={final['status']} approver_id={final['approver_id']}")
        else:
            print(f"  status         : {result['status']} (no pause - auto-approved)")

        print()

"""
EXPECTED OUTPUT
---------------
nodes: ['auto_approve', 'compare_to_playbook', 'draft_redline', 'extract', 'finalize', 'human_approval']

ShopSense M5 - the finished ticket-review workflow
----------------------------------------------------
(graph picture written to shopsense_graph.png - open it to view)
  - or, if mermaid.ink is unreachable, the Mermaid source is printed instead.

SHOPSENSE-00001: classification=standard, status=finalized, no pause
SHOPSENSE-00002: classification=non_standard, PAUSED, then RESUMED -> status=finalized
SHOPSENSE-00003: classification=non_standard, PAUSED, then RESUMED -> status=finalized
"""