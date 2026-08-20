"""
ShopSense M5 - Parts 5 & 6 tests: workflow/graph.py (build_graph).

Pure router tests live separately in test_routing.py; human_approval's own
logic tests live in test_human_approval.py (both fully executed, no
langgraph needed). Everything in THIS file needs a real compiled
StateGraph, so it's gated behind `pytest.importorskip("langgraph")`.

STATUS: langgraph is not installable in the sandbox this was built in (no
PyPI access there). Every test below is WRITTEN AND REASONED THROUGH
against the confirmed langgraph API the Day3 Session1 notebook itself uses
(StateGraph/START/END/add_conditional_edges/interrupt/Command/compile/
invoke), but NOT yet executed. Please run this file in your real
shopsensevenv and report back what you see.
"""

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed in this environment")

from langgraph.types import Command  # noqa: E402  (after importorskip, deliberately)

from workflow.graph import build_graph  # noqa: E402
from workflow.state import seed_state  # noqa: E402


def _standard_ticket():
    return dict(
        ticket_id="T-STD", raw_text="My headphones arrived broken, refund Rs. 500.",
        customer_ref="CUST-1", order_id="ORD-1",
    )


def _non_standard_ticket():
    return dict(
        ticket_id="T-NSTD", raw_text="This laptop bag is defective, refund me Rs. 3500.",
        customer_ref="CUST-2", order_id="ORD-2",
    )


def test_graph_compiles_with_the_expected_nodes():
    graph = build_graph()
    nodes = set(n for n in graph.get_graph().nodes if not n.startswith("__"))
    assert nodes == {"extract", "compare_to_playbook", "draft_redline", "auto_approve", "human_approval", "finalize"}


def test_standard_ticket_skips_the_interrupt_entirely():
    """The whole point of the standard/non_standard split: a clean ticket
    reaches finalize in ONE invoke() call, no pause, no Command(resume=...)
    needed."""
    graph = build_graph()
    demo = _standard_ticket()
    cfg = {"configurable": {"thread_id": demo["ticket_id"]}}
    result = graph.invoke(
        seed_state(demo["ticket_id"], demo["raw_text"], customer_ref=demo["customer_ref"], order_id=demo["order_id"]),
        cfg,
    )

    assert "__interrupt__" not in result
    assert result["classification"] == "standard"
    assert result["approver_id"] == "system:auto-approval"
    assert result["status"] == "finalized"


def test_non_standard_ticket_pauses_with_a_useful_payload():
    graph = build_graph()
    demo = _non_standard_ticket()
    cfg = {"configurable": {"thread_id": demo["ticket_id"]}}
    result = graph.invoke(
        seed_state(demo["ticket_id"], demo["raw_text"], customer_ref=demo["customer_ref"], order_id=demo["order_id"]),
        cfg,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["ticket_id"] == demo["ticket_id"]
    assert payload["concerns"]
    assert payload["redline_draft"]

    snap = graph.get_state(cfg)
    assert snap.next == ("human_approval",)


def test_resuming_with_approval_and_approver_id_reaches_finalize():
    graph = build_graph()
    demo = _non_standard_ticket()
    cfg = {"configurable": {"thread_id": demo["ticket_id"]}}
    graph.invoke(
        seed_state(demo["ticket_id"], demo["raw_text"], customer_ref=demo["customer_ref"], order_id=demo["order_id"]),
        cfg,
    )

    final = graph.invoke(
        Command(resume={"action": "approved", "note": "Confirmed with finance.", "approver_id": "reviewer-42"}),
        cfg,
    )

    assert final["status"] == "finalized"
    assert final["approver_id"] == "reviewer-42"
    assert graph.get_state(cfg).next == ()


def test_resuming_with_approval_but_no_approver_id_does_not_finalize():
    """route_after_human's authz enforcement, proven end-to-end through a
    real pause/resume cycle, not just the router in isolation."""
    graph = build_graph()
    demo = _non_standard_ticket()
    cfg = {"configurable": {"thread_id": demo["ticket_id"] + "-noauth"}}
    graph.invoke(
        seed_state(demo["ticket_id"], demo["raw_text"], customer_ref=demo["customer_ref"], order_id=demo["order_id"]),
        cfg,
    )

    final = graph.invoke(Command(resume={"action": "approved", "note": "looks fine"}), cfg)

    assert final["status"] == "approved"
    assert final["status"] != "finalized"


def test_resuming_with_rejection_stops_without_finalizing():
    graph = build_graph()
    demo = _non_standard_ticket()
    cfg = {"configurable": {"thread_id": demo["ticket_id"] + "-reject"}}
    graph.invoke(
        seed_state(demo["ticket_id"], demo["raw_text"], customer_ref=demo["customer_ref"], order_id=demo["order_id"]),
        cfg,
    )

    final = graph.invoke(
        Command(resume={"action": "rejected", "note": "Not eligible.", "approver_id": "reviewer-42"}),
        cfg,
    )

    assert final["status"] == "rejected"
    assert graph.get_state(cfg).next == ()


def test_resuming_with_an_edited_draft_replaces_the_redline():
    graph = build_graph()
    demo = _non_standard_ticket()
    cfg = {"configurable": {"thread_id": demo["ticket_id"] + "-edit"}}
    graph.invoke(
        seed_state(demo["ticket_id"], demo["raw_text"], customer_ref=demo["customer_ref"], order_id=demo["order_id"]),
        cfg,
    )

    final = graph.invoke(
        Command(resume={
            "action": "approved", "approver_id": "reviewer-42",
            "edited_draft": "A manually rewritten resolution message.",
        }),
        cfg,
    )

    assert final["redline_draft"] == "A manually rewritten resolution message."


def test_audit_log_accumulates_across_every_node_including_after_resume():
    graph = build_graph()
    demo = _non_standard_ticket()
    cfg = {"configurable": {"thread_id": demo["ticket_id"] + "-audit"}}
    graph.invoke(
        seed_state(demo["ticket_id"], demo["raw_text"], customer_ref=demo["customer_ref"], order_id=demo["order_id"]),
        cfg,
    )
    final = graph.invoke(
        Command(resume={"action": "approved", "approver_id": "reviewer-42"}), cfg,
    )

    # NOTE: routers (route_after_review / route_after_human) never appear
    # here - a LangGraph conditional-edge function's signature is
    # `state -> str`, it cannot return a partial state update, so it is
    # architecturally impossible for a router to write its own audit entry.
    # Which route was taken is inferrable from which node ran next, not
    # from a dedicated log line.
    prefixes = [entry.split(":")[0] for entry in final["audit_log"]]
    for expected in ("extract", "compare_to_playbook", "draft_redline", "human_approval", "finalize"):
        assert expected in prefixes, f"missing audit entries from {expected!r}: {final['audit_log']}"


def test_malformed_ticket_still_completes_without_crashing():
    graph = build_graph()
    result = graph.invoke(seed_state("T-EMPTY", "   "), {"configurable": {"thread_id": "T-EMPTY"}})

    assert "__interrupt__" in result
    assert result["classification"] == "non_standard"