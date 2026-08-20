"""
ShopSense M5 - Step 7 tests: workflow/nodes/finalize.py

Fully executed here, NO langgraph needed - build_finalize_node takes a
plain evaluate_refund callable, same dependency-injection shape as every
other node factory in this project.

The most important test in this file is not "does finalize work" but "does
finalize call evaluate_refund with commit=True while compare_to_playbook
calls it with commit=False" - that distinction is the entire point of
Step 7's design. A single shared spy proves both halves of that contract at
once.
"""

from workflow.nodes.compare_to_playbook import build_compare_to_playbook_node
from workflow.nodes.finalize import build_finalize_node
from workflow.state import seed_state


def _spy_evaluate_refund(eligible_amount=500.0, fraud_flag=False, amount_mismatch=False):
    """Records every call's (order_id, claimed_amount, commit) so tests can
    assert exactly what each node passed - not just that SOME call
    happened."""
    calls = []

    def evaluate_refund(order_id, claimed_amount, commit=False):
        calls.append({"order_id": order_id, "claimed_amount": claimed_amount, "commit": commit})
        return {
            "eligible_amount": eligible_amount,
            "action": "refund",
            "fraud_flag": fraud_flag,
            "amount_mismatch": amount_mismatch,
        }

    evaluate_refund.calls = calls
    return evaluate_refund


def _refund_state(**overrides):
    state = seed_state("T1", "My headphones arrived broken, refund Rs. 500.", order_id="ORD-1")
    state["parsed_ticket"] = {"issue_type": "REFUND", "claimed_refund_amount": 500.0}
    state["classification"] = "standard"
    state["status"] = "approved"
    state["approver_id"] = "system:auto-approval"
    state["redline_draft"] = "Your refund of ₹500.00 has been approved."
    state.update(overrides)
    return state


def _non_refund_state(**overrides):
    state = seed_state("T2", "Where is my order?", order_id="ORD-2")
    state["parsed_ticket"] = {"issue_type": "DELIVERY", "claimed_refund_amount": None}
    state["classification"] = "standard"
    state["status"] = "approved"
    state["approver_id"] = "system:auto-approval"
    state["redline_draft"] = "Your order is on the way."
    state.update(overrides)
    return state


# --------------------------------------------------------------------------
# The commit=False vs commit=True contract - the reason Step 7 exists
# --------------------------------------------------------------------------

def test_finalize_calls_evaluate_refund_with_commit_true():
    spy = _spy_evaluate_refund()
    finalize = build_finalize_node(spy)

    finalize(_refund_state())

    assert len(spy.calls) == 1
    assert spy.calls[0]["commit"] is True


def test_compare_to_playbook_calls_evaluate_refund_with_commit_false():
    """The other half of the contract, proven against the ACTUAL
    compare_to_playbook node (Step 3), not a re-implementation of it -
    guards against the two nodes silently drifting apart in the future."""
    spy = _spy_evaluate_refund()
    compare = build_compare_to_playbook_node(spy, lambda query, k: [])

    state = seed_state("T1", "My headphones arrived broken, refund Rs. 500.", order_id="ORD-1")
    state["parsed_ticket"] = {"issue_type": "REFUND", "claimed_refund_amount": 500.0}
    compare(state)

    assert len(spy.calls) == 1
    assert spy.calls[0]["commit"] is False


def test_a_shared_evaluate_refund_instance_sees_false_then_true_in_order():
    """End-to-end proof using ONE spy for both nodes, the same way
    build_graph() wires ONE evaluate_refund value to both - simulates the
    real classify-then-finalize sequence a ticket goes through."""
    spy = _spy_evaluate_refund()
    compare = build_compare_to_playbook_node(spy, lambda query, k: [])
    finalize = build_finalize_node(spy)

    state = _refund_state()
    state.update(compare(state))
    state.update(finalize(state))

    assert [c["commit"] for c in spy.calls] == [False, True]


# --------------------------------------------------------------------------
# Commit-needed vs no-commit-needed tickets
# --------------------------------------------------------------------------

def test_refund_ticket_commits_and_records_final_result():
    spy = _spy_evaluate_refund(eligible_amount=500.0)
    finalize = build_finalize_node(spy)

    update = finalize(_refund_state())

    assert update["status"] == "finalized"
    assert update["final_result"]["committed"] is True
    assert update["final_result"]["eligible_amount"] == 500.0
    assert update["final_result"]["ticket_id"] == "T1"
    assert update["final_result"]["approver_id"] == "system:auto-approval"


def test_non_refund_ticket_finalizes_without_calling_evaluate_refund():
    spy = _spy_evaluate_refund()
    finalize = build_finalize_node(spy)

    update = finalize(_non_refund_state())

    assert spy.calls == []
    assert update["status"] == "finalized"
    assert update["final_result"]["committed"] is False


def test_refund_ticket_with_no_order_id_fails_closed_without_committing():
    """Should be unreachable given compare_to_playbook's own routing, but
    finalize does not trust that invariant blindly - it re-checks rather
    than crashing or guessing an order_id."""
    spy = _spy_evaluate_refund()
    finalize = build_finalize_node(spy)

    state = _refund_state(order_id=None)
    update = finalize(state)

    assert spy.calls == []
    assert update["status"] == "finalized"
    assert update["final_result"]["committed"] is False
    assert "commit_error" in update["final_result"]


def test_empty_parsed_ticket_finalizes_without_committing():
    spy = _spy_evaluate_refund()
    finalize = build_finalize_node(spy)

    state = _refund_state(parsed_ticket={})
    update = finalize(state)

    assert spy.calls == []
    assert update["final_result"]["committed"] is False


# --------------------------------------------------------------------------
# Node contract basics
# --------------------------------------------------------------------------

def test_final_result_always_carries_the_approved_redline():
    finalize = build_finalize_node(_spy_evaluate_refund())

    update = finalize(_non_refund_state())

    assert update["final_result"]["resolution"] == "Your order is on the way."


def test_default_evaluate_refund_is_the_offline_fixture_and_works_unwired():
    """build_finalize_node() with zero args must still work, same posture
    as every other build_X_node default - callable with no external
    services wired."""
    finalize = build_finalize_node()

    update = finalize(_refund_state())

    assert update["status"] == "finalized"
    assert update["final_result"]["committed"] is True


def test_node_returns_only_the_expected_keys():
    finalize = build_finalize_node(_spy_evaluate_refund())

    update = finalize(_refund_state())

    assert set(update.keys()) == {"status", "final_result", "audit_log"}


def test_audit_log_mentions_commit_for_a_refund_ticket():
    finalize = build_finalize_node(_spy_evaluate_refund())

    update = finalize(_refund_state())

    assert any("committed refund" in line for line in update["audit_log"])


def test_audit_log_mentions_no_commit_for_a_non_refund_ticket():
    finalize = build_finalize_node(_spy_evaluate_refund())

    update = finalize(_non_refund_state())

    assert any("no commit needed" in line for line in update["audit_log"])