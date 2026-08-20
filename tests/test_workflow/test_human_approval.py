"""
ShopSense M5 - Part 6 tests: workflow/nodes/human_approval.py

Fully executed here, NO langgraph needed - build_human_approval_node's
lazy-import means injecting a fake `interrupt_fn` (a plain callable that
returns a canned decision dict synchronously) exercises the node's real
logic without ever touching the langgraph package. This is the payoff of
the lazy-import fix described in graph.py/human_approval.py's docstrings:
unlike test_graph.py, none of this needs `pytest.importorskip`.

The fake interrupt_fn stands in for a live pause/resume cycle the same way
every other node's fixtures stand in for a live dependency (fixture_parse_
ticket for a live LLM, fixture_evaluate_refund for a live order table).
"""

from workflow.nodes.human_approval import build_human_approval_node
from workflow.state import seed_state


def _reviewed_state(**overrides):
    state = seed_state("T1", "This laptop bag is defective, refund me Rs. 3500.",
                        customer_ref="CUST-1", order_id="ORD-1")
    state["parsed_ticket"] = {"issue_type": "REFUND", "claimed_refund_amount": 3500.0}
    state["policy_eligible_amount"] = 3500.0
    state["citations"] = [{"doc_slug": "refund-authority", "clause_number": "4.1"}]
    state["concerns"] = ["refund-authority.md §4.1/4.2: eligible amount ₹3500.00 exceeds the ₹2000 auto-approval cap"]
    state["classification"] = "non_standard"
    state["redline_draft"] = "You requested a refund of ₹3500.00. This has been flagged for review."
    state.update(overrides)
    return state


def fake_interrupt(canned_decision):
    """Factory for a fake interrupt_fn: returns `canned_decision`
    regardless of the payload it's called with, and records the payload it
    was called with for assertions."""
    calls = []

    def interrupt_fn(payload):
        calls.append(payload)
        return canned_decision

    interrupt_fn.calls = calls
    return interrupt_fn


# --------------------------------------------------------------------------
# Payload shape - what a real reviewer would see
# --------------------------------------------------------------------------

def test_payload_carries_everything_a_reviewer_needs():
    interrupt_fn = fake_interrupt({"action": "approved", "approver_id": "rev-1"})
    human_approval = build_human_approval_node(interrupt_fn)

    human_approval(_reviewed_state())

    assert len(interrupt_fn.calls) == 1
    payload = interrupt_fn.calls[0]
    assert payload["ticket_id"] == "T1"
    assert payload["redline_draft"]
    assert payload["concerns"]
    assert payload["citations"]
    assert payload["policy_eligible_amount"] == 3500.0
    assert payload["issue_type"] == "REFUND"


def test_payload_is_the_only_thing_that_happens_before_the_pause():
    """Lab B Part B8's lesson: on resume, a node re-runs from the top -
    anything before interrupt() executes twice. This node has nothing
    before it but state reads. Proven indirectly: calling human_approval
    TWICE (simulating first-pass + resume-replay) with the same fake
    interrupt_fn produces IDENTICAL payloads both times - no counter, no
    accumulating side effect anywhere in this node."""
    interrupt_fn = fake_interrupt({"action": "approved", "approver_id": "rev-1"})
    human_approval = build_human_approval_node(interrupt_fn)
    state = _reviewed_state()

    human_approval(state)
    human_approval(state)

    assert interrupt_fn.calls[0] == interrupt_fn.calls[1]


# --------------------------------------------------------------------------
# Decision handling
# --------------------------------------------------------------------------

def test_approved_decision_sets_status_and_records_approver():
    interrupt_fn = fake_interrupt({"action": "approved", "note": "Looks fine.", "approver_id": "rev-1"})
    human_approval = build_human_approval_node(interrupt_fn)

    update = human_approval(_reviewed_state())

    assert update["status"] == "approved"
    assert update["approver_note"] == "Looks fine."
    assert update["approver_id"] == "rev-1"


def test_rejected_decision_sets_status_rejected():
    interrupt_fn = fake_interrupt({"action": "rejected", "note": "Not valid.", "approver_id": "rev-1"})
    human_approval = build_human_approval_node(interrupt_fn)

    update = human_approval(_reviewed_state())

    assert update["status"] == "rejected"


def test_edited_draft_overrides_redline_draft():
    interrupt_fn = fake_interrupt({
        "action": "approved", "approver_id": "rev-1",
        "edited_draft": "A manually rewritten resolution message.",
    })
    human_approval = build_human_approval_node(interrupt_fn)

    update = human_approval(_reviewed_state())

    assert update["redline_draft"] == "A manually rewritten resolution message."


def test_no_edited_draft_keeps_the_original_redline():
    interrupt_fn = fake_interrupt({"action": "approved", "approver_id": "rev-1"})
    human_approval = build_human_approval_node(interrupt_fn)
    state = _reviewed_state()

    update = human_approval(state)

    assert update["redline_draft"] == state["redline_draft"]


def test_missing_approver_id_is_recorded_as_none_not_invented():
    """The notebook's own warning: interrupt() gives back a value, not
    authorization. If the resume call didn't supply an approver_id, this
    node must record that honestly (None), never fabricate one -
    enforcement of "don't finalize without it" is route_after_human's job,
    not this node's."""
    interrupt_fn = fake_interrupt({"action": "approved", "note": "ok"})
    human_approval = build_human_approval_node(interrupt_fn)

    update = human_approval(_reviewed_state())

    assert update["approver_id"] is None


def test_missing_note_defaults_to_empty_string():
    interrupt_fn = fake_interrupt({"action": "approved", "approver_id": "rev-1"})
    human_approval = build_human_approval_node(interrupt_fn)

    update = human_approval(_reviewed_state())

    assert update["approver_note"] == ""


def test_node_returns_only_the_expected_keys():
    interrupt_fn = fake_interrupt({"action": "approved", "approver_id": "rev-1"})
    human_approval = build_human_approval_node(interrupt_fn)

    update = human_approval(_reviewed_state())

    assert set(update.keys()) == {"status", "approver_note", "approver_id", "redline_draft", "audit_log"}