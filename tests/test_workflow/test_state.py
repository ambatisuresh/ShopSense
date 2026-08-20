"""
ShopSense M5 - Part 1 tests: workflow/state.py

Deterministic, no graph, no model, no I/O - pure introspection of the
TypedDict + one call to seed_state(). This is the regression guard for the
one invariant the whole workflow depends on: which fields overwrite and
which accumulate. Get this wrong and either the router misfires or the
audit trail silently loses steps - see the module docstring in state.py.

Same pytest convention as M4's tests/test_rag/ - plain functions, plain
assert, pytest.mark.parametrize for the per-field checks.
"""

import json
from operator import add

import pytest

from workflow.state import MAX_REVISIONS, TicketReviewState, has_reducer, seed_state

# Every field in TicketReviewState EXCEPT audit_log. Kept as an explicit list
# (not "everything but audit_log" computed from __annotations__) so that
# adding a new field to the schema without deciding control-vs-audit for it
# fails test_schema_has_no_untracked_fields loudly, instead of silently
# defaulting to "control".
CONTROL_FIELDS = [
    "ticket_id",
    "raw_text",
    "customer_ref",
    "order_id",
    "parsed_ticket",
    "citations",
    "policy_eligible_amount",
    "policy_action",
    "concerns",
    "classification",
    "redline_draft",
    "revision_count",
    "status",
    "approver_note",
    "approver_id",
    "final_result",
]


# --------------------------------------------------------------------------
# Reducer assignment - the one thing this schema exists to get right.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", CONTROL_FIELDS)
def test_control_field_has_no_reducer(field):
    assert not has_reducer(field), (
        f"{field} must overwrite, not accumulate - an accumulating control "
        f"field can never report 'nothing wrong' again once it has ever had "
        f"one entry, which would loop the graph forever."
    )


def test_audit_log_has_a_reducer():
    assert has_reducer("audit_log")


def test_audit_log_reducer_is_list_concatenation():
    assert TicketReviewState.__annotations__["audit_log"].__metadata__[0] is add


def test_schema_has_no_untracked_fields():
    """Every field in the TypedDict must be accounted for in either
    CONTROL_FIELDS or be audit_log - this is what makes the two tests above
    an exhaustive check, not a spot check."""
    all_fields = set(TicketReviewState.__annotations__)
    accounted_for = set(CONTROL_FIELDS) | {"audit_log"}
    assert all_fields == accounted_for, (
        f"Unaccounted field(s): {all_fields - accounted_for}. Add them to "
        f"CONTROL_FIELDS above (or handle as audit) and decide deliberately "
        f"whether they should have a reducer."
    )


# --------------------------------------------------------------------------
# seed_state()
# --------------------------------------------------------------------------

def test_seed_state_populates_identity_fields():
    s = seed_state("SHOPSENSE-00042", "I want a refund for my headphones.")
    assert s["ticket_id"] == "SHOPSENSE-00042"
    assert s["raw_text"] == "I want a refund for my headphones."


def test_seed_state_defaults_are_empty_not_none():
    """Lists/dicts default to empty containers, not None - every node
    downstream can assume state["concerns"] is iterable without a
    None-check, e.g. `if state["concerns"] and ...` in a router."""
    s = seed_state("T1", "raw")
    assert s["citations"] == []
    assert s["concerns"] == []
    assert s["audit_log"] == []
    assert s["parsed_ticket"] == {}
    assert s["final_result"] == {}


def test_seed_state_control_counters_start_at_zero():
    s = seed_state("T1", "raw")
    assert s["revision_count"] == 0
    assert s["classification"] == ""
    assert s["status"] == ""


def test_seed_state_unresolved_identity_is_none_not_missing():
    s = seed_state("T1", "raw")
    assert s["customer_ref"] is None
    assert s["order_id"] is None
    assert s["approver_id"] is None


def test_seed_state_is_json_serialisable():
    """Guards the checkpoint-safety claim in state.py's docstring: every
    field must be a JSON-round-trippable primitive so SqliteSaver (Step 8)
    never has to deserialise anything beyond str/int/float/bool/None/dict/
    list. A future field holding a Pydantic model or a datetime would fail
    this test immediately, rather than surfacing three steps later as an
    obscure checkpoint error."""
    s = seed_state("T1", "raw")
    round_tripped = json.loads(json.dumps(s))
    assert round_tripped == s


# --------------------------------------------------------------------------
# MAX_REVISIONS
# --------------------------------------------------------------------------

def test_max_revisions_is_a_small_positive_bound():
    """Not a magic-number check - a guard against someone 'fixing' a stuck
    workflow by setting this to a huge number instead of finding the actual
    bug, which is exactly the failure mode Part B4 of the notebook warns
    about (a guard that doesn't guard anything)."""
    assert 1 <= MAX_REVISIONS <= 5