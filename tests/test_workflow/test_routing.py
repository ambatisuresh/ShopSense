"""
ShopSense M5 - Part 5 tests: workflow/routing.py (route_after_review).

Split into its OWN file, separate from test_graph.py, for a concrete
reason confirmed while building this step: `pytest.importorskip("langgraph")`
placed inside a module skips that ENTIRE module on failure - Python cannot
partially import a module, so even test functions defined BEFORE the
importorskip call never get collected once it raises. Verified directly:
an earlier single-file version reported "collected 0 items / 1 skipped"
for ALL of test_graph.py's tests, including these four, which have nothing
to do with langgraph.

These four have ZERO dependency on langgraph - route_after_review is a
plain `dict -> str` function - and must run in every environment
regardless of whether langgraph is installed. Same point Lab B makes about
route_after_checks: "no graph, no model, no tokens" needed to test a router.
"""

from workflow.routing import route_after_human, route_after_review
from workflow.state import seed_state


def test_standard_classification_routes_to_auto_approve():
    state = seed_state("T1", "raw")
    state["classification"] = "standard"
    assert route_after_review(state) == "auto_approve"


def test_non_standard_classification_routes_to_human_approval():
    state = seed_state("T1", "raw")
    state["classification"] = "non_standard"
    assert route_after_review(state) == "human_approval"


def test_unset_classification_fails_closed_to_human_approval():
    """seed_state()'s default classification is "" - a graph that somehow
    reached routing without compare_to_playbook/draft_redline running must
    NOT auto-approve. Fail closed, not fail open."""
    state = seed_state("T1", "raw")
    assert state["classification"] == ""
    assert route_after_review(state) == "human_approval"


def test_unrecognized_classification_value_fails_closed():
    """A future new classification value nobody's wired a branch for yet
    must also fail closed, not silently auto-approve."""
    state = seed_state("T1", "raw")
    state["classification"] = "some_new_value_nobody_handled"
    assert route_after_review(state) == "human_approval"


# --------------------------------------------------------------------------
# route_after_human() - Step 6
# --------------------------------------------------------------------------

def test_approved_with_approver_id_routes_to_finalize():
    state = seed_state("T1", "raw")
    state["status"] = "approved"
    state["approver_id"] = "rev-1"
    assert route_after_human(state) == "finalize"


def test_approved_without_approver_id_fails_closed_to_end():
    """Enforces the notebook's authz warning as executable policy: a
    workflow must not record 'approved' with no record of who approved
    it. Even a status of 'approved' does not reach finalize without one."""
    state = seed_state("T1", "raw")
    state["status"] = "approved"
    state["approver_id"] = None
    assert route_after_human(state) == "end"


def test_rejected_routes_to_end():
    state = seed_state("T1", "raw")
    state["status"] = "rejected"
    state["approver_id"] = "rev-1"
    assert route_after_human(state) == "end"


def test_unrecognized_status_fails_closed_to_end():
    """An unhandled outcome (e.g. a future 'changes_requested' this
    milestone doesn't wire a branch for) must never accidentally finalize -
    it just doesn't proceed, which is the safe default."""
    state = seed_state("T1", "raw")
    state["status"] = "changes_requested"
    state["approver_id"] = "rev-1"
    assert route_after_human(state) == "end"