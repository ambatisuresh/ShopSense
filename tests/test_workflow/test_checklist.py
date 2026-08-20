"""
ShopSense M5 - Step 9 tests: workflow/checklist.py

Fully executed here, NO langgraph needed - every check operates on plain
synthetic result dicts shaped like what scripts/run_workflow.py produces,
not on a real graph run. Same posture as test_routing.py: this is testing
the CHECKING logic itself, independent of whether a real graph is
available to produce results to check.
"""

from workflow.checklist import (
    check_all_reached_a_terminal_state,
    check_audit_log_completeness,
    check_commit_matches_refund_need,
    check_final_result_is_json_serializable,
    check_no_exceptions,
    run_all_checks,
    summarize_requires_human_agreement,
)


def _ok_result(unique_id="T1-L0001", record_id="SHOPSENSE-1", committed=True, needs_refund=True, has_order_id=True, classification="standard", requires_human=False):
    parsed = {"issue_type": "REFUND"} if needs_refund else {"issue_type": "DELIVERY"}
    return {
        "unique_id": unique_id,
        "record_id": record_id,
        "ground_truth": {"requires_human": requires_human},
        "exception": None,
        "was_interrupted": False,
        "final_state": {
            "status": "finalized",
            "audit_log": ["extract: ok", "compare_to_playbook: ok", "draft_redline: ok", "finalize: ok"],
            "parsed_ticket": parsed,
            "order_id": "ORD-1" if has_order_id else None,
            "classification": classification,
            "final_result": {"committed": committed, "resolution": "ok"},
        },
    }


def _crashed_result(unique_id="T-CRASH"):
    return {
        "unique_id": unique_id,
        "record_id": "SHOPSENSE-CRASH",
        "ground_truth": {},
        "exception": "ValueError: boom",
        "was_interrupted": False,
        "final_state": None,
    }


# --------------------------------------------------------------------------
# check_no_exceptions
# --------------------------------------------------------------------------

def test_no_exceptions_passes_when_nothing_crashed():
    result = check_no_exceptions([_ok_result(), _ok_result(unique_id="T2")])
    assert result["passed"] is True


def test_no_exceptions_fails_and_names_the_crashed_ticket():
    result = check_no_exceptions([_ok_result(), _crashed_result()])
    assert result["passed"] is False
    assert "T-CRASH" in result["detail"]


# --------------------------------------------------------------------------
# check_all_reached_a_terminal_state
# --------------------------------------------------------------------------

def test_terminal_state_passes_for_finalized():
    result = check_all_reached_a_terminal_state([_ok_result()])
    assert result["passed"] is True


def test_terminal_state_passes_for_rejected():
    r = _ok_result()
    r["final_state"]["status"] = "rejected"
    result = check_all_reached_a_terminal_state([r])
    assert result["passed"] is True


def test_terminal_state_fails_for_stuck_ticket():
    r = _ok_result()
    r["final_state"]["status"] = "pending_human"
    result = check_all_reached_a_terminal_state([r])
    assert result["passed"] is False


def test_terminal_state_ignores_crashed_tickets():
    """A crashed ticket has no final_state to judge - check_no_exceptions
    is what flags it, not this check."""
    result = check_all_reached_a_terminal_state([_crashed_result()])
    assert result["passed"] is True


# --------------------------------------------------------------------------
# check_audit_log_completeness
# --------------------------------------------------------------------------

def test_audit_log_completeness_passes_when_all_prefixes_present():
    result = check_audit_log_completeness([_ok_result()])
    assert result["passed"] is True


def test_audit_log_completeness_fails_when_a_node_never_logged():
    r = _ok_result()
    r["final_state"]["audit_log"] = ["extract: ok", "draft_redline: ok"]  # missing compare_to_playbook
    result = check_audit_log_completeness([r])
    assert result["passed"] is False
    assert "compare_to_playbook" in result["detail"]


# --------------------------------------------------------------------------
# check_final_result_is_json_serializable
# --------------------------------------------------------------------------

def test_final_result_json_serializable_passes_for_plain_dict():
    result = check_final_result_is_json_serializable([_ok_result()])
    assert result["passed"] is True


def test_final_result_json_serializable_fails_for_a_non_serializable_value():
    r = _ok_result()
    r["final_state"]["final_result"] = {"committed": True, "bad": object()}
    result = check_final_result_is_json_serializable([r])
    assert result["passed"] is False


def test_final_result_json_serializable_ignores_non_finalized_tickets():
    r = _ok_result()
    r["final_state"]["status"] = "rejected"
    r["final_state"]["final_result"] = {"bad": object()}
    result = check_final_result_is_json_serializable([r])
    assert result["passed"] is True


# --------------------------------------------------------------------------
# check_commit_matches_refund_need
# --------------------------------------------------------------------------

def test_commit_matches_refund_need_passes_for_refund_with_order_id_committed():
    r = _ok_result(needs_refund=True, has_order_id=True, committed=True)
    result = check_commit_matches_refund_need([r])
    assert result["passed"] is True


def test_commit_matches_refund_need_passes_for_non_refund_not_committed():
    r = _ok_result(needs_refund=False, has_order_id=True, committed=False)
    result = check_commit_matches_refund_need([r])
    assert result["passed"] is True


def test_commit_matches_refund_need_passes_for_refund_with_no_order_id_not_committed():
    r = _ok_result(needs_refund=True, has_order_id=False, committed=False)
    result = check_commit_matches_refund_need([r])
    assert result["passed"] is True


def test_commit_matches_refund_need_fails_when_committed_but_should_not_have():
    r = _ok_result(needs_refund=False, has_order_id=True, committed=True)
    result = check_commit_matches_refund_need([r])
    assert result["passed"] is False


def test_commit_matches_refund_need_fails_when_not_committed_but_should_have():
    r = _ok_result(needs_refund=True, has_order_id=True, committed=False)
    result = check_commit_matches_refund_need([r])
    assert result["passed"] is False


# --------------------------------------------------------------------------
# run_all_checks
# --------------------------------------------------------------------------

def test_run_all_checks_returns_five_checks():
    results = run_all_checks([_ok_result()])
    assert len(results) == 5
    assert all(c["passed"] for c in results)


# --------------------------------------------------------------------------
# summarize_requires_human_agreement (informational, not pass/fail)
# --------------------------------------------------------------------------

def test_agreement_summary_full_agreement():
    results = [
        _ok_result(unique_id="T1", classification="non_standard", requires_human=True),
        _ok_result(unique_id="T2", classification="standard", requires_human=False),
    ]
    summary = summarize_requires_human_agreement(results)
    assert summary["total_compared"] == 2
    assert summary["agree"] == 2
    assert summary["agreement_rate"] == 1.0


def test_agreement_summary_partial_agreement():
    results = [
        _ok_result(unique_id="T1", classification="standard", requires_human=True),  # disagree
        _ok_result(unique_id="T2", classification="standard", requires_human=False),  # agree
    ]
    summary = summarize_requires_human_agreement(results)
    assert summary["agree"] == 1
    assert summary["agreement_rate"] == 0.5


def test_agreement_summary_excludes_crashed_tickets():
    results = [_ok_result(), _crashed_result()]
    summary = summarize_requires_human_agreement(results)
    assert summary["total_compared"] == 1


def test_agreement_summary_handles_empty_results():
    summary = summarize_requires_human_agreement([])
    assert summary["total_compared"] == 0
    assert summary["agreement_rate"] is None