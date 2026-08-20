"""
ShopSense M5 - Step 9: the Lab-B-style pass/fail checklist for a full
batch run (scripts/run_workflow.py) against real records.jsonl tickets.

Split into its own PURE module - like workflow/routing.py before it - so
these checks are testable with plain synthetic result dicts, no langgraph,
no real graph run needed. scripts/run_workflow.py is the only place that
actually builds a compiled graph and produces the `results` list these
functions consume.

Each check operates on a LIST of per-ticket result dicts, one per ticket
processed by run_workflow.py, shaped like:
    {
        "unique_id": str,              # see run_workflow.py's unique_id() - NOT records.jsonl's own record_id, see below
        "record_id": str,              # records.jsonl's own "record_id", for display only
        "ground_truth": dict,          # records.jsonl's own "ground_truth"
        "exception": Optional[str],    # str(exc) if graph.invoke() raised, else None
        "final_state": Optional[dict], # the final merged state dict, or None if it crashed
        "was_interrupted": bool,       # True if the ticket paused at human_approval at least once
    }

WHY "unique_id" and not just "record_id": run_workflow.py discovered, by
actually loading this project's own records.jsonl, that "record_id" is NOT
globally unique across the file - it repeats across what look like
separate paraphrase batches (confirmed directly: SHOPSENSE-00020 through
SHOPSENSE-00038 each appear twice in even the 30-record sample shipped in
data/records.jsonl). Using record_id as the LangGraph thread_id would
silently collide two different tickets' checkpointed state onto one
thread. run_workflow.py's unique_id() works around this; this module just
reports whatever record_id it's given, unmodified, for human-readable
output.

Two kinds of check, deliberately kept SEPARATE and never conflated:
    - HARD checks (check_*): structural invariants the GRAPH itself
      guarantees regardless of parser accuracy - "did every ticket reach a
      terminal state", "is final_result JSON-serializable" - these are
      PASS/FAIL, and a failure here means an actual bug in this project's
      code.
    - INFORMATIONAL summaries (summarize_*): e.g. how often this workflow's
      classification agreed with records.jsonl's own requires_human label.
      NOT pass/fail - the fixture parser (workflow/nodes/extract.py's
      fixture_parse_ticket) is a keyword stub standing in for M1's real
      LLM, and this workflow's non_standard triggers (refund cap,
      escalation-tone keywords - see compare_to_playbook.py) are a
      NARROWER, policy-compliance-focused concept than records.jsonl's
      requires_human label, which appears to encode a broader
      support-routing judgment covering many intents (wrong_item,
      missing_item, general complaints) this workflow was never designed
      to catch on keyword grounds alone. Treating a disagreement here as a
      FAILURE would conflate "the graph is broken" with "the fixture
      parser/this milestone's narrower scope diverges from a differently-
      scoped ground truth label" - two different things. The second is
      EXPECTED, not a bug.
"""

import json
from typing import Optional


def check_no_exceptions(results: list) -> dict:
    failures = [r["unique_id"] for r in results if r.get("exception")]
    return {
        "name": "no ticket raised an exception",
        "passed": not failures,
        "detail": "all clear" if not failures else f"{len(failures)} ticket(s) raised: {failures}",
    }


def check_all_reached_a_terminal_state(results: list) -> dict:
    """Every ticket that didn't crash must end EITHER finalized (auto-
    approve, or approved-then-finalized) or, for a rejected ticket,
    "rejected" - a real, intentional stop. Anything else (stuck mid-graph,
    an unrecognized status) is a bug, not a data-quality issue."""
    ok_statuses = {"finalized", "rejected"}
    bad = []
    for r in results:
        if r.get("exception"):
            continue
        status = (r.get("final_state") or {}).get("status")
        if status not in ok_statuses:
            bad.append((r["unique_id"], status))
    return {
        "name": "every non-crashed ticket reached a terminal state (finalized/rejected)",
        "passed": not bad,
        "detail": "all clear" if not bad else f"{len(bad)} ticket(s) stuck: {bad}",
    }


def check_audit_log_completeness(results: list) -> dict:
    """Every non-crashed ticket's audit_log must show extract,
    compare_to_playbook, and draft_redline ran - they run on EVERY path,
    unconditionally. A missing entry means a node silently failed to log,
    not a parser-accuracy issue."""
    required_always = ("extract", "compare_to_playbook", "draft_redline")
    bad = []
    for r in results:
        if r.get("exception"):
            continue
        prefixes = {entry.split(":")[0] for entry in (r.get("final_state") or {}).get("audit_log", [])}
        missing = [p for p in required_always if p not in prefixes]
        if missing:
            bad.append((r["unique_id"], missing))
    return {
        "name": "audit_log has entries from every always-run node",
        "passed": not bad,
        "detail": "all clear" if not bad else f"{len(bad)} ticket(s) missing entries: {bad}",
    }


def check_final_result_is_json_serializable(results: list) -> dict:
    bad = []
    for r in results:
        final_state = r.get("final_state") or {}
        if final_state.get("status") != "finalized":
            continue
        try:
            json.dumps(final_state.get("final_result"))
        except TypeError as e:
            bad.append((r["unique_id"], str(e)))
    return {
        "name": "final_result is JSON-serializable for every finalized ticket",
        "passed": not bad,
        "detail": "all clear" if not bad else f"{len(bad)} ticket(s): {bad}",
    }


def check_commit_matches_refund_need(results: list) -> dict:
    """finalize's own invariant (unit-tested in isolation in
    test_finalize.py), re-checked here against REAL graph output on messy
    data: `committed` must be True exactly when the ticket both needed a
    refund AND had a resolvable order_id - the two conditions
    build_finalize_node's `_needs_commit` + its fail-closed no-order_id
    branch actually implement (see workflow/nodes/finalize.py)."""
    bad = []
    for r in results:
        final_state = r.get("final_state") or {}
        if final_state.get("status") != "finalized":
            continue
        parsed = final_state.get("parsed_ticket") or {}
        needs_refund = bool(parsed) and (
            parsed.get("issue_type") == "REFUND" or parsed.get("claimed_refund_amount") is not None
        )
        has_order_id = bool(final_state.get("order_id"))
        expected_committed = needs_refund and has_order_id
        committed = (final_state.get("final_result") or {}).get("committed")
        if committed is not expected_committed:
            bad.append((
                r["unique_id"],
                f"needs_refund={needs_refund} has_order_id={has_order_id} committed={committed}",
            ))
    return {
        "name": "committed flag matches whether the ticket needed (and could reach) a refund commit",
        "passed": not bad,
        "detail": "all clear" if not bad else f"{len(bad)} ticket(s): {bad}",
    }


def run_all_checks(results: list) -> list:
    return [
        check_no_exceptions(results),
        check_all_reached_a_terminal_state(results),
        check_audit_log_completeness(results),
        check_final_result_is_json_serializable(results),
        check_commit_matches_refund_need(results),
    ]


def summarize_requires_human_agreement(results: list) -> dict:
    """INFORMATIONAL ONLY - see module docstring for why a low agreement
    rate is EXPECTED, not a bug. Compares records.jsonl's
    ground_truth.requires_human to whether this ticket's final
    classification was non_standard."""
    comparable = [r for r in results if not r.get("exception") and r.get("final_state")]
    agree = 0
    for r in comparable:
        expected_human = bool((r.get("ground_truth") or {}).get("requires_human"))
        ours_non_standard = r["final_state"].get("classification") == "non_standard"
        if expected_human == ours_non_standard:
            agree += 1
    total = len(comparable)
    return {
        "total_compared": total,
        "agree": agree,
        "agreement_rate": (agree / total) if total else None,
    }