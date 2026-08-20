"""
ShopSense M5 - Step 9: the full-batch demo + Lab-B-style pass/fail
checklist. Runs EVERY ticket in records.jsonl through the compiled graph
end to end (extract -> compare_to_playbook -> draft_redline -> route ->
auto_approve/human_approval -> finalize), then runs workflow/checklist.py's
structural checks against the whole batch.

DATA SOURCE: data/records.jsonl in this repo is a SAMPLE (30 records, ~1.5
of what look like paraphrase batches) of the records.jsonl project doc -
included so this script is runnable standalone, NOT the full file. If your
repo already has the full records.jsonl somewhere from M1-M4 (very likely -
M1's parser and M4's eval set were both built against it), point this
script at that file instead:
    python3 -m scripts.run_workflow /path/to/your/records.jsonl
Optionally cap how many records to process (useful for a quick smoke run):
    python3 -m scripts.run_workflow /path/to/your/records.jsonl 20

IDENTITY NOTE: records.jsonl's own "record_id" field is NOT globally
unique (confirmed directly - see workflow/records_loader.py's docstring
for the full explanation and the exact duplicate ids found). This script
uses workflow.records_loader.unique_id() as the real LangGraph thread_id /
TicketReviewState ticket_id; record_id/ticket_id from the source stay
visible in the printed report for cross-referencing, but never double as
the thread identity.

DEPENDENCIES: every node defaults to its offline fixture (fixture_parse_
ticket, fixture_evaluate_refund, fixture_retrieve_citations) - same as
every other demo in this project. This means the classification results
below reflect THIS WORKFLOW'S OWN routing logic (refund cap, escalation-
tone keywords) exercised against real messy free text, NOT M1's real LLM
parser's accuracy. See workflow/checklist.py's module docstring for why
the informational agreement-rate summary at the end is expected to be well
under 100% and why that is not a bug.

BATCH-DEMO REVIEWER POLICY: every ticket that pauses at human_approval is
auto-approved by a fake reviewer here. This is a SMOKE TEST of the full
pause/resume/finalize path across real data, not a claim that every ticket
SHOULD be approved - a real deployment puts an actual human at that
decision (see workflow/nodes/human_approval.py).

Needs the real `langgraph` package - not executable in the sandbox this
was built in, so this hasn't been run by me. Written directly against the
confirmed API used throughout scripts/run_graph.py (already exercised via
its own EXPECTED OUTPUT contract) - run it in your real shopsensevenv and
report back what you see.
"""

import sys

from langgraph.types import Command

from workflow.checklist import run_all_checks, summarize_requires_human_agreement
from workflow.checkpointing import thread_config
from workflow.graph import build_graph
from workflow.records_loader import load_records, unique_id
from workflow.state import seed_state

RECORDS_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/intake/records.jsonl"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None


def run_one_ticket(graph, line_no, record):
    record_id = record["record_id"]
    uid = unique_id(record_id, line_no)
    seed = seed_state(
        uid, record["raw_text"],
        customer_ref=record.get("customer_ref"), order_id=record.get("order_ref"),
    )
    cfg = thread_config(uid)
    was_interrupted = False
    try:
        result = graph.invoke(seed, cfg)
        if "__interrupt__" in result:
            was_interrupted = True
            result = graph.invoke(
                Command(resume={
                    "action": "approved", "note": "batch demo auto-approval",
                    "approver_id": "batch-demo-reviewer",
                }),
                cfg,
            )
        return {
            "unique_id": uid, "record_id": record_id,
            "ground_truth": record.get("ground_truth", {}),
            "exception": None, "final_state": result, "was_interrupted": was_interrupted,
        }
    except Exception as e:
        return {
            "unique_id": uid, "record_id": record_id,
            "ground_truth": record.get("ground_truth", {}),
            "exception": f"{type(e).__name__}: {e}", "final_state": None, "was_interrupted": was_interrupted,
        }


if __name__ == "__main__":
    records = load_records(RECORDS_PATH)
    if LIMIT:
        records = records[:LIMIT]
    print(f"Loaded {len(records)} ticket(s) from {RECORDS_PATH}")
    print()

    graph = build_graph()  # every dependency defaults to an offline fixture; InMemorySaver checkpointer
    results = []

    for line_no, record in records:
        r = run_one_ticket(graph, line_no, record)
        results.append(r)
        if r["exception"]:
            line = f"CRASHED: {r['exception']}"
        else:
            fs = r["final_state"] or {}
            paused = "yes" if r["was_interrupted"] else "no"
            line = (
                f"classification={fs.get('classification', ''):<12} "
                f"status={fs.get('status', ''):<10} "
                f"paused={paused:<3} "
                f"ground_truth.requires_human={r['ground_truth'].get('requires_human')}"
            )
        print(f"{r['unique_id']:<24} (record_id={r['record_id']:<16}) {line}")

    print()
    print("=" * 70)
    print("HARD CHECKS (structural invariants - a failure here is a bug)")
    print("=" * 70)
    checks = run_all_checks(results)
    all_passed = True
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        all_passed = all_passed and check["passed"]
        print(f"[{status}] {check['name']}")
        if not check["passed"]:
            print(f"       {check['detail']}")

    print()
    print("=" * 70)
    print("INFORMATIONAL (not pass/fail - see workflow/checklist.py docstring)")
    print("=" * 70)
    agreement = summarize_requires_human_agreement(results)
    if agreement["total_compared"]:
        pct = agreement["agreement_rate"] * 100
        print(
            f"classification-vs-requires_human agreement: {agreement['agree']}/{agreement['total_compared']} "
            f"({pct:.0f}%) - expected to be well under 100%; this workflow's non_standard "
            f"triggers are a NARROWER concept than records.jsonl's requires_human label, and "
            f"fixture_parse_ticket is a keyword stub, not M1's real LLM."
        )

    print()
    print("OVERALL:", "ALL HARD CHECKS PASSED" if all_passed else "SOME HARD CHECKS FAILED - see above")

"""
EXPECTED OUTPUT
---------------
Loaded 30 ticket(s) from data/records.jsonl

SHOPSENSE-00000-L0001   (record_id=SHOPSENSE-00000) classification=standard     status=finalized  paused=no  ground_truth.requires_human=False
... (one line per ticket) ...

======================================================================
HARD CHECKS (structural invariants - a failure here is a bug)
======================================================================
[PASS] no ticket raised an exception
[PASS] every non-crashed ticket reached a terminal state (finalized/rejected)
[PASS] audit_log has entries from every always-run node
[PASS] final_result is JSON-serializable for every finalized ticket
[PASS] committed flag matches whether the ticket needed (and could reach) a refund commit

======================================================================
INFORMATIONAL (not pass/fail - see workflow/checklist.py docstring)
======================================================================
classification-vs-requires_human agreement: NN/30 (XX%) - expected to be well under 100%; ...

OVERALL: ALL HARD CHECKS PASSED
"""