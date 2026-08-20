"""
ShopSense M5 - Step 8 demo, PART 2 of 2: reattach in a BRAND NEW PROCESS and
resume.

Run scripts/run_checkpointing_pause.py FIRST, as a separate `python3`
invocation, then run this one. This script imports nothing from that one
and shares no Python state with it whatsoever - the only thing connecting
them is `shopsense_checkpoints.sqlite` sitting on disk. That is deliberate:
it is the whole point of Part B7 / Step 8. "A checkpointer persists your
STATE, never your CODE" - this script rebuilds the CODE (imports the node
functions, recompiles the graph) and reattaches to the STATE (the sqlite
file), same as a real redeployed service would on startup.

Needs the real `langgraph` + `langgraph-checkpoint-sqlite` packages. Not
executable in the sandbox this was built in - written directly against the
confirmed API the notebook itself uses (Part B7's own "run this AFTER
restarting the kernel" cell).
"""

from langgraph.types import Command

from workflow.checkpointing import checkpoint_file_size, thread_config
from workflow.graph import build_graph

DB_PATH = "shopsense_checkpoints.sqlite"
TICKET_ID = "SHOPSENSE-DURABLE-1"

if __name__ == "__main__":
    # checkpoint_db_path with fresh=False (build_graph's own default
    # behaviour) - REATTACH, do not wipe. This is the one call site in this
    # whole demo that must never pass fresh=True.
    graph = build_graph(checkpoint_db_path=DB_PATH)
    cfg = thread_config(TICKET_ID)

    snap = graph.get_state(cfg)
    print("recovered from disk :", bool(snap.values))
    print("parked at           :", snap.next)
    print("ticket_id           :", snap.values.get("ticket_id"))
    print("classification      :", snap.values.get("classification"))
    print("checkpoint file     :", DB_PATH, f"({checkpoint_file_size(DB_PATH)} bytes on disk)")

    if not snap.values:
        raise SystemExit(
            "No state recovered - did you run scripts/run_checkpointing_pause.py "
            "first, from the same working directory?"
        )

    # The human finally answers - in a brand-new process, exactly per the
    # notebook's own script for this cell.
    final = graph.invoke(
        Command(resume={"action": "approved", "note": "Approved after restart.", "approver_id": "reviewer-42"}),
        cfg,
    )

    print()
    print("status              :", final["status"])
    print("approver_id         :", final["approver_id"])
    print("final_result        :", final["final_result"])
    print("audit trail         :", final["audit_log"])
    print()
    print("MILESTONE: paused in one process, resumed in another, via a real")
    print("filesystem checkpoint - not a Python variable, not a mock.")

"""
EXPECTED OUTPUT
---------------
recovered from disk : True
parked at           : ('human_approval',)
ticket_id           : SHOPSENSE-DURABLE-1
classification      : non_standard
checkpoint file     : shopsense_checkpoints.sqlite (NNNNN bytes on disk)

status              : finalized
approver_id         : reviewer-42
final_result        : {...committed=True...}
audit trail         : [...]

MILESTONE: paused in one process, resumed in another, via a real
filesystem checkpoint - not a Python variable, not a mock.
"""