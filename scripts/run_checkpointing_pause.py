"""
ShopSense M5 - Step 8 demo, PART 1 of 2: start a durable run and pause it.

Mirrors the notebook's Part B7 milestone cell exactly: build a real,
disk-backed SqliteSaver (fresh=True - clean slate, this script is meant to
be re-run from scratch), compile the graph against it, invoke a
non_standard ticket so it parks at human_approval, and print proof that the
checkpoint is now sitting on disk rather than in this process's heap.

Needs the real `langgraph` package PLUS the separate
`langgraph-checkpoint-sqlite` package:
    pip install "langgraph-checkpoint-sqlite>=3.0,<4.0"
Not executable in the sandbox this was built in (no langgraph at all there)
- written directly against the confirmed API the notebook itself uses.

HOW TO RUN THIS DEMO (the whole point is proving a REAL process boundary,
not just a kernel-restart-in-one-notebook-cell):

    python3 -m scripts.run_checkpointing_pause      # this script - PAUSES
    python3 -m scripts.run_checkpointing_resume      # separate process - RESUMES

Run them as two separate commands, not import calls from the same script -
each one is a genuinely distinct OS process, which is a strictly harder bar
to clear than the notebook's own "restart the kernel" instruction.
"""

from workflow.checkpointing import checkpoint_file_size, thread_config
from workflow.graph import build_graph
from workflow.state import seed_state

DB_PATH = "shopsense_checkpoints.sqlite"
TICKET = dict(
    ticket_id="SHOPSENSE-DURABLE-1",
    raw_text="This laptop bag is defective, refund me Rs. 3500.",
    customer_ref="CUST-durable", order_id="ORD-durable",
)

if __name__ == "__main__":
    # fresh=True happens INSIDE build_graph via checkpoint_db_path only on
    # first use of that path in this run - but build_graph() itself always
    # reattaches (fresh=False). For this demo's "clean slate" requirement
    # we delete the file ourselves first, same as the notebook's own
    # `if os.path.exists(DB_PATH): os.remove(DB_PATH)` line.
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    graph = build_graph(checkpoint_db_path=DB_PATH)
    cfg = thread_config(TICKET["ticket_id"])

    result = graph.invoke(
        seed_state(TICKET["ticket_id"], TICKET["raw_text"],
                   customer_ref=TICKET["customer_ref"], order_id=TICKET["order_id"]),
        cfg,
    )

    snap = graph.get_state(cfg)
    print("paused at        :", snap.next)
    print("checkpoint file  :", DB_PATH, f"({checkpoint_file_size(DB_PATH)} bytes on disk)")
    print("thread id        :", TICKET["ticket_id"])
    print()
    print("This run now exists ON DISK. Run scripts/run_checkpointing_resume.py")
    print("as a SEPARATE process (a new `python3` invocation, not a function call")
    print("from here) to prove it survives - this process's variables, node")
    print("functions, and compiled graph are about to be gone the moment this")
    print("script exits, exactly as if the service had been redeployed.")

"""
EXPECTED OUTPUT
---------------
paused at        : ('human_approval',)
checkpoint file  : shopsense_checkpoints.sqlite (NNNNN bytes on disk)
thread id        : SHOPSENSE-DURABLE-1

This run now exists ON DISK. Run scripts/run_checkpointing_resume.py ...
"""