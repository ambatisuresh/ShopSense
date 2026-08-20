"""
ShopSense M5 - Step 8: durable checkpointing (SqliteSaver).

Everything through Step 7 used an `InMemorySaver` by default - fine for a
single process, but the notebook's own Part B7 is explicit about what that
means: "InMemorySaver stores checkpoints in the process's heap. A restart
is a wipe." A human reviewer may approve a ticket three days later, from a
web form, after the service has been redeployed twice. `InMemorySaver`
cannot survive that under any graph design; only a durable checkpointer can.

`SqliteSaver` writes each checkpoint to a file instead of the heap, so a
paused run outlives the process. This module wraps the exact construction
sequence Part B7 uses (`sqlite3.connect(..., check_same_thread=False)` ->
`SqliteSaver(conn)` -> `saver.setup()`), lazily imported for the same reason
every other langgraph-touching factory in this project lazily imports it:
`workflow.checkpointing` must remain importable in an environment with no
langgraph installed, so tests that don't need it (`thread_config`'s tests
below) still run.

Requires the SEPARATE `langgraph-checkpoint-sqlite` package - the notebook's
own setup cell calls this out explicitly, it does not ship inside the base
`langgraph` install:
    pip install "langgraph-checkpoint-sqlite>=3.0,<4.0"

WHY `build_graph()`'s bare default stays `InMemorySaver`, not `SqliteSaver`
(a deliberate refinement of what Step 6/7's docstrings said was coming):
a bare-default `SqliteSaver` would mean every test and every quick demo
that calls `build_graph()` with no arguments silently starts writing (and
reusing, across runs) a checkpoint file on disk - the "clean slate" a
test needs is exactly what a durable store must NOT give you by default.
Durability has to be something a caller opts into with an explicit path,
same as every other production dependency in this project (make_m1_parser_
adapter, make_m2_refund_adapter, etc. are all opt-in, never silently
defaulted). `build_graph(checkpoint_db_path=...)` is that opt-in - see
graph.py.
"""

import os
from typing import Optional

# thread_config() is a plain dict-builder - zero langgraph dependency,
# usable (and tested) in every environment, same posture as workflow/routing.py.
def thread_config(ticket_id: str) -> dict:
    """The single source of truth for how a ticket_id becomes a thread_id.

    Every call site in this project (scripts/run_graph.py,
    scripts/run_checkpointing_pause.py/_resume.py, and any future caller)
    should build its config through this function rather than re-typing
    `{"configurable": {"thread_id": ...}}` - one typo'd key name in one
    call site is exactly how a resume silently starts a fresh thread
    instead (the notebook's own pitfall table: "Resume starts a fresh run -
    Different or missing thread_id").

    ShopSense's mapping is the simplest one that works: one thread per
    ticket, thread_id == ticket_id. A ticket is never re-opened under a
    different id once created, so there is no need for anything fancier
    (e.g. a suffix or a UUID) here.
    """
    if not ticket_id:
        raise ValueError("thread_config() requires a non-empty ticket_id")
    return {"configurable": {"thread_id": ticket_id}}


def build_sqlite_checkpointer(db_path: str, *, fresh: bool = False):
    """Build a real, disk-backed `SqliteSaver`, wired exactly as Part B7's
    milestone cell does it.

    `db_path`: file path for the sqlite database. Relative paths are
    resolved against the current working directory, same as `sqlite3.
    connect()` itself - callers running via `python3 -m scripts.run_X` from
    the repo root will get a file at the repo root unless given an absolute
    path.

    `fresh`: if True, deletes any existing file at `db_path` first - "clean
    slate so the lab is repeatable", the notebook's own phrase for this
    exact line. Use `fresh=True` only when STARTING a new durable run (see
    scripts/run_checkpointing_pause.py). Use `fresh=False` (the default) to
    REATTACH to a run already on disk - the whole point of Step 8 is that
    reattaching must be possible from a brand-new process; wiping the file
    on every call would make that impossible to demonstrate.

    `check_same_thread=False`: the notebook's own comment - "the runtime
    may touch the connection from a worker thread." Not optional for
    langgraph's internal use of this connection.

    `saver.setup()`: creates the checkpoint tables if they don't exist yet.
    Idempotent - safe to call every time, including on reattach.
    """
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    if fresh and os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def checkpoint_file_size(db_path: str) -> Optional[int]:
    """Convenience for demo/print output only (e.g. "checkpoint file: X
    bytes on disk", mirroring the notebook's own EXPECTED OUTPUT). Returns
    None if the file doesn't exist yet rather than raising - a graph that
    hasn't run a single superstep yet may not have created the file."""
    return os.path.getsize(db_path) if os.path.exists(db_path) else None