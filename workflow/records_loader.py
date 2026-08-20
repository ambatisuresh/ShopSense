"""
ShopSense M5 - Step 9: loading records.jsonl and assigning each record a
safe, unique graph identity.

Split out of scripts/run_workflow.py into its own PURE module - same
reason workflow/routing.py and the langgraph-independent half of
workflow/checkpointing.py were split out: `scripts/run_workflow.py` does
`from langgraph.types import Command` at module level, so nothing defined
in that file can be imported (or tested) without a real langgraph install.
`load_records()` / `unique_id()` have zero langgraph dependency and should
be testable regardless - see tests/test_workflow/test_records_loader.py.

WHY unique_id() EXISTS AT ALL: loading even the 30-record sample shipped
in data/records.jsonl surfaced a real data-quality issue in this project's
own source file - "record_id" is NOT globally unique. It repeats across
what look like separate paraphrase batches: SHOPSENSE-00020 through
SHOPSENSE-00038 each appear TWICE in data/records.jsonl, each time with
different raw_text/received_at - genuinely different ticket instances, not
a duplicate line to drop. Using record_id as-is for LangGraph's thread_id
would silently collide two different tickets' checkpointed state onto one
thread. unique_id() combines record_id with the record's 1-indexed line
number in the file, which is unique by construction regardless of what
record_id/ticket_id pattern the source data happens to use.
"""

import json


def load_records(path: str) -> "list[tuple[int, dict]]":
    """Returns a list of (line_no, record) pairs, 1-indexed by line.
    EVERY line is kept, including ones sharing a record_id with an earlier
    line - see the module docstring for why dropping "duplicates" here
    would silently discard real, distinct tickets."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            records.append((line_no, json.loads(line)))
    return records


def unique_id(record_id: str, line_no: int) -> str:
    """The actual LangGraph thread_id / TicketReviewState ticket_id used
    for this record - see the module docstring's explanation of why
    record_id alone is not safe to use directly."""
    return f"{record_id}-L{line_no:04d}"