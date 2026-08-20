"""The Supervisor: reads the whole team's shared state and decides which
agent runs next. This is the piece that turns four independent per-clause
nodes (Extraction, Playbook RAG, Redline Drafter, Legal Reviewer) into an
actual review pipeline with a bounded revision loop — same role as the
Day3 Session 2 notebook's supervisor_node().

The routing policy (`decide_next_agent`) is plain deterministic Python, not
an LLM call — this is a control-flow decision with a small number of
correct answers derivable directly from state, exactly the kind of thing
that should NOT be delegated to a model (an LLM router here would add
latency and a new failure mode for zero benefit: there's nothing genuinely
ambiguous to interpret). Contrast with the *content* decisions inside
Extraction/Redline Drafter/Legal Reviewer, where an LLM call is tried first
because real judgment is involved.

`next_agent` is the supervisor's sole write-scope (team/scopes.py). It never
touches `draft`, `legal_review`, or `revision_count` itself — it only reads
them to decide where to route.
"""
from __future__ import annotations

from team.scopes import scoped


def decide_next_agent(state: dict) -> str:
    """Pure routing policy — no side effects, easy to unit test directly
    against hand-built state dicts without running the whole pipeline.

    Returns one of: "extraction", "playbook_rag", "redline_drafter",
    "legal_reviewer", "escalate", "done".
    """
    if not state["clauses"]:
        return "extraction"

    if len(state["playbook_findings"]) < len(state["clauses"]):
        return "playbook_rag"

    if len(state["draft"]) < len(state["clauses"]):
        return "redline_drafter"

    redlined = {cid for cid, entry in state["draft"].items() if entry["action"] == "redline_proposed"}
    pending_review = redlined - set(state["legal_review"])
    if pending_review:
        return "legal_reviewer"

    needs_revision = [
        cid for cid, review in state["legal_review"].items()
        if review["verdict"] == "changes_requested"
        and state["draft"][cid].get("drafted_at_revision", 0) == review["revision_count_at_review"]
    ]
    if needs_revision:
        return "redline_drafter"

    if any(review["verdict"] == "escalated" for review in state["legal_review"].values()):
        return "escalate"

    return "done"


@scoped("supervisor")
def supervisor_node(state: dict) -> dict:
    next_agent = decide_next_agent(state)
    return {
        "next_agent": next_agent,
        "log": [f"supervisor: routing to '{next_agent}'"],
    }