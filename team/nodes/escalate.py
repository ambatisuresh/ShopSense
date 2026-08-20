"""The terminal escalation node: marks the whole contract review as needing
a human, once the Supervisor has determined at least one clause's
`legal_review` verdict is "escalated" and nothing else is outstanding.

Deliberately the simplest node in the whole team — it doesn't decide
*whether* to escalate (that's the Supervisor's routing policy, reading
`legal_review`) or draft anything; it just stamps the terminal state. Kept
as its own node (rather than folding the status write into the Supervisor)
because `status` and `next_agent` are two different agents' write-scopes in
team/scopes.py, matching the Day3 Session 2 notebook's own pattern of a
distinct terminal node rather than a decision node quietly taking terminal
actions itself.
"""
from __future__ import annotations

from team.scopes import scoped


@scoped("escalate")
def escalate_node(state: dict) -> dict:
    escalated_clauses = sorted(
        (cid for cid, review in state["legal_review"].items() if review["verdict"] == "escalated"),
        key=lambda cid: tuple(int(p) for p in cid.split(".")),
    )
    return {
        "status": "escalated_to_human",
        "log": [
            f"escalate: contract review escalated to human — "
            f"{len(escalated_clauses)} clause(s) need manual sign-off: {', '.join(escalated_clauses)}"
        ],
    }