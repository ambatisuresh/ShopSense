"""Agent write-scopes for the M6 team, and the decorator that enforces them.

Direct port of the Day3 Session 2 notebook's A1 pattern (scoped() +
AGENT_SCOPES) onto the four M6 specialists. An agent boundary is a
PERMISSION, enforced by this decorator raising PermissionError, not a
sentence in a prompt.
"""
from __future__ import annotations

import inspect

# Which state keys each node is allowed to return an update for. Every role
# may also always write "log" (the shared audit trail) — added below.
AGENT_SCOPES = {
    "extraction":      {"contract_text", "clauses"},
    "playbook_rag":    {"playbook_findings"},
    # Redline Drafter may reset legal_review when it redrafts — a rewrite
    # VOIDS the prior verdict, same rule M5's draft_redline/writer nodes use.
    "redline_drafter": {"draft", "revision_count", "legal_review"},
    "legal_reviewer":  {"legal_review"},
    "supervisor":      {"next_agent"},
    "escalate":        {"status"},
}


def scoped(role: str):
    """Enforce write permissions for `role`. A violation raises rather than
    silently corrupting state.

    Handles both sync and async node functions — Extraction and Playbook RAG
    become async in Step 10 once they call out to the MCP contract-repository
    tool, and a decorator that only understood sync functions would wrap the
    coroutine object itself and fail somewhere unrecognizable, not at the
    call site where the bug actually is.
    """
    if role not in AGENT_SCOPES:
        raise KeyError(f"unknown role {role!r}; add it to AGENT_SCOPES first")
    allowed = AGENT_SCOPES[role] | {"log"}

    def enforce(update: dict) -> dict:
        illegal = set(update) - allowed
        if illegal:
            raise PermissionError(
                f"agent {role!r} wrote outside its scope: {sorted(illegal)}; "
                f"allowed={sorted(allowed)}"
            )
        return update

    def decorate(fn):
        if inspect.iscoroutinefunction(fn):
            async def awrapper(state):
                return enforce(await fn(state))
            awrapper.__name__ = fn.__name__
            return awrapper

        def wrapper(state):
            return enforce(fn(state))
        wrapper.__name__ = fn.__name__
        return wrapper

    return decorate