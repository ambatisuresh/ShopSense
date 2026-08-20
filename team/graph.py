"""Step 8: the real langgraph wiring, replacing scripts/run_supervisor.py's
manual while-loop with an actual compiled StateGraph.

Directly mirrors the Day3 Session 2 notebook's Lab A build_team() pattern:
a star topology with the Supervisor at the hub, add_conditional_edges()
routing on state["next_agent"] (the exact same field decide_next_agent()
already computes in team/nodes/supervisor.py), and every specialist edging
straight back to the Supervisor so it re-evaluates after every single node
call.

Nothing about this file's OWN wiring changed for Step 10 — build_team()
wires together the same six node callables it always did. What changed is
two of those callables: as of Step 10, extraction_node and playbook_rag_node
are `async def` (they call the MCP contract-repository server instead of
reading local disk — see team/mcp_client.py). langgraph runs sync and async
nodes in the same StateGraph without any special wiring on this end; the
one thing that changes for CALLERS is `team.invoke(...)` -> `await
team.ainvoke(...)`, exactly the notebook's own note on its MCP-backed
researcher swap ("The node becomes async... invoke the graph with await
team.ainvoke(...) instead of team.invoke(...)"). See scripts/run_graph.py.

IMPORTANT — read before running: langgraph could not be installed in the
sandbox this file was written in (pip / uv pip / uv tool all refused with
403/not-found errors for this package), so this module has NOT been run or
verified the way every prior step's code was. It was written by careful,
direct translation of the notebook's own working build_team() pattern, but
you are the first one to actually execute it — see the Step 8 delivery
message for what to check and report back.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from team.nodes.escalate import escalate_node
from team.nodes.extraction import extraction_node
from team.nodes.legal_reviewer import legal_reviewer_node
from team.nodes.playbook_rag import playbook_rag_node
from team.nodes.redline_drafter import redline_drafter_node
from team.nodes.supervisor import supervisor_node
from team.state import ContractReviewState

# The four specialists the Supervisor can route to. "escalate" and "done"
# are handled separately below (escalate is a real node with its own edge
# to END; "done" maps directly to END, no node needed).
SPECIALISTS = ["extraction", "playbook_rag", "redline_drafter", "legal_reviewer"]

NODE_FN = {
    "extraction": extraction_node,
    "playbook_rag": playbook_rag_node,
    "redline_drafter": redline_drafter_node,
    "legal_reviewer": legal_reviewer_node,
}


def build_team():
    """Compile the M6 contract-review team into a runnable langgraph app.

    Star topology: START -> supervisor -> {one specialist | escalate | END},
    every specialist -> supervisor, escalate -> END. Same shape as the
    Day3 Session 2 notebook's build_team(), with ShopSense's 4 specialists
    standing in for the notebook's planner/researcher/writer/fact_checker.
    """
    tb = StateGraph(ContractReviewState)

    tb.add_node("supervisor", supervisor_node)
    for name, fn in NODE_FN.items():
        tb.add_node(name, fn)
    tb.add_node("escalate", escalate_node)

    tb.add_edge(START, "supervisor")

    tb.add_conditional_edges(
        "supervisor",
        lambda s: s["next_agent"],
        {**{name: name for name in SPECIALISTS}, "escalate": "escalate", "done": END},
    )

    for name in SPECIALISTS:
        tb.add_edge(name, "supervisor")

    tb.add_edge("escalate", END)

    return tb.compile()