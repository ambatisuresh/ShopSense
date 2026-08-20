"""Step 7 demo: the full team, wired by the Supervisor's routing policy,
run against all 4 contracts end to end — no manual per-node loops left in
this script at all, unlike Steps 4-6's demos.

Run:
    python3 scripts/run_supervisor.py

Still no langgraph dependency — this is a plain while-loop calling
supervisor_node() to decide what's next and dispatching to the matching
node function. Step 8 replaced this loop with a real langgraph StateGraph
for the actual deliverable (scripts/run_graph.py) — this script is kept
running anyway, deliberately, because it's a second, langgraph-independent
way to verify Step 10's MCP wiring end to end (useful since langgraph
itself couldn't be verified in the build sandbox at all).

Step 10 made extraction_node and playbook_rag_node async (MCP contract
server calls). This script predates that; the imports below wrap each with
asyncio.run() under the same name so the rest of the loop needed zero
changes.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.nodes.escalate import escalate_node
from team.nodes.extraction import extraction_node as _extraction_node_async
from team.nodes.legal_reviewer import legal_reviewer_node
from team.nodes.playbook_rag import playbook_rag_node as _playbook_rag_node_async
from team.nodes.redline_drafter import redline_drafter_node
from team.nodes.supervisor import supervisor_node
from team.state import seed_state


def extraction_node(state):
    return asyncio.run(_extraction_node_async(state))


def playbook_rag_node(state):
    return asyncio.run(_playbook_rag_node_async(state))

CONTRACTS = [
    "data/contracts/vendor_payments_processor_agreement.md",
    "data/contracts/vendor_fulfillment_logistics_agreement.md",
    "data/contracts/vendor_warranty_repair_partner_agreement.md",
    "data/contracts/vendor_returns_processing_agreement.md",
]

NODE_MAP = {
    "extraction": extraction_node,
    "playbook_rag": playbook_rag_node,
    "redline_drafter": redline_drafter_node,
    "legal_reviewer": legal_reviewer_node,
    "escalate": escalate_node,
}

# Audit fields accumulate (mirrors the Annotated[list, add] reducer these
# state keys have in team/state.py); everything else is a plain overwrite.
# This manual merge is exactly what a real langgraph StateGraph does for
# us automatically once Step 8 wires one up.
AUDIT_FIELDS = {"playbook_findings", "log"}

MAX_ITERATIONS = 200


def merge_update(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key in AUDIT_FIELDS:
            state[key] = state[key] + value
        else:
            state[key] = value


def run_one_contract(contract_path: str) -> dict:
    state = dict(seed_state(contract_path))
    for _ in range(MAX_ITERATIONS):
        merge_update(state, supervisor_node(state))
        if state["next_agent"] == "done":
            break
        merge_update(state, NODE_MAP[state["next_agent"]](state))
        if state.get("status") == "escalated_to_human":
            break
    else:
        raise AssertionError(f"{contract_path}: exceeded {MAX_ITERATIONS} supervisor iterations")
    return state


def main() -> None:
    for contract_path in CONTRACTS:
        state = run_one_contract(contract_path)

        routing_trace = [
            line.split("routing to ")[1] for line in state["log"] if line.startswith("supervisor:")
        ]
        final_status = state["status"] or "completed"
        redlined = [cid for cid, e in state["draft"].items() if e["action"] == "redline_proposed"]
        verdicts = {cid: r["verdict"] for cid, r in state["legal_review"].items()}

        print(f"\n=== {contract_path} ===")
        print(f"  status: {final_status}   revision_count: {state['revision_count']}   "
              f"supervisor calls: {len(routing_trace)}")
        print(f"  routing trace: {' -> '.join(routing_trace)}")
        for cid in sorted(redlined, key=lambda c: tuple(int(p) for p in c.split("."))):
            print(f"    {cid:<5} {verdicts.get(cid, 'PENDING'):<18} {state['draft'][cid]['playbook_clause']}")

        assert len(state["playbook_findings"]) == len(state["clauses"])
        assert len(state["draft"]) == len(state["clauses"])
        assert set(state["legal_review"]) == set(redlined)

    print("\nPASS - supervisor routed all 4 contracts to completion.")


if __name__ == "__main__":
    main()