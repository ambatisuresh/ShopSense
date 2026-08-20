"""Step 5 demo: Extraction -> Playbook RAG -> Redline Drafter, run against
all 4 contracts.

Run:
    python3 scripts/run_redline_drafter.py

Still no langgraph dependency — redline_drafter_node() is a plain function,
driven here one finding per call, exactly as Step 7's supervisor will
eventually drive it.

Step 10 made extraction_node and playbook_rag_node async (MCP contract
server calls). This script predates that; the imports below wrap each with
asyncio.run() under the same name so nothing else here needed to change.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.nodes.extraction import extraction_node as _extraction_node_async
from team.nodes.playbook_rag import playbook_rag_node as _playbook_rag_node_async
from team.nodes.redline_drafter import redline_drafter_node
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


def main() -> None:
    for contract_path in CONTRACTS:
        state = dict(seed_state(contract_path))
        state.update(extraction_node(state))

        for _ in range(len(state["clauses"])):
            update = playbook_rag_node(state)
            state["playbook_findings"] = (
                state["playbook_findings"] + update.get("playbook_findings", [])
            )
            state["log"] = state["log"] + update.get("log", [])

        for _ in range(len(state["clauses"])):
            update = redline_drafter_node(state)
            if "draft" in update:
                state["draft"] = update["draft"]
            state["log"] = state["log"] + update.get("log", [])

        print(f"\n=== {contract_path} ===")
        for clause_id in sorted(state["draft"], key=lambda c: tuple(int(p) for p in c.split("."))):
            entry = state["draft"][clause_id]
            if entry["action"] == "no_action":
                tag = entry.get("compliance") or entry.get("reason", "")
                print(f"  {clause_id:<5} [no action]      {tag}")
            else:
                print(
                    f"  {clause_id:<5} [REDLINE]         "
                    f"{entry['compliance']:<14} (matched: {entry['matched_tier']}, "
                    f"via {entry['assessment_method']}) -> playbook {entry['playbook_clause']}"
                )

        outstanding = len(state["clauses"]) - len(state["draft"])
        assert outstanding == 0, f"{outstanding} clause(s) never got a draft entry"

    print("\nPASS - redline_drafter ran to completion against all 4 contracts, "
          "one clause at a time.")


if __name__ == "__main__":
    main()