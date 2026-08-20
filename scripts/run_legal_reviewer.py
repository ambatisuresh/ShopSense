"""Step 6 demo: Extraction -> Playbook RAG -> Redline Drafter -> Legal
Reviewer, run against all 4 contracts.

Run:
    python3 scripts/run_legal_reviewer.py

Still no langgraph dependency — legal_reviewer_node() is a plain function,
driven here one draft entry per call.

Step 10 made extraction_node and playbook_rag_node async (MCP contract
server calls). This script predates that; the imports below wrap each with
asyncio.run() under the same name so nothing else here needed to change.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.nodes.extraction import extraction_node as _extraction_node_async
from team.nodes.legal_reviewer import legal_reviewer_node
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

        for _ in range(len(state["clauses"])):
            update = redline_drafter_node(state)
            if "draft" in update:
                state["draft"] = update["draft"]

        redlined = [c for c, e in state["draft"].items() if e["action"] == "redline_proposed"]
        for _ in range(len(redlined)):
            update = legal_reviewer_node(state)
            if "legal_review" in update:
                state["legal_review"] = update["legal_review"]

        print(f"\n=== {contract_path} ===")
        for clause_id in sorted(redlined, key=lambda c: tuple(int(p) for p in c.split("."))):
            review = state["legal_review"][clause_id]
            print(f"  {clause_id:<5} {review['verdict']:<18} ({review['method']:<13}) {review['reason']}")

        outstanding = len(redlined) - len(state["legal_review"])
        assert outstanding == 0, f"{outstanding} redlined clause(s) never got a review"

    print("\nPASS - legal_reviewer ran to completion against all 4 contracts, "
          "one draft entry at a time.")


if __name__ == "__main__":
    main()