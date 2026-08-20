"""Step 4 demo: Extraction -> Playbook RAG, run against all 4 contracts.

Run:
    python3 scripts/run_playbook_rag.py

No langgraph dependency — playbook_rag_node() is driven here in a loop one
clause per call, exactly the way Step 7's supervisor will eventually drive
it.

Step 10 made both extraction_node and playbook_rag_node async (they call
the MCP contract-repository server instead of reading local disk). This
script predates that and was written to call them as plain sync functions
in a loop, so the imports below wrap each with asyncio.run() under the same
name — nothing else in this file needed to change.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.nodes.extraction import extraction_node as _extraction_node_async
from team.nodes.playbook_rag import playbook_rag_node as _playbook_rag_node_async
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

        print(f"\n=== {contract_path} ===")

        # Drive playbook_rag_node one clause per call until every clause
        # (classified or not) has a finding.
        for _ in range(len(state["clauses"])):
            update = playbook_rag_node(state)
            state["playbook_findings"] = (
                state["playbook_findings"] + update.get("playbook_findings", [])
            )
            state["log"] = state["log"] + update.get("log", [])

        for finding in state["playbook_findings"]:
            if finding["status"] == "skipped":
                print(f"  {finding['clause_id']:<5} [skipped - unclassified]")
                continue
            pos = finding["position"]
            passages = finding["retrieved_passages"]
            top = passages[0] if passages else None
            top_desc = f"{top['doc']} §{top['section_id']}" if top else "no match"
            print(
                f"  {finding['clause_id']:<5} {finding['clause_type']:<40} "
                f"-> playbook {pos['clause_number']:<4} | top match: {top_desc}"
            )

        outstanding = len(state["clauses"]) - len(state["playbook_findings"])
        assert outstanding == 0, f"{outstanding} clause(s) never got a finding"

    print("\nPASS - playbook_rag ran to completion against all 4 contracts, "
          "one clause at a time.")


if __name__ == "__main__":
    main()