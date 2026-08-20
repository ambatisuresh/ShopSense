"""Step 3 demo: the Extraction agent, run against all 4 sample contracts.

Run:
    python3 scripts/run_extraction.py

No langgraph dependency — extraction_node() is called directly here exactly
like the notebook's own per-step demo cells call planner_node()/
researcher_node() before any graph exists.

Step 10 made extraction_node async (it calls the MCP contract-repository
server instead of reading local disk). This script predates that and was
written to call it as a plain sync function in a loop, so the import below
wraps it with asyncio.run() under the same name — nothing else in this file
needed to change.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.nodes.extraction import extraction_node as _extraction_node_async
from team.state import seed_state


def extraction_node(state):
    return asyncio.run(_extraction_node_async(state))

CONTRACTS = [
    "data/contracts/vendor_payments_processor_agreement.md",
    "data/contracts/vendor_fulfillment_logistics_agreement.md",
    "data/contracts/vendor_warranty_repair_partner_agreement.md",
    "data/contracts/vendor_returns_processing_agreement.md",
]


def main() -> None:
    grand_total = 0
    grand_classified = 0

    for contract_path in CONTRACTS:
        state = seed_state(contract_path)
        result = extraction_node(state)

        classified = [c for c in result["clauses"] if c["clause_type"] != "unclassified"]
        unclassified = [c for c in result["clauses"] if c["clause_type"] == "unclassified"]
        grand_total += len(result["clauses"])
        grand_classified += len(classified)

        print(f"\n=== {contract_path} ===")
        print(f"  {len(result['clauses'])} clause(s) total: "
              f"{len(classified)} classified, {len(unclassified)} unclassified")
        for c in result["clauses"]:
            print(f"    {c['clause_id']:<5} {c['title']:<45} -> {c['clause_type']}")

    print(f"\nTOTAL: {grand_classified}/{grand_total} clauses classified against "
          f"the 23-entry playbook taxonomy.")
    print("PASS - extraction ran against all 4 contracts with zero exceptions.")


if __name__ == "__main__":
    main()