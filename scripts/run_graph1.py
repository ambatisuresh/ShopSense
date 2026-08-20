"""Step 8/10 demo: the real compiled langgraph team, run against all 4
contracts end to end — this replaces scripts/run_supervisor.py's manual
while-loop with await team.ainvoke(seed, {"recursion_limit": ...}).

Run:
    python3 scripts/run_graph.py

Needs langgraph AND mcp/fastmcp installed (see requirements.txt) — Step 10
wired extraction_node and playbook_rag_node to the MCP contract-repository
server (mcp_server/contract_server.py gets spawned as a subprocess per call
they make), so this script now needs BOTH dependencies, not just langgraph.

This is `await team.ainvoke(...)` now, not `team.invoke(...)`, because two
of the six nodes the graph runs (extraction, playbook_rag) are `async def`
as of Step 10 — same mechanical note the Day3 Session 2 notebook makes
about its own MCP-backed researcher swap: "invoke the graph with await
team.ainvoke(...) instead of team.invoke(...)". langgraph runs the sync
nodes (supervisor, redline_drafter, legal_reviewer, escalate) exactly as
before; nothing about THEM changed.

recursion_limit note: the Day3 Session 2 notebook flags that a star
topology costs 2 supersteps per unit of work (specialist -> supervisor is
2 hops), and needed recursion_limit=50 against langgraph's default of 25
for its own 5-specialist team. Step 7's manual-loop demo saw up to ~28
supervisor calls for a single ShopSense contract (5 clauses x roughly
extraction + playbook_rag + redline_drafter + legal_reviewer, plus any
revision-loop redraws) which alone would need a limit north of 56; 200 is
used here to match scripts/run_supervisor.py's MAX_ITERATIONS with room
to spare, including for contracts that hit the revision loop.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.graph import build_team  # noqa: E402
from team.state import seed_state  # noqa: E402

CONTRACTS = [
    "data/contracts/vendor_payments_processor_agreement.md",
    "data/contracts/vendor_fulfillment_logistics_agreement.md",
    "data/contracts/vendor_warranty_repair_partner_agreement.md",
    "data/contracts/vendor_returns_processing_agreement.md",
]

RECURSION_LIMIT = 200


async def main() -> None:
    team = build_team()

    for contract_path in CONTRACTS:
        seed = seed_state(contract_path)
        state = await team.ainvoke(seed, {"recursion_limit": RECURSION_LIMIT})

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

    print("\nPASS - the compiled langgraph team (MCP-backed extraction + playbook_rag) "
          "routed all 4 contracts to completion.")


if __name__ == "__main__":
    asyncio.run(main())