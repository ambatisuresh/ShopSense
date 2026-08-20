"""Step 2 demo: ContractReviewState, AGENT_SCOPES, and the scoped() guard.

Run:
    python3 scripts/run_state_demo.py

Zero external dependencies (no langgraph needed yet) — this only exercises
team/state.py and team/scopes.py, both pure Python.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.scopes import AGENT_SCOPES, scoped
from team.state import MAX_REVISIONS, seed_state


def main() -> None:
    print("AGENT SCOPES")
    print("------------")
    for role, keys in AGENT_SCOPES.items():
        print(f"  {role:<16} may write: {sorted(keys)}")
    print(f"\nMAX_REVISIONS = {MAX_REVISIONS}")

    print("\nSEED STATE for vendor_payments_processor_agreement.md")
    print("-------------------------------------------------------")
    state = seed_state("data/contracts/vendor_payments_processor_agreement.md")
    for key, value in state.items():
        print(f"  {key:<16} = {value!r}")

    # --- Self-check 1: a legitimate write passes straight through. -----------
    @scoped("legal_reviewer")
    def honest_legal_reviewer(state):
        return {"legal_review": {"approved": False, "notes": ["clause 3.1 exceeds cap"]},
                "log": ["legal_reviewer: rejected — see notes"]}

    result = honest_legal_reviewer(state)
    assert result["legal_review"]["approved"] is False
    print("\nPASS - a legitimate legal_reviewer write passed through unchanged.")

    # --- Self-check 2: a rogue reviewer that also edits the draft is caught. -
    @scoped("legal_reviewer")
    def rogue_legal_reviewer(state):
        # A reviewer that "helpfully" fixes the draft itself instead of just
        # judging it. This is exactly the failure mode A1 warns about: a
        # critic that can edit the artefact is no longer a critic.
        return {"legal_review": {"approved": True}, "draft": {"redlines": []}}

    try:
        rogue_legal_reviewer(state)
        raise AssertionError("scoped() did not fire — this must not happen")
    except PermissionError as e:
        print("PASS - rogue legal_reviewer write caught:")
        print(f"  {e}")

    # --- Self-check 3: scoped() also enforces async nodes correctly. ---------
    # Playbook RAG becomes async in Step 10 once it calls the MCP contract
    # tool; this proves the decorator handles that transparently, before any
    # of the real async plumbing exists.
    @scoped("playbook_rag")
    async def rogue_async_playbook_rag(state):
        await asyncio.sleep(0)  # stand-in for a real awaited call later
        return {"playbook_findings": [{"clause_id": "3.1"}], "next_agent": "writer"}

    try:
        asyncio.run(rogue_async_playbook_rag(state))
        raise AssertionError("scoped() did not fire on the async node")
    except PermissionError as e:
        print("PASS - rogue ASYNC playbook_rag write caught:")
        print(f"  {e}")

    print("\nAll self-checks passed. State + scope enforcement are ready for Step 3.")


if __name__ == "__main__":
    main()