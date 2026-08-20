"""M6 team state: ContractReviewState.

Same foundational split M5's workflow/state.py and the Day3 Session 2
notebook's TeamState both use: CONTROL fields are plain overwrites (exactly
one node owns each decision, and a control field must be resettable to its
"unset" value or a loop can never exit); AUDIT fields accumulate via
Annotated[list, add] and are never routed on.
"""
# NOTE: deliberately NOT using `from __future__ import annotations` here.
# That flag turns every annotation into a string (PEP 563), which would hide
# the Annotated[list, add] reducers from runtime introspection — and Step 2's
# whole test suite for the control-vs-audit split depends on inspecting the
# real Annotated objects in ContractReviewState.__annotations__.
from operator import add
from typing import Annotated, TypedDict

# The loop budget for Legal Reviewer -> Redline Drafter revisions. Matches the
# Day3 Session 2 notebook's Lab A (MAX_REVISIONS = 2) rather than M5's
# reserved-but-unused 3, since M6's graph actually has a revision loop and
# this milestone is explicitly built on that notebook's pattern.
MAX_REVISIONS = 2


class ContractReviewState(TypedDict):
    # --- CONTROL fields: plain overwrite, exactly one node writes each ---
    contract_path: str          # which contract file is under review (seed input)
    contract_text: str          # raw contract text; Extraction sets this once
    clauses: list[dict]         # Extraction's output: [{clause_id, clause_type, text}, ...]
    draft: dict                 # Redline Drafter's current draft; {} = not yet drafted
    legal_review: dict          # Legal Reviewer's verdict; {} = not yet reviewed
    revision_count: int         # the loop guard
    next_agent: str             # the supervisor's routing decision
    status: str                 # "" while in progress; "escalated_to_human" on budget exhaustion

    # --- AUDIT fields: accumulate, never routed on ---
    playbook_findings: Annotated[list[dict], add]  # one entry per clause looked up
    log: Annotated[list[str], add]                 # the team's full trajectory


def seed_state(contract_path: str, contract_text: str = "") -> ContractReviewState:
    """Build a fresh ContractReviewState for one contract review run.

    contract_text may be pre-loaded (e.g. by a script reading the file
    directly) or left empty for the Extraction node to fill in itself (e.g.
    once Extraction reads it via the MCP contract-repository tool in Step 10).
    """
    return ContractReviewState(
        contract_path=contract_path,
        contract_text=contract_text,
        clauses=[],
        draft={},
        legal_review={},
        revision_count=0,
        next_agent="",
        status="",
        playbook_findings=[],
        log=[],
    )