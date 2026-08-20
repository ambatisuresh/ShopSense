"""Step 6 tests: team/nodes/legal_reviewer.py.

Run:
    pytest tests/test_team/test_legal_reviewer.py -v

Like test_redline_drafter.py, every test in this file forces the
deterministic fallback path via an autouse fixture — see that file's module
docstring for why (Step 5's test suite initially broke on a machine with
real litellm credentials configured, because the LLM path legitimately
produces different, better results than the no-LLM fallback). This node has
its own LLM path (_llm_review_redline); the same reasoning applies.

EXPECTED_REVIEW below is the exact, verified output of a real end-to-end run
(scripts/run_legal_reviewer.py) with the deterministic fallback forced.

Step 10 made extraction_node() and playbook_rag_node() async and MCP-backed.
Same hermetic treatment as test_redline_drafter.py: extraction_node here
pre-populates contract_text from local disk, and playbook_rag's positions
cache is primed from a local disk parse before any test runs.
"""
import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from team.nodes.extraction import extraction_node as _extraction_node_async  # noqa: E402
from team.nodes.legal_reviewer import (  # noqa: E402
    _ADEQUACY_THRESHOLD,
    _keyword_adequacy,
    _named_escalation_role,
    legal_reviewer_node,
)
from team.nodes.playbook_rag import playbook_rag_node as _playbook_rag_node_async  # noqa: E402
import team.nodes.playbook_rag as _playbook_rag_module  # noqa: E402
from team.nodes.redline_drafter import redline_drafter_node  # noqa: E402
from team.playbook_index import (  # noqa: E402
    PLAYBOOK_PATH,
    load_playbook_positions,
    parse_playbook_positions,
)
from team.scopes import AGENT_SCOPES  # noqa: E402
from team.state import MAX_REVISIONS, seed_state  # noqa: E402

_playbook_rag_module._POSITIONS_CACHE = parse_playbook_positions(
    PLAYBOOK_PATH.read_text(encoding="utf-8")
)


def extraction_node(state):
    state = dict(state)
    if not state.get("contract_text"):
        state["contract_text"] = (REPO_ROOT / state["contract_path"]).read_text(encoding="utf-8")
    return asyncio.run(_extraction_node_async(state))


def playbook_rag_node(state):
    return asyncio.run(_playbook_rag_node_async(state))


@pytest.fixture(autouse=True)
def _force_deterministic_review(monkeypatch):
    monkeypatch.setattr(
        "team.nodes.redline_drafter._llm_assess_compliance",
        lambda clause, position: None,
    )
    monkeypatch.setattr(
        "team.nodes.redline_drafter._llm_compose_redline",
        lambda clause, position, compliance, feedback=None: None,
    )
    monkeypatch.setattr(
        "team.nodes.legal_reviewer._llm_review_redline",
        lambda clause_id, entry, position: None,
    )


# {filename: {clause_id: verdict}} — only clauses that reached
# action == "redline_proposed" appear here (everything else was never
# routed to Legal Reviewer in this design).
EXPECTED_REVIEW = {
    "vendor_payments_processor_agreement.md": {
        "3.1": "escalated",   # named Operations Manager sign-off (playbook 1.1)
        "4.1": "approved",
        "5.1": "approved",
        "6.1": "approved",
    },
    "vendor_fulfillment_logistics_agreement.md": {
        "2.1": "approved", "3.1": "approved", "4.1": "approved",
        "5.1": "approved", "6.1": "approved",
    },
    "vendor_warranty_repair_partner_agreement.md": {
        "2.1": "approved", "2.2": "approved", "3.2": "approved", "5.1": "approved",
    },
    "vendor_returns_processing_agreement.md": {
        "2.1": "approved", "3.1": "approved",
    },
}


def _run_full_pipeline(filename: str) -> dict:
    state = dict(seed_state(f"data/contracts/{filename}"))
    state.update(extraction_node(state))
    for _ in range(len(state["clauses"])):
        update = playbook_rag_node(state)
        state["playbook_findings"] = state["playbook_findings"] + update.get("playbook_findings", [])
    for _ in range(len(state["clauses"])):
        update = redline_drafter_node(state)
        if "draft" in update:
            state["draft"] = update["draft"]
    redlined = [c for c, e in state["draft"].items() if e["action"] == "redline_proposed"]
    for _ in range(len(redlined)):
        update = legal_reviewer_node(state)
        if "legal_review" in update:
            state["legal_review"] = update["legal_review"]
    return state


# ---------------------------------------------------------------------------
# _named_escalation_role
# ---------------------------------------------------------------------------

def test_named_escalation_role_finds_operations_manager():
    positions = load_playbook_positions()
    parts = positions["refund_settlement_authority"]["parts"]
    assert _named_escalation_role("unacceptable", parts) == "Operations Manager"


def test_named_escalation_role_none_for_ordinary_unacceptable_tiers():
    positions = load_playbook_positions()
    parts = positions["limitation_of_liability"]["parts"]
    assert _named_escalation_role("unacceptable", parts) is None


def test_named_escalation_role_none_when_matched_tier_is_none():
    positions = load_playbook_positions()
    parts = positions["indemnification"]["parts"]
    assert _named_escalation_role(None, parts) is None


# ---------------------------------------------------------------------------
# _keyword_adequacy
# ---------------------------------------------------------------------------

def test_keyword_adequacy_approves_a_redline_that_echoes_the_target():
    parts = {"fallback": "settle refunds within three business days of instruction"}
    verdict = _keyword_adequacy(
        "Revise the clause to settle refunds within three business days of instruction.",
        parts,
    )
    assert verdict == "approved"


def test_keyword_adequacy_requests_changes_for_unrelated_text():
    parts = {"fallback": "settle refunds within three business days of instruction"}
    verdict = _keyword_adequacy("This clause discusses an entirely different topic.", parts)
    assert verdict == "changes_requested"


def test_keyword_adequacy_returns_none_with_no_target_tier():
    assert _keyword_adequacy("anything at all", {}) is None


def test_adequacy_threshold_is_a_fraction():
    assert 0 < _ADEQUACY_THRESHOLD < 1


# ---------------------------------------------------------------------------
# legal_reviewer_node — end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", list(EXPECTED_REVIEW))
def test_legal_reviewer_end_to_end_per_contract(filename):
    state = _run_full_pipeline(filename)
    expected = EXPECTED_REVIEW[filename]

    redlined = {c for c, e in state["draft"].items() if e["action"] == "redline_proposed"}
    assert set(state["legal_review"]) == redlined == set(expected)

    for clause_id, want_verdict in expected.items():
        assert state["legal_review"][clause_id]["verdict"] == want_verdict, (
            f"{filename} clause {clause_id}: expected {want_verdict!r}, "
            f"got {state['legal_review'][clause_id]['verdict']!r}"
        )


def test_named_escalation_takes_priority_regardless_of_redline_quality():
    """Clause 1.1's Unacceptable tier names Operations Manager sign-off —
    this must escalate even though the template composer's redline text
    trivially echoes the target position (which would otherwise approve)."""
    state = _run_full_pipeline("vendor_payments_processor_agreement.md")
    review = state["legal_review"]["3.1"]
    assert review["verdict"] == "escalated"
    assert review["method"] == "playbook_rule"
    assert "Operations Manager" in review["reason"]


# ---------------------------------------------------------------------------
# legal_reviewer_node — revision cap
# ---------------------------------------------------------------------------

def _state_with_one_redline(filename: str, clause_id: str) -> dict:
    """Run the pipeline through Redline Drafter only, for a clause with no
    named-escalation position, so the revision-cap path can be tested in
    isolation from the named-escalation path."""
    state = dict(seed_state(f"data/contracts/{filename}"))
    state.update(extraction_node(state))
    for _ in range(len(state["clauses"])):
        update = playbook_rag_node(state)
        state["playbook_findings"] = state["playbook_findings"] + update.get("playbook_findings", [])
    for _ in range(len(state["clauses"])):
        update = redline_drafter_node(state)
        if "draft" in update:
            state["draft"] = update["draft"]
    assert state["draft"][clause_id]["action"] == "redline_proposed"
    return state


def _review_until(state: dict, clause_id: str) -> dict:
    """legal_reviewer_node always picks the lowest outstanding clause_id
    first (same ordering as every other node in this project), so reaching
    a specific later clause means driving the loop past whatever comes
    before it — clause 3.1 (named escalation) always goes first in the
    payments contract."""
    for _ in range(len(state["draft"])):
        update = legal_reviewer_node(state)
        if "legal_review" not in update:
            break
        state["legal_review"] = {**state["legal_review"], **update["legal_review"]}
        if clause_id in state["legal_review"]:
            return state["legal_review"][clause_id]
    raise AssertionError(f"clause {clause_id} was never reviewed")


def test_revision_cap_forces_escalation():
    state = _state_with_one_redline("vendor_payments_processor_agreement.md", "4.1")
    state["revision_count"] = MAX_REVISIONS
    review = _review_until(state, "4.1")
    assert review["verdict"] == "escalated"
    assert review["method"] == "revision_cap"


def test_below_revision_cap_reviews_normally():
    state = _state_with_one_redline("vendor_payments_processor_agreement.md", "4.1")
    state["revision_count"] = MAX_REVISIONS - 1
    review = _review_until(state, "4.1")
    assert review["verdict"] != "escalated" or review["method"] != "revision_cap"


# ---------------------------------------------------------------------------
# legal_reviewer_node — shape/scope
# ---------------------------------------------------------------------------

def test_legal_reviewer_processes_exactly_one_entry_per_call():
    state = _state_with_one_redline("vendor_payments_processor_agreement.md", "4.1")
    update = legal_reviewer_node(state)
    assert len(update["legal_review"]) == 1


def test_legal_reviewer_no_ops_once_everything_is_done():
    state = _run_full_pipeline("vendor_payments_processor_agreement.md")
    update = legal_reviewer_node(state)
    assert "legal_review" not in update
    assert "nothing outstanding" in update["log"][0]


def test_legal_reviewer_output_stays_inside_its_write_scope():
    state = _state_with_one_redline("vendor_payments_processor_agreement.md", "4.1")
    update = legal_reviewer_node(state)
    allowed = AGENT_SCOPES["legal_reviewer"] | {"log"}
    assert set(update) <= allowed


def test_legal_reviewer_skips_clauses_that_were_never_redlined():
    """Only draft entries with action == 'redline_proposed' should ever
    reach Legal Reviewer — a 'no_action' clause was already cleared and
    doesn't need sign-off."""
    state = _state_with_one_redline("vendor_payments_processor_agreement.md", "4.1")
    no_action_ids = {c for c, e in state["draft"].items() if e["action"] == "no_action"}
    assert no_action_ids  # sanity: this contract does have no_action clauses

    for _ in range(len(state["draft"])):
        update = legal_reviewer_node(state)
        if "legal_review" not in update:
            break
        state["legal_review"] = {**state["legal_review"], **update["legal_review"]}

    assert no_action_ids.isdisjoint(state["legal_review"])