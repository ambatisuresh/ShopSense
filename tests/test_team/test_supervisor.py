"""Step 7 tests: team/nodes/supervisor.py, team/nodes/escalate.py, and the
full team wired together end to end.

Run:
    pytest tests/test_team/test_supervisor.py -v

Same hermetic-testing lesson as Steps 5 and 6: an autouse fixture forces
both redline_drafter's and legal_reviewer's LLM paths off, so results don't
depend on whatever litellm credentials happen to be configured wherever
these tests run.

Step 10 made extraction_node() and playbook_rag_node() async and MCP-backed.
Same hermetic treatment as test_redline_drafter.py/test_legal_reviewer.py:
extraction_node here pre-populates contract_text from local disk, and
playbook_rag's positions cache is primed from a local disk parse before any
test runs — this file's manual supervisor loop (_run_pipeline) then calls
both under the same names as before, needing zero changes to the loop
itself.
"""
import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from team.nodes.escalate import escalate_node  # noqa: E402
from team.nodes.extraction import extraction_node as _extraction_node_async  # noqa: E402
from team.nodes.legal_reviewer import legal_reviewer_node  # noqa: E402
from team.nodes.playbook_rag import playbook_rag_node as _playbook_rag_node_async  # noqa: E402
import team.nodes.playbook_rag as _playbook_rag_module  # noqa: E402
from team.nodes.redline_drafter import redline_drafter_node  # noqa: E402
from team.nodes.supervisor import decide_next_agent, supervisor_node  # noqa: E402
from team.playbook_index import PLAYBOOK_PATH, parse_playbook_positions  # noqa: E402
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
def _force_deterministic_pipeline(monkeypatch):
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


NODE_MAP = {
    "extraction": extraction_node,
    "playbook_rag": playbook_rag_node,
    "redline_drafter": redline_drafter_node,
    "legal_reviewer": legal_reviewer_node,
    "escalate": escalate_node,
}
AUDIT_FIELDS = {"playbook_findings", "log"}
MAX_ITERATIONS = 300


def _merge(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key in AUDIT_FIELDS:
            state[key] = state[key] + value
        else:
            state[key] = value


def _run_pipeline(filename: str) -> dict:
    state = dict(seed_state(f"data/contracts/{filename}"))
    for _ in range(MAX_ITERATIONS):
        _merge(state, supervisor_node(state))
        if state["next_agent"] == "done":
            break
        _merge(state, NODE_MAP[state["next_agent"]](state))
        if state.get("status") == "escalated_to_human":
            break
    else:
        raise AssertionError(f"{filename}: exceeded {MAX_ITERATIONS} supervisor iterations")
    return state


# ---------------------------------------------------------------------------
# decide_next_agent — unit tests against hand-built state
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    state = {
        "clauses": [], "playbook_findings": [], "draft": {}, "legal_review": {},
        "revision_count": 0,
    }
    state.update(overrides)
    return state


def test_routes_to_extraction_when_no_clauses():
    assert decide_next_agent(_base_state()) == "extraction"


def test_routes_to_playbook_rag_when_findings_incomplete():
    state = _base_state(clauses=[{"clause_id": "1.1"}, {"clause_id": "2.1"}], playbook_findings=[{"clause_id": "1.1"}])
    assert decide_next_agent(state) == "playbook_rag"


def test_routes_to_redline_drafter_when_draft_incomplete():
    state = _base_state(
        clauses=[{"clause_id": "1.1"}, {"clause_id": "2.1"}],
        playbook_findings=[{"clause_id": "1.1"}, {"clause_id": "2.1"}],
        draft={"1.1": {"action": "no_action"}},
    )
    assert decide_next_agent(state) == "redline_drafter"


def test_routes_to_legal_reviewer_when_redlines_pending_review():
    state = _base_state(
        clauses=[{"clause_id": "1.1"}],
        playbook_findings=[{"clause_id": "1.1"}],
        draft={"1.1": {"action": "redline_proposed"}},
    )
    assert decide_next_agent(state) == "legal_reviewer"


def test_routes_to_redline_drafter_for_a_fresh_changes_requested_verdict():
    state = _base_state(
        clauses=[{"clause_id": "1.1"}],
        playbook_findings=[{"clause_id": "1.1"}],
        draft={"1.1": {"action": "redline_proposed", "drafted_at_revision": 0}},
        legal_review={"1.1": {"verdict": "changes_requested", "revision_count_at_review": 0}},
    )
    assert decide_next_agent(state) == "redline_drafter"


def test_does_not_reroute_a_changes_requested_verdict_already_redrafted():
    """drafted_at_revision has moved past revision_count_at_review — this
    clause was already redrafted since that particular rejection and is
    correctly waiting on Legal Reviewer's NEXT review, not another redraft."""
    state = _base_state(
        clauses=[{"clause_id": "1.1"}],
        playbook_findings=[{"clause_id": "1.1"}],
        draft={"1.1": {"action": "redline_proposed", "drafted_at_revision": 1}},
        legal_review={"1.1": {"verdict": "changes_requested", "revision_count_at_review": 0}},
    )
    assert decide_next_agent(state) == "done"


def test_routes_to_escalate_when_any_verdict_is_escalated():
    state = _base_state(
        clauses=[{"clause_id": "1.1"}],
        playbook_findings=[{"clause_id": "1.1"}],
        draft={"1.1": {"action": "redline_proposed", "drafted_at_revision": 0}},
        legal_review={"1.1": {"verdict": "escalated", "revision_count_at_review": 0}},
    )
    assert decide_next_agent(state) == "escalate"


def test_routes_to_done_when_everything_is_clean():
    state = _base_state(
        clauses=[{"clause_id": "1.1"}],
        playbook_findings=[{"clause_id": "1.1"}],
        draft={"1.1": {"action": "no_action"}},
    )
    assert decide_next_agent(state) == "done"


def test_supervisor_node_output_stays_inside_its_write_scope():
    update = supervisor_node(_base_state())
    allowed = AGENT_SCOPES["supervisor"] | {"log"}
    assert set(update) <= allowed


# ---------------------------------------------------------------------------
# escalate_node
# ---------------------------------------------------------------------------

def test_escalate_node_sets_status():
    state = _base_state(legal_review={"3.1": {"verdict": "escalated"}, "4.1": {"verdict": "approved"}})
    update = escalate_node(state)
    assert update["status"] == "escalated_to_human"
    assert "3.1" in update["log"][0]
    assert "4.1" not in update["log"][0]


def test_escalate_node_output_stays_inside_its_write_scope():
    update = escalate_node(_base_state(legal_review={"3.1": {"verdict": "escalated"}}))
    allowed = AGENT_SCOPES["escalate"] | {"log"}
    assert set(update) <= allowed


# ---------------------------------------------------------------------------
# Full pipeline, supervisor-driven
# ---------------------------------------------------------------------------

CONTRACTS = [
    "vendor_payments_processor_agreement.md",
    "vendor_fulfillment_logistics_agreement.md",
    "vendor_warranty_repair_partner_agreement.md",
    "vendor_returns_processing_agreement.md",
]


@pytest.mark.parametrize("filename", CONTRACTS)
def test_full_pipeline_terminates_and_covers_every_clause(filename):
    state = _run_pipeline(filename)
    assert len(state["playbook_findings"]) == len(state["clauses"])
    assert len(state["draft"]) == len(state["clauses"])
    redlined = {c for c, e in state["draft"].items() if e["action"] == "redline_proposed"}
    assert set(state["legal_review"]) == redlined


def test_payments_contract_escalates_on_the_named_sign_off_clause():
    """End-to-end confirmation that the whole wired team reaches the same
    outcome Step 6 verified in isolation: clause 3.1's Operations Manager
    sign-off requirement propagates all the way to status."""
    state = _run_pipeline("vendor_payments_processor_agreement.md")
    assert state["status"] == "escalated_to_human"
    assert state["legal_review"]["3.1"]["verdict"] == "escalated"


def test_contracts_with_no_escalation_complete_cleanly():
    for filename in ("vendor_fulfillment_logistics_agreement.md",
                      "vendor_warranty_repair_partner_agreement.md",
                      "vendor_returns_processing_agreement.md"):
        state = _run_pipeline(filename)
        assert state["status"] == ""  # never set -> "completed" in the demo's display, "" in state


# ---------------------------------------------------------------------------
# The revision loop, forced — the real corpus never naturally produces a
# "changes_requested" verdict (every template-composed redline trivially
# passes the keyword-adequacy check — see Step 6's build note), so exercising
# the redraft loop and the MAX_REVISIONS cap requires forcing rejections.
# ---------------------------------------------------------------------------

def test_revision_loop_redrafts_then_respects_the_cap(monkeypatch):
    """Force every adequacy check to request changes and drive the full
    pipeline. Regression test for a real bug this step caught: an earlier
    version of redline_drafter kept redrafting every stale rejection
    regardless of the cap, letting revision_count overshoot MAX_REVISIONS
    (5 rejected clauses drove it to 6 against a cap of 2) instead of
    stopping once the shared budget was spent."""
    monkeypatch.setattr("team.nodes.legal_reviewer._keyword_adequacy", lambda *a, **k: "changes_requested")

    state = _run_pipeline("vendor_fulfillment_logistics_agreement.md")

    assert state["status"] == "escalated_to_human"
    assert state["revision_count"] == MAX_REVISIONS
    # Every redlined clause ends up escalated -- some via an actual redraft
    # attempt, others (once the budget was spent) sent straight to Legal
    # Reviewer for a cap-based verdict without ever being redrafted.
    for review in state["legal_review"].values():
        assert review["verdict"] == "escalated"
        assert review["method"] == "revision_cap"
    redrafted = [cid for cid, e in state["draft"].items()
                 if e["action"] == "redline_proposed" and e.get("drafted_at_revision", 0) > 0]
    assert redrafted  # at least one clause actually consumed a redraft attempt
    never_redrafted = [cid for cid, e in state["draft"].items()
                        if e["action"] == "redline_proposed" and e.get("drafted_at_revision", 0) == 0]
    assert never_redrafted  # and at least one had its rejection cleared without ever being redrafted


def test_revision_loop_actually_changes_the_proposed_text(monkeypatch):
    """The clause that actually consumes a redraft attempt (2.1 — the lowest
    clause_id, always selected first by _select_next_revision) should carry
    the template composer's revision marker and Legal Reviewer's feedback,
    not just be relabeled with the same rejected text."""
    monkeypatch.setattr("team.nodes.legal_reviewer._keyword_adequacy", lambda *a, **k: "changes_requested")

    state = _run_pipeline("vendor_fulfillment_logistics_agreement.md")

    entry = state["draft"]["2.1"]
    assert entry["drafted_at_revision"] > 0
    assert "(revised" in entry["proposed_text"]
    assert "Legal Reviewer feedback" in entry["proposed_text"]