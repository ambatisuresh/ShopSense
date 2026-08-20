"""Step 8 tests: team/graph.py's compiled langgraph team.

Run:
    pytest tests/test_team/test_graph.py -v

This whole file is skipped with pytest.importorskip if langgraph isn't
installed wherever it runs — that's expected in the sandbox this was
written in (langgraph could not be installed there at all), and this file
will simply report as skipped rather than erroring. It should genuinely
run on your machine, where langgraph is presumably already installed as
part of the M6 milestone's requirements. Please run it for real and report
back the output.

Same hermetic-testing lesson as Steps 5-7: an autouse fixture forces every
LLM path off so results don't depend on whatever litellm credentials
happen to be configured in the environment these tests run in.

Step 10 note: _run() below awaits team.ainvoke(), not team.invoke() — a
real bug caught on a real machine after this file could finally actually
run (it couldn't in the build sandbox, no langgraph there). Step 10 made
extraction_node/playbook_rag_node async, and the FIRST version of this file
still called the sync team.invoke(), which fails with "No synchronous
function provided" the moment the graph reaches an async node. No
pytest-asyncio plugin is assumed to be available, so _run() stays a plain
sync function that internally asyncio.run()s the real async call.
"""
import asyncio
import sys
from pathlib import Path

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed in this environment")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from team.graph import build_team  # noqa: E402
from team.state import MAX_REVISIONS, seed_state  # noqa: E402


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


RECURSION_LIMIT = 200

CONTRACTS = [
    "vendor_payments_processor_agreement.md",
    "vendor_fulfillment_logistics_agreement.md",
    "vendor_warranty_repair_partner_agreement.md",
    "vendor_returns_processing_agreement.md",
]


async def _run_async(filename: str) -> dict:
    team = build_team()
    seed = seed_state(f"data/contracts/{filename}")
    return await team.ainvoke(seed, {"recursion_limit": RECURSION_LIMIT})


def _run(filename: str) -> dict:
    return asyncio.run(_run_async(filename))


def test_build_team_compiles():
    # If build_team() has any wiring mistake (a routing key with no matching
    # node, a node name typo, a missing edge), this raises at compile time
    # rather than only failing on a specific invoke() path later.
    team = build_team()
    assert team is not None


@pytest.mark.parametrize("filename", CONTRACTS)
def test_full_pipeline_terminates_and_covers_every_clause(filename):
    state = _run(filename)
    assert len(state["playbook_findings"]) == len(state["clauses"])
    assert len(state["draft"]) == len(state["clauses"])
    redlined = {c for c, e in state["draft"].items() if e["action"] == "redline_proposed"}
    assert set(state["legal_review"]) == redlined


def test_payments_contract_escalates_on_the_named_sign_off_clause():
    """Cross-check against Steps 6 and 7: the same outcome the manual
    while-loop in scripts/run_supervisor.py produced should come out of the
    real compiled graph too, since the node functions themselves are
    unchanged — only the orchestration mechanism is different."""
    state = _run("vendor_payments_processor_agreement.md")
    assert state["status"] == "escalated_to_human"
    assert state["legal_review"]["3.1"]["verdict"] == "escalated"


def test_contracts_with_no_escalation_complete_cleanly():
    for filename in ("vendor_fulfillment_logistics_agreement.md",
                      "vendor_warranty_repair_partner_agreement.md",
                      "vendor_returns_processing_agreement.md"):
        state = _run(filename)
        assert state["status"] == ""


def test_revision_loop_redrafts_then_respects_the_cap(monkeypatch):
    """Same forced-rejection scenario Step 7 used to catch the real
    revision_count overshoot bug (5 rejections used to drive it to 6
    against a cap of 2) — re-run here through the actual compiled graph
    rather than the manual loop, since langgraph's own reducers now do the
    state-merging that used to be done by hand in merge_update()."""
    monkeypatch.setattr("team.nodes.legal_reviewer._keyword_adequacy", lambda *a, **k: "changes_requested")

    state = _run("vendor_fulfillment_logistics_agreement.md")

    assert state["status"] == "escalated_to_human"
    assert state["revision_count"] == MAX_REVISIONS
    for review in state["legal_review"].values():
        assert review["verdict"] == "escalated"
        assert review["method"] == "revision_cap"