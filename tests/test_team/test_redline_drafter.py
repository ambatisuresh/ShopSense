"""Step 5 tests: team/compliance.py, team/nodes/redline_drafter.py.

Run:
    pytest tests/test_team/test_redline_drafter.py -v

The EXPECTED_DRAFT table below is the exact, corrected output of a real
end-to-end run of the DETERMINISTIC fallback path (no LLM). Three entries
(returns 2.2, 2.3, 4.1) are DELIBERATELY left as their verified-but-wrong
"compliant" result: the deterministic keyword fallback can't bridge a
paraphrase gap ("limited to a refund of the processing fee" vs the
playbook's "capped at the processing fee alone" mean the same thing but
share almost no vocabulary). See test_known_keyword_heuristic_false_negatives
below and the Step 5 build note.

Every test in this file runs with the LLM paths force-disabled (see the
autouse `_force_deterministic_assessment` fixture below), regardless of
whether litellm + working API credentials happen to be present in whatever
environment runs these tests. Without that, this file's expectations would
be environment-dependent: on a machine with real credentials configured,
_llm_assess_compliance correctly resolves the returns 2.2/2.3/4.1 paraphrase
gap that the deterministic fallback misses — which is the system working
exactly as designed ("let the model produce, let deterministic code
decide"), but would make EXPECTED_DRAFT wrong on that machine and right on
this one. scripts/run_redline_drafter.py is where the live LLM path
actually runs when credentials are configured — run it yourself to see the
LLM-assisted results.

Step 10 made extraction_node() and playbook_rag_node() async and MCP-backed.
Same hermetic treatment as test_playbook_rag.py: extraction_node here
pre-populates contract_text from local disk, and playbook_rag's positions
cache is primed from a local disk parse before any test runs, so this file
(which is really testing Redline Drafter, not the data-fetch layer) never
needs a live MCP server either.
"""
import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from team.compliance import (  # noqa: E402
    REDLINE_REQUIRED,
    _keyword_assess,
    _numeric_assess,
    assess_clause,
    classify_compliance,
    tokenize_words,
)
from team.nodes.extraction import extraction_node as _extraction_node_async  # noqa: E402
from team.nodes.playbook_rag import playbook_rag_node as _playbook_rag_node_async  # noqa: E402
import team.nodes.playbook_rag as _playbook_rag_module  # noqa: E402
from team.nodes.redline_drafter import redline_drafter_node  # noqa: E402
from team.playbook_index import (  # noqa: E402
    PLAYBOOK_PATH,
    load_playbook_positions,
    parse_playbook_positions,
)
from team.scopes import AGENT_SCOPES  # noqa: E402
from team.state import seed_state  # noqa: E402

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
def _force_deterministic_assessment(monkeypatch):
    """Disable both LLM paths for every test in this file. See the module
    docstring above for why this file must be hermetic to the environment's
    litellm/API-credential availability."""
    monkeypatch.setattr(
        "team.nodes.redline_drafter._llm_assess_compliance",
        lambda clause, position: None,
    )
    monkeypatch.setattr(
        "team.nodes.redline_drafter._llm_compose_redline",
        lambda clause, position, compliance, feedback=None: None,
    )

# {filename: {clause_id: {"action", "compliance", "matched_tier", "method"}}}
EXPECTED_DRAFT = {
    "vendor_payments_processor_agreement.md": {
        "1.1": {"action": "no_action", "compliance": None},
        "2.1": {"action": "no_action", "compliance": None},
        "3.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "keyword"},
        "3.2": {"action": "no_action", "compliance": "acceptable_fallback"},
        "4.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "keyword"},
        "5.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "6.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "keyword"},
        "7.1": {"action": "no_action", "compliance": "compliant"},
        "8.1": {"action": "no_action", "compliance": "compliant"},
        "9.1": {"action": "no_action", "compliance": "advisory"},
        "10.1": {"action": "no_action", "compliance": "compliant"},
    },
    "vendor_fulfillment_logistics_agreement.md": {
        "1.1": {"action": "no_action", "compliance": None},
        "2.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "3.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "4.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "5.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "6.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "keyword"},
        "7.1": {"action": "no_action", "compliance": None},
        "8.1": {"action": "no_action", "compliance": "acceptable_fallback"},
        "9.1": {"action": "no_action", "compliance": "compliant"},
    },
    "vendor_warranty_repair_partner_agreement.md": {
        "1.1": {"action": "no_action", "compliance": None},
        "2.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "2.2": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "keyword"},
        "3.1": {"action": "no_action", "compliance": "compliant"},
        "3.2": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "keyword"},
        "4.1": {"action": "no_action", "compliance": None},
        "5.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "6.1": {"action": "no_action", "compliance": None},
        # Negative control (Step 1): this clause's 90-day notice was
        # deliberately written to MATCH the playbook's Preferred position —
        # it must clear compliant, not get redlined.
        "7.1": {"action": "no_action", "compliance": "compliant",
                "matched_tier": "preferred"},
        "8.1": {"action": "no_action", "compliance": "compliant"},
    },
    "vendor_returns_processing_agreement.md": {
        "1.1": {"action": "no_action", "compliance": None},
        "2.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "keyword"},
        # Known keyword-heuristic false negatives — see module docstring.
        "2.2": {"action": "no_action", "compliance": "compliant"},
        "2.3": {"action": "no_action", "compliance": "compliant"},
        "3.1": {"action": "redline_proposed", "compliance": "non_compliant",
                "matched_tier": "unacceptable", "method": "numeric"},
        "4.1": {"action": "no_action", "compliance": "compliant"},
        "5.1": {"action": "no_action", "compliance": None},
        "6.1": {"action": "no_action", "compliance": None},
        "7.1": {"action": "no_action", "compliance": "acceptable_fallback"},
    },
}

CONTRACTS_DIR = REPO_ROOT / "data" / "contracts"


def _run_full_redline(filename: str) -> dict:
    """Extraction -> Playbook RAG -> Redline Drafter, each driven one unit
    per call until exhausted — the same loop scripts/run_redline_drafter.py
    uses. Returns the final state dict."""
    state = dict(seed_state(f"data/contracts/{filename}"))
    state.update(extraction_node(state))
    for _ in range(len(state["clauses"])):
        update = playbook_rag_node(state)
        state["playbook_findings"] = state["playbook_findings"] + update.get("playbook_findings", [])
    for _ in range(len(state["clauses"])):
        update = redline_drafter_node(state)
        if "draft" in update:
            state["draft"] = update["draft"]
    return state


# ---------------------------------------------------------------------------
# team/compliance.py — tokenize_words
# ---------------------------------------------------------------------------

def test_tokenize_words_strips_short_stopwords():
    words = tokenize_words("The vendor shall not exceed this amount for any claim.")
    assert "the" not in words
    assert "shall" not in words
    assert "vendor" not in words  # in the domain stopword list
    assert "exceed" in words
    assert "claim" in words


def test_tokenize_words_normalizes_numbers():
    words = tokenize_words("USD 50.00 or INR 2,000 per transaction")
    assert "50" in words
    assert "2000" in words


# ---------------------------------------------------------------------------
# team/compliance.py — numeric threshold assessment
# ---------------------------------------------------------------------------

def test_numeric_assess_picks_preferred_when_contract_beats_the_bar():
    parts = {"preferred": "at most 5 business days", "fallback": "up to 7 business days"}
    tier = _numeric_assess("we commit to 3 business days", parts, r"(\d+)\s*business\s*days?", "lower_better")
    assert tier == "preferred"


def test_numeric_assess_picks_fallback_when_between_bars():
    parts = {"preferred": "at most 5 business days", "fallback": "up to 7 business days"}
    tier = _numeric_assess("we commit to 6 business days", parts, r"(\d+)\s*business\s*days?", "lower_better")
    assert tier == "fallback"


def test_numeric_assess_defaults_to_unacceptable_beyond_both_bars():
    parts = {"preferred": "at most 5 business days", "fallback": "up to 7 business days"}
    tier = _numeric_assess("we commit to 20 business days", parts, r"(\d+)\s*business\s*days?", "lower_better")
    assert tier == "unacceptable"


def test_numeric_assess_returns_none_when_clause_has_no_number():
    parts = {"preferred": "at most 5 business days", "fallback": "up to 7 business days"}
    tier = _numeric_assess("best efforts, no fixed commitment", parts, r"(\d+)\s*business\s*days?", "lower_better")
    assert tier is None


def test_numeric_assess_returns_none_when_playbook_gives_no_numeric_guidance():
    parts = {"preferred": "Kartway's home jurisdiction", "fallback": "a neutral jurisdiction"}
    tier = _numeric_assess("30 business days", parts, r"(\d+)\s*business\s*days?", "lower_better")
    assert tier is None


def test_numeric_assess_handles_parenthesized_digits():
    """Regression test: this corpus consistently writes 'ten (10) business
    days' — the digit sits right before a closing paren, not the unit."""
    parts = {"preferred": "five (5) business days", "fallback": "seven (7) business days"}
    tier = _numeric_assess(
        "Fixwell shall complete repairs within fifteen (15) business days",
        parts, r"(\d+(?:\.\d+)?)\)?\s*business\s*days?", "lower_better",
    )
    assert tier == "unacceptable"


def test_numeric_assess_higher_better_direction():
    parts = {"preferred": "at least 90 calendar days", "fallback": "60 calendar days"}
    tier = _numeric_assess("30 calendar days notice", parts, r"(\d+)\s*calendar\s*days?", "higher_better")
    assert tier == "unacceptable"
    tier = _numeric_assess("90 calendar days notice", parts, r"(\d+)\s*calendar\s*days?", "higher_better")
    assert tier == "preferred"


# ---------------------------------------------------------------------------
# team/compliance.py — keyword assessment
# ---------------------------------------------------------------------------

def test_keyword_assess_finds_the_currency_mismatch_conflict():
    """The corpus's central deliberate conflict (Step 1): a USD 50
    auto-settlement ceiling against a playbook that requires INR 2,000 and
    explicitly calls out the USD 50 figure as unacceptable."""
    positions = load_playbook_positions()
    parts = positions["refund_settlement_authority"]["parts"]
    clause_text = (
        "Meridian shall auto-settle any refund instruction up to USD 50.00 "
        "per transaction without additional written authorization."
    )
    assert _keyword_assess(clause_text, parts) == "unacceptable"


def test_keyword_assess_returns_none_with_no_overlap():
    parts = {"preferred": "alpha beta gamma", "unacceptable": "delta epsilon zeta"}
    assert _keyword_assess("nothing relevant whatsoever here", parts) is None


# ---------------------------------------------------------------------------
# team/compliance.py — classify_compliance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier,expected", [
    ("preferred", "compliant"),
    ("acceptable_as_is", "compliant"),
    ("fallback", "acceptable_fallback"),
    ("note", "advisory"),
    ("unacceptable", "non_compliant"),
])
def test_classify_compliance_maps_every_known_tier(tier, expected):
    parts = {"preferred": "x", "fallback": "y", "unacceptable": "z", "note": "n", "acceptable_as_is": "a"}
    assert classify_compliance(tier, parts) == expected


def test_classify_compliance_unmatched_with_blocking_tier_needs_review():
    parts = {"preferred": "x", "fallback": "y", "unacceptable": "z"}
    assert classify_compliance(None, parts) == "needs_review"


def test_classify_compliance_unmatched_without_blocking_tier_is_advisory():
    """governing_law_and_venue has no 'unacceptable' tier at all — the
    playbook itself says it should be noted, not escalated."""
    parts = {"preferred": "x", "fallback": "y", "note": "n"}
    assert classify_compliance(None, parts) == "advisory"


def test_redline_required_set_matches_the_blocking_verdicts():
    assert REDLINE_REQUIRED == {"non_compliant", "needs_review"}


# ---------------------------------------------------------------------------
# team/nodes/redline_drafter.py — end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", list(EXPECTED_DRAFT))
def test_redline_drafter_end_to_end_per_contract(filename):
    state = _run_full_redline(filename)
    expected = EXPECTED_DRAFT[filename]

    assert len(state["draft"]) == len(state["clauses"])

    for clause_id, want in expected.items():
        got = state["draft"][clause_id]
        assert got["action"] == want["action"], (
            f"{filename} clause {clause_id}: expected action={want['action']!r}, "
            f"got {got['action']!r}"
        )
        assert got.get("compliance") == want["compliance"], (
            f"{filename} clause {clause_id}: expected compliance={want['compliance']!r}, "
            f"got {got.get('compliance')!r}"
        )
        if "matched_tier" in want:
            assert got.get("matched_tier") == want["matched_tier"]
        if "method" in want:
            assert got.get("assessment_method") == want["method"]


def test_redline_proposed_entries_always_include_proposed_text():
    for filename in EXPECTED_DRAFT:
        state = _run_full_redline(filename)
        for entry in state["draft"].values():
            if entry["action"] == "redline_proposed":
                assert entry["proposed_text"]
                assert entry["composer"] in ("llm", "template")
                # No LLM credentials in this environment — the fallback
                # composer is what's actually exercised here.
                assert entry["composer"] == "template"


def test_negative_control_clause_is_never_redlined():
    """warranty-partner clause 7.1 (90-day termination notice) was written
    in Step 1 specifically to match the playbook's Preferred position, as a
    check against a drafter that flags everything indiscriminately."""
    state = _run_full_redline("vendor_warranty_repair_partner_agreement.md")
    entry = state["draft"]["7.1"]
    assert entry["action"] == "no_action"
    assert entry["compliance"] == "compliant"


def test_known_keyword_heuristic_false_negatives():
    """Documents, rather than hides, a real limitation: the no-LLM keyword
    fallback misses 3 clauses whose contract language paraphrases the
    playbook's Unacceptable tier without sharing its vocabulary (e.g.
    'limited to a refund of the processing fee' vs 'capped at the
    processing fee alone'). See this module's docstring and the Step 5
    build note. If this test ever starts failing because these clauses got
    correctly flagged, that's an improvement — update the expectation, not
    the code."""
    state = _run_full_redline("vendor_returns_processing_agreement.md")
    for clause_id in ("2.2", "2.3", "4.1"):
        assert state["draft"][clause_id]["compliance"] == "compliant"


def test_redline_drafter_processes_exactly_one_clause_per_call():
    state = dict(seed_state("data/contracts/vendor_payments_processor_agreement.md"))
    state.update(extraction_node(state))
    for _ in range(len(state["clauses"])):
        update = playbook_rag_node(state)
        state["playbook_findings"] += update.get("playbook_findings", [])

    update = redline_drafter_node(state)
    assert len(update["draft"]) == 1


def test_redline_drafter_no_ops_once_everything_is_done():
    state = _run_full_redline("vendor_payments_processor_agreement.md")
    update = redline_drafter_node(state)
    assert "draft" not in update
    assert "nothing outstanding" in update["log"][0]


def test_redline_drafter_output_stays_inside_its_write_scope():
    state = dict(seed_state("data/contracts/vendor_payments_processor_agreement.md"))
    state.update(extraction_node(state))
    for _ in range(len(state["clauses"])):
        update = playbook_rag_node(state)
        state["playbook_findings"] += update.get("playbook_findings", [])

    update = redline_drafter_node(state)
    allowed = AGENT_SCOPES["redline_drafter"] | {"log"}
    assert set(update) <= allowed


def test_redline_drafter_handles_skipped_findings_without_crashing():
    """Clause 1.1 (Scope of Services) is unclassified in every contract in
    this corpus — playbook_rag marks it 'skipped', and redline_drafter must
    record a no_action entry rather than trying to look up a position that
    doesn't exist."""
    state = dict(seed_state("data/contracts/vendor_payments_processor_agreement.md"))
    state.update(extraction_node(state))
    update = playbook_rag_node(state)  # first clause is always 1.1
    state["playbook_findings"] += update.get("playbook_findings", [])
    assert state["playbook_findings"][0]["status"] == "skipped"

    update = redline_drafter_node(state)
    entry = update["draft"]["1.1"]
    assert entry["action"] == "no_action"
    assert entry["compliance"] is None