"""Step 2 tests: ContractReviewState + AGENT_SCOPES + scoped().

Run:
    pytest tests/test_team/test_state.py -v

Zero external dependencies (no langgraph/fastmcp needed) — pure Python,
same as team/state.py and team/scopes.py themselves.
"""
import asyncio
import sys
from operator import add
from pathlib import Path
from typing import Annotated, get_args, get_origin

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from team.scopes import AGENT_SCOPES, scoped  # noqa: E402
from team.state import ContractReviewState, MAX_REVISIONS, seed_state  # noqa: E402

AUDIT_FIELDS = {"playbook_findings", "log"}
CONTROL_FIELDS = set(ContractReviewState.__annotations__) - AUDIT_FIELDS


# ---------------------------------------------------------------------------
# seed_state()
# ---------------------------------------------------------------------------

def test_seed_state_has_every_declared_field():
    state = seed_state("data/contracts/vendor_payments_processor_agreement.md")
    assert set(state.keys()) == set(ContractReviewState.__annotations__)


def test_seed_state_defaults_are_empty_not_none():
    state = seed_state("some/path.md")
    assert state["clauses"] == []
    assert state["draft"] == {}
    assert state["legal_review"] == {}
    assert state["revision_count"] == 0
    assert state["next_agent"] == ""
    assert state["status"] == ""
    assert state["playbook_findings"] == []
    assert state["log"] == []


def test_seed_state_carries_the_contract_path_and_optional_text():
    state = seed_state("data/contracts/x.md", contract_text="raw text here")
    assert state["contract_path"] == "data/contracts/x.md"
    assert state["contract_text"] == "raw text here"


def test_max_revisions_is_a_small_positive_int():
    assert isinstance(MAX_REVISIONS, int)
    assert 1 <= MAX_REVISIONS <= 5


# ---------------------------------------------------------------------------
# Control vs. audit field split (the invariant M5 and the notebook both rely
# on: a reducer on a control field means it can never be reset to "unset",
# which means a revision loop can never exit).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", sorted(AUDIT_FIELDS))
def test_audit_fields_have_an_add_reducer(field):
    annotation = ContractReviewState.__annotations__[field]
    assert get_origin(annotation) is Annotated, f"{field} must be Annotated[..., add]"
    assert add in get_args(annotation), f"{field} must reduce with operator.add"


@pytest.mark.parametrize("field", sorted(CONTROL_FIELDS))
def test_control_fields_have_no_reducer(field):
    annotation = ContractReviewState.__annotations__[field]
    assert get_origin(annotation) is not Annotated, (
        f"{field} is a control field and must be a plain overwrite — "
        f"a reducer here means it can never be reset to its unset value"
    )


# ---------------------------------------------------------------------------
# AGENT_SCOPES coverage
# ---------------------------------------------------------------------------

def test_agent_scopes_covers_every_role():
    assert set(AGENT_SCOPES) == {
        "extraction", "playbook_rag", "redline_drafter",
        "legal_reviewer", "supervisor", "escalate",
    }


def test_agent_scopes_only_reference_real_state_fields():
    all_state_fields = set(ContractReviewState.__annotations__)
    for role, keys in AGENT_SCOPES.items():
        unknown = keys - all_state_fields
        assert not unknown, f"{role} scope references non-existent fields: {unknown}"


def test_only_redline_drafter_may_reset_legal_review():
    """Mirrors the notebook's writer-may-reset-fact_check-and-review rule: a
    rewrite voids prior approvals, and exactly one role should hold that
    power."""
    holders = [role for role, keys in AGENT_SCOPES.items() if "legal_review" in keys]
    assert set(holders) == {"redline_drafter", "legal_reviewer"}


def test_legal_reviewer_cannot_touch_the_draft():
    """The critic must not be able to edit the artefact it's judging."""
    assert "draft" not in AGENT_SCOPES["legal_reviewer"]


def test_extraction_and_playbook_rag_cannot_touch_next_agent():
    """Only the supervisor may write next_agent — specialists decide their
    own output, never who acts next."""
    for role in ("extraction", "playbook_rag", "redline_drafter", "legal_reviewer"):
        assert "next_agent" not in AGENT_SCOPES[role]


# ---------------------------------------------------------------------------
# scoped() enforcement
# ---------------------------------------------------------------------------

def test_scoped_allows_a_legitimate_write():
    @scoped("legal_reviewer")
    def honest(state):
        return {"legal_review": {"approved": True}, "log": ["ok"]}

    result = honest({})
    assert result == {"legal_review": {"approved": True}, "log": ["ok"]}


def test_scoped_blocks_a_write_outside_the_role_scope():
    @scoped("legal_reviewer")
    def rogue(state):
        return {"legal_review": {"approved": True}, "draft": {"redlines": []}}

    with pytest.raises(PermissionError, match=r"legal_reviewer.*draft"):
        rogue({})


def test_scoped_every_role_is_individually_enforced():
    for role, allowed_keys in AGENT_SCOPES.items():
        @scoped(role)
        def rogue(state, _role=role):
            return {"__not_a_real_field__": True}

        with pytest.raises(PermissionError, match=role):
            rogue({})


def test_scoped_unknown_role_raises_at_decoration_time():
    with pytest.raises(KeyError):
        scoped("not_a_real_agent")


def test_scoped_handles_async_nodes():
    @scoped("playbook_rag")
    async def honest_async(state):
        await asyncio.sleep(0)
        return {"playbook_findings": [{"clause_id": "1.1"}]}

    result = asyncio.run(honest_async({}))
    assert result["playbook_findings"] == [{"clause_id": "1.1"}]


def test_scoped_blocks_async_nodes_too():
    @scoped("playbook_rag")
    async def rogue_async(state):
        await asyncio.sleep(0)
        return {"next_agent": "writer"}

    with pytest.raises(PermissionError, match="playbook_rag"):
        asyncio.run(rogue_async({}))