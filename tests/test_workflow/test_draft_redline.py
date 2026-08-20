"""
ShopSense M5 - Part 4 tests: workflow/nodes/draft_redline.py

Layered the same way Step 3's tests were:
  1. Pure deterministic guards (sanitize_liability_language,
     _detect_cap_conflict) - no node, no state, called directly.
  2. deterministic_compose_redline - pure function of state, no LLM.
  3. The node itself, proving the partial-update trap doesn't happen
     (existing concerns are preserved, not erased) and that the liability
     guard applies even when a composer deliberately violates it.

make_llm_compose_redline is not tested here - it wraps a live LLMClient
call, out of scope for a deterministic suite (same reason M1/M4's own
pytest suites don't call a real model either).
"""

from workflow.nodes.draft_redline import (
    COMPLIANT_APOLOGY,
    _detect_cap_conflict,
    build_draft_redline_node,
    deterministic_compose_redline,
    sanitize_liability_language,
)
from workflow.state import seed_state


# --------------------------------------------------------------------------
# sanitize_liability_language() - escalation-tone.md §4.4.2
# --------------------------------------------------------------------------

def test_clean_text_is_unchanged():
    text = "Thank you for reaching out. We can process your refund."
    sanitized, redactions = sanitize_liability_language(text)
    assert sanitized == text
    assert redactions == 0


def test_liability_admission_sentence_is_replaced():
    text = "Hello. I'm sorry our product failed you. We will refund you."
    sanitized, redactions = sanitize_liability_language(text)
    assert redactions == 1
    assert COMPLIANT_APOLOGY in sanitized
    assert "our product failed" not in sanitized.lower()


def test_only_the_offending_sentence_is_replaced_not_the_whole_text():
    text = "Thank you for your patience. Our product failed you. We appreciate your business."
    sanitized, _ = sanitize_liability_language(text)
    assert "Thank you for your patience." in sanitized
    assert "We appreciate your business." in sanitized


def test_matching_is_case_insensitive():
    text = "OUR PRODUCT FAILED and we are sorry."
    _, redactions = sanitize_liability_language(text)
    assert redactions == 1


def test_multiple_offending_sentences_all_replaced():
    text = "This was our fault. Our product is defective. Thanks for reaching out."
    sanitized, redactions = sanitize_liability_language(text)
    assert redactions == 2
    assert sanitized.count(COMPLIANT_APOLOGY) == 2


# --------------------------------------------------------------------------
# _detect_cap_conflict() - the M2/M4 flagged-open ₹2,000 vs $50 conflict
# --------------------------------------------------------------------------

def test_no_conflict_when_only_refund_authority_cited():
    citations = [{"doc_slug": "refund-authority", "clause_number": "4.1"}]
    assert _detect_cap_conflict(citations) is None


def test_no_conflict_when_neither_doc_cited():
    citations = [{"doc_slug": "shipping-policy", "clause_number": "3"}]
    assert _detect_cap_conflict(citations) is None


def test_conflict_detected_when_both_docs_cited_together():
    citations = [
        {"doc_slug": "refund-authority", "clause_number": "4.1"},
        {"doc_slug": "escalation-tone", "clause_number": "4.3.6"},
    ]
    note = _detect_cap_conflict(citations)
    assert note is not None
    assert "₹2,000" in note and "$50" in note


# --------------------------------------------------------------------------
# deterministic_compose_redline() - pure function of state
# --------------------------------------------------------------------------

def _refund_state(**overrides):
    state = seed_state("T1", "My item was broken, refund please.", order_id="ORD-1")
    state["parsed_ticket"] = {"issue_type": "REFUND", "claimed_refund_amount": 500.0}
    state["policy_eligible_amount"] = 500.0
    state["policy_action"] = "refund"
    state["citations"] = [{"doc_slug": "refund-authority", "clause_number": "4.1"}]
    state["concerns"] = []
    state["classification"] = "standard"
    state.update(overrides)
    return state


def test_compose_mentions_eligible_amount_when_it_matches_claim():
    draft = deterministic_compose_redline(_refund_state())
    assert "₹500.00" in draft
    assert "approved" in draft.lower()


def test_compose_surfaces_the_gap_when_claimed_exceeds_eligible():
    state = _refund_state(policy_eligible_amount=200.0)
    state["parsed_ticket"]["claimed_refund_amount"] = 800.0
    draft = deterministic_compose_redline(state)
    assert "₹800.00" in draft
    assert "₹200.00" in draft


def test_compose_flags_pending_review_when_concerns_present():
    state = _refund_state(concerns=["refund-authority.md §4.1/4.2: over cap"], classification="non_standard")
    draft = deterministic_compose_redline(state)
    assert "flagged for a quick review" in draft
    assert "approved" not in draft.lower()


def test_compose_does_not_assert_approval_ahead_of_a_pending_review():
    """Regression test: an earlier version said 'We can process a refund of
    ₹X' AND 'flagged for review' in the same message - contradictory, since
    it asserted an action that hasn't actually happened yet."""
    state = _refund_state(
        policy_eligible_amount=3500.0,
        concerns=["refund-authority.md §4.1/4.2: over cap"],
        classification="non_standard",
    )
    state["parsed_ticket"]["claimed_refund_amount"] = 3500.0  # claimed == eligible
    draft = deterministic_compose_redline(state)
    assert "we can process" not in draft.lower()
    assert "₹3500.00" in draft


def test_compose_cites_the_retrieved_sources():
    draft = deterministic_compose_redline(_refund_state())
    assert "refund-authority §4.1" in draft


def test_compose_never_needs_its_own_sanitizer_pass():
    """The template itself uses the compliant apology line verbatim - it
    should never trip its own liability-language guard."""
    draft = deterministic_compose_redline(_refund_state())
    _, redactions = sanitize_liability_language(draft)
    assert redactions == 0


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------

def test_node_preserves_existing_concerns_not_overwrite_them():
    """The partial-update trap this module's docstring warns about: a node
    must start from state["concerns"], never build a fresh list, or it
    silently erases what compare_to_playbook already found."""
    draft_redline = build_draft_redline_node()
    state = _refund_state(concerns=["refund-authority.md §4.1/4.2: over cap"], classification="non_standard")

    update = draft_redline(state)

    assert "refund-authority.md §4.1/4.2: over cap" in update["concerns"]


def test_node_adds_conflict_concern_and_flips_classification_to_non_standard():
    """A ticket that LOOKED standard from compare_to_playbook can still be
    caught here - the conflict-detection safety net runs independently."""
    draft_redline = build_draft_redline_node()
    state = _refund_state(
        citations=[
            {"doc_slug": "refund-authority", "clause_number": "4.1"},
            {"doc_slug": "escalation-tone", "clause_number": "4.3.6"},
        ],
        concerns=[],
        classification="standard",
    )

    update = draft_redline(state)

    assert update["classification"] == "non_standard"
    assert any("conflict" in c for c in update["concerns"])


def test_node_applies_liability_guard_even_when_composer_violates_it():
    """Proves the guard is unconditional, not composer-dependent - a stub
    composer that deliberately writes an admission of fault still gets
    caught, same as an uncooperative real LLM would."""
    def bad_composer(state):
        return "I'm sorry our product failed. We'll fix it."

    draft_redline = build_draft_redline_node(compose_redline=bad_composer)
    state = _refund_state()

    update = draft_redline(state)

    assert "our product failed" not in update["redline_draft"].lower()
    assert COMPLIANT_APOLOGY in update["redline_draft"]
    assert any("liability" in a for a in update["audit_log"])


def test_node_increments_revision_count():
    draft_redline = build_draft_redline_node()
    state = _refund_state()
    assert state["revision_count"] == 0

    update = draft_redline(state)

    assert update["revision_count"] == 1


def test_node_returns_only_the_expected_keys():
    """Partial-update discipline: this node must not accidentally return
    keys it has no business touching (e.g. citations, policy_action)."""
    draft_redline = build_draft_redline_node()
    update = draft_redline(_refund_state())

    assert set(update.keys()) == {
        "redline_draft", "concerns", "classification", "revision_count", "audit_log",
    }