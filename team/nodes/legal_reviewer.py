"""The Legal Reviewer agent: for each clause Redline Drafter proposed a
redline for, decide whether that redline is good enough to execute, needs
another drafting pass, or must be escalated to a human.

One draft entry per invocation — same per-unit-of-work design as Playbook
RAG and Redline Drafter. `legal_review` is a plain-overwrite state field, so
each call reads the existing dict, adds one clause's verdict, and returns
the whole thing back.

Three possible verdicts:
  - "approved" — the proposed redline adequately aligns the clause with the
    playbook's target position. Ready to execute once applied.
  - "changes_requested" — the proposed redline doesn't clearly get there.
    Step 7's supervisor is what actually routes this back to Redline
    Drafter for another pass (bumping `revision_count`, which Legal
    Reviewer can read but not write — see team/scopes.py) — this node's job
    ends at recording the verdict and why.
  - "escalated" — either the playbook itself names a specific human sign-off
    for this exact position (only refund_settlement_authority's Unacceptable
    tier does this in the current playbook: "Flag for Operations Manager
    sign-off..."), or the revision budget (MAX_REVISIONS, team/state.py) is
    already spent and another "changes_requested" round would just loop.

"Let the model produce, let deterministic code decide" a third time: an LLM
review is tried first (`_llm_review_redline`), because judging whether
freshly-drafted language actually satisfies a playbook position is exactly
the kind of semantic call bag-of-words can't reliably make. The deterministic
fallback (`_keyword_adequacy`) is a much blunter proxy — it checks whether
the proposed text shares meaningful vocabulary with the position's target
tier text, which the template composer path will (almost tautologically)
always pass, since that composer quotes the target text directly. It's a
real check when the composer was the LLM instead. No LLM credentials are
configured in this sandbox, so the deterministic path is what's actually
exercised and tested here.
"""
from __future__ import annotations

import re

from team.compliance import tokenize_words
from team.scopes import scoped
from team.state import MAX_REVISIONS

# Matches playbook language that names a specific human sign-off for this
# exact position — e.g. clause 1.1's Unacceptable tier: "Flag for Operations
# Manager sign-off per refund-authority.md §4.4.1 before this clause is
# executed." This is deliberately narrower than the playbook's general
# "flagged for redline and Legal Reviewer sign-off" purpose statement (every
# clause reaching this node already has that, implicitly) — it only fires
# when the position calls out a role BEYOND ordinary Legal Reviewer review.
_NAMED_ESCALATION_RE = re.compile(r"flag for ([a-z][a-z ]*?) sign[- ]off", re.IGNORECASE)

_ADEQUACY_THRESHOLD = 0.3  # fraction of target-tier words the redline must echo


def _clause_sort_key(clause_id: str) -> tuple:
    return tuple(int(part) for part in clause_id.split("."))


def _select_next_entry(draft: dict, done_ids: set) -> dict | None:
    candidates = [
        entry for clause_id, entry in draft.items()
        if entry["action"] == "redline_proposed" and clause_id not in done_ids
    ]
    candidates.sort(key=lambda e: _clause_sort_key(e["clause_id"]))
    return candidates[0] if candidates else None


def _target_tier_text(parts: dict) -> str:
    for tier in ("fallback", "preferred", "acceptable_as_is", "note"):
        if parts.get(tier):
            return parts[tier]
    return ""


def _named_escalation_role(matched_tier: str | None, parts: dict) -> str | None:
    if matched_tier is None:
        return None
    m = _NAMED_ESCALATION_RE.search(parts.get(matched_tier, ""))
    return m.group(1).strip().title() if m else None


def _keyword_adequacy(proposed_text: str, parts: dict) -> str | None:
    """Deterministic fallback verdict, or None if there's nothing to compare
    the redline against (a position with no defined tiers at all)."""
    target = _target_tier_text(parts)
    target_words = tokenize_words(target)
    if not target_words:
        return None
    proposed_words = tokenize_words(proposed_text)
    overlap = len(target_words & proposed_words) / len(target_words)
    return "approved" if overlap >= _ADEQUACY_THRESHOLD else "changes_requested"


def _llm_review_redline(clause_id: str, entry: dict, position: dict) -> str | None:
    """Best-effort LLM adequacy review. Returns None on ANY failure — same
    contract as every other _llm_* helper in this project: never raises,
    caller always has the deterministic fallback ready."""
    try:
        import litellm
    except ImportError:
        return None

    target = _target_tier_text(position["parts"])
    prompt = (
        "You are a Legal Reviewer checking a proposed contract redline "
        "against Kartway's negotiation playbook. Reply with EXACTLY one "
        "label — approved or changes_requested — and nothing else.\n\n"
        f"Playbook {position['clause_number']} ({position['title']}) target position:\n"
        f"{target}\n\n"
        f"Proposed redline for clause {clause_id}:\n{entry['proposed_text']}\n\n"
        "Does the proposed redline adequately align the clause with the "
        "target position?\nLabel:"
    )
    try:
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=15,
        )
        label = response["choices"][0]["message"]["content"].strip().lower().replace(" ", "_")
    except Exception:
        return None

    return label if label in ("approved", "changes_requested") else None


@scoped("legal_reviewer")
def legal_reviewer_node(state: dict) -> dict:
    done_ids = set(state["legal_review"].keys())
    entry = _select_next_entry(state["draft"], done_ids)

    if entry is None:
        return {"log": ["legal_reviewer: nothing outstanding to review"]}

    clause_id = entry["clause_id"]
    position = None
    for finding in state["playbook_findings"]:
        if finding["clause_id"] == clause_id:
            position = finding["position"]
            break

    escalation_role = _named_escalation_role(entry.get("matched_tier"), position["parts"]) if position else None

    if escalation_role:
        verdict = "escalated"
        reason = (
            f"playbook {entry['playbook_clause']} names a specific sign-off "
            f"authority for this position: {escalation_role}"
        )
        method = "playbook_rule"
    elif state.get("revision_count", 0) >= MAX_REVISIONS:
        verdict = "escalated"
        reason = (
            f"revision budget exhausted (revision_count="
            f"{state.get('revision_count', 0)} >= MAX_REVISIONS={MAX_REVISIONS}) "
            "without a resolved redline — needs manual sign-off"
        )
        method = "revision_cap"
    else:
        llm_verdict = _llm_review_redline(clause_id, entry, position) if position else None
        if llm_verdict is not None:
            verdict, method = llm_verdict, "llm"
        else:
            keyword_verdict = _keyword_adequacy(entry["proposed_text"], position["parts"]) if position else None
            verdict = keyword_verdict or "changes_requested"
            method = "keyword" if keyword_verdict is not None else "no_target"
        reason = (
            f"proposed redline {'echoes' if verdict == 'approved' else 'does not clearly echo'} "
            f"playbook {entry['playbook_clause']}'s target position (assessed via {method})"
        )

    review = {
        "clause_id": clause_id,
        "verdict": verdict,
        "reason": reason,
        "method": method,
        "revision_count_at_review": state.get("revision_count", 0),
    }
    new_legal_review = {**state["legal_review"], clause_id: review}
    return {
        "legal_review": new_legal_review,
        "log": [f"legal_reviewer: clause {clause_id} -> {verdict} ({method})"],
    }