"""The Redline Drafter agent: for each clause Playbook RAG resolved to a
position, decide whether the contract's actual language is acceptable and,
if not, draft specific replacement language.

One finding per invocation — same per-unit-of-work design as Playbook RAG
(Step 4). `draft` is a plain-overwrite state field (not an audit-log
reducer), so each call reads the existing draft dict, adds or updates one
clause's entry, and returns the whole dict back — the same pattern
extraction_node uses for `clauses`.

Two phases, checked in order on every call:

1. **Initial drafting** — any clause with no `draft` entry yet at all. This
   is everything Step 5 built.
2. **Revision drafting** (Step 7) — clauses Legal Reviewer sent back with
   `verdict == "changes_requested"` that haven't been redrafted since that
   specific rejection. Each draft entry is stamped with the `revision_count`
   value it was (re)drafted at (`drafted_at_revision`); a clause is eligible
   for revision when that stamp still matches the `revision_count_at_review`
   its rejection was recorded under — once redrafted, the stamp moves past
   that value, so the same rejection can't be reprocessed twice. This node
   bumps `revision_count` by one per redraft, and clears the stale
   `legal_review` entry so Legal Reviewer reviews the new text fresh — both
   fields are in this node's write-scope (see team/scopes.py) specifically
   for this purpose.

   `revision_count` is a single budget shared across every clause needing
   rework, not a separate per-clause allowance — so once it reaches
   MAX_REVISIONS (team/state.py), this node stops attempting further
   redrafts entirely: any remaining stale rejection is sent straight back to
   Legal Reviewer (by clearing its `legal_review` entry without touching
   `draft` or `revision_count`) for a cap-based `escalated` verdict instead
   of burning another redraft on a budget that's already spent. An earlier
   version of this node kept redrafting every stale rejection regardless of
   the cap, which let `revision_count` overshoot MAX_REVISIONS by however
   many clauses needed rework in the same round (verified by running: 5
   rejected clauses drove revision_count to 6 against a cap of 2) — caught
   by actually running the forced-rejection scenario in
   tests/test_team/test_supervisor.py, not by inspection.

"Let the model produce, let deterministic code decide" applies twice here:

1. Compliance *tier* (does this clause read as Preferred/Fallback/
   Unacceptable/etc.?) — an LLM call is tried first (`_llm_assess_compliance`
   below), because several playbook positions are purely qualitative
   (indemnification, category exclusions, liability phrasing) and genuinely
   need paraphrase understanding a bag-of-words comparison can't do — e.g.
   "liability is limited to a refund of the processing fee" vs the
   playbook's "capped at the processing fee alone" mean the same thing to a
   reader but share almost no vocabulary. When the LLM call fails or isn't
   configured, team/compliance.py's `assess_clause` (numeric threshold
   comparison, then keyword overlap) is the deterministic fallback — always
   available, but honestly weaker on exactly this kind of paraphrase. No LLM
   credentials are configured in this environment, so the deterministic path
   is what's actually exercised and tested here; its known false-negative
   cases are documented in tests/test_team/test_redline_drafter.py and the
   Step 5 build note.
2. Redline *text* — an LLM composer is tried first for the actual proposed
   language; a template built from the playbook's own Preferred/Fallback
   wording is the fallback. Never trusted to make the compliance call
   itself, regardless of which path produced it.
"""
from __future__ import annotations

from team.compliance import REDLINE_REQUIRED, assess_clause, classify_compliance
from team.scopes import scoped
from team.state import MAX_REVISIONS

_EXCERPT_LEN = 320


def _clause_sort_key(clause_id: str) -> tuple:
    return tuple(int(part) for part in clause_id.split("."))


def _select_next_finding(findings: list[dict], done_ids: set) -> dict | None:
    todo = [f for f in findings if f["clause_id"] not in done_ids]
    todo.sort(key=lambda f: _clause_sort_key(f["clause_id"]))
    return todo[0] if todo else None


def _lookup_clause(clauses: list[dict], clause_id: str) -> dict | None:
    for clause in clauses:
        if clause["clause_id"] == clause_id:
            return clause
    return None


def _target_text(parts: dict) -> tuple[str, str]:
    """The most lenient defined position to redline *toward*, preferring
    Fallback (the actual acceptable-but-not-ideal bar) over Preferred (the
    ideal, which may be an unrealistic ask mid-negotiation)."""
    for tier in ("fallback", "preferred", "acceptable_as_is", "note"):
        if parts.get(tier):
            return tier, parts[tier]
    return "", ""


def _llm_assess_compliance(clause: dict, position: dict) -> str | None:
    """Ask a live LLM which playbook tier this contract clause matches.

    Returns None on ANY failure: missing package, missing/invalid API key,
    network error, timeout, or a response outside this position's actual
    tier labels (preferred / fallback / unacceptable / acceptable_as_is /
    note — whichever subset this specific playbook clause defines). Same
    contract as extraction.py's _llm_classify_clause_type: never raises,
    caller always has a deterministic fallback ready."""
    try:
        import litellm
    except ImportError:
        return None

    tiers = list(position["parts"].keys())
    tier_descriptions = "\n".join(
        f"- {tier}: {text}" for tier, text in position["parts"].items()
    )
    prompt = (
        "You are comparing ONE contract clause against a negotiation "
        "playbook's stated positions for that clause type. Reply with "
        f"EXACTLY one label from this list, and nothing else: {', '.join(tiers)}\n\n"
        f"Playbook clause {position['clause_number']} ({position['title']}):\n"
        f"{tier_descriptions}\n\n"
        f"Contract clause {clause['clause_id']} ({clause['title']}):\n"
        f"{clause['text'][:1000]}\n\n"
        "Which playbook position does the contract clause actually match?\n"
        "Label:"
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

    return label if label in tiers else None


def _llm_compose_redline(clause: dict, position: dict, compliance: str, feedback: str | None = None) -> str | None:
    """Best-effort LLM redline drafting. Returns None on ANY failure so the
    caller always has a deterministic fallback — no LLM credentials are
    configured in this environment, so this path is untested here and the
    template composer is what's actually verified."""
    try:
        import litellm

        tier, target = _target_text(position["parts"])
        feedback_block = (
            f"\nLegal Reviewer rejected a prior draft of this clause: {feedback}\n"
            "Address that specifically in your revised proposal.\n"
            if feedback else ""
        )
        prompt = (
            "You are drafting a contract redline for a procurement/legal team.\n"
            f"Playbook clause {position['clause_number']} ({position['title']}):\n"
            f"Target position ({tier}): {target}\n"
            f"{feedback_block}\n"
            f"Current contract clause {clause['clause_id']} ({clause['title']}):\n"
            f"{clause['text']}\n\n"
            "In 2-3 sentences, propose specific replacement language for this "
            "clause that would satisfy the target position. Return only the "
            "proposed clause text, no preamble."
        )
        response = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=10,
        )
        text = response["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None


def _template_compose_redline(clause: dict, position: dict, compliance: str, feedback: str | None = None) -> str:
    tier, target = _target_text(position["parts"])
    revision_note = f" (revised — Legal Reviewer feedback: {feedback})" if feedback else ""
    if compliance == "needs_review":
        return (
            f"Automated comparison could not confidently place Clause "
            f"{clause['clause_id']} against Playbook {position['clause_number']} "
            f"({position['title']}) — manual Legal Reviewer comparison needed."
            f"{revision_note} Playbook Fallback position for reference: "
            f"{position['parts'].get('fallback') or position['parts'].get('preferred') or 'n/a'}"
        )
    return (
        f"Clause {clause['clause_id']} ({clause['title']}) as drafted does not "
        f"meet Playbook {position['clause_number']} ({position['title']}).{revision_note} "
        f"Proposed redline: revise the clause to match the {tier or 'playbook'} "
        f"position — \"{target}\""
    )


def _build_entry(clause: dict, finding: dict, position: dict, revision: int, feedback: str | None = None) -> dict:
    """Assess and (if needed) draft a redline for one clause. Shared by both
    the initial-drafting and revision-drafting phases below — the only
    difference between them is which clause gets selected and whether
    `feedback` from a prior rejection is threaded into the composers."""
    clause_id = finding["clause_id"]

    llm_tier = _llm_assess_compliance(clause, position)
    if llm_tier is not None:
        matched_tier, method = llm_tier, "llm"
    else:
        assessment = assess_clause(clause["text"], position, finding["clause_type"])
        matched_tier, method = assessment["matched_tier"], assessment["method"]
    compliance = classify_compliance(matched_tier, position["parts"])

    if compliance in REDLINE_REQUIRED:
        redline_text = _llm_compose_redline(clause, position, compliance, feedback)
        composer = "llm"
        if redline_text is None:
            redline_text = _template_compose_redline(clause, position, compliance, feedback)
            composer = "template"
        return {
            "clause_id": clause_id,
            "action": "redline_proposed",
            "compliance": compliance,
            "matched_tier": matched_tier,
            "assessment_method": method,
            "current_excerpt": clause["text"][:_EXCERPT_LEN],
            "proposed_text": redline_text,
            "composer": composer,
            "playbook_clause": position["clause_number"],
            "drafted_at_revision": revision,
        }
    return {
        "clause_id": clause_id,
        "action": "no_action",
        "compliance": compliance,
        "matched_tier": matched_tier,
        "assessment_method": method,
        "playbook_clause": position["clause_number"],
        "reason": f"assessed as '{compliance}' — within acceptable range",
        "drafted_at_revision": revision,
    }


def _select_next_revision(draft: dict, legal_review: dict) -> str | None:
    """A clause is due for revision when Legal Reviewer rejected it and the
    draft hasn't been touched since that specific rejection (comparing the
    draft's `drafted_at_revision` stamp against the review's
    `revision_count_at_review`) — prevents reprocessing the same rejection
    forever once it's been addressed."""
    candidates = [
        clause_id for clause_id, review in legal_review.items()
        if review["verdict"] == "changes_requested"
        and draft[clause_id].get("drafted_at_revision", 0) == review["revision_count_at_review"]
    ]
    candidates.sort(key=_clause_sort_key)
    return candidates[0] if candidates else None


@scoped("redline_drafter")
def redline_drafter_node(state: dict) -> dict:
    revision_count = state.get("revision_count", 0)

    # Phase 1: initial drafting — any clause with no draft entry at all.
    done_ids = set(state["draft"].keys())
    finding = _select_next_finding(state["playbook_findings"], done_ids)

    if finding is not None:
        clause_id = finding["clause_id"]

        if finding["status"] != "found":
            # skipped (unclassified) or no_position — nothing to redline against.
            entry = {
                "clause_id": clause_id,
                "action": "no_action",
                "compliance": None,
                "reason": f"playbook_rag status was '{finding['status']}', not 'found'",
                "drafted_at_revision": revision_count,
            }
        else:
            clause = _lookup_clause(state["clauses"], clause_id)
            entry = _build_entry(clause, finding, finding["position"], revision_count)

        new_draft = {**state["draft"], clause_id: entry}
        return {
            "draft": new_draft,
            "log": [f"redline_drafter: clause {clause_id} -> {entry['action']} ({entry.get('compliance')})"],
        }

    # Phase 2: revision drafting — clauses Legal Reviewer rejected that
    # haven't been redrafted since that specific rejection.
    redo_id = _select_next_revision(state["draft"], state["legal_review"])
    if redo_id is None:
        return {"log": ["redline_drafter: nothing outstanding to draft"]}

    if revision_count >= MAX_REVISIONS:
        # Budget's already spent — don't burn another redraft on it. Clear
        # the stale rejection so the supervisor routes this clause back to
        # Legal Reviewer, which will finalize it as "escalated" via its own
        # revision-cap check rather than requesting changes again.
        new_legal_review = {k: v for k, v in state["legal_review"].items() if k != redo_id}
        return {
            "legal_review": new_legal_review,
            "log": [
                f"redline_drafter: revision budget spent (revision_count={revision_count} "
                f">= MAX_REVISIONS={MAX_REVISIONS}) — clause {redo_id} sent back for "
                "final escalation without another redraft"
            ],
        }

    review = state["legal_review"][redo_id]
    finding = next(f for f in state["playbook_findings"] if f["clause_id"] == redo_id)
    clause = _lookup_clause(state["clauses"], redo_id)
    new_revision_count = revision_count + 1

    entry = _build_entry(clause, finding, finding["position"], new_revision_count, feedback=review["reason"])
    new_draft = {**state["draft"], redo_id: entry}
    new_legal_review = {k: v for k, v in state["legal_review"].items() if k != redo_id}

    return {
        "draft": new_draft,
        "legal_review": new_legal_review,
        "revision_count": new_revision_count,
        "log": [
            f"redline_drafter: clause {redo_id} redrafted (revision {new_revision_count}) "
            f"after changes_requested"
        ],
    }