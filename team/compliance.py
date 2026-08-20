"""Deterministic contract-clause compliance assessment against a negotiation
playbook position.

This is the "deterministic code decides" half of Redline Drafter (Step 5) —
see team/nodes/redline_drafter.py for the optional LLM-composed redline text,
which is always attempted first but never trusted to make the compliance
call itself.

Two assessment methods, tried in order:

1. Numeric threshold comparison (`_numeric_assess`) — for clause types whose
   playbook position states a concrete number (days, %, USD, months). A
   single representative number is extracted from the contract clause and
   compared against the same figure extracted from the Preferred and
   Fallback position text, using a per-type "which direction is better"
   rule. This is precise where it applies, but it is a *first-number-found*
   heuristic: it does not understand ranges ("5 to 7 business days"), nor
   relative language ("1 business day slower than the SLA"). Both of those
   shapes appear in this corpus (delivery_sla_alignment, respectively) and
   are documented inline below where they matter.

2. Keyword overlap (`_keyword_assess`) — for everything numeric comparison
   can't resolve (either the clause has no number, or the playbook position
   is purely qualitative). Each tier's *discriminating* words — words that
   appear in that tier's text and no other tier's — are compared against the
   clause body's words. This deliberately ignores words shared across tiers
   (which are usually just the clause topic restated), so a match only
   counts when the clause echoes language specific to one tier.

Neither method is a substitute for a human reading the clause. Where both
methods come back empty, `assess_clause` returns matched_tier=None and the
node flags the clause for manual review rather than guessing.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "is", "are",
    "this", "that", "these", "those", "shall", "must", "may", "any", "all",
    "with", "from", "by", "as", "at", "on", "per", "its", "their", "other",
    "not", "no", "than", "then", "than", "will", "would", "should", "can",
    "such", "each", "party", "parties", "agreement", "kartway", "vendor",
    "customer", "clause", "under", "into", "upon", "within", "without",
    "prior", "provided", "only", "before", "after", "been", "being", "has",
    "have", "had", "who", "which", "whose", "own", "over", "out", "than",
}

_WORD_RE = re.compile(r"[a-z]{3,}")
_NUM_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def tokenize_words(text: str) -> set:
    """Word tokens (stopword-filtered) plus normalized number tokens ("50.00"
    and "50" both become "50", "2,000" becomes "2000"). Numbers matter a lot
    for keyword-overlap discrimination in this corpus — e.g. clause 1.1's
    contract-vs-playbook conflict is literally "USD 50" vs "INR 2,000", which
    pure word-overlap can't see since digits aren't letters."""
    text_l = (text or "").lower()
    words = {w for w in _WORD_RE.findall(text_l) if w not in _STOPWORDS}
    numbers = {n.replace(",", "").split(".")[0] for n in _NUM_TOKEN_RE.findall(text_l)}
    numbers.discard("")
    return words | numbers


def _extract_num(text: str, pattern: str) -> float | None:
    if not text:
        return None
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


# Number-extraction patterns. Each requires the digit to sit next to its
# unit (allowing an optional closing paren, since this corpus consistently
# spells numbers as "ten (10) business days") so "five (5) to seven (7)
# business days" resolves to the figure actually adjacent to the unit
# words, not an arbitrary one in a range.
_BDAY_RE = r"(\d+(?:\.\d+)?)\)?\s*business\s*days?"
_CDAY_RE = r"(\d+(?:\.\d+)?)\)?\s*calendar\s*days?"
_PCT_RE = r"(\d+(?:\.\d+)?)\s*(?:%|percent)"
_MONTH_RE = r"(\d+(?:\.\d+)?)\)?\s*months?"
_USD_RE = r"(?:USD|US\$|\$)\s*([\d,]+(?:\.\d+)?)"

# clause_type -> (regex, direction). direction is "lower_better" when a
# smaller number is the more Kartway-favorable outcome (turnaround times,
# investigation windows, loss-declaration thresholds) or "higher_better"
# when a larger number is (compensation %, liability caps, notice periods,
# months of fee coverage).
#
# Clause types not listed here have no clean single-number shape in this
# corpus's playbook text (pure qualitative positions, or the number lives on
# the "wrong side" — e.g. refund_settlement_authority's conflict is a
# currency mismatch, not a threshold) and go straight to keyword matching.
NUMERIC_SPECS = {
    "refund_settlement_timing": (_BDAY_RE, "lower_better"),
    "delivery_sla_alignment": (_BDAY_RE, "lower_better"),
    "delay_compensation_alignment": (_PCT_RE, "higher_better"),
    "lost_in_transit_threshold": (_CDAY_RE, "lower_better"),
    "disputed_delivery_investigation": (_BDAY_RE, "lower_better"),
    "carrier_liability_cap": (_USD_RE, "higher_better"),
    "repair_turnaround": (_BDAY_RE, "lower_better"),
    "claim_status_reporting_cadence": (_BDAY_RE, "lower_better"),
    "return_intake_window": (_CDAY_RE, "higher_better"),
    "inspection_to_refund_buffer": (_BDAY_RE, "lower_better"),
    "termination_and_renewal": (_CDAY_RE, "higher_better"),
    "limitation_of_liability": (_MONTH_RE, "higher_better"),
}


def _numeric_assess(clause_body: str, parts: dict, regex: str, direction: str) -> str | None:
    contract_val = _extract_num(clause_body, regex)
    if contract_val is None:
        return None  # clause states nothing in this unit; can't compare numerically

    preferred_val = _extract_num(parts.get("preferred", ""), regex)
    fallback_val = _extract_num(parts.get("fallback", ""), regex)
    if preferred_val is None and fallback_val is None:
        return None  # playbook gives no numeric guidance for this metric either

    def satisfies(bound):
        if bound is None:
            return False
        return contract_val <= bound if direction == "lower_better" else contract_val >= bound

    if satisfies(preferred_val):
        return "preferred"
    if satisfies(fallback_val):
        return "fallback"
    return "unacceptable"


# When two tiers tie on keyword-overlap score, prefer flagging over clearing
# — a false "needs a closer look" costs a human a few minutes; a false
# "compliant" ships a bad clause.
_TIE_BREAK_PRIORITY = {
    "unacceptable": 4,
    "needs_review": 3,
    "fallback": 2,
    "note": 1,
    "preferred": 0,
    "acceptable_as_is": 0,
}


def _keyword_assess(clause_body: str, parts: dict) -> str | None:
    clause_words = tokenize_words(clause_body)
    if not clause_words:
        return None

    tier_words = {tier: tokenize_words(text) for tier, text in parts.items()}
    discriminating = {}
    for tier, words in tier_words.items():
        others = set()
        for other_tier, other_words in tier_words.items():
            if other_tier != tier:
                others |= other_words
        discriminating[tier] = words - others

    scores = {
        tier: len(clause_words & words)
        for tier, words in discriminating.items()
        if clause_words & words
    }
    if not scores:
        return None
    return max(scores, key=lambda t: (scores[t], _TIE_BREAK_PRIORITY.get(t, 0)))


def assess_clause(clause_body: str, position: dict, clause_type: str) -> dict:
    """Return {"matched_tier": str|None, "method": "numeric"|"keyword"|"none"}."""
    spec = NUMERIC_SPECS.get(clause_type)
    if spec:
        regex, direction = spec
        tier = _numeric_assess(clause_body, position["parts"], regex, direction)
        if tier:
            return {"matched_tier": tier, "method": "numeric"}

    tier = _keyword_assess(clause_body, position["parts"])
    if tier:
        return {"matched_tier": tier, "method": "keyword"}

    return {"matched_tier": None, "method": "none"}


# matched_tier -> compliance verdict.
_TIER_TO_COMPLIANCE = {
    "preferred": "compliant",
    "acceptable_as_is": "compliant",
    "fallback": "acceptable_fallback",
    "note": "advisory",
    "unacceptable": "non_compliant",
}


def classify_compliance(matched_tier: str | None, parts: dict) -> str:
    """Map an assessed tier to a compliance verdict.

    When nothing matched (matched_tier is None), the verdict depends on what
    kind of position this is: if the playbook position doesn't even define
    an "unacceptable" tier (e.g. governing_law_and_venue, which is Preferred
    / Fallback / Note only — a pure negotiation preference the playbook
    itself says "should be noted rather than escalated"), an unmatched
    clause is advisory, not a violation. If the position does define a
    blocking tier, an unmatched clause is inconclusive rather than
    automatically a violation — it's flagged for manual review, not
    silently passed and not silently failed.
    """
    if matched_tier is not None:
        return _TIER_TO_COMPLIANCE.get(matched_tier, "needs_review")

    if "unacceptable" not in parts:
        # No blocking tier defined for this position at all (e.g.
        # governing_law_and_venue is Preferred/Fallback/Note only, and the
        # playbook explicitly says it "should be noted rather than
        # escalated") — an unmatched clause here is a non-issue, not a gap.
        return "advisory"
    return "needs_review"


# Compliance verdicts that require a redline to be drafted.
REDLINE_REQUIRED = {"non_compliant", "needs_review"}