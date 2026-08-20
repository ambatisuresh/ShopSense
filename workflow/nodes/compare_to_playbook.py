"""
ShopSense M5 - Part 3: the `compare_to_playbook` node.

Reads `parsed_ticket` (extract's output) and decides two things, both
CONTROL fields the router in Step 5 will read:
    - `concerns`       : why this ticket is NOT safe to auto-approve
                          (empty list = nothing standing in the way)
    - `classification` : "standard" | "non_standard"

Also populates `citations` (for draft_redline, Step 4, to cite) and
`policy_eligible_amount` / `policy_action` (the deterministic refund verdict,
if this is a refund-type ticket).

WHAT'S NEW vs. WHAT'S REUSED:
    - escalation_tone_concerns() is NEW for M5. M2's summary is explicit
      that escalation-tone.md's triggers (4.3.1-4.3.6) were left to the
      agent's own judgment via system-prompt guidance, not enforced as
      deterministic tool logic - fine for an agent, but this workflow's
      router (Step 5) needs a deterministic field to read, not a model's
      opinion re-derived on every run. Same "let deterministic code decide"
      principle Lab B used for check_document().
    - refund_policy_concerns() REUSES M2's refund_calculator/process_refund
      via an adapter (make_m2_refund_adapter), same pattern as Step 2's M1
      parser adapter - not new policy logic, just new wiring.
    - Citation retrieval REUSES M4's retrieval pipeline via
      make_m4_retrieval_adapter - again, wiring, not new retrieval logic.

WHAT "standard" MEANS HERE: not just "refund amount is small". A ticket
classifies as standard when the proposed resolution (whatever it is - a
refund, a delivery-status reply, a replacement) can go to the customer
without a human sign-off. non_standard means a human must review it before
anything is sent. This node decides that; it does not decide WHAT the
resolution says - that's draft_redline (Step 4).

The refund-authority.md (₹2,000) vs escalation-tone.md ($50) auto-approval
cap conflict - flagged open since M2, still open per M4's summary - is
resolved HERE the same way M2 resolved it: refund-authority.md's ₹2,000 is
authoritative for the classification decision. This node does not silently
erase the conflict, though - Step 4's draft_redline is where a human-facing
surfacing of that conflict belongs, since compare_to_playbook's job is to
decide routing, not to write prose about a policy inconsistency.

Production wiring - ASSUMED, not yet confirmed against real source (only
M2's/M4's SUMMARIES were available while building this - same flagged-not-
verified status Step 2 gave tools/order_lookup.py until the real file
showed up):
    from tools.refund_calculator import calculate_refund as _m2_calculate_refund
    from tools.refund_replace import process_refund as _m2_process_refund
    from rag.bm25_index import bm25_search as _m4_bm25_search
    from rag.rerank import CrossEncoderReranker

    evaluate_refund = make_m2_refund_adapter(_m2_calculate_refund, _m2_process_refund)
    retrieve_citations = make_m4_retrieval_adapter(_m4_bm25_search, CrossEncoderReranker())
    compare_to_playbook_node = build_compare_to_playbook_node(evaluate_refund, retrieve_citations)

STEP 7 ADDITION - the `commit` flag:
    This node calls `evaluate_refund` purely to CLASSIFY a ticket, before
    any human has approved anything. `workflow/nodes/finalize.py` (Step 7)
    calls the SAME `evaluate_refund` again, after approval, to actually
    resolve the ticket. If M2's real `process_refund` only computes a
    verdict (fraud_flag/amount_mismatch/eligible_amount) that's safe to
    call twice. If it actually COMMITS - writes to a ledger, moves money,
    mutates an order record - then this node calling it pre-approval would
    be silently refunding tickets that later get rejected by a human. M2's
    summary is not conclusive either way ("refund/replace API... approval
    tiers + fraud triggers"), so this is UNRESOLVED, not guessed at.

    Mitigation: `evaluate_refund` now takes a `commit: bool = False`
    keyword. `compare_to_playbook` (this node) always calls it with
    `commit=False` - "just tell me what would happen, don't do it yet".
    `finalize` calls it with `commit=True` - "now actually do it". This
    makes the intent explicit in code and gives `make_m2_refund_adapter` a
    single place to pass `commit` through to the real `process_refund` -
    but it does NOT resolve the ambiguity. Confirm against the real
    tools/refund_replace.py source before this adapter is used in
    production: (a) does `process_refund` accept a `commit` kwarg at all,
    and (b) if not, does calling it unconditionally commit on every call -
    in which case `make_m2_refund_adapter` must be rewritten so that a
    `commit=False` call never reaches `process_refund` at all, and falls
    back to `calculate_refund` alone for the pre-approval verdict.
"""

from typing import Callable, Optional

from workflow.state import TicketReviewState

EvaluateRefundFn = Callable[..., dict]
RetrieveCitationsFn = Callable[[str, int], "list[dict]"]

# refund-authority.md §4.1/4.2: automated system auto-approves refunds up
# to this amount; above it, manual approval is required "regardless of
# customer tier or sentiment" (§4.2.2). This is the authoritative figure
# per M2's key decision #2 - NOT escalation-tone.md §4.3.6's "$50" line,
# which is a known corpus inconsistency, not a second valid cap.
AUTO_APPROVAL_CAP_INR = 2000.0

HUMAN_REQUEST_KEYWORDS = (
    "speak to a human", "talk to a human", "talk to a person", "connect me to a human",
    "get me a human", "human agent", "human support", "real person", "a representative",
)
LEGAL_KEYWORDS = ("lawyer", "legal action", "sue ", "consumer court", "regulatory complaint", "file a complaint")
SAFETY_KEYWORDS = ("injur", "unsafe", "hazard", "caught fire", "electric shock", "choking", "burn")


# ---------------------------------------------------------------------------
# Pure policy functions - no I/O, no state, no graph. Same testing posture
# as Lab B's check_document(): call these directly with plain values.
# ---------------------------------------------------------------------------

def escalation_tone_concerns(raw_text: str, parsed_ticket: dict) -> list[str]:
    """Deterministic implementation of escalation-tone.md §4.3's triggers,
    to the extent a SINGLE ticket can prove them.

    NOT implemented: §4.3.3 "third contact on the same issue" - that needs
    cross-ticket history this node has no access to (no ticket-history
    lookup exists anywhere in this project yet). Flagged here deliberately,
    same spirit as M4's "no silent narrowing" scope note - rather than
    silently treating it as "never triggers", it is documented as a known
    gap so a future milestone (or this one, if you want it now) can close
    it once ticket history is available.
    """
    text = (raw_text or "").lower()
    concerns = []

    if any(p in text for p in HUMAN_REQUEST_KEYWORDS):
        concerns.append("escalation-tone.md §4.3.1: explicit request for human support")

    if parsed_ticket.get("sentiment") == "threatening":
        concerns.append("escalation-tone.md §4.3.2: abusive/threatening language")

    if any(p in text for p in LEGAL_KEYWORDS):
        concerns.append("escalation-tone.md §4.3.4: mention of legal action or regulatory complaint")

    if any(p in text for p in SAFETY_KEYWORDS):
        concerns.append("escalation-tone.md §4.3.5: mention of physical harm or product safety")

    return concerns


def refund_policy_concerns(refund_eval: dict) -> list[str]:
    """Applies refund-authority.md §4.1/4.2/4.4.2 to an already-computed
    refund evaluation (M2's calculate_refund/process_refund output, via
    make_m2_refund_adapter). Does not call any tool itself - pure function
    of the dict it's handed."""
    concerns = []

    if refund_eval.get("fraud_flag"):
        # §4.4.2: fraud/abuse escalates to Finance regardless of amount.
        concerns.append("refund-authority.md §4.4.2: fraud/abuse trigger requires Finance escalation")

    eligible = refund_eval.get("eligible_amount")
    if eligible is not None and eligible > AUTO_APPROVAL_CAP_INR:
        concerns.append(
            f"refund-authority.md §4.1/4.2: eligible amount ₹{eligible:.2f} "
            f"exceeds the ₹{AUTO_APPROVAL_CAP_INR:.0f} auto-approval cap"
        )

    if refund_eval.get("amount_mismatch"):
        concerns.append("amount_mismatch: claimed refund amount does not match the policy-eligible amount")

    return concerns


def classify(
    parsed_ticket: dict,
    raw_text: str,
    order_id: Optional[str],
    refund_eval: Optional[dict],
) -> "tuple[str, list[str]]":
    """The one function this whole node exists to get right. Returns
    (classification, concerns). Empty concerns <=> "standard" - this is the
    ONLY place that equivalence is decided; the router (Step 5) just reads
    the result, it never re-derives it."""
    if not parsed_ticket:
        return "non_standard", ["ticket could not be parsed into a structured request"]

    concerns: list[str] = []

    if parsed_ticket.get("contains_suspicious_instructions"):
        concerns.append("intake flagged contains_suspicious_instructions (possible prompt injection)")

    concerns += escalation_tone_concerns(raw_text, parsed_ticket)

    needs_refund_eval = (
        parsed_ticket.get("issue_type") == "REFUND"
        or parsed_ticket.get("claimed_refund_amount") is not None
    )
    if needs_refund_eval and not order_id:
        # M2 decision #5's principle carried over: dirty/incomplete data
        # gets routed to a human, not guessed at.
        concerns.append(
            "refund requested but no order_id could be resolved - cannot "
            "verify eligibility against the order record"
        )
    elif refund_eval:
        concerns += refund_policy_concerns(refund_eval)

    classification = "non_standard" if concerns else "standard"
    return classification, concerns


def _build_retrieval_query(parsed_ticket: dict, raw_text: str) -> str:
    issue_type = parsed_ticket.get("issue_type", "")
    snippet = " ".join(raw_text.split())[:200]
    return f"{issue_type} policy: {snippet}".strip()


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def build_compare_to_playbook_node(
    evaluate_refund: EvaluateRefundFn,
    retrieve_citations: RetrieveCitationsFn,
) -> Callable[[TicketReviewState], dict]:
    """Returns the `compare_to_playbook` node: `TicketReviewState -> partial
    update`.

    `evaluate_refund(order_id, claimed_amount, commit=False) -> dict` with at
    least `eligible_amount` / `action` / `fraud_flag` / `amount_mismatch`
    keys. This node always calls it with `commit=False` - see the module
    docstring's "STEP 7 ADDITION" note for why the flag exists.

    `retrieve_citations(query, k) -> list[dict]` with citation dicts shaped
    like `state.py`'s `citations` field: `{doc_slug, clause_number,
    clause_title, chunk_id}` at minimum.
    """

    def compare_to_playbook(state: TicketReviewState) -> dict:
        parsed = state["parsed_ticket"]
        order_id = state.get("order_id")
        audit: list[str] = []

        needs_refund_eval = bool(parsed) and (
            parsed.get("issue_type") == "REFUND"
            or parsed.get("claimed_refund_amount") is not None
        )
        refund_eval = None
        if needs_refund_eval and order_id:
            # commit=False: this node only classifies. finalize.py is the
            # only place that ever calls evaluate_refund(commit=True). See
            # the module docstring's "STEP 7 ADDITION" note.
            refund_eval = evaluate_refund(
                order_id=order_id, claimed_amount=parsed.get("claimed_refund_amount"), commit=False,
            )
            audit.append(
                f"compare_to_playbook: refund evaluation -> "
                f"eligible=₹{refund_eval.get('eligible_amount')} "
                f"fraud_flag={refund_eval.get('fraud_flag')} "
                f"amount_mismatch={refund_eval.get('amount_mismatch')}"
            )

        classification, concerns = classify(parsed, state["raw_text"], order_id, refund_eval)

        citations: list[dict] = []
        if parsed:
            query = _build_retrieval_query(parsed, state["raw_text"])
            citations = retrieve_citations(query, 5)
            audit.append(f"compare_to_playbook: retrieved {len(citations)} citation(s) for query={query!r}")

        audit.append(f"compare_to_playbook: classification={classification} concerns={concerns}")

        return {
            "citations": citations,
            "policy_eligible_amount": (refund_eval or {}).get("eligible_amount"),
            "policy_action": (refund_eval or {}).get("action"),
            "concerns": concerns,
            "classification": classification,
            "audit_log": audit,
        }

    return compare_to_playbook


# ---------------------------------------------------------------------------
# Production adapters - ASSUMED call shapes, not yet confirmed against real
# source. Deferring dedicated adapter tests until the real
# tools/refund_calculator.py, tools/refund_replace.py, rag/bm25_index.py,
# rag/rerank.py are available - same "flag, don't guess-and-hide" treatment
# order_lookup got in Step 2 until llm_client.py/parser.py were uploaded.
# ---------------------------------------------------------------------------

def make_m2_refund_adapter(calculate_refund, process_refund) -> EvaluateRefundFn:
    """Wrap M2's `tools/refund_calculator.py::calculate_refund` +
    `tools/refund_replace.py::process_refund`. Exact signatures unconfirmed;
    written against M2's summary description ("refund-amount calculator...
    return windows + condition-based partial refund" /
    "refund/replace API... approval tiers + fraud triggers... never trusts
    the requested amount... flags amount_mismatch").

    `commit` is threaded straight through to `process_refund(..., commit=)`
    on the ASSUMPTION that the real function accepts such a flag. This is
    UNCONFIRMED - see the module docstring's "STEP 7 ADDITION" note. If the
    real `process_refund` has no `commit` parameter, this adapter is not
    safe to use as-is for the `commit=False` (pre-approval, compare_to_
    playbook) call path and must be rewritten to skip calling
    `process_refund` entirely when `commit=False`, falling back to
    `calculate_refund` alone for the verdict."""

    def evaluate_refund(order_id: str, claimed_amount, commit: bool = False) -> dict:
        try:
            calc = calculate_refund(order_id, condition="unopened")
        except Exception as e:
            return {
                "eligible_amount": None, "action": "escalate",
                "fraud_flag": False, "amount_mismatch": False, "error": str(e),
            }
        result = process_refund(
            order_id, requested_amount=claimed_amount, condition="unopened", commit=commit,
        )
        return {
            "eligible_amount": calc.get("eligible_amount"),
            "action": result.get("action", "refund"),
            "fraud_flag": result.get("fraud_flag", False),
            "amount_mismatch": result.get("amount_mismatch", False),
        }

    return evaluate_refund


def make_m4_retrieval_adapter(bm25_search, reranker) -> RetrieveCitationsFn:
    """Wrap M4's retrieval pipeline: BM25 candidate search -> cross-encoder
    rerank - the same Qdrant-free path M4's own run_bm25.py /
    run_eval_retrieval.py demos use. Exact call shape unconfirmed against
    the real rag/bm25_index.py / rag/rerank.py source."""

    def retrieve(query: str, k: int = 5) -> list[dict]:
        candidates = bm25_search(query, k=max(k * 3, 10))
        reranked = reranker.rerank(query, candidates, top_k=k)
        return [
            {
                "doc_slug": c.get("doc_slug"),
                "clause_number": c.get("clause_number"),
                "clause_title": c.get("clause_title"),
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
            }
            for c in reranked
        ]

    return retrieve


# ---------------------------------------------------------------------------
# Offline fixtures - deterministic, no order table, no corpus/index. Used by
# tests/test_workflow/test_compare_to_playbook.py and
# scripts/run_compare_to_playbook.py.
# ---------------------------------------------------------------------------

def fixture_evaluate_refund(order_id: Optional[str], claimed_amount, commit: bool = False) -> dict:
    """No mock order/refund tables wired in this sandbox. eligible_amount
    mirrors claimed_amount 1:1 (no proration, no return-window check) -
    enough to exercise compare_to_playbook's OWN cap/classification logic,
    NOT a substitute for M2's real condition/window-based calculation.

    `commit` is accepted (so this fixture satisfies the same EvaluateRefundFn
    contract the real adapter does) but has no observable effect here - a
    stateless fixture has nothing to commit. finalize.py's own tests use a
    dedicated spy, not this fixture, to prove commit=True/False is actually
    threaded correctly - see test_finalize.py."""
    return {
        "eligible_amount": claimed_amount,
        "action": "refund" if claimed_amount else None,
        "fraud_flag": False,
        "amount_mismatch": False,
    }


_FIXTURE_CITATIONS_BY_ISSUE_TYPE = {
    "REFUND": [
        {"doc_slug": "refund-authority", "clause_number": "4.1", "clause_title": "Refund Approval Tiers", "chunk_id": "refund-authority#4.1"},
        {"doc_slug": "returns-policy", "clause_number": "2", "clause_title": "Return Windows", "chunk_id": "returns-policy#2"},
    ],
    "DELIVERY": [
        {"doc_slug": "shipping-policy", "clause_number": "3", "clause_title": "Delay Handling", "chunk_id": "shipping-policy#3"},
    ],
    "PRODUCT": [
        {"doc_slug": "warranty-policy", "clause_number": "1", "clause_title": "Warranty Coverage", "chunk_id": "warranty-policy#1"},
    ],
}


def fixture_retrieve_citations(query: str, k: int = 5) -> list[dict]:
    """No corpus/index wired in this sandbox. Deterministic keyword lookup
    keyed by issue_type (parsed out of the query string built by
    _build_retrieval_query) - enough structure to prove compare_to_playbook
    consumes citations correctly, NOT a substitute for M4's real hybrid
    search + rerank."""
    for issue_type, citations in _FIXTURE_CITATIONS_BY_ISSUE_TYPE.items():
        if query.startswith(issue_type):
            return citations[:k]
    return []