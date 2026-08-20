"""
ShopSense M5 - Part 2: the `extract` node.

Turns a raw ticket into the structured data the rest of the workflow
reasons over. This node is an ADAPTER, not new parsing logic - it reuses
M1's intake parser and M3's customer-resolution priority rather than
rebuilding either.

Design decisions:

1. `extract` only extracts. It never classifies standard/non-standard and
   never touches `concerns` - a failed parse just means `parsed_ticket`
   stays `{}`. Deciding what an empty parse MEANS for approval routing
   belongs to `compare_to_playbook` (Step 3), whose entire job is producing
   `concerns`/`classification`. One node, one decision - same discipline
   Lab B used to keep `checks_node` the only place that reads BANNED_TERMS.

2. Dependency injection via a factory, not a hardcoded import. Same shape
   M3 used for `build_customer_memory_tool` and M4 used for `get_embedder`:
   `build_extract_node(parse_ticket, resolve_customer_ref)` takes two plain
   callables and returns the actual `state -> partial update` node function.
   That is what makes this testable without a live LLM or a real order
   table - see the fixtures at the bottom of this file, used by
   tests/test_workflow/test_extract.py and scripts/run_extract.py.

3. Customer/order resolution follows M3's decision #4 EXACTLY: the ticket's
   own intake metadata (already sitting in `state["customer_ref"]` /
   `state["order_id"]`, seeded from records.jsonl by the caller - see
   `seed_state()`'s docstring) is authoritative. `resolve_customer_ref` is
   invoked ONLY as a fallback, and ONLY the customer_ref side needs a
   fallback tool call (order_lookup) - order_id has no equivalent lookup,
   so an unresolved order_id just stays unresolved.

Production wiring:

    from core.llm_client import LLMClient
    from intake.parser import parse_ticket_safe as _m1_parse_ticket_safe
    from tools.order_lookup import order_lookup as _m2_order_lookup

    llm_client = LLMClient()
    extract_node = build_extract_node(
        parse_ticket=make_m1_parser_adapter(llm_client, _m1_parse_ticket_safe),
        resolve_customer_ref=make_m3_resolver_adapter(_m2_order_lookup),
    )

Confirmed against the REAL source (core/llm_client.py, intake/parser.py, as
of this build):
    - `parse_ticket_safe(ticket_id, raw_text, client, max_retries=2) ->
      tuple[SupportTicket | None, list[str]]` - signature matches exactly
      what `make_m1_parser_adapter` calls below.
    - `LLMClient()` takes no required args (reads `LLM_PROVIDER` /
      `GEMINI_API_KEY` / `OPENAI_API_KEY` etc. from the environment) - so
      `LLMClient()` at the call site above is correct, not a placeholder.
    - On success AFTER one or more repair-loop retries, `parse_ticket_safe`
      returns `(ticket, errors)` with `errors` NON-EMPTY - a successful
      parse can still carry warnings. `make_m1_parser_adapter` passes those
      through unchanged, and `extract`'s audit_log line for the
      "parsed with N repair-loop warning(s)" case (see below) exists
      specifically to surface this. See
      test_extract_adapters.py::test_adapter_passes_through_success_after_repair_loop_warnings.
    - `ticket.raw_text`/`ticket.ticket_id` are overwritten from the source
      args before being returned, never trusted from the model - so this
      adapter doesn't need to re-guard against that; M1 already does.

Still UNCONFIRMED (not uploaded/available while building this - flagged the
same way M2/M3 flagged their own open assumptions, not silently assumed):
    - `core/schema.py`'s actual `SupportTicket` field names/enum values -
      still going on the M1 summary's description, not the source. If a
      field name here turns out to differ, only the two constants near the
      top of the fixtures section below and this docstring's field list
      need to change - the node logic itself (extract()) never hardcodes a
      field name from SupportTicket.
    - `tools/order_lookup.py`'s real signature/return shape for
      `make_m3_resolver_adapter`.
"""

import re
from typing import Callable, Optional

from workflow.state import TicketReviewState

ParseTicketFn = Callable[[str, str], "tuple[dict, list[str]]"]
ResolveCustomerRefFn = Callable[[Optional[str]], Optional[str]]


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def build_extract_node(
    parse_ticket: ParseTicketFn,
    resolve_customer_ref: ResolveCustomerRefFn,
) -> Callable[[TicketReviewState], dict]:
    """Returns the `extract` node: `TicketReviewState -> partial update`.

    `parse_ticket(ticket_id, raw_text) -> (dict, list[str])` must return an
    already-plain-dict, SupportTicket-shaped result ({} on failure) plus any
    repair-loop warnings/errors - the pydantic v1/v2 handling and enum
    stringification happen inside the adapter (`make_m1_parser_adapter`),
    not in this node, so the node itself never imports pydantic.

    `resolve_customer_ref(order_id) -> Optional[str]` is the FALLBACK path
    only. The primary path (state's own customer_ref) is read directly
    below and never goes through this callable.
    """

    def extract(state: TicketReviewState) -> dict:
        parsed, errors = parse_ticket(state["ticket_id"], state["raw_text"])

        if not parsed:
            reason = "; ".join(errors) if errors else "no errors reported"
            return {
                "parsed_ticket": {},
                "audit_log": [
                    f"extract: parser returned no structured ticket ({reason})"
                ],
            }

        audit: list[str] = []
        if errors:
            audit.append(
                f"extract: parsed with {len(errors)} repair-loop warning(s): {errors}"
            )

        # M3 decision #4: known intake metadata beats anything derived from
        # free text. order_id: prefer the record's own order_ref (already in
        # state) over the LLM-parsed order_id, which M3 found is frequently
        # None. customer_ref: same priority, but customer_ref has a real
        # fallback (order_lookup) when intake metadata didn't carry one.
        order_id = state.get("order_id") or parsed.get("order_id")

        customer_ref = state.get("customer_ref")
        if customer_ref:
            resolution_note = "from intake metadata"
        elif order_id:
            customer_ref = resolve_customer_ref(order_id)
            resolution_note = "via order_lookup fallback" if customer_ref else "unresolved after fallback"
        else:
            resolution_note = "unresolved (no order_id to fall back on)"

        audit.append(
            f"extract: issue_type={parsed.get('issue_type')} "
            f"urgency={parsed.get('urgency')} sentiment={parsed.get('sentiment')} "
            f"customer_ref={resolution_note}"
        )

        return {
            "parsed_ticket": parsed,
            "customer_ref": customer_ref,
            "order_id": order_id,
            "audit_log": audit,
        }

    return extract


# ---------------------------------------------------------------------------
# Production adapters - wrap the REAL M1/M2/M3 code. Not exercised in this
# sandbox (no live LLM, no real order table here); wire these in the actual
# repo and let tests/test_workflow/test_extract.py's fixture-based cases
# stand in until then.
# ---------------------------------------------------------------------------

def make_m1_parser_adapter(llm_client, parse_ticket_safe) -> ParseTicketFn:
    """Wrap M1's `intake.parser.parse_ticket_safe(ticket_id, raw_text,
    client, max_retries=2) -> (SupportTicket | None, list[str])`.

    `parse_ticket_safe` is passed in (not imported here) so this module has
    zero hard dependency on the real M1 package existing - swap in the real
    import at the call site shown in the module docstring above.
    """

    def parse(ticket_id: str, raw_text: str):
        ticket, errors = parse_ticket_safe(ticket_id, raw_text, llm_client)
        if ticket is None:
            return {}, errors

        # Same v1/v2 compatibility handling as M2's scripts/run_agent.py's
        # _model_to_dict(), plus stringifying enums the same way, so
        # format_ticket_for_agent-style code downstream sees "REFUND" not
        # "IssueType.REFUND".
        data = ticket.model_dump(mode="json") if hasattr(ticket, "model_dump") else ticket.dict()
        for key, value in list(data.items()):
            if hasattr(value, "value"):
                data[key] = value.value
        return data, errors

    return parse


def make_m3_resolver_adapter(order_lookup) -> ResolveCustomerRefFn:
    """Wrap M2's `tools.order_lookup.order_lookup(order_ref)` as the
    fallback customer-resolution path M3 describes. Never trusted as the
    primary source - see the node's docstring."""

    def resolve(order_id: Optional[str]) -> Optional[str]:
        if not order_id:
            return None
        try:
            result = order_lookup(order_id)
        except Exception:
            return None
        return (result or {}).get("customer_ref")

    return resolve


# ---------------------------------------------------------------------------
# Offline fixtures - deterministic, no LLM, no order table. Used by
# tests/test_workflow/test_extract.py and scripts/run_extract.py. Same
# fallback spirit as the notebook's `ask()` returning `None`: NOT a
# substitute for the real M1 parser, just enough structure to keep this
# node's own logic (not the parser's accuracy) testable and demoable
# without live dependencies.
# ---------------------------------------------------------------------------

_REFUND_WORDS = ("refund", "money back", "reimburse")
_DELIVERY_WORDS = ("deliver", "shipping", "shipment", "late", "tracking")
_PRODUCT_WORDS = ("broken", "defect", "damaged", "wrong item", "doesn't work", "not working")
_THREAT_WORDS = ("lawyer", "legal action", "sue", "consumer court")
_ANGRY_WORDS = ("furious", "unacceptable", "terrible", "ridiculous")
_FRUSTRATED_WORDS = ("disappointed", "frustrated", "again", "still waiting")
_URGENT_WORDS = ("urgent", "immediately", "asap", "unsafe", "injur", "hazard")
_INJECTION_PHRASES = ("ignore previous", "ignore all previous", "system override", "you are now", "disregard the above")
_AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹)\s?([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def fixture_parse_ticket(ticket_id: str, raw_text: str) -> "tuple[dict, list[str]]":
    """Deterministic, keyword-based stand-in for M1's real LLM parser.

    Deliberately never guesses an `order_id` from free text - M1's real
    parser can (imperfectly), but a keyword fixture inventing order numbers
    would be a worse stand-in than admitting it doesn't know. `confidence`
    is pinned low (0.5) precisely so nothing downstream mistakes this for a
    real accuracy signal.
    """
    if not raw_text.strip():
        return {}, ["empty raw_text"]

    text = raw_text.lower()

    if any(w in text for w in _REFUND_WORDS):
        issue_type = "REFUND"
    elif any(w in text for w in _DELIVERY_WORDS):
        issue_type = "DELIVERY"
    elif any(w in text for w in _PRODUCT_WORDS):
        issue_type = "PRODUCT"
    else:
        issue_type = "ORDER"

    if any(w in text for w in _THREAT_WORDS):
        sentiment = "threatening"
    elif any(w in text for w in _ANGRY_WORDS):
        sentiment = "angry"
    elif any(w in text for w in _FRUSTRATED_WORDS):
        sentiment = "frustrated"
    else:
        sentiment = "neutral"

    if any(w in text for w in _URGENT_WORDS) or sentiment == "threatening":
        urgency = "high"
    elif issue_type in ("REFUND", "DELIVERY"):
        urgency = "medium"
    else:
        urgency = "low"

    amount_match = _AMOUNT_RE.search(text)
    claimed_refund_amount = float(amount_match.group(1).replace(",", "")) if amount_match else None

    contains_suspicious_instructions = any(p in text for p in _INJECTION_PHRASES)

    return {
        "issue_type": issue_type,
        "order_id": None,
        "sentiment": sentiment,
        "urgency": urgency,
        "claimed_refund_amount": claimed_refund_amount,
        "contains_suspicious_instructions": contains_suspicious_instructions,
        "confidence": 0.5,
    }, []


def fixture_resolve_customer_ref(order_id: Optional[str]) -> Optional[str]:
    """No mock order table wired in this sandbox - the fallback path always
    reports 'unresolved'. Fine for demo/tests: real records.jsonl tickets
    take the PRIMARY path (state's own customer_ref) per M3 decision #4, so
    this fallback exists to prove the node handles 'nothing to fall back on'
    gracefully rather than crashing - not to prove real order lookups work."""
    return None