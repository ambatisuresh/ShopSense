"""
Mock refund/replace API.

This is the one tool that must never trust a customer-claimed amount at face
value (SupportTicket.claimed_refund_amount can be inflated or fabricated —
see golden_set.json's injection/guardrail cases). Every request is
cross-checked against refund_calculator's policy-eligible amount and routed
through refund-authority.md's approval matrix + fraud-abuse.md's triggers.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from tools.data_loader import customers_by_ref, orders_by_ref, refunds_by_order_ref
from tools.refund_calculator import Condition, calculate_refund

Action = Literal["refund", "replace", "goodwill_credit"]

# refund-authority.md 4.1
AUTO_APPROVAL_CAP_INR = 2000
TEAM_LEAD_CAP_INR = 10_000
OPS_MANAGER_CAP_INR = 50_000
GOODWILL_CREDIT_CAP_INR = 1000  # 4.3.1

# fraud-abuse.md 2.1
SERIAL_REQUESTER_THRESHOLD = 5
SERIAL_REQUESTER_WINDOW_DAYS = 30


def _approval_tier(amount_inr: int) -> str:
    if amount_inr <= AUTO_APPROVAL_CAP_INR:
        return "auto"
    if amount_inr <= TEAM_LEAD_CAP_INR:
        return "team_lead"
    if amount_inr <= OPS_MANAGER_CAP_INR:
        return "ops_manager"
    return "finance"


def _is_serial_requester(customer_ref: str, as_of: datetime) -> bool:
    """fraud-abuse.md 2.1 — >5 refund requests by the same customer in a rolling 30 days."""
    window_start = as_of - timedelta(days=SERIAL_REQUESTER_WINDOW_DAYS)
    count = 0
    for order_ref, order in orders_by_ref().items():
        if order["customer_ref"] != customer_ref:
            continue
        for r in refunds_by_order_ref().get(order_ref, []):
            try:
                requested = datetime.fromisoformat(r["requested_on"])
            except ValueError:
                continue
            if window_start <= requested <= as_of:
                count += 1
    return count > SERIAL_REQUESTER_THRESHOLD


def process_refund(
    order_ref: str,
    action: Action,
    requested_amount_inr: int,
    reason_code: str,
    condition: Condition = "unopened",
    safety_or_legal_concern: bool = False,
) -> dict[str, Any]:
    """
    Process a refund, replacement, or goodwill-credit request.

    Args:
        order_ref: e.g. 'KW-O-000123'
        action: 'refund' | 'replace' | 'goodwill_credit'
        requested_amount_inr: what the customer is asking for (verified, not trusted)
        reason_code: short code, e.g. 'defective', 'wrong_item', 'not_delivered'
        condition: passed through to refund_calculator
        safety_or_legal_concern: True if the ticket mentions physical harm, product
            safety, legal action, or a regulatory complaint (escalation-tone.md 4.3.4/4.3.5).
            The agent must set this from ticket content — this tool can't read raw_text.

    Returns:
        {status, approval_tier, approved_amount_inr, requested_amount_inr,
         eligible_amount_inr, reason, requires_human}
        status is one of: 'auto_approved', 'requires_human_review', 'denied'
    """
    order = orders_by_ref().get(order_ref)
    if order is None:
        return {"status": "denied", "order_ref": order_ref, "reason": f"No order found with reference '{order_ref}'."}

    now = datetime.now()
    customer = customers_by_ref().get(order["customer_ref"], {})

    if action == "goodwill_credit":
        if requested_amount_inr > GOODWILL_CREDIT_CAP_INR:
            return {
                "status": "requires_human_review",
                "approval_tier": "team_lead",
                "requested_amount_inr": requested_amount_inr,
                "reason": f"Goodwill credit exceeds the ₹{GOODWILL_CREDIT_CAP_INR} cap (refund-authority.md 4.3.1).",
                "requires_human": True,
            }
        return {
            "status": "auto_approved",
            "approval_tier": "auto",
            "approved_amount_inr": requested_amount_inr,
            "requested_amount_inr": requested_amount_inr,
            "reason": "Goodwill credit within cap; note this cannot be combined with a full refund on this order (4.3.2).",
            "requires_human": False,
        }

    calc = calculate_refund(order_ref, condition=condition)
    if not calc.get("eligible"):
        return {
            "status": "denied",
            "order_ref": order_ref,
            "requested_amount_inr": requested_amount_inr,
            "eligible_amount_inr": 0,
            "reason": calc.get("reason", "Not eligible per policy."),
            "requires_human": False,
        }

    eligible_amount = calc["eligible_amount_inr"]
    # Cross-check: never take the requested amount at face value — cap at what
    # policy actually supports, and flag the mismatch for human review rather
    # than silently correcting it.
    amount_mismatch = requested_amount_inr > eligible_amount
    verdict_amount = min(requested_amount_inr, eligible_amount)

    fraud_flag = bool(customer.get("flagged_for_abuse")) or _is_serial_requester(order["customer_ref"], now)

    if fraud_flag:
        # fraud-abuse.md 4.1 — escalate to Finance regardless of amount.
        # Do NOT surface the fraud flag itself to the customer (fraud-abuse.md 3.1) —
        # the agent should phrase this as a routine account review, nothing more.
        return {
            "status": "requires_human_review",
            "approval_tier": "finance",
            "order_ref": order_ref,
            "requested_amount_inr": requested_amount_inr,
            "eligible_amount_inr": eligible_amount,
            "reason": "Escalated for account review (internal: fraud/abuse trigger — do not disclose to customer).",
            "requires_human": True,
        }

    if safety_or_legal_concern:
        # escalation-tone.md 4.3.4 / 4.3.5 — immediate human escalation regardless of amount.
        return {
            "status": "requires_human_review",
            "approval_tier": _approval_tier(verdict_amount),
            "order_ref": order_ref,
            "requested_amount_inr": requested_amount_inr,
            "eligible_amount_inr": eligible_amount,
            "reason": "Escalated: ticket references safety, injury, legal action, or a regulatory complaint.",
            "requires_human": True,
        }

    tier = _approval_tier(verdict_amount)
    if tier == "auto":
        return {
            "status": "auto_approved",
            "approval_tier": "auto",
            "action": action,
            "order_ref": order_ref,
            "approved_amount_inr": verdict_amount,
            "requested_amount_inr": requested_amount_inr,
            "eligible_amount_inr": eligible_amount,
            "amount_mismatch": amount_mismatch,
            "reason": calc["reason"],
            "requires_human": False,
        }

    return {
        "status": "requires_human_review",
        "approval_tier": tier,
        "action": action,
        "order_ref": order_ref,
        "requested_amount_inr": requested_amount_inr,
        "eligible_amount_inr": eligible_amount,
        "amount_mismatch": amount_mismatch,
        "reason": f"₹{verdict_amount} exceeds the ₹{AUTO_APPROVAL_CAP_INR} auto-approval cap (refund-authority.md 4.1/4.2).",
        "requires_human": True,
    }