"""
Refund-amount calculator.

Deterministic policy math — return-window eligibility and the condition-based
partial-refund schedule from returns-policy.md — kept out of the model's hands
entirely, same rationale as calculator tool. This is a pure
function: no mock API, no network, just orders/products data + policy constants.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from tools.data_loader import orders_by_ref, products_by_sku, shipments_by_order_ref

Condition = Literal["unopened", "opened_unused", "opened_used", "damaged"]

# returns-policy.md 2.2 — category-specific return windows (days). Anything not
# listed falls back to the 30-day standard window (2.1).
CATEGORY_RETURN_WINDOW_DAYS: dict[str, int | None] = {
    "Electronics": 15,
    "Groceries": 0,  # non-returnable, 2.2b
    "Apparel": 45,
}
STANDARD_RETURN_WINDOW_DAYS = 30

# returns-policy.md 2.5 — partial refund schedule for opened items
CONDITION_REFUND_PCT: dict[Condition, float] = {
    "unopened": 1.00,
    "opened_unused": 0.75,
    "opened_used": 0.50,
    "damaged": 0.0,
}


def _resolve_delivery_date(order: dict) -> datetime | None:
    """Prefer actual_delivery_days when present; fall back to the shipment's
    'delivered' scan timestamp when the orders table has a null (dirty-data case)."""
    placed = datetime.fromisoformat(order["placed_at"])
    if order.get("actual_delivery_days") is not None:
        return placed + timedelta(days=order["actual_delivery_days"])
    shipment = shipments_by_order_ref().get(order["order_ref"])
    if shipment and shipment.get("last_scan") == "delivered" and shipment.get("last_scan_at"):
        try:
            return datetime.fromisoformat(shipment["last_scan_at"])
        except ValueError:
            return None
    return None


def calculate_refund(
    order_ref: str,
    condition: Condition = "unopened",
    as_of: str | None = None,
) -> dict[str, Any]:
    """
    Compute the policy-eligible refund amount for an order.

    Args:
        order_ref: e.g. 'KW-O-000123'
        condition: 'unopened' | 'opened_unused' | 'opened_used' | 'damaged'.
            Infer this from the ticket text; default to 'unopened' only when
            the ticket gives no signal either way.
        as_of: ISO datetime to evaluate the return window against; defaults to now.

    Returns:
        {found, eligible, eligible_amount_inr, refund_pct, reason,
         category, return_window_days, days_since_delivery}
    """
    order = orders_by_ref().get(order_ref)
    if order is None:
        return {"found": False, "order_ref": order_ref, "reason": f"No order found with reference '{order_ref}'."}

    product = products_by_sku().get(order["sku"], {})
    category = product.get("category", "Unknown")
    window_days = CATEGORY_RETURN_WINDOW_DAYS.get(category, STANDARD_RETURN_WINDOW_DAYS)

    if order["status"] != "delivered":
        return {
            "found": True,
            "order_ref": order_ref,
            "eligible": False,
            "eligible_amount_inr": 0,
            "refund_pct": 0.0,
            "category": category,
            "reason": f"Order status is '{order['status']}', not 'delivered' — not yet eligible for return.",
        }

    if window_days == 0:
        return {
            "found": True,
            "order_ref": order_ref,
            "eligible": False,
            "eligible_amount_inr": 0,
            "refund_pct": 0.0,
            "category": category,
            "reason": f"{category} items are non-returnable and non-refundable (returns-policy.md 2.2b).",
        }

    now = datetime.fromisoformat(as_of) if as_of else datetime.now()
    delivered_at = _resolve_delivery_date(order)
    if delivered_at is None:
        return {
            "found": True,
            "order_ref": order_ref,
            "eligible": False,
            "eligible_amount_inr": 0,
            "refund_pct": 0.0,
            "category": category,
            "reason": (
                "Order is marked delivered but no delivery date could be determined "
                "(missing actual_delivery_days and no shipment scan on record). "
                "Route to human review to confirm delivery date before proceeding."
            ),
        }
    days_since_delivery = (now - delivered_at).days

    if days_since_delivery > window_days:
        return {
            "found": True,
            "order_ref": order_ref,
            "eligible": False,
            "eligible_amount_inr": 0,
            "refund_pct": 0.0,
            "category": category,
            "return_window_days": window_days,
            "days_since_delivery": days_since_delivery,
            "reason": (
                f"{days_since_delivery} days since delivery exceeds the {window_days}-day "
                f"return window for {category} (returns-policy.md 2.1/2.2)."
            ),
        }

    if condition == "damaged":
        return {
            "found": True,
            "order_ref": order_ref,
            "eligible": False,
            "eligible_amount_inr": 0,
            "refund_pct": 0.0,
            "category": category,
            "return_window_days": window_days,
            "days_since_delivery": days_since_delivery,
            "reason": "Heavily used or damaged items are not refundable (returns-policy.md 2.5.iii).",
        }

    pct = CONDITION_REFUND_PCT[condition]
    eligible_amount = round(order["order_value_inr"] * pct)

    return {
        "found": True,
        "order_ref": order_ref,
        "eligible": True,
        "eligible_amount_inr": eligible_amount,
        "refund_pct": pct,
        "category": category,
        "return_window_days": window_days,
        "days_since_delivery": days_since_delivery,
        "reason": f"Eligible for {int(pct * 100)}% refund based on condition '{condition}' (returns-policy.md 2.5).",
    }