"""
Mock shipping-tracker API.

Encodes shipping-policy.md Section 3 (delay compensation ladder) and Section 4
(lost-in-transit threshold) so the agent gets a policy-grounded verdict, not
just raw tracking events, and doesn't have to do date arithmetic itself.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from tools.data_loader import orders_by_ref, shipments_by_order_ref

LOST_IN_TRANSIT_DAYS = 15


def _delay_compensation_pct(days_late: int) -> float:
    # shipping-policy.md 3.1
    if days_late <= 0:
        return 0.0
    if days_late <= 2:
        return 0.10
    if days_late <= 4:
        return 0.25
    return 0.50


def track_shipment(order_ref: str) -> dict[str, Any]:
    """
    Track a shipment by order reference.

    Returns:
        {found: bool}
        if the order has no shipment record yet: {found: True, shipped: False, ...}
        if shipped: carrier, last_scan, last_scan_at, scan_location, is_delivered,
        days_late (vs. delivery_promise_days), delay_compensation_pct, is_lost_in_transit
    """
    order = orders_by_ref().get(order_ref)
    if order is None:
        return {"found": False, "order_ref": order_ref, "reason": f"No order found with reference '{order_ref}'."}

    shipment = shipments_by_order_ref().get(order_ref)
    if shipment is None:
        return {
            "found": True,
            "order_ref": order_ref,
            "shipped": False,
            "status": order["status"],
            "reason": "No shipment record yet for this order.",
        }

    is_delivered = shipment["last_scan"] == "delivered"
    days_late = 0
    is_lost_in_transit = False
    days_late_unknown = False

    if is_delivered:
        if order.get("actual_delivery_days") is not None:
            days_late = max(0, order["actual_delivery_days"] - order["delivery_promise_days"])
        else:
            # dirty-data fallback: derive actual delivery days from the shipment scan
            try:
                placed = datetime.fromisoformat(order["placed_at"])
                last_scan = datetime.fromisoformat(shipment["last_scan_at"])
                actual_days = (last_scan - placed).days
                days_late = max(0, actual_days - order["delivery_promise_days"])
            except (ValueError, TypeError):
                days_late = 0
                days_late_unknown = True
    else:
        try:
            placed = datetime.fromisoformat(order["placed_at"])
            last_scan = datetime.fromisoformat(shipment["last_scan_at"])
            elapsed_days = (last_scan - placed).days
            days_late = max(0, elapsed_days - order["delivery_promise_days"])
            is_lost_in_transit = elapsed_days >= LOST_IN_TRANSIT_DAYS
        except (ValueError, TypeError):
            pass  # unparseable timestamps -> leave conservative defaults

    return {
        "found": True,
        "order_ref": order_ref,
        "shipped": True,
        "carrier": shipment["carrier"],
        "last_scan": shipment["last_scan"],
        "last_scan_at": shipment["last_scan_at"],
        "scan_location": shipment["scan_location"],
        "is_delivered": is_delivered,
        "delivery_promise_days": order["delivery_promise_days"],
        "days_late": days_late,
        "days_late_unknown": days_late_unknown,
        "delay_compensation_pct": _delay_compensation_pct(days_late) if not days_late_unknown else None,
        "is_lost_in_transit": is_lost_in_transit,
    }