"""
Mock order-lookup API.

Returns a typed "not found" result rather than raising, because SupportTicket.order_id
is Optional and frequently wrong or missing (see records.jsonl e.g. SHOPSENSE-00004,
order_ref: null) — the agent has to be able to handle that as data, not a crash.
"""
from __future__ import annotations

from typing import Any

from tools.data_loader import customers_by_ref, orders_by_ref, products_by_sku


def order_lookup(order_ref: str) -> dict[str, Any]:
    """
    Look up an order by its reference (e.g. 'KW-O-000123').

    Returns:
        {found: bool}
        if found, also: order_ref, customer_ref, customer_tier, sku, product_title,
        product_category, quantity, order_value_inr, placed_at, payment_method,
        status, delivery_promise_days, actual_delivery_days
        if not found: {found: False, order_ref, reason}
    """
    order = orders_by_ref().get(order_ref)
    if order is None:
        return {
            "found": False,
            "order_ref": order_ref,
            "reason": f"No order found with reference '{order_ref}'.",
        }

    product = products_by_sku().get(order["sku"], {})
    customer = customers_by_ref().get(order["customer_ref"], {})

    return {
        "found": True,
        "order_ref": order["order_ref"],
        "customer_ref": order["customer_ref"],
        "customer_tier": customer.get("tier"),
        "sku": order["sku"],
        "product_title": product.get("title"),
        "product_category": product.get("category"),
        "quantity": order["quantity"],
        "order_value_inr": order["order_value_inr"],
        "placed_at": order["placed_at"],
        "payment_method": order["payment_method"],
        "status": order["status"],
        "delivery_promise_days": order["delivery_promise_days"],
        "actual_delivery_days": order["actual_delivery_days"],
    }