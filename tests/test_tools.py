"""
Tests for tools/order_lookup.py, shipping_tracker.py, refund_calculator.py,
refund_replace.py.

Design note: these tests do NOT read the real orders.json/etc. They monkeypatch
each tool module's *already-imported* data-access functions (e.g.
tools.order_lookup.orders_by_ref) with small, hand-built fixtures instead.

Why: the real dataset is 1500 randomly-generated rows that can be regenerated
at any point in the course. A test that says "find some delivered order and
assert its refund is denied" is really testing whatever the generator happened
to produce that day -- fragile, and it doesn't document intent. A test that
builds one exact order, with an exact `order_value_inr` and an exact
`placed_at` relative to "now", and asserts an exact expected output, is
testing the *policy logic itself*, independent of the data.

Import path note: each tool does `from tools.data_loader import orders_by_ref`,
which binds a *local* name in that tool's own module namespace. Patching
tools.data_loader.orders_by_ref would NOT affect tools.order_lookup, because
order_lookup already has its own reference to the original function. So every
monkeypatch below targets the tool module directly, e.g.
"tools.order_lookup.orders_by_ref" -- not "tools.data_loader.orders_by_ref".
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tools.order_lookup import order_lookup
from tools.refund_calculator import calculate_refund
from tools.refund_replace import process_refund
from tools.shipping_tracker import track_shipment

NOW = datetime.now()


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="minutes")


# ---------------------------------------------------------------------------
# Fixture builders -- small, explicit, hand-built rows (not real dataset rows)
# ---------------------------------------------------------------------------

def make_order(
    order_ref="KW-O-TEST01",
    customer_ref="KW-C-TEST01",
    sku="KW-SKU-TEST01",
    order_value_inr=10_000,
    status="delivered",
    delivery_promise_days=5,
    actual_delivery_days=3,
    placed_days_ago=10,
) -> dict:
    return {
        "order_ref": order_ref,
        "customer_ref": customer_ref,
        "sku": sku,
        "quantity": 1,
        "order_value_inr": order_value_inr,
        "placed_at": iso(NOW - timedelta(days=placed_days_ago)),
        "payment_method": "card",
        "status": status,
        "delivery_promise_days": delivery_promise_days,
        "actual_delivery_days": actual_delivery_days,
    }


def make_product(sku="KW-SKU-TEST01", category="Electronics", title="Test Widget") -> dict:
    return {
        "sku": sku, "title": title, "category": category,
        "price_inr": 10_000, "seller_id": "KW-S-TEST", "in_stock": True, "warranty_months": 12,
    }


def make_customer(customer_ref="KW-C-TEST01", flagged_for_abuse=False, tier="standard") -> dict:
    return {
        "customer_ref": customer_ref, "display_name": "Test Customer", "city": "Testville",
        "tier": tier, "joined_on": "2020-01-01", "lifetime_orders": 10,
        "return_rate": 0.05, "flagged_for_abuse": flagged_for_abuse,
    }


def make_shipment(order_ref="KW-O-TEST01", last_scan="delivered", last_scan_at=None, carrier="Bluedart") -> dict:
    return {
        "tracking_id": "KW-TRK-TEST01", "order_ref": order_ref, "carrier": carrier,
        "last_scan": last_scan,
        "last_scan_at": last_scan_at or iso(NOW - timedelta(days=7)),
        "scan_location": "Testville",
    }


# ---------------------------------------------------------------------------
# order_lookup.py
# ---------------------------------------------------------------------------

class TestOrderLookup:
    def test_found_returns_joined_fields(self, monkeypatch):
        order = make_order()
        monkeypatch.setattr("tools.order_lookup.orders_by_ref", lambda: {order["order_ref"]: order})
        monkeypatch.setattr("tools.order_lookup.products_by_sku", lambda: {order["sku"]: make_product()})
        monkeypatch.setattr("tools.order_lookup.customers_by_ref", lambda: {order["customer_ref"]: make_customer()})

        result = order_lookup(order["order_ref"])
        print(result)

        assert result["found"] is True
        assert result["order_value_inr"] == 10_000
        assert result["product_title"] == "Test Widget"
        assert result["product_category"] == "Electronics"
        assert result["customer_tier"] == "standard"

    def test_not_found_does_not_raise(self, monkeypatch):
        monkeypatch.setattr("tools.order_lookup.orders_by_ref", lambda: {})
        monkeypatch.setattr("tools.order_lookup.products_by_sku", lambda: {})
        monkeypatch.setattr("tools.order_lookup.customers_by_ref", lambda: {})

        result = order_lookup("KW-O-NOPE")

        assert result["found"] is False
        assert "KW-O-NOPE" in result["reason"]

    def test_found_order_with_missing_product_does_not_crash(self, monkeypatch):
        """orders_by_ref has a sku that products_by_sku doesn't -- dirty-join case."""
        order = make_order()
        monkeypatch.setattr("tools.order_lookup.orders_by_ref", lambda: {order["order_ref"]: order})
        monkeypatch.setattr("tools.order_lookup.products_by_sku", lambda: {})  # sku missing
        monkeypatch.setattr("tools.order_lookup.customers_by_ref", lambda: {order["customer_ref"]: make_customer()})

        result = order_lookup(order["order_ref"])

        assert result["found"] is True
        assert result["product_title"] is None
        assert result["product_category"] is None


# ---------------------------------------------------------------------------
# shipping_tracker.py
# ---------------------------------------------------------------------------

class TestShippingTracker:
    def test_order_not_found(self, monkeypatch):
        monkeypatch.setattr("tools.shipping_tracker.orders_by_ref", lambda: {})
        monkeypatch.setattr("tools.shipping_tracker.shipments_by_order_ref", lambda: {})

        result = track_shipment("KW-O-NOPE")
        assert result["found"] is False

    def test_no_shipment_record_yet(self, monkeypatch):
        order = make_order(status="placed")
        monkeypatch.setattr("tools.shipping_tracker.orders_by_ref", lambda: {order["order_ref"]: order})
        monkeypatch.setattr("tools.shipping_tracker.shipments_by_order_ref", lambda: {})

        result = track_shipment(order["order_ref"])
        assert result["found"] is True
        assert result["shipped"] is False

    def test_delivered_on_time_no_compensation(self, monkeypatch):
        order = make_order(delivery_promise_days=7, actual_delivery_days=3)
        shipment = make_shipment(order["order_ref"], last_scan="delivered")
        monkeypatch.setattr("tools.shipping_tracker.orders_by_ref", lambda: {order["order_ref"]: order})
        monkeypatch.setattr("tools.shipping_tracker.shipments_by_order_ref", lambda: {order["order_ref"]: shipment})

        result = track_shipment(order["order_ref"])
        assert result["is_delivered"] is True
        assert result["days_late"] == 0
        assert result["delay_compensation_pct"] == 0.0

    def test_delivered_5_days_late_gets_50pct_compensation(self, monkeypatch):
        order = make_order(delivery_promise_days=3, actual_delivery_days=8)
        shipment = make_shipment(order["order_ref"], last_scan="delivered")
        monkeypatch.setattr("tools.shipping_tracker.orders_by_ref", lambda: {order["order_ref"]: order})
        monkeypatch.setattr("tools.shipping_tracker.shipments_by_order_ref", lambda: {order["order_ref"]: shipment})

        result = track_shipment(order["order_ref"])
        assert result["days_late"] == 5
        assert result["delay_compensation_pct"] == 0.50  # shipping-policy.md 3.1c

    def test_delivered_with_null_actual_delivery_days_falls_back_to_shipment_scan(self, monkeypatch):
        """The dirty-data case: actual_delivery_days is None, must derive from last_scan_at."""
        order = make_order(delivery_promise_days=3, actual_delivery_days=None, placed_days_ago=10)
        shipment = make_shipment(
            order["order_ref"], last_scan="delivered",
            last_scan_at=iso(NOW - timedelta(days=4)),  # took 6 days -> 3 days late
        )
        monkeypatch.setattr("tools.shipping_tracker.orders_by_ref", lambda: {order["order_ref"]: order})
        monkeypatch.setattr("tools.shipping_tracker.shipments_by_order_ref", lambda: {order["order_ref"]: shipment})

        result = track_shipment(order["order_ref"])
        assert result["days_late"] == 3
        assert result["days_late_unknown"] is False
        assert result["delay_compensation_pct"] == 0.25  # shipping-policy.md 3.1b

    def test_not_delivered_after_15_days_is_lost_in_transit(self, monkeypatch):
        order = make_order(status="shipped", delivery_promise_days=5, placed_days_ago=20)
        shipment = make_shipment(
            order["order_ref"], last_scan="in_transit",
            last_scan_at=iso(NOW - timedelta(days=1)),  # last scan recent, but placed 20 days ago
        )
        monkeypatch.setattr("tools.shipping_tracker.orders_by_ref", lambda: {order["order_ref"]: order})
        monkeypatch.setattr("tools.shipping_tracker.shipments_by_order_ref", lambda: {order["order_ref"]: shipment})

        result = track_shipment(order["order_ref"])
        assert result["is_delivered"] is False
        assert result["is_lost_in_transit"] is True  # shipping-policy.md 4.1


# ---------------------------------------------------------------------------
# refund_calculator.py
# ---------------------------------------------------------------------------

def patch_refund_calc_data(monkeypatch, order, product=None, shipment=None):
    monkeypatch.setattr("tools.refund_calculator.orders_by_ref", lambda: {order["order_ref"]: order})
    monkeypatch.setattr("tools.refund_calculator.products_by_sku",
                         lambda: {order["sku"]: product or make_product(order["sku"])})
    monkeypatch.setattr("tools.refund_calculator.shipments_by_order_ref",
                         lambda: {order["order_ref"]: shipment} if shipment else {})


class TestRefundCalculator:
    def test_order_not_found(self, monkeypatch):
        monkeypatch.setattr("tools.refund_calculator.orders_by_ref", lambda: {})
        monkeypatch.setattr("tools.refund_calculator.products_by_sku", lambda: {})
        monkeypatch.setattr("tools.refund_calculator.shipments_by_order_ref", lambda: {})

        result = calculate_refund("KW-O-NOPE")
        assert result["found"] is False

    def test_not_yet_delivered_is_ineligible(self, monkeypatch):
        order = make_order(status="shipped")
        patch_refund_calc_data(monkeypatch, order)

        result = calculate_refund(order["order_ref"])
        assert result["eligible"] is False
        assert "not 'delivered'" in result["reason"]

    def test_groceries_are_never_returnable(self, monkeypatch):
        order = make_order(status="delivered", placed_days_ago=2, actual_delivery_days=1)
        product = make_product(order["sku"], category="Groceries")
        patch_refund_calc_data(monkeypatch, order, product=product)

        result = calculate_refund(order["order_ref"])
        assert result["eligible"] is False
        assert "non-returnable" in result["reason"]

    def test_electronics_within_15_day_window_is_eligible(self, monkeypatch):
        order = make_order(status="delivered", placed_days_ago=10, actual_delivery_days=3)  # 7 days since delivery
        product = make_product(order["sku"], category="Electronics")
        patch_refund_calc_data(monkeypatch, order, product=product)

        result = calculate_refund(order["order_ref"], condition="unopened")
        assert result["eligible"] is True
        assert result["eligible_amount_inr"] == order["order_value_inr"]  # 100% unopened

    def test_electronics_past_15_day_window_is_ineligible(self, monkeypatch):
        order = make_order(status="delivered", placed_days_ago=20, actual_delivery_days=3)  # 17 days since delivery
        product = make_product(order["sku"], category="Electronics")
        patch_refund_calc_data(monkeypatch, order, product=product)

        result = calculate_refund(order["order_ref"])
        assert result["eligible"] is False
        assert "exceeds the 15-day" in result["reason"]

    def test_apparel_45_day_window_wider_than_electronics(self, monkeypatch):
        order = make_order(status="delivered", placed_days_ago=20, actual_delivery_days=3)  # 17 days since delivery
        product = make_product(order["sku"], category="Apparel")
        patch_refund_calc_data(monkeypatch, order, product=product)

        result = calculate_refund(order["order_ref"])
        assert result["eligible"] is True  # 17 days is fine within Apparel's 45-day window

    @pytest.mark.parametrize("condition,expected_pct", [
        ("unopened", 1.00),
        ("opened_unused", 0.75),
        ("opened_used", 0.50),
    ])
    def test_condition_based_partial_refund_schedule(self, monkeypatch, condition, expected_pct):
        order = make_order(status="delivered", placed_days_ago=5, actual_delivery_days=1, order_value_inr=8_000)
        patch_refund_calc_data(monkeypatch, order)

        result = calculate_refund(order["order_ref"], condition=condition)
        assert result["refund_pct"] == expected_pct
        assert result["eligible_amount_inr"] == round(8_000 * expected_pct)

    def test_damaged_condition_is_ineligible_even_within_window(self, monkeypatch):
        order = make_order(status="delivered", placed_days_ago=5, actual_delivery_days=1)
        patch_refund_calc_data(monkeypatch, order)

        result = calculate_refund(order["order_ref"], condition="damaged")
        assert result["eligible"] is False
        assert "damaged" in result["reason"].lower()

    def test_null_actual_delivery_days_falls_back_to_shipment_scan(self, monkeypatch):
        order = make_order(status="delivered", placed_days_ago=10, actual_delivery_days=None)
        shipment = make_shipment(order["order_ref"], last_scan="delivered",
                                  last_scan_at=iso(NOW - timedelta(days=5)))  # delivered 5 days ago
        patch_refund_calc_data(monkeypatch, order, shipment=shipment)

        result = calculate_refund(order["order_ref"])
        assert result["eligible"] is True
        assert result["days_since_delivery"] == 5

    def test_null_actual_delivery_days_and_no_shipment_scan_routes_to_human(self, monkeypatch):
        order = make_order(status="delivered", placed_days_ago=10, actual_delivery_days=None)
        patch_refund_calc_data(monkeypatch, order, shipment=None)

        result = calculate_refund(order["order_ref"])
        assert result["eligible"] is False
        assert "route to human review" in result["reason"].lower()


# ---------------------------------------------------------------------------
# refund_replace.py
# ---------------------------------------------------------------------------

def patch_refund_replace_data(monkeypatch, order, product=None, customer=None, refunds_by_order=None):
    """Patches BOTH refund_replace's own data access AND refund_calculator's
    (since process_refund calls calculate_refund internally, which reads from
    its own module-level names, not refund_replace's)."""
    monkeypatch.setattr("tools.refund_replace.orders_by_ref", lambda: {order["order_ref"]: order})
    monkeypatch.setattr("tools.refund_replace.customers_by_ref",
                         lambda: {order["customer_ref"]: customer or make_customer(order["customer_ref"])})
    monkeypatch.setattr("tools.refund_replace.refunds_by_order_ref", lambda: refunds_by_order or {})

    monkeypatch.setattr("tools.refund_calculator.orders_by_ref", lambda: {order["order_ref"]: order})
    monkeypatch.setattr("tools.refund_calculator.products_by_sku",
                         lambda: {order["sku"]: product or make_product(order["sku"])})
    monkeypatch.setattr("tools.refund_calculator.shipments_by_order_ref", lambda: {})


class TestProcessRefund:
    def test_order_not_found(self, monkeypatch):
        monkeypatch.setattr("tools.refund_replace.orders_by_ref", lambda: {})
        result = process_refund("KW-O-NOPE", "refund", 500, "defective")
        assert result["status"] == "denied"

    def test_small_amount_within_window_auto_approves(self, monkeypatch):
        order = make_order(placed_days_ago=5, actual_delivery_days=1, order_value_inr=8_000)
        patch_refund_replace_data(monkeypatch, order)

        result = process_refund(order["order_ref"], "refund", 1500, "defective")
        assert result["status"] == "auto_approved"
        assert result["approval_tier"] == "auto"
        assert result["approved_amount_inr"] == 1500

    def test_amount_above_2000_cap_requires_human(self, monkeypatch):
        order = make_order(placed_days_ago=5, actual_delivery_days=1, order_value_inr=24_999)
        patch_refund_replace_data(monkeypatch, order)

        result = process_refund(order["order_ref"], "refund", 24_999, "defective")
        assert result["status"] == "requires_human_review"
        assert result["approval_tier"] == "ops_manager"  # 10,001-50,000 tier

    def test_requested_amount_above_eligible_is_capped_not_trusted(self, monkeypatch):
        """The core security property: a customer asking for more than policy
        supports must never get that inflated amount approved."""
        order = make_order(placed_days_ago=5, actual_delivery_days=1, order_value_inr=1_500)
        patch_refund_replace_data(monkeypatch, order)

        result = process_refund(order["order_ref"], "refund", 48_000, "defective")
        assert result["amount_mismatch"] is True
        assert result["eligible_amount_inr"] == 1_500
        # requested (48,000) is never used as the approved/verdict basis
        assert result.get("approved_amount_inr", 0) != 48_000

    def test_ineligible_order_is_denied_regardless_of_amount(self, monkeypatch):
        order = make_order(placed_days_ago=40, actual_delivery_days=1)  # past 30-day window
        product = make_product(order["sku"], category="Home & Kitchen")
        patch_refund_replace_data(monkeypatch, order, product=product)

        result = process_refund(order["order_ref"], "refund", 500, "changed_mind")
        assert result["status"] == "denied"

    def test_safety_concern_forces_human_review_even_for_tiny_amount(self, monkeypatch):
        order = make_order(placed_days_ago=5, actual_delivery_days=1, order_value_inr=500)
        patch_refund_replace_data(monkeypatch, order)

        result = process_refund(order["order_ref"], "refund", 100, "defective", safety_or_legal_concern=True)
        assert result["status"] == "requires_human_review"
        assert result["requires_human"] is True

    def test_flagged_customer_escalates_to_finance_regardless_of_small_amount(self, monkeypatch):
        order = make_order(placed_days_ago=5, actual_delivery_days=1, order_value_inr=500)
        flagged_customer = make_customer(order["customer_ref"], flagged_for_abuse=True)
        patch_refund_replace_data(monkeypatch, order, customer=flagged_customer)

        result = process_refund(order["order_ref"], "refund", 500, "defective")
        assert result["status"] == "requires_human_review"
        assert result["approval_tier"] == "finance"
        assert "do not disclose" in result["reason"].lower()  # internal-only marker present

    def test_serial_requester_escalates_to_finance(self, monkeypatch):
        """fraud-abuse.md 2.1: >5 refund requests by the same customer in 30 days."""
        order = make_order(placed_days_ago=5, actual_delivery_days=1, order_value_inr=500)
        # 6 prior refund requests on this same order, all within the last 30 days
        prior_refunds = [
            {"refund_id": f"KW-RF-TEST0{i}", "order_ref": order["order_ref"], "amount_inr": 100,
             "reason_code": "test", "status": "processed", "approved_by": "auto",
             "requested_on": (NOW - timedelta(days=i)).date().isoformat()}
            for i in range(6)
        ]
        patch_refund_replace_data(monkeypatch, order, refunds_by_order={order["order_ref"]: prior_refunds})

        result = process_refund(order["order_ref"], "refund", 500, "defective")
        assert result["status"] == "requires_human_review"
        assert result["approval_tier"] == "finance"

    def test_goodwill_credit_within_cap_auto_approves(self, monkeypatch):
        order = make_order()
        patch_refund_replace_data(monkeypatch, order)

        result = process_refund(order["order_ref"], "goodwill_credit", 800, "goodwill")
        assert result["status"] == "auto_approved"

    def test_goodwill_credit_above_1000_cap_requires_human(self, monkeypatch):
        order = make_order()
        patch_refund_replace_data(monkeypatch, order)

        result = process_refund(order["order_ref"], "goodwill_credit", 1500, "goodwill")
        assert result["status"] == "requires_human_review"

    def test_goodwill_credit_ignores_eligibility_window(self, monkeypatch):
        """Goodwill credits don't route through calculate_refund at all -- an
        order past its return window can still get a goodwill credit."""
        order = make_order(placed_days_ago=40, actual_delivery_days=1)  # past 30-day window
        patch_refund_replace_data(monkeypatch, order)

        result = process_refund(order["order_ref"], "goodwill_credit", 500, "goodwill")
        assert result["status"] == "auto_approved"

