"""
Loads and indexes the mock data tables (orders, shipments, refunds,
customers, products) so the tools can do O(1) lookups instead of scanning
JSON lists on every call.

DATA_DIR defaults to data/mock_api relative to the repo root; override with the
SHOPSENSE_DATA_DIR env var if you keep the tables elsewhere.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("SHOPSENSE_DATA_DIR", "data/mock_api"))


def _load(filename: str) -> list[dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Mock data file not found: {path}. "
            f"Set SHOPSENSE_DATA_DIR if your tables live somewhere else."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def orders_by_ref() -> dict[str, dict]:
    #return {row["order_ref"]: row for row in _load("orders.json")}
    result = {}
    for row in _load("orders.json"):
        result[row["order_ref"]] = row
    return result


@lru_cache(maxsize=1)
def shipments_by_order_ref() -> dict[str, dict]:
    # one shipment per order in this dataset; if that ever changes, switch to list-valued
    return {row["order_ref"]: row for row in _load("shipments.json")}


@lru_cache(maxsize=1)
def refunds_by_order_ref() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in _load("refunds.json"):
        out.setdefault(row["order_ref"], []).append(row)
    return out


@lru_cache(maxsize=1)
def customers_by_ref() -> dict[str, dict]:
    return {row["customer_ref"]: row for row in _load("customers.json")}


@lru_cache(maxsize=1)
def products_by_sku() -> dict[str, dict]:
    return {row["sku"]: row for row in _load("products.json")}


def clear_cache() -> None:
    """Call after editing the underlying JSON files mid-session (e.g. in tests)."""
    for fn in (orders_by_ref, shipments_by_order_ref, refunds_by_order_ref,
               customers_by_ref, products_by_sku):
        fn.cache_clear()