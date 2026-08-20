"""
ShopSense M5 - tests for the PRODUCTION adapters in workflow/nodes/extract.py
(make_m1_parser_adapter, make_m3_resolver_adapter).

These exercise the adapters' own logic - dict conversion, enum
stringification, error passthrough, None-safety - against FAKE
parse_ticket_safe / order_lookup callables shaped exactly like the real
ones, confirmed against the uploaded core/llm_client.py and
intake/parser.py (M1) source. No live LLM call, no real dataset - same
"fake stub standing in for a live dependency" convention M2/M3 used for
their own tool/agent tests.

What was CONFIRMED by reading the real intake/parser.py:
  - parse_ticket_safe(ticket_id, raw_text, client, max_retries=2)
      -> tuple[SupportTicket | None, list[str]]
  - a ticket recovered by the repair loop returns (ticket, errors) where
    `errors` is NON-EMPTY even though the parse ultimately succeeded -
    the loop appends to `errors` on every failed attempt BEFORE the
    eventual successful `return ticket, errors`.
  - final failure (all max_retries+1 attempts invalid) returns
    (None, errors).
  - ticket.raw_text / ticket.ticket_id are overwritten from the source
    args, never trusted from the model output.
"""

from workflow.nodes.extract import make_m1_parser_adapter, make_m3_resolver_adapter


# --------------------------------------------------------------------------
# Fakes shaped like a real SupportTicket, for the two pydantic API shapes
# the adapter has to support (v2's model_dump, v1's dict()).
# --------------------------------------------------------------------------

class _FakeEnum:
    """Stands in for a real Enum member (e.g. IssueType.REFUND) - anything
    with a `.value` attribute, which is the adapter's actual detection
    mechanism (`hasattr(value, "value")`)."""
    def __init__(self, value):
        self.value = value


class _FakeTicketV2:
    """Mimics pydantic v2: has model_dump(mode=...)."""
    def __init__(self, data):
        self._data = data

    def model_dump(self, mode="python"):
        return dict(self._data)


class _FakeTicketV1:
    """Mimics pydantic v1: no model_dump, only dict()."""
    def __init__(self, data):
        self._data = data

    def dict(self):
        return dict(self._data)


# --------------------------------------------------------------------------
# make_m1_parser_adapter
# --------------------------------------------------------------------------

def test_adapter_converts_successful_v2_style_ticket_to_plain_dict():
    ticket = _FakeTicketV2({
        "ticket_id": "T1", "raw_text": "raw", "issue_type": "REFUND",
        "order_id": None, "sentiment": "neutral", "urgency": "low",
        "claimed_refund_amount": None, "contains_suspicious_instructions": False,
        "confidence": 0.9,
    })

    def fake_parse_ticket_safe(ticket_id, raw_text, client, max_retries=2):
        return ticket, []

    parse = make_m1_parser_adapter(llm_client=object(), parse_ticket_safe=fake_parse_ticket_safe)
    data, errors = parse("T1", "raw")

    assert data["issue_type"] == "REFUND"
    assert data["confidence"] == 0.9
    assert errors == []


def test_adapter_falls_back_to_dict_for_v1_style_ticket():
    ticket = _FakeTicketV1({"issue_type": "ORDER", "confidence": 0.4})

    def fake_parse_ticket_safe(ticket_id, raw_text, client, max_retries=2):
        return ticket, []

    parse = make_m1_parser_adapter(llm_client=object(), parse_ticket_safe=fake_parse_ticket_safe)
    data, errors = parse("T1", "raw")

    assert data == {"issue_type": "ORDER", "confidence": 0.4}


def test_adapter_stringifies_enum_valued_fields():
    """If SupportTicket's model_dump ever leaves a live Enum member in a
    field (pydantic v1, or mode="python" instead of "json"), the adapter
    must still hand extract() a plain string - never "IssueType.REFUND"."""
    ticket = _FakeTicketV2({"issue_type": _FakeEnum("REFUND"), "confidence": 0.9})

    def fake_parse_ticket_safe(ticket_id, raw_text, client, max_retries=2):
        return ticket, []

    parse = make_m1_parser_adapter(llm_client=object(), parse_ticket_safe=fake_parse_ticket_safe)
    data, errors = parse("T1", "raw")

    assert data["issue_type"] == "REFUND"
    assert isinstance(data["issue_type"], str)


def test_adapter_passes_through_success_after_repair_loop_warnings():
    """Confirmed real behaviour: a ticket recovered by the repair loop
    returns (ticket, errors) with errors NON-EMPTY despite success.
    extract_node's audit_log must surface this, not silently drop it."""
    ticket = _FakeTicketV2({"issue_type": "REFUND", "confidence": 0.7})

    def fake_parse_ticket_safe(ticket_id, raw_text, client, max_retries=2):
        return ticket, ["Expecting value: line 1 column 1 (char 0)"]

    parse = make_m1_parser_adapter(llm_client=object(), parse_ticket_safe=fake_parse_ticket_safe)
    data, errors = parse("T1", "raw")

    assert data["issue_type"] == "REFUND"
    assert len(errors) == 1


def test_adapter_passes_through_final_failure_as_empty_dict():
    """Confirmed real behaviour: after max_retries+1 failed attempts,
    parse_ticket_safe returns (None, errors). The adapter must translate
    that None into {} - extract_node's contract is "empty dict = failure",
    never None (which would need an extra None-check everywhere else)."""
    def fake_parse_ticket_safe(ticket_id, raw_text, client, max_retries=2):
        return None, ["err1", "err2", "err3"]

    parse = make_m1_parser_adapter(llm_client=object(), parse_ticket_safe=fake_parse_ticket_safe)
    data, errors = parse("T1", "raw")

    assert data == {}
    assert errors == ["err1", "err2", "err3"]


def test_adapter_calls_parse_ticket_safe_with_the_confirmed_argument_order():
    """Pins the exact call shape confirmed from intake/parser.py:
    parse_ticket_safe(ticket_id, raw_text, client, max_retries=2)."""
    seen = {}

    def fake_parse_ticket_safe(ticket_id, raw_text, client, max_retries=2):
        seen["args"] = (ticket_id, raw_text, client, max_retries)
        return _FakeTicketV2({"issue_type": "ORDER"}), []

    sentinel_client = object()
    parse = make_m1_parser_adapter(llm_client=sentinel_client, parse_ticket_safe=fake_parse_ticket_safe)
    parse("T-42", "raw text here")

    ticket_id, raw_text, client, max_retries = seen["args"]
    assert ticket_id == "T-42"
    assert raw_text == "raw text here"
    assert client is sentinel_client
    assert max_retries == 2


# --------------------------------------------------------------------------
# make_m3_resolver_adapter
# --------------------------------------------------------------------------

def test_m3_resolver_adapter_returns_none_on_missing_order_id():
    resolve = make_m3_resolver_adapter(order_lookup=lambda oid: {"customer_ref": "SHOULD-NOT-BE-CALLED"})
    assert resolve(None) is None


def test_m3_resolver_adapter_extracts_customer_ref_from_order_lookup_result():
    resolve = make_m3_resolver_adapter(order_lookup=lambda oid: {"customer_ref": "CUST-9", "order_id": oid})
    assert resolve("ORD-1") == "CUST-9"


def test_m3_resolver_adapter_swallows_order_lookup_exceptions():
    """order_lookup is a mock API in M2 - if it raises, the workflow should
    treat that as 'unresolved', not crash the whole ticket review."""
    def flaky_order_lookup(oid):
        raise RuntimeError("mock API down")

    resolve = make_m3_resolver_adapter(order_lookup=flaky_order_lookup)
    assert resolve("ORD-1") is None
