"""
Tests for agent/support_agent.py -- the tool-calling loop itself.

Design note: run_support_agent calls a real LLM via _get_llm(). These tests
never do that -- they monkeypatch agent.support_agent._get_llm to return a
FakeChatModel that returns pre-scripted AIMessage responses instead. This
tests the LOOP's behavior (does it call the right tool, does it stop when it
should, does it handle a bad tool call gracefully, does it respect the
iteration cap) without needing an API key, network access, or nondeterministic
real model output.

Where a scripted response includes a tool_call for order_lookup_tool /
calculate_refund_tool, the underlying tool's data layer is monkeypatched the
same way as in test_tools.py, so those tool calls actually execute against
controlled fixtures rather than the real dataset.

M3 update: run_support_agent now requires customer_ref and a CustomerMemory
instance. Rather than wiring up a real CustomerMemory (which would mean
faking the Supermemory SDK here too -- that's tests/test_memory.py's job,
not this file's), these tests use FakeMemory: a minimal stand-in exposing
only the two methods support_agent.py and tools/customer_memory_tool.py
actually call (get_context_block, remember_preference). This keeps this
file's fakes scoped to what the AGENT LOOP touches, the same separation of
concerns as FakeChatModel only faking what the loop needs from an LLM.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

import agent.support_agent as agent_module
from agent.support_agent import run_support_agent
from tools.reliability import TOOL_CALL_LOG


class FakeChatModel:
    """Stands in for ChatLiteLLM. Returns responses from a scripted queue,
    one per .invoke() call, and records every message list it was given.
    Also records the tools it was bound with, so tests can assert on WHICH
    tools were made available for a given customer_ref (e.g. whether
    note_customer_preference was included) without needing bind_tools to
    actually do anything."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.invocations: list[list] = []
        self.bound_tools: list = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self  # tool binding is a no-op for the fake -- return self so .invoke still works

    def invoke(self, messages):
        self.invocations.append(list(messages))
        if not self._responses:
            raise AssertionError(
                f"FakeChatModel ran out of scripted responses after {len(self.invocations)} calls -- "
                f"the loop asked for more turns than the test scripted."
            )
        return self._responses.pop(0)


class FakeMemory:
    """Minimal stand-in for memory.customer_memory.CustomerMemory -- only the
    two methods the agent loop and the memory tool actually call. Doesn't
    touch Supermemory at all, matching how test_tools.py/test_agent.py never
    hit real external services for the M2 tools either.

    context_block: what get_context_block() returns -- "" (default) means
    "no known history", matching CustomerMemory's real default for a new
    customer.
    """

    def __init__(self, context_block: str = ""):
        self._context_block = context_block
        self.get_context_block_calls: list[tuple[str, str]] = []
        self.preferences_written: list[tuple[str, str, dict]] = []

    def get_context_block(self, customer_ref: str, query: str, k: int = 3) -> str:
        self.get_context_block_calls.append((customer_ref, query))
        return self._context_block

    def remember_preference(self, customer_ref: str, fact: str, **extra) -> None:
        self.preferences_written.append((customer_ref, fact, extra))


def install_fake_llm(monkeypatch, responses: list[AIMessage]) -> FakeChatModel:
    fake = FakeChatModel(responses)
    monkeypatch.setattr(agent_module, "_get_llm", lambda: fake)
    return fake


def disable_retry_delays(monkeypatch):
    """Tests that deliberately trigger a tool failure would otherwise sit
    through reliability.py's real exponential backoff (~3.5s per failure)."""
    monkeypatch.setattr("tools.reliability.time.sleep", lambda seconds: None)


SAMPLE_TICKET = {
    "ticket_id": "SHOPSENSE-TEST01",
    "order_id": "KW-O-TEST01",
    "issue_type": "REFUND",
    "sentiment": "frustrated",
    "urgency": "medium",
    "claimed_refund_amount": 500,
    "contains_suspicious_instructions": False,
    "confidence": 0.92,
    "raw_text": "My order arrived damaged, please refund me.",
}

# Matches the customer_ref in patch_order_lookup_data's fixture data below --
# kept consistent so tests reflect a realistic order/customer pairing, even
# though FakeMemory itself doesn't care about real customer data.
SAMPLE_CUSTOMER_REF = "KW-C-TEST01"


def patch_order_lookup_data(monkeypatch, order_ref="KW-O-TEST01"):
    order = {
        "order_ref": order_ref, "customer_ref": "KW-C-TEST01", "sku": "KW-SKU-TEST01",
        "quantity": 1, "order_value_inr": 5000, "placed_at": "2026-08-01T10:00",
        "payment_method": "card", "status": "delivered",
        "delivery_promise_days": 5, "actual_delivery_days": 3,
    }
    product = {"sku": "KW-SKU-TEST01", "title": "Test Widget", "category": "Electronics",
               "price_inr": 5000, "seller_id": "KW-S-TEST", "in_stock": True, "warranty_months": 12}
    customer = {"customer_ref": "KW-C-TEST01", "display_name": "Test Customer", "city": "Testville",
                "tier": "standard", "joined_on": "2020-01-01", "lifetime_orders": 5,
                "return_rate": 0.05, "flagged_for_abuse": False}
    monkeypatch.setattr("tools.order_lookup.orders_by_ref", lambda: {order_ref: order})
    monkeypatch.setattr("tools.order_lookup.products_by_sku", lambda: {order["sku"]: product})
    monkeypatch.setattr("tools.order_lookup.customers_by_ref", lambda: {order["customer_ref"]: customer})


class TestRunSupportAgent:
    def test_resolves_immediately_when_no_tool_calls(self, monkeypatch):
        install_fake_llm(monkeypatch, [AIMessage(content="Thanks, here's your answer.")])

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())

        assert result["resolution_status"] == "resolved"
        assert result["final_answer"] == "Thanks, here's your answer."
        assert result["iterations_used"] == 1
        assert result["tool_calls_made"] == []

    def test_one_tool_call_then_resolves(self, monkeypatch):
        patch_order_lookup_data(monkeypatch)
        install_fake_llm(monkeypatch, [
            AIMessage(content="", tool_calls=[
                {"name": "order_lookup_tool", "args": {"order_ref": "KW-O-TEST01"}, "id": "call_1"}
            ]),
            AIMessage(content="Your order was delivered on time."),
        ])

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())

        assert result["resolution_status"] == "resolved"
        assert result["iterations_used"] == 2
        assert len(result["tool_calls_made"]) == 1
        assert result["tool_calls_made"][0]["tool"] == "order_lookup"

    def test_multiple_tool_calls_in_a_single_turn(self, monkeypatch):
        patch_order_lookup_data(monkeypatch)
        install_fake_llm(monkeypatch, [
            AIMessage(content="", tool_calls=[
                {"name": "order_lookup_tool", "args": {"order_ref": "KW-O-TEST01"}, "id": "call_1"},
                {"name": "calculate_refund_tool", "args": {"order_ref": "KW-O-TEST01", "condition": "unopened"}, "id": "call_2"},
            ]),
            AIMessage(content="Checked both."),
        ])
        # calculate_refund_tool reads from tools.refund_calculator's own imports
        monkeypatch.setattr("tools.refund_calculator.orders_by_ref", lambda: {
            "KW-O-TEST01": {"order_ref": "KW-O-TEST01", "customer_ref": "KW-C-TEST01", "sku": "KW-SKU-TEST01",
                             "quantity": 1, "order_value_inr": 5000, "placed_at": "2026-08-01T10:00",
                             "payment_method": "card", "status": "delivered",
                             "delivery_promise_days": 5, "actual_delivery_days": 3}
        })
        monkeypatch.setattr("tools.refund_calculator.products_by_sku", lambda: {
            "KW-SKU-TEST01": {"sku": "KW-SKU-TEST01", "title": "Test Widget", "category": "Electronics",
                               "price_inr": 5000, "seller_id": "KW-S-TEST", "in_stock": True, "warranty_months": 12}
        })
        monkeypatch.setattr("tools.refund_calculator.shipments_by_order_ref", lambda: {})

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())

        assert result["iterations_used"] == 2  # both calls happened in ONE model turn
        assert len(result["tool_calls_made"]) == 2
        tool_names = {tc["tool"] for tc in result["tool_calls_made"]}
        assert tool_names == {"order_lookup", "calculate_refund"}

    def test_tool_exception_is_caught_and_fed_back_not_crashed(self, monkeypatch):
        disable_retry_delays(monkeypatch)
        # deliberately broken data source -> order_lookup will raise inside the tool
        monkeypatch.setattr("tools.order_lookup.orders_by_ref", lambda: (_ for _ in ()).throw(RuntimeError("data source down")))
        monkeypatch.setattr("tools.order_lookup.products_by_sku", lambda: {})
        monkeypatch.setattr("tools.order_lookup.customers_by_ref", lambda: {})

        install_fake_llm(monkeypatch, [
            AIMessage(content="", tool_calls=[
                {"name": "order_lookup_tool", "args": {"order_ref": "KW-O-TEST01"}, "id": "call_1"}
            ]),
            AIMessage(content="I ran into an issue looking that up, escalating to a human."),
        ])

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())  # must not raise

        assert result["resolution_status"] == "resolved"
        assert "escalating" in result["final_answer"].lower()
        # the failure was logged, not silently dropped
        assert any("error" in tc for tc in result["tool_calls_made"])

    def test_unknown_tool_name_does_not_crash_the_run(self, monkeypatch):
        """Regression test for a real bug found during M2 review: TOOLS_BY_NAME[name]
        was originally outside the try/except, so a hallucinated tool name would
        raise an uncaught KeyError and crash run_support_agent entirely. M3 adds a
        fifth possible tool name (note_customer_preference); this still covers a
        name that matches none of the five."""
        install_fake_llm(monkeypatch, [
            AIMessage(content="", tool_calls=[
                {"name": "this_tool_does_not_exist", "args": {}, "id": "call_1"}
            ]),
            AIMessage(content="Recovered gracefully."),
        ])

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())  # must not raise KeyError

        assert result["resolution_status"] == "resolved"
        assert result["final_answer"] == "Recovered gracefully."

    def test_max_iterations_exceeded_when_model_never_stops(self, monkeypatch):
        patch_order_lookup_data(monkeypatch)
        # every single turn requests another tool call, never a plain answer
        responses = [
            AIMessage(content="", tool_calls=[
                {"name": "order_lookup_tool", "args": {"order_ref": "KW-O-TEST01"}, "id": f"call_{i}"}
            ])
            for i in range(6)
        ]
        install_fake_llm(monkeypatch, responses)

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory(), max_iterations=6)

        assert result["resolution_status"] == "max_iterations_exceeded"
        assert result["final_answer"] is None
        assert result["iterations_used"] == 6
        assert len(result["tool_calls_made"]) == 6

    def test_respects_custom_max_iterations(self, monkeypatch):
        patch_order_lookup_data(monkeypatch)
        responses = [
            AIMessage(content="", tool_calls=[
                {"name": "order_lookup_tool", "args": {"order_ref": "KW-O-TEST01"}, "id": f"call_{i}"}
            ])
            for i in range(3)
        ]
        install_fake_llm(monkeypatch, responses)

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory(), max_iterations=3)

        assert result["resolution_status"] == "max_iterations_exceeded"
        assert result["iterations_used"] == 3

    def test_tool_call_log_clears_between_separate_runs(self, monkeypatch):
        patch_order_lookup_data(monkeypatch)
        install_fake_llm(monkeypatch, [
            AIMessage(content="", tool_calls=[
                {"name": "order_lookup_tool", "args": {"order_ref": "KW-O-TEST01"}, "id": "call_1"}
            ]),
            AIMessage(content="First run done."),
        ])
        first_result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())
        assert len(first_result["tool_calls_made"]) == 1

        install_fake_llm(monkeypatch, [AIMessage(content="Second run, no tools needed.")])
        second_result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())

        assert second_result["tool_calls_made"] == []  # not carrying over the first run's call

    def test_ticket_fields_reach_the_model_in_the_first_message(self, monkeypatch):
        """Confirms format_ticket_for_agent's output is actually what gets sent --
        not testing prompts.py's formatting logic itself (that's prompts.py's own
        concern), just that support_agent wires it in correctly."""
        fake = install_fake_llm(monkeypatch, [AIMessage(content="Acknowledged.")])

        run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())

        first_call_messages = fake.invocations[0]
        human_message_content = first_call_messages[1].content  # [0]=system, [1]=human
        assert SAMPLE_TICKET["order_id"] in human_message_content
        assert SAMPLE_TICKET["raw_text"] in human_message_content
        assert "UNTRUSTED" in human_message_content


class TestM3MemoryIntegration:
    """New for M3: the memory tool's availability/scoping, and the customer
    history block actually reaching the model. Deliberately separate from
    TestRunSupportAgent so it's obvious at a glance which coverage is M2's
    loop behavior vs. what M3 added on top of it."""

    def test_customer_context_block_reaches_the_model(self, monkeypatch):
        fake = install_fake_llm(monkeypatch, [AIMessage(content="Acknowledged.")])
        memory = FakeMemory(context_block="Known customer history:\n- [semantic] Prefers replacement over refund.")

        run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, memory)

        human_message_content = fake.invocations[0][1].content
        assert "Prefers replacement over refund." in human_message_content
        assert "known customer history" in human_message_content.lower()
        # confirms get_context_block was actually called with THIS ticket's
        # customer_ref and raw_text, not defaulted/hardcoded somewhere
        assert memory.get_context_block_calls == [(SAMPLE_CUSTOMER_REF, SAMPLE_TICKET["raw_text"])]

    def test_no_history_block_when_customer_has_none(self, monkeypatch):
        """FakeMemory()'s default context_block="" mirrors a brand-new
        customer -- confirms the empty case doesn't leave a stray labeled
        section with nothing in it."""
        fake = install_fake_llm(monkeypatch, [AIMessage(content="Acknowledged.")])

        run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())

        human_message_content = fake.invocations[0][1].content
        assert "known customer history" not in human_message_content.lower()

    def test_note_customer_preference_tool_available_when_customer_resolved(self, monkeypatch):
        fake = install_fake_llm(monkeypatch, [AIMessage(content="Acknowledged.")])

        run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, FakeMemory())

        tool_names = {t.name for t in fake.bound_tools}
        assert "note_customer_preference" in tool_names
        assert len(fake.bound_tools) == 5  # the four M2 tools + this one

    def test_note_customer_preference_tool_absent_when_customer_ref_is_none(self, monkeypatch):
        """M3's graceful-degradation case: a ticket with no resolvable
        customer_ref (e.g. missing order_id) still runs, just without the
        memory tool -- and must never call into memory at all, since there's
        no namespace to scope a call to."""
        fake = install_fake_llm(monkeypatch, [AIMessage(content="Acknowledged.")])
        memory = FakeMemory()

        result = run_support_agent(SAMPLE_TICKET, None, memory)

        tool_names = {t.name for t in fake.bound_tools}
        assert "note_customer_preference" not in tool_names
        assert len(fake.bound_tools) == 4  # only the M2 tools
        assert memory.get_context_block_calls == []  # never called with customer_ref=None
        assert result["resolution_status"] == "resolved"  # ticket still runs to completion

    def test_agent_calling_note_customer_preference_writes_to_the_right_customer(self, monkeypatch):
        install_fake_llm(monkeypatch, [
            AIMessage(content="", tool_calls=[
                {"name": "note_customer_preference",
                 "args": {"fact": "Prefers replacement over refund."},
                 "id": "call_1"}
            ]),
            AIMessage(content="Noted your preference for next time."),
        ])
        memory = FakeMemory()

        result = run_support_agent(SAMPLE_TICKET, SAMPLE_CUSTOMER_REF, memory)

        assert result["resolution_status"] == "resolved"
        assert len(memory.preferences_written) == 1
        written_customer_ref, fact, extra = memory.preferences_written[0]
        # the model was NEVER given a customer_ref argument to supply (see
        # tools/customer_memory_tool.py) -- this confirms the closure bound
        # it correctly to THIS ticket's customer, not something model-supplied
        assert written_customer_ref == SAMPLE_CUSTOMER_REF
        assert fact == "Prefers replacement over refund."
        assert extra.get("source_ticket_id") == SAMPLE_TICKET["ticket_id"]
        assert any(tc["tool"] == "note_customer_preference" for tc in result["tool_calls_made"])