"""
Single-agent tool-calling loop for ShopSense M2.

Uses LangChain's ChatLiteLLM + bind_tools, kept deliberately separate from
core/llm_client.py (M1's sole call site for structured ticket extraction).
That split is intentional: LLMClient stays reserved for structured extraction,
the agent layer (M2+) standardizes on LangChain, since M6's Langfuse tracing
gets tool-call spans for free from LangChain's callback handler.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agent.prompts import SYSTEM_PROMPT, format_ticket_for_agent
from tools.order_lookup import order_lookup
from tools.refund_calculator import calculate_refund
from tools.refund_replace import process_refund
from tools.reliability import TOOL_CALL_LOG, clear_log, make_robust_tool
from tools.shipping_tracker import track_shipment

MAX_ITERATIONS = 6

robust_order_lookup = make_robust_tool(order_lookup, "order_lookup")
robust_track_shipment = make_robust_tool(track_shipment, "track_shipment")
robust_calculate_refund = make_robust_tool(calculate_refund, "calculate_refund")
robust_process_refund = make_robust_tool(process_refund, "process_refund")


@tool
def order_lookup_tool(order_ref: str) -> dict:
    """Look up an order by reference (e.g. 'KW-O-000123'). Returns status, product, customer tier."""
    return robust_order_lookup(order_ref)


@tool
def track_shipment_tool(order_ref: str) -> dict:
    """Get shipping/delivery status for an order, including any delay compensation owed."""
    return robust_track_shipment(order_ref)


@tool
def calculate_refund_tool(order_ref: str, condition: str = "unopened") -> dict:
    """Compute the policy-eligible refund amount for an order. condition: unopened|opened_unused|opened_used|damaged."""
    return robust_calculate_refund(order_ref, condition=condition)


@tool
def process_refund_tool(
    order_ref: str,
    action: str,
    requested_amount_inr: int,
    reason_code: str,
    condition: str = "unopened",
    safety_or_legal_concern: bool = False,
) -> dict:
    """Submit a refund/replace/goodwill_credit request. action: refund|replace|goodwill_credit."""
    return robust_process_refund(
        order_ref, action, requested_amount_inr, reason_code,
        condition=condition, safety_or_legal_concern=safety_or_legal_concern,
    )


TOOLS = [order_lookup_tool, track_shipment_tool, calculate_refund_tool, process_refund_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def _resolve_model() -> str:
    """
    Mirrors core/llm_client.py's provider switch: LLM_PROVIDER picks which
    *_LLM_MODEL env var is authoritative, so this agent and M1's intake parser
    stay in sync if you swap providers -- change LLM_PROVIDER once, not per file.
    """
    provider = os.environ.get("LLM_PROVIDER", "GEMINI").upper()

    if provider == "GEMINI":
        model = os.environ.get("GEMINI_LLM_MODEL")
    elif provider == "OPENAI":
        model = os.environ.get("OPENAI_LLM_MODEL")
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}' -- expected 'GEMINI' or 'OPENAI'."
        )

    if not model:
        raise ValueError(
            f"LLM_PROVIDER is '{provider}' but "
            f"{'GEMINI_LLM_MODEL' if provider == 'GEMINI' else 'OPENAI_LLM_MODEL'} "
            f"is not set in the environment."
        )
    return model


def _get_llm():
    from langchain_litellm import ChatLiteLLM
    return ChatLiteLLM(model=_resolve_model(), max_retries=5)


def run_support_agent(ticket: dict, max_iterations: int = MAX_ITERATIONS) -> dict[str, Any]:
    """
    Run the tool-calling loop for a single SupportTicket.

    Returns:
        {final_answer, tool_calls_made: [...], resolution_status, iterations_used}
        resolution_status is 'resolved' if the loop ended with a final answer,
        or 'max_iterations_exceeded' if it hit the cap without one.
    """
    clear_log()
    llm = _get_llm().bind_tools(TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=format_ticket_for_agent(ticket)),
    ]

    for i in range(max_iterations):
        ai_msg: AIMessage = llm.invoke(messages)
        messages.append(ai_msg)

        if not getattr(ai_msg, "tool_calls", None):
            return {
                "final_answer": ai_msg.content,
                "tool_calls_made": list(TOOL_CALL_LOG),
                "resolution_status": "resolved",
                "iterations_used": i + 1,
            }

        for tc in ai_msg.tool_calls:
            try:
                tool_fn = TOOLS_BY_NAME[tc["name"]]
                result = tool_fn.invoke(tc["args"])
            except Exception as e:
                result = {"error": str(e)}
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return {
        "final_answer": None,
        "tool_calls_made": list(TOOL_CALL_LOG),
        "resolution_status": "max_iterations_exceeded",
        "iterations_used": max_iterations,
    }