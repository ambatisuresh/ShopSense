"""
ShopSense M5 - Parts 5 & 6: routing + graph wiring.

Assembles extract -> compare_to_playbook -> draft_redline -> route ->
(auto_approve | human_approval) -> finalize into a compiled graph -
mirroring Lab B's own incremental build order: Part B3 wired
checks -> revise/finalize with no checkpointer; Part B6 introduced the
real interrupt()-based human_approval node AND a checkpointer at the same
time (interrupt() cannot run without one - see below); Part B7 upgraded
that checkpointer from InMemorySaver to SqliteSaver specifically for real
process-restart durability. Same staging here:
    - Step 5: real routing, PLACEHOLDER terminal nodes, no checkpointer.
    - Step 6 (this file, updated): `human_approval` is now the real
      interrupt() node, so a checkpointer is now REQUIRED - added as a
      parameter, defaulting to a lazily-constructed InMemorySaver. `auto_
      approve` and `human_approval`'s "approved" outcome both now flow
      into a shared `finalize` node, still a PLACEHOLDER.
    - Step 7 (this file, updated): `finalize` is now the real node from
      workflow/nodes/finalize.py - actually commits the refund/replace via
      M2's process_refund (evaluate_refund called with commit=True; see
      compare_to_playbook.py's "STEP 7 ADDITION" docstring note for why).
    - Step 8 (this file, updated): a new `checkpoint_db_path` parameter
      opts into a real, disk-backed `SqliteSaver` (workflow/checkpointing.
      py) instead of the default `InMemorySaver` - see that module's
      docstring for why the BARE default deliberately stays InMemorySaver
      rather than silently switching, which is a refinement of what this
      docstring used to say here. A real process-restart is demonstrated in
      scripts/run_checkpointing_pause.py / _resume.py.

WHY a checkpointer is now required, not optional: the notebook lists three
hard preconditions for interrupt() to work at all, and the first one is "a
checkpointer wired at compile() - before the pause, not after". Calling
interrupt() inside a graph compiled with no checkpointer does not just fail
to pause - it errors. Step 5 could defer checkpointing entirely because
nothing in that graph paused yet; this step can't.

`route_after_review` / `route_after_human` (workflow/routing.py) are PURE
functions of state - decide in a node, route in an edge. Kept in their own
module because THIS file imports the real `langgraph` package, and a
router that has nothing to do with langgraph shouldn't become untestable
just by living next to code that does.

LAZY IMPORT inside build_graph(), not at module top level - same reasoning
as workflow/nodes/human_approval.py's lazy interrupt import (itself
following M4's rag/rerank.py precedent): importing `workflow.graph` (e.g.
`from workflow.graph import build_graph`) now succeeds even without
langgraph installed. Only CALLING build_graph() requires it - which is
exactly when it's actually needed.
"""

from workflow.nodes.compare_to_playbook import (
    build_compare_to_playbook_node,
    fixture_evaluate_refund,
    fixture_retrieve_citations,
)
from workflow.nodes.draft_redline import build_draft_redline_node, deterministic_compose_redline
from workflow.nodes.extract import build_extract_node, fixture_parse_ticket, fixture_resolve_customer_ref
from workflow.nodes.finalize import build_finalize_node
from workflow.nodes.human_approval import build_human_approval_node
from workflow.routing import route_after_human, route_after_review
from workflow.state import TicketReviewState


def _auto_approve(state: TicketReviewState) -> dict:
    """The standard-path equivalent of a reviewer's approval, minus the
    pause - no interrupt() here, no human involved, by design (that's the
    entire point of the standard/non_standard split). Sets approver_id to
    a fixed system identity rather than leaving it blank, so finalize
    (Step 7) never has to special-case "who approved this" between the two
    paths - both always carry a real value."""
    return {
        "status": "approved",
        "approver_id": "system:auto-approval",
        "approver_note": "auto-approved: classification=standard, no concerns raised",
        "audit_log": ["auto_approve: approved without human review (classification=standard)"],
    }


def build_graph(
    parse_ticket=fixture_parse_ticket,
    resolve_customer_ref=fixture_resolve_customer_ref,
    evaluate_refund=fixture_evaluate_refund,
    retrieve_citations=fixture_retrieve_citations,
    compose_redline=deterministic_compose_redline,
    interrupt_fn=None,
    checkpointer=None,
    checkpoint_db_path=None,
):
    """Every extract/compare/draft dependency is a parameter with an
    offline-fixture default, so this is callable with zero external
    services - no LLM key, no order table, no corpus/index - same posture
    as every demo script so far. Swap in the production adapters
    (make_m1_parser_adapter, make_m3_resolver_adapter,
    make_m2_refund_adapter, make_m4_retrieval_adapter,
    make_llm_compose_redline) for a real deployment.

    `interrupt_fn`: passed straight through to build_human_approval_node -
    leave as None for the real langgraph interrupt() (production), or
    inject a fake for tests that don't want a live pause/resume cycle.

    `checkpointer`: an already-built saver (InMemorySaver, SqliteSaver, or
    anything else conforming to langgraph's checkpointer interface) - use
    this to reattach to a specific existing saver instance, e.g. one built
    once and reused across multiple build_graph() calls. Takes priority
    over `checkpoint_db_path` if both are given.

    `checkpoint_db_path`: convenience opt-in for durability without
    constructing a saver yourself - if given (and `checkpointer` is None),
    builds a real SqliteSaver at this path via workflow.checkpointing.
    build_sqlite_checkpointer(db_path, fresh=False) (REATTACHES if the file
    already exists, never wipes it - see that function's docstring for why
    wipe-vs-reattach must be an explicit, separate choice).

    If NEITHER is given, falls back to a lazily-constructed InMemorySaver -
    REQUIRED for human_approval's interrupt() to work at all (see module
    docstring), but ephemeral: state does not survive this process exiting.
    Fine for tests and quick demos; not fine for "a human approves three
    days later after a redeploy" - that needs `checkpoint_db_path`.
    """
    from langgraph.graph import END, START, StateGraph

    if checkpointer is None:
        if checkpoint_db_path is not None:
            from workflow.checkpointing import build_sqlite_checkpointer
            checkpointer = build_sqlite_checkpointer(checkpoint_db_path, fresh=False)
        else:
            from langgraph.checkpoint.memory import InMemorySaver
            checkpointer = InMemorySaver()

    g = StateGraph(TicketReviewState)

    g.add_node("extract", build_extract_node(parse_ticket, resolve_customer_ref))
    g.add_node("compare_to_playbook", build_compare_to_playbook_node(evaluate_refund, retrieve_citations))
    g.add_node("draft_redline", build_draft_redline_node(compose_redline))
    g.add_node("auto_approve", _auto_approve)
    g.add_node("human_approval", build_human_approval_node(interrupt_fn))
    # Same `evaluate_refund` instance passed to compare_to_playbook above -
    # required so commit=False (classify) and commit=True (finalize) both
    # go through the same underlying adapter/order-table wiring.
    g.add_node("finalize", build_finalize_node(evaluate_refund))

    g.add_edge(START, "extract")
    g.add_edge("extract", "compare_to_playbook")
    g.add_edge("compare_to_playbook", "draft_redline")
    g.add_conditional_edges(
        "draft_redline",
        route_after_review,
        {"auto_approve": "auto_approve", "human_approval": "human_approval"},
    )
    g.add_edge("auto_approve", "finalize")
    g.add_conditional_edges(
        "human_approval",
        route_after_human,
        {"finalize": "finalize", "end": END},
    )
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)