"""The Extraction agent: raw contract text -> structured, classified clauses.

Same "let the model produce, let deterministic code decide" shape as every
other node in this project (M1's parser, the Day3 Session 2 notebook's
planner/researcher, M5's compare_to_playbook): an LLM call is attempted
first, but it can never be the only path — if it's unavailable, times out,
or returns something outside the allowed clause-type taxonomy, a
deterministic keyword classifier decides instead. extraction_node() never
raises on a classification failure; the worst case is a clause tagged
"unclassified" and flagged in the log.

Step 10: contract text now comes from the MCP contract-repository server
(mcp_server/contract_server.py) via read_contract, not a local disk read —
which is why this node is `async def` now. Same mechanical note as the
Day3 Session 2 notebook's own MCP swap: the node becomes async because the
MCP call is awaited, and @scoped already handles that (it checks
inspect.iscoroutinefunction and wraps accordingly). Nothing about clause
parsing/classification below changed at all.
"""
from __future__ import annotations

import os
from pathlib import Path

from team.mcp_client import read_contract_via_mcp
from team.parsing import split_into_clauses
from team.scopes import scoped
from team.taxonomy import CLAUSE_TYPES, classify_clause_type_fallback


def _llm_classify_clause_type(title: str, body: str) -> str | None:
    """Ask a live LLM which clause_type this contract clause matches, if any.

    Returns None on ANY failure: missing package, missing/invalid API key,
    network error, timeout, or a response that isn't one of the allowed
    labels. This function is not permitted to raise — callers must always
    have a deterministic fallback ready regardless of why this returned None.
    """
    try:
        import litellm
    except ImportError:
        return None

    allowed = ", ".join(CLAUSE_TYPES) + ", unclassified"
    prompt = (
        "You are classifying ONE clause from a vendor contract against a fixed "
        "taxonomy of clause types used by a legal negotiation playbook. Reply "
        "with EXACTLY one label from this list, and nothing else:\n"
        f"{allowed}\n\n"
        f"Clause title: {title}\n"
        f"Clause text: {body[:800]}\n\n"
        "Label:"
    )
    try:
        response = litellm.completion(
            model=os.environ.get("SHOPSENSE_LLM_MODEL", "gemini/gemini-2.0-flash"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=15,
        )
        label = response["choices"][0]["message"]["content"].strip().lower()
    except Exception:
        return None

    if label == "unclassified":
        return None
    return label if label in CLAUSE_TYPES else None


@scoped("extraction")
async def extraction_node(state: dict) -> dict:
    """Parse `state["contract_text"]` (or fetch it from the MCP contract
    repository via `state["contract_path"]`'s bare filename, if not already
    set) into clauses, and classify each one's clause_type.

    Returns a partial state update: {"contract_text", "clauses", "log"} —
    exactly extraction's write-scope from team/scopes.py, enforced by the
    @scoped decorator above.
    """
    if state.get("contract_text"):
        text = state["contract_text"]
    else:
        # mcp_server/contract_server.py's read_contract tool takes a bare
        # filename (its _safe_path() sandbox rejects directory components),
        # so this strips whatever directory state["contract_path"] carries
        # (e.g. "data/contracts/vendor_x.md" -> "vendor_x.md").
        text = await read_contract_via_mcp(Path(state["contract_path"]).name)
    raw_clauses = split_into_clauses(text)

    clauses = []
    log_lines = [
        f"extraction: parsed {len(raw_clauses)} raw clause(s) from {state['contract_path']}"
    ]

    for rc in raw_clauses:
        llm_label = _llm_classify_clause_type(rc["title"], rc["body"])
        if llm_label is not None:
            clause_type, source = llm_label, "llm"
        else:
            clause_type = classify_clause_type_fallback(rc["title"], rc["body"]) or "unclassified"
            source = "fallback"

        clauses.append({
            "clause_id": rc["clause_id"],
            "title": rc["title"],
            "clause_type": clause_type,
            "text": rc["body"],
        })
        log_lines.append(
            f"extraction[{source}]: clause {rc['clause_id']} ({rc['title']}) -> {clause_type}"
        )

    return {"contract_text": text, "clauses": clauses, "log": log_lines}