"""The Playbook RAG agent: for each extracted clause, retrieve its matching
negotiation-playbook position and the real policy passage(s) backing it.

One clause per invocation — same design as the Day3 Session 2 notebook's
researcher_node(): the supervisor's routing trace shows each lookup as its
own step, and "which clauses are still outstanding" becomes a plain set
difference downstream, not a hidden internal loop.

This node does NOT judge whether a contract clause complies with the
playbook position — that's Redline Drafter's (Step 5) and Legal Reviewer's
(Step 6) job. Playbook RAG only retrieves.

Step 10: the playbook itself now comes from the MCP contract-repository
server's contracts://playbook/negotiation resource, not a local disk read —
which is why this node is `async def` now. Note what did NOT move to MCP:
the five general policy docs (refund-authority.md, shipping-policy.md,
etc.) backing each playbook position stay a local BM25 index — Step 9's
server was scoped to the contract repository only (contracts + the
negotiation playbook), so _policy_index() below is unchanged and still
synchronous.
"""
from __future__ import annotations

from team.bm25 import BM25
from team.mcp_client import read_playbook_via_mcp
from team.playbook_index import parse_playbook_positions
from team.policy_corpus import load_policy_corpus
from team.scopes import scoped

# Module-level caches: the playbook positions and the policy BM25 index are
# fixed for the lifetime of a process, so there's no reason to
# refetch-and-reparse/reindex on every clause. Mirrors the Day3 Session 2
# notebook's own note on caching with_structured_output() runnables per
# schema rather than rebuilding them every call.
_POSITIONS_CACHE: dict | None = None
_POLICY_INDEX_CACHE: tuple | None = None


async def _positions() -> dict:
    global _POSITIONS_CACHE
    if _POSITIONS_CACHE is None:
        text = await read_playbook_via_mcp()
        _POSITIONS_CACHE = parse_playbook_positions(text)
    return _POSITIONS_CACHE


def _policy_index() -> tuple:
    global _POLICY_INDEX_CACHE
    if _POLICY_INDEX_CACHE is None:
        # Drop empty-body chunks (e.g. warranty-policy.md's "## 5. Replacement
        # vs. Repair Decision Rule" parent header, whose real content lives in
        # its "### 5.1 Decision Criteria" child). An empty-body chunk has
        # nothing to ground a citation in, but indexing "heading + body" let
        # its heading text alone out-score the real content chunk whenever
        # the heading happened to echo the query's words back at it —
        # caught by actually inspecting retrieval output, not by inspection.
        chunks = [c for c in load_policy_corpus() if c["body"]]
        bm25 = BM25([f"{c['heading']} {c['body']}" for c in chunks])
        _POLICY_INDEX_CACHE = (chunks, bm25)
    return _POLICY_INDEX_CACHE


def _clause_sort_key(clause_id: str) -> tuple:
    """Numeric sort key so '10.1' sorts after '2.1', not before it."""
    return tuple(int(part) for part in clause_id.split("."))


def _select_next_clause(clauses: list[dict], done_ids: set) -> dict | None:
    todo = [c for c in clauses if c["clause_id"] not in done_ids]
    todo.sort(key=lambda c: _clause_sort_key(c["clause_id"]))
    return todo[0] if todo else None


@scoped("playbook_rag")
async def playbook_rag_node(state: dict) -> dict:
    done_ids = {f["clause_id"] for f in state["playbook_findings"]}
    clause = _select_next_clause(state["clauses"], done_ids)

    if clause is None:
        # Nothing left to look up. The supervisor shouldn't route here again
        # once every clause has a finding, but this is a harmless no-op
        # rather than a crash if it ever does.
        return {"log": ["playbook_rag: nothing outstanding to look up"]}

    clause_type = clause["clause_type"]

    if clause_type == "unclassified":
        finding = {
            "clause_id": clause["clause_id"],
            "clause_type": clause_type,
            "status": "skipped",
            "reason": "no playbook position covers this clause type",
        }
        return {
            "playbook_findings": [finding],
            "log": [f"playbook_rag: clause {clause['clause_id']} unclassified — skipped"],
        }

    position = (await _positions()).get(clause_type)
    retrieved = []

    # Some playbook positions (e.g. Term/Renewal, Limitation of Liability,
    # Indemnification, Governing Law) are pure negotiation stances with no
    # backing policy doc — a genuine, expected case, not a retrieval miss.
    # Searching the whole corpus for one anyway just surfaces noise (an
    # unrelated section that happens to share a few words), so retrieval
    # only runs when there's something to ground against.
    if position and position["cited_docs"]:
        chunks, bm25 = _policy_index()
        query = " ".join([clause["title"], *position["parts"].values()])
        # Search a wider pool, then keep only chunks from the doc(s) this
        # playbook clause actually cites — the corpus-wide top-3 for a query
        # like "Repair vs. Replacement Determination" can easily be
        # dominated by a same-worded but wrong-document chunk otherwise.
        candidates = bm25.search(query, top_k=10)
        candidates = [
            (i, score) for i, score in candidates
            if chunks[i]["doc"] in position["cited_docs"]
        ][:3]
        retrieved = [
            {
                "doc": chunks[i]["doc"],
                "section_id": chunks[i]["section_id"],
                "heading": chunks[i]["heading"],
                "snippet": chunks[i]["body"][:400],
                "score": round(score, 3),
            }
            for i, score in candidates
        ]

    finding = {
        "clause_id": clause["clause_id"],
        "clause_type": clause_type,
        "status": "found" if position else "no_position",
        "position": position,
        "retrieved_passages": retrieved,
    }
    playbook_ref = position["clause_number"] if position else "?"
    if position and not position["cited_docs"]:
        detail = "no policy doc citation (pure negotiation term)"
    else:
        detail = f"{len(retrieved)} supporting passage(s) retrieved"
    return {
        "playbook_findings": [finding],
        "log": [
            f"playbook_rag: clause {clause['clause_id']} ({clause_type}) -> "
            f"playbook {playbook_ref}, {detail}"
        ],
    }