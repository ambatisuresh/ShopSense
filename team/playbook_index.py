"""Maps each clause_type to its negotiation-playbook position
(Preferred/Fallback/Unacceptable) and the policy docs it cites, by parsing
data/playbook/negotiation-playbook.md.
"""
from __future__ import annotations

import re
from pathlib import Path

from team.parsing import split_into_clauses

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK_PATH = REPO_ROOT / "data" / "playbook" / "negotiation-playbook.md"

# clause_type -> playbook clause number. Kept as an explicit mapping (not
# derived by zipping CLAUSE_TYPES against playbook order) so a future edit
# to team/taxonomy.py can't silently desync this without a test catching it
# — see test_playbook_rag.py's cross-check against the real playbook file.
CLAUSE_TYPE_TO_PLAYBOOK_CLAUSE = {
    "refund_settlement_authority": "1.1",
    "refund_settlement_timing": "1.2",
    "delivery_sla_alignment": "2.1",
    "delay_compensation_alignment": "2.2",
    "lost_in_transit_threshold": "2.3",
    "disputed_delivery_investigation": "2.4",
    "carrier_liability_cap": "2.5",
    "repair_turnaround": "3.1",
    "replacement_turnaround": "3.2",
    "warranty_void_criteria_alignment": "3.3",
    "repair_vs_replacement_decision_authority": "3.4",
    "claim_status_reporting_cadence": "3.5",
    "return_intake_window": "4.1",
    "condition_grading_alignment": "4.2",
    "category_exclusions": "4.3",
    "inspection_to_refund_buffer": "4.4",
    "return_mishandling_liability": "4.5",
    "termination_and_renewal": "5.1",
    "limitation_of_liability": "5.2",
    "indemnification": "5.3",
    "governing_law_and_venue": "5.4",
    "confidentiality_and_data_security": "5.5",
    "force_majeure": "5.6",
}

_LABELS = ["Preferred", "Fallback", "Unacceptable", "Acceptable as-is", "Note"]
_DOC_CITATION_RE = re.compile(r"\b([a-z][a-z-]+\.md)\b")


def _parse_position_body(body: str) -> dict:
    """Split a playbook clause's body into its Preferred/Fallback/
    Unacceptable/Acceptable-as-is/Note sub-parts (line-prefix based, not
    regex-lookahead — each label always starts its own line in this
    corpus), plus which policy docs the whole clause cites."""
    parts: dict[str, str] = {}
    current_key = None
    current_lines: list[str] = []

    def flush():
        if current_key is not None:
            parts[current_key] = " ".join(current_lines).strip()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        matched_label = next((lbl for lbl in _LABELS if line.startswith(lbl + ":")), None)
        if matched_label:
            flush()
            current_key = matched_label.lower().replace(" ", "_").replace("-", "_")
            current_lines = [line[len(matched_label) + 1:].strip()]
        elif current_key is not None and line:
            current_lines.append(line)
    flush()

    cited_docs = sorted(set(_DOC_CITATION_RE.findall(body)))
    return {"parts": parts, "cited_docs": cited_docs}


def parse_playbook_positions(text: str) -> dict[str, dict]:
    """Return {clause_type: {clause_number, title, parts, cited_docs}} for
    every clause_type that has a matching playbook clause, given the
    playbook's raw text.

    Split out from load_playbook_positions() in Step 10 so Playbook RAG can
    parse text fetched from the MCP contracts://playbook/negotiation
    resource (team/nodes/playbook_rag.py) the exact same way
    load_playbook_positions() parses a local disk read below — one parser,
    two sources.
    """
    by_number = {c["clause_id"]: c for c in split_into_clauses(text)}

    positions = {}
    for clause_type, clause_number in CLAUSE_TYPE_TO_PLAYBOOK_CLAUSE.items():
        clause = by_number.get(clause_number)
        if clause is None:
            continue
        positions[clause_type] = {
            "clause_number": clause_number,
            "title": clause["title"],
            **_parse_position_body(clause["body"]),
        }
    return positions


def load_playbook_positions() -> dict[str, dict]:
    """Local-disk convenience wrapper around parse_playbook_positions().

    Playbook RAG itself no longer calls this as of Step 10 (it fetches the
    playbook through the MCP server instead — see
    team/nodes/playbook_rag.py's _positions()); this is kept for anything
    that legitimately reads the playbook straight off disk, e.g.
    mcp_server/contract_server.py's own negotiation_playbook() resource,
    and for tests/scripts that don't need MCP in the loop at all.
    """
    text = PLAYBOOK_PATH.read_text(encoding="utf-8")
    return parse_playbook_positions(text)