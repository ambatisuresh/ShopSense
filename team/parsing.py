"""Shared clause-header parsing for the M6 contract corpus.

Generalizes the header-shape check prototyped in Step 1's
scripts/validate_contract_corpus.py (which only needed clause_id -> title)
into a full clause_id -> title -> body split, which Step 3's Extraction node
needs to classify each clause's content, not just confirm its heading exists.
"""
from __future__ import annotations

import re

# Matches lines like "**3.1 Refund Settlement Authority**" — the one header
# shape every M6 contract + the playbook use (see Step 1's build notes for
# why this shape was chosen over the other 3 rag/chunking.py supports).
CLAUSE_HEADER_RE = re.compile(r"^\*\*(\d+\.\d+) ([^*]+)\*\*\s*$", re.MULTILINE)


def split_into_clauses(text: str) -> list[dict]:
    """Split `text` into clauses at each '**N.N Title**' header.

    Returns a list of {clause_id, title, body} in document order. `body` is
    everything between this header and the next header (or end of file),
    NOT including the header line itself.
    """
    matches = list(CLAUSE_HEADER_RE.finditer(text))
    clauses = []
    for i, match in enumerate(matches):
        clause_id, title = match.group(1), match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        clauses.append({"clause_id": clause_id, "title": title, "body": body})
    return clauses