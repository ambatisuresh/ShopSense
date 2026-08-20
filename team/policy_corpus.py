"""Chunks the 5 real ShopSense policy docs the negotiation playbook cites,
and exposes them for BM25 retrieval.

M4's real rag/chunking.py (not available in this session — it lives in your
project's own codebase, not in the docs this session can read) is documented
as needing to support 4 different clause-header shapes across its 14-document
corpus. Only 2 of the 5 docs relevant here (refund-authority.md,
returns-policy.md) use the "**N.N Title**" shape team/parsing.py already
supports; the other 3 use shapes of their own, handled below. Chunking is
deliberately kept at each document's top-level (or near-top-level) section
granularity rather than attempting full clause-number precision — enough for
BM25 to surface the right supporting passage, without re-deriving the whole
of M4's real chunker sight unseen.
"""
from __future__ import annotations

import re
from pathlib import Path

from team.parsing import CLAUSE_HEADER_RE

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = REPO_ROOT / "data" / "policy"


def _flat_split(text: str, matches: list[tuple[int, int, str, str]]) -> list[dict]:
    """Split `text` into chunks between consecutive heading matches.

    `matches`: (match_start, match_end, section_id, heading_title) tuples,
    any order — sorted by position here. Each chunk's body is the text
    between that heading's end and the next heading's start.
    """
    matches = sorted(matches, key=lambda m: m[0])
    chunks = []
    for i, (_start, end, section_id, title) in enumerate(matches):
        body_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        body = text[end:body_end].strip()
        chunks.append({"section_id": section_id, "heading": title, "body": body})
    return chunks


def _chunk_bold_nn(text: str) -> list[dict]:
    """'**N.N Title**' shape — refund-authority.md, returns-policy.md."""
    matches = [
        (m.start(), m.end(), m.group(1), m.group(2).strip())
        for m in CLAUSE_HEADER_RE.finditer(text)
    ]
    return _flat_split(text, matches)


_TOP_LEVEL_BOLD_N_RE = re.compile(r"^\*\*(\d+)\.\s+([^*]+)\*\*\s*$", re.MULTILINE)


def _chunk_shipping(text: str) -> list[dict]:
    """'**N. Title**' top-level shape — shipping-policy.md. Sub-items (1.1.,
    3.1, lettered a/b/c) stay inside their parent section's chunk rather
    than being split out further."""
    matches = [
        (m.start(), m.end(), m.group(1), m.group(2).strip())
        for m in _TOP_LEVEL_BOLD_N_RE.finditer(text)
    ]
    return _flat_split(text, matches)


_ATX_RE = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.MULTILINE)
_LEADING_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*(.*)$")


def _chunk_warranty(text: str) -> list[dict]:
    """'##'/'###' ATX headers, with the clause number embedded in the
    heading text itself — warranty-policy.md."""
    matches = []
    for m in _ATX_RE.finditer(text):
        heading_text = m.group(1).strip()
        num_match = _LEADING_NUMBER_RE.match(heading_text)
        if num_match and num_match.group(1):
            section_id = num_match.group(1)
            title = num_match.group(2).strip() or heading_text
        else:
            section_id, title = "", heading_text
        matches.append((m.start(), m.end(), section_id, title))
    return _flat_split(text, matches)


_SECTION_BOLD_RE = re.compile(
    r"^\*\*Section\s+(\d+(?:\.\d+)*):\s*([^*]+)\*\*\s*$", re.MULTILINE
)
_NUMBER_OUTSIDE_BOLD_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+\*\*([^*]+)\*\*", re.MULTILINE
)


def _chunk_escalation(text: str) -> list[dict]:
    """'**Section N.N: Title**' (top) + 'N.N.N **Title**' (sub, number
    outside the bold) shapes — escalation-tone.md. Chunks at the finer
    sub-clause level, since that's what the playbook's citations point to
    (e.g. escalation-tone.md §4.3.6)."""
    matches = [
        (m.start(), m.end(), m.group(1), m.group(2).strip())
        for m in _SECTION_BOLD_RE.finditer(text)
    ]
    matches += [
        (m.start(), m.end(), m.group(1), m.group(2).strip())
        for m in _NUMBER_OUTSIDE_BOLD_RE.finditer(text)
    ]
    return _flat_split(text, matches)


CHUNKERS = {
    "refund-authority.md": _chunk_bold_nn,
    "returns-policy.md": _chunk_bold_nn,
    "shipping-policy.md": _chunk_shipping,
    "warranty-policy.md": _chunk_warranty,
    "escalation-tone.md": _chunk_escalation,
}


def load_policy_corpus(policy_dir: Path = POLICY_DIR) -> list[dict]:
    """Return a flat list of {doc, section_id, heading, body} chunks across
    all 5 policy docs, in CHUNKERS' key order."""
    chunks = []
    for filename, chunk_fn in CHUNKERS.items():
        text = (policy_dir / filename).read_text(encoding="utf-8")
        for chunk in chunk_fn(text):
            chunks.append({"doc": filename, **chunk})
    return chunks