"""Clause-aware chunking for the Kartway policy corpus.

Why not `MarkdownHeaderTextSplitter` (M3's summary assumed this would be
used): it only recognizes real `#`/`##`/`###` ATX headers, and only
`warranty-policy.md` actually uses those. Every other document in the
corpus marks its clause headings with **bold** text instead — sometimes
"**N.N Title**", sometimes "**N. Title**", sometimes the clause number
sits *outside* the bold ("4.3.1 **Explicit Request for Human Support**"),
and sometimes it's "**Section N.N: Title**". `MarkdownHeaderTextSplitter`
run as-is against this corpus would treat 13 of the 14 documents as a
single unbroken chunk each. This module recognizes both heading styles
and normalizes them to a common `(clause_number, clause_title)` pair, so
every chunk can be cited at clause granularity — which is what "evaluate
whether cited clauses are grounded in the source playbook" (M4's brief)
actually needs: a chunk id alone isn't a clause citation, "returns-policy
clause 2.3" is.

Any clause whose body text is still too long after this split is run
through `RecursiveCharacterTextSplitter` (same chunk_size/overlap as the
Day 2 Session 2 notebook) so no single embedded chunk balloons past what a
retriever should reasonably match against.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# A real Markdown ATX heading: "## 2.1 Manufacturer Warranty"
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Clause number OUTSIDE the bold: "4.3.1 **Explicit Request for Human Support**"
_NUM_THEN_BOLD_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\*\*(.+?)\*\*\s*$")
# A line that is entirely bold and nothing else: "**2.1 Standard Return Window**"
_BOLD_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")
# Extract a leading clause number from an already-isolated heading title.
_LEADING_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")
# "Section 4.3: Triggers for Escalation ..." style (escalation-tone.md).
_SECTION_RE = re.compile(r"^Section\s+(\d+(?:\.\d+)*)\s*:?\s*(.*)$", re.IGNORECASE)


@dataclass
class Clause:
    doc_slug: str
    doc_title: str
    section: str
    clause_number: Optional[str]
    clause_title: str
    text: str


def _extract_number(title: str) -> tuple[Optional[str], str]:
    m = _LEADING_NUM_RE.match(title)
    if m:
        return m.group(1), m.group(2).strip()
    return None, title


def _detect_heading(line: str) -> Optional[tuple[Optional[str], str]]:
    """Return (clause_number, clause_title) if `line` is a heading, else None."""
    stripped = line.strip()
    if not stripped:
        return None

    m = _MD_HEADING_RE.match(stripped)
    if m:
        num, title = _extract_number(m.group(2).strip())
        return num, title or m.group(2).strip()

    m = _NUM_THEN_BOLD_RE.match(stripped)
    if m:
        return m.group(1), m.group(2).strip()

    m = _BOLD_LINE_RE.match(stripped)
    if m:
        title = m.group(1).strip()
        num, clean_title = _extract_number(title)
        if num is None:
            sm = _SECTION_RE.match(title)
            if sm:
                num = sm.group(1)
                clean_title = sm.group(2).strip() or clean_title
        return num, clean_title

    return None


def parse_document(md_path: Path, doc_slug: str, doc_title: str, section: str) -> list[Clause]:
    """Split one policy markdown file into clause-level `Clause` objects.

    Any heading (bold or ATX) with no body text before the next heading is
    dropped — this is what naturally filters out subtitle lines like the
    "**Kartway Shipping Policy**" restatement right under the H1, and
    metadata lines like "**Effective Date: October 1, 2023**", without
    needing a separate list of things to ignore.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    clauses: list[Clause] = []
    current: Optional[tuple[Optional[str], str, list[str]]] = None

    def flush():
        if current is None:
            return
        num, title, body_lines = current
        body = "\n".join(body_lines).strip()
        if body:
            clauses.append(Clause(doc_slug, doc_title, section, num, title or "(untitled)", body))

    for line in lines:
        heading = _detect_heading(line)
        if heading is not None:
            flush()
            num, title = heading
            current = (num, title, [])
        elif current is not None:
            current[2].append(line)
        # else: preamble before the first heading (blank lines / stray H1) — dropped.
    flush()
    return clauses


def load_index(corpus_dir: Path) -> list[dict]:
    """Read index.json. Checks directly inside `corpus_dir` first (a flat
    layout); falls back to `corpus_dir/corpus/` if that's where it
    actually lives -- this project's real layout is `data/corpus/index.json`,
    one level deeper than a flat `data/index.json` would be."""
    corpus_dir = Path(corpus_dir)
    direct = corpus_dir / "index.json"
    nested = corpus_dir / "corpus" / "index.json"
    if direct.exists():
        return json.loads(direct.read_text())
    if nested.exists():
        return json.loads(nested.read_text())
    raise FileNotFoundError(f"index.json not found at {direct} or {nested}")


def _simple_recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph/sentence-aware fallback splitter, same contract as
    LangChain's RecursiveCharacterTextSplitter (try each separator in
    order, only descend to the next when a piece is still too long)."""
    separators = ["\n\n", "\n", ". ", " "]

    def split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size or not seps:
            return [t]
        sep, rest = seps[0], seps[1:]
        parts = t.split(sep)
        pieces, buf = [], ""
        for part in parts:
            candidate = (buf + sep + part) if buf else part
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    pieces.append(buf)
                buf = part
        if buf:
            pieces.append(buf)
        out = []
        for p in pieces:
            out.extend(split(p, rest) if len(p) > chunk_size else [p])
        return out

    raw = split(text, separators)
    chunks = []
    for i, piece in enumerate(raw):
        if i > 0 and overlap > 0:
            piece = raw[i - 1][-overlap:] + piece
        piece = piece.strip()
        if piece:
            chunks.append(piece)
    return chunks


def _split_long_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Use LangChain's RecursiveCharacterTextSplitter when it's installed
    (matches the Day 2 Session 2 notebook exactly); otherwise fall back to
    `_simple_recursive_split` so this module has no hard dependency on
    `langchain-text-splitters` — same fallback philosophy as the
    notebook's PyMuPDFLoader -> PyPDFLoader path."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)
    except ImportError:
        return _simple_recursive_split(text, chunk_size, overlap)


def build_chunks(corpus_dir: Path) -> list[dict]:
    """Parse every doc in `corpus_dir`'s index.json into embeddable chunks.

    Each chunk dict: cid (== its position, so it can double as a Qdrant
    point id), text, doc_slug, doc_title, section, clause_number,
    clause_title, part (sub-split index within its clause, 0 if the clause
    wasn't split further).
    """
    corpus_dir = Path(corpus_dir)
    entries = load_index(corpus_dir)
    chunks: list[dict] = []
    for entry in entries:
        path = corpus_dir / entry["markdown"]
        for clause in parse_document(path, entry["slug"], entry["title"], entry["section"]):
            pieces = [clause.text] if len(clause.text) <= CHUNK_SIZE else _split_long_text(clause.text)
            for i, piece in enumerate(pieces):
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append({
                    "cid": len(chunks),
                    "text": piece,
                    "doc_slug": clause.doc_slug,
                    "doc_title": clause.doc_title,
                    "section": clause.section,
                    "clause_number": clause.clause_number,
                    "clause_title": clause.clause_title,
                    "part": i,
                })
    return chunks