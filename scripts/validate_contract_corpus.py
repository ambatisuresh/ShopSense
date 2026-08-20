"""Step 1 validator for the M6 contract repository + negotiation playbook.

Run standalone:
    python3 scripts/validate_contract_corpus.py

This does NOT depend on rag/chunking.py (that real compatibility check happens
in Step 4, once the Playbook RAG agent actually indexes this corpus). This is
a cheap, dependency-free pre-check that:
  1. every expected file exists and is non-empty,
  2. every file's clauses parse under the "**N.N Title**" header shape already
     supported by rag/chunking.py (same shape used in refund-authority.md and
     returns-policy.md, per the M4 build notes),
  3. the specific clauses this corpus was DESIGNED around (the ones meant to
     trip the Playbook RAG / Legal Reviewer agents later) are actually present,
  4. every playbook position that is supposed to cite an internal policy doc
     still cites it (catches an accidental edit silently dropping a citation).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "data" / "contracts"
PLAYBOOK_DIR = REPO_ROOT / "data" / "playbook"

CONTRACT_FILES = [
    "vendor_payments_processor_agreement.md",
    "vendor_fulfillment_logistics_agreement.md",
    "vendor_warranty_repair_partner_agreement.md",
    "vendor_returns_processing_agreement.md",
]
PLAYBOOK_FILE = "negotiation-playbook.md"

# The clauses each contract was deliberately written around. If any of these
# go missing, the redline scenarios Steps 3-6 are built to exercise silently
# disappear.
EXPECTED_CLAUSES = {
    "vendor_payments_processor_agreement.md": {
        "3.1": "Refund Settlement Authority",   # the $50-vs-INR-2,000 conflict
        "4.1": "Term and Renewal",              # 30-day notice, below playbook floor
        "5.1": "Limitation of Liability",       # 1-month-fees cap, below playbook floor
        "6.1": "Indemnification",               # one-sided, favors vendor
    },
    "vendor_fulfillment_logistics_agreement.md": {
        "2.1": "Delivery Service Levels",       # slower than shipping-policy.md SLA
        "3.1": "Delay Compensation to Kartway", # flat 5%, below customer-owed tiers
        "4.1": "Lost-in-Transit Declaration",   # 21 days vs. 15-day customer threshold
        "5.1": "Disputed Delivery Investigation", # 10 days vs. 5-day customer commitment
        "6.1": "Liability for Lost or Damaged Goods", # flat $100 cap
    },
    "vendor_warranty_repair_partner_agreement.md": {
        "2.1": "Repair Turnaround",             # 15 days vs. 10-day customer commitment
        "2.2": "Replacement Turnaround",        # vague "best efforts", no fixed SLA
        "3.1": "Claim Eligibility Assessment",  # broader void criteria than policy
        "3.2": "Repair vs. Replacement Determination", # vendor discretion, not Kartway's
        "5.1": "Claim Status Reporting",        # 10 days vs. 5-day customer commitment
    },
    "vendor_returns_processing_agreement.md": {
        "2.1": "Return Intake Window",          # 5 days, tighter than 7-day customer promise
        "2.2": "Condition Grading",             # binary grading, no tiered mapping
        "2.3": "Excluded Categories",           # no grocery/gift-card exclusion
        "3.1": "Inspection Turnaround",         # buffer risk vs. digital-wallet refund SLA
        "4.1": "Liability for Mishandled Returns", # capped at processing fee only
    },
}

# Playbook clauses that MUST cite the internal policy doc backing that position.
PLAYBOOK_CITATIONS = {
    "1.1": ["refund-authority.md", "escalation-tone.md"],
    "1.2": ["returns-policy.md"],
    "2.1": ["shipping-policy.md"],
    "2.2": ["shipping-policy.md"],
    "2.3": ["shipping-policy.md"],
    "2.4": ["shipping-policy.md"],
    "3.1": ["warranty-policy.md"],
    "3.2": ["warranty-policy.md"],
    "3.3": ["warranty-policy.md"],
    "3.4": ["warranty-policy.md"],
    "3.5": ["warranty-policy.md"],
    "4.1": ["returns-policy.md"],
    "4.2": ["returns-policy.md"],
    "4.3": ["returns-policy.md"],
    "4.4": ["returns-policy.md"],
    "5.6": ["shipping-policy.md"],
}

# Matches lines like "**3.1 Refund Settlement Authority**" — the same
# "**N.N Title**" shape rag/chunking.py already supports (see refund-authority.md,
# returns-policy.md). Deliberately NOT using the other 3 header shapes M4 had to
# support, so this corpus needs zero chunker changes in Step 4.
CLAUSE_HEADER_RE = re.compile(r"^\*\*(\d+\.\d+) ([^*]+)\*\*\s*$", re.MULTILINE)


def parse_clauses(text: str) -> dict[str, str]:
    """Return {clause_id: title} for every '**N.N Title**' header in text."""
    return {cid: title.strip() for cid, title in CLAUSE_HEADER_RE.findall(text)}


def validate() -> list[str]:
    """Run every check; return a list of human-readable failures (empty = pass)."""
    failures: list[str] = []

    for filename, expected in EXPECTED_CLAUSES.items():
        path = CONTRACTS_DIR / filename
        if not path.is_file():
            failures.append(f"missing contract file: {filename}")
            continue
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 200:
            failures.append(f"{filename}: suspiciously short ({len(text)} chars)")
        clauses = parse_clauses(text)
        if not clauses:
            failures.append(f"{filename}: no '**N.N Title**' clause headers found at all")
        for cid, title_fragment in expected.items():
            if cid not in clauses:
                failures.append(f"{filename}: expected clause {cid} is missing")
            elif title_fragment.lower() not in clauses[cid].lower():
                failures.append(
                    f"{filename}: clause {cid} title changed — "
                    f"expected to contain {title_fragment!r}, found {clauses[cid]!r}"
                )

    playbook_path = PLAYBOOK_DIR / PLAYBOOK_FILE
    if not playbook_path.is_file():
        failures.append(f"missing playbook file: {PLAYBOOK_FILE}")
    else:
        text = playbook_path.read_text(encoding="utf-8")
        clauses = parse_clauses(text)
        if not clauses:
            failures.append(f"{PLAYBOOK_FILE}: no '**N.N Title**' clause headers found at all")
        for cid, required_docs in PLAYBOOK_CITATIONS.items():
            if cid not in clauses:
                failures.append(f"{PLAYBOOK_FILE}: expected clause {cid} is missing")
                continue
            # Re-extract this clause's body (from its header to the next header).
            start = text.index(f"**{cid} ")
            next_header = CLAUSE_HEADER_RE.search(text, start + 1)
            body = text[start: next_header.start()] if next_header else text[start:]
            for doc in required_docs:
                if doc not in body:
                    failures.append(
                        f"{PLAYBOOK_FILE}: clause {cid} no longer cites {doc}"
                    )

    return failures


def main() -> None:
    failures = validate()
    total_contract_clauses = sum(
        len(parse_clauses((CONTRACTS_DIR / f).read_text(encoding="utf-8")))
        for f in EXPECTED_CLAUSES
        if (CONTRACTS_DIR / f).is_file()
    )
    playbook_clause_count = (
        len(parse_clauses((PLAYBOOK_DIR / PLAYBOOK_FILE).read_text(encoding="utf-8")))
        if (PLAYBOOK_DIR / PLAYBOOK_FILE).is_file()
        else 0
    )
    print(f"contract files checked : {len(EXPECTED_CLAUSES)}")
    print(f"total contract clauses : {total_contract_clauses}")
    print(f"playbook clauses       : {playbook_clause_count}")
    print()
    if failures:
        print(f"FAIL - {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("PASS - contract repository and negotiation playbook are structurally sound.")


if __name__ == "__main__":
    main()
