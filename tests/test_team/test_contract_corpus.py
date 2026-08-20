"""Step 1 tests: the M6 contract repository + negotiation playbook.

Run:
    pytest tests/test_team/test_contract_corpus.py -v

Zero external dependencies (no langgraph/fastmcp/rag needed) — this only
exercises scripts/validate_contract_corpus.py, so it can run before any of
the M6 pip installs happen.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_contract_corpus as vcc  # noqa: E402


@pytest.mark.parametrize("filename", list(vcc.EXPECTED_CLAUSES))
def test_contract_file_exists_and_is_nonempty(filename):
    path = vcc.CONTRACTS_DIR / filename
    assert path.is_file(), f"{filename} not found under {vcc.CONTRACTS_DIR}"
    assert len(path.read_text(encoding="utf-8").strip()) > 200


def test_playbook_file_exists_and_is_nonempty():
    path = vcc.PLAYBOOK_DIR / vcc.PLAYBOOK_FILE
    assert path.is_file(), f"{vcc.PLAYBOOK_FILE} not found under {vcc.PLAYBOOK_DIR}"
    assert len(path.read_text(encoding="utf-8").strip()) > 500


@pytest.mark.parametrize("filename", list(vcc.EXPECTED_CLAUSES))
def test_clause_headers_parse_in_supported_shape(filename):
    """Every contract must use the '**N.N Title**' header shape rag/chunking.py
    already supports — this is what lets Step 4 index this corpus with zero
    chunker changes."""
    text = (vcc.CONTRACTS_DIR / filename).read_text(encoding="utf-8")
    clauses = vcc.parse_clauses(text)
    assert clauses, f"{filename}: no clause headers matched the supported shape"


@pytest.mark.parametrize(
    "filename,clause_id,title_fragment",
    [
        (fname, cid, frag)
        for fname, clauses in vcc.EXPECTED_CLAUSES.items()
        for cid, frag in clauses.items()
    ],
)
def test_expected_redline_target_clauses_present(filename, clause_id, title_fragment):
    """Each clause this corpus was deliberately built around (the ones meant to
    trip Playbook RAG / Legal Reviewer later) must still be present and titled
    the way the later steps expect."""
    text = (vcc.CONTRACTS_DIR / filename).read_text(encoding="utf-8")
    clauses = vcc.parse_clauses(text)
    assert clause_id in clauses, f"{filename}: clause {clause_id} missing"
    assert title_fragment.lower() in clauses[clause_id].lower()


@pytest.mark.parametrize("clause_id,required_docs", list(vcc.PLAYBOOK_CITATIONS.items()))
def test_playbook_clause_cites_its_backing_policy_doc(clause_id, required_docs):
    text = (vcc.PLAYBOOK_DIR / vcc.PLAYBOOK_FILE).read_text(encoding="utf-8")
    clauses = vcc.parse_clauses(text)
    assert clause_id in clauses, f"playbook clause {clause_id} missing"
    start = text.index(f"**{clause_id} ")
    next_header = vcc.CLAUSE_HEADER_RE.search(text, start + 1)
    body = text[start: next_header.start()] if next_header else text[start:]
    for doc in required_docs:
        assert doc in body, f"playbook clause {clause_id} no longer cites {doc}"


def test_refund_authority_vs_escalation_tone_conflict_is_preserved():
    """This is THE carried-forward item from M2/M4/M5 (flagged, never resolved):
    refund-authority.md's INR 2,000 auto-approval cap vs. escalation-tone.md's
    $50 automated-refund-cap line. M6's payments contract (Clause 3.1) and
    playbook (Clause 1.1) are deliberately built to force this into the open —
    if either loses its reference to the conflict, Step 4-6 lose their reason
    to exist."""
    contract_text = (
        vcc.CONTRACTS_DIR / "vendor_payments_processor_agreement.md"
    ).read_text(encoding="utf-8")
    assert "USD 50.00" in contract_text

    playbook_text = (vcc.PLAYBOOK_DIR / vcc.PLAYBOOK_FILE).read_text(encoding="utf-8")
    assert "INR 2,000" in playbook_text
    assert "escalation-tone.md" in playbook_text
    assert "conflict" in playbook_text.lower()


def test_full_validator_reports_no_failures():
    failures = vcc.validate()
    assert failures == [], "\n".join(failures)
