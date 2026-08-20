"""
ShopSense M5 - Step 9 tests: workflow/records_loader.py

Fully executed here, NO langgraph needed - load_records()/unique_id() have
zero langgraph dependency, unlike scripts/run_workflow.py itself (which
does `from langgraph.types import Command` at module level and so cannot
be imported at all without a real langgraph install - this split is why
these tests can run regardless).
"""

import json

from workflow.records_loader import load_records, unique_id


def _write_jsonl(tmp_path, lines):
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return str(path)


def test_load_records_returns_one_to_one_indexed_pairs(tmp_path):
    path = _write_jsonl(tmp_path, [{"record_id": "A"}, {"record_id": "B"}])
    records = load_records(path)
    assert records == [(1, {"record_id": "A"}), (2, {"record_id": "B"})]


def test_load_records_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"record_id": "A"}\n\n{"record_id": "B"}\n', encoding="utf-8")
    records = load_records(str(path))
    assert [r["record_id"] for _, r in records] == ["A", "B"]


def test_load_records_preserves_duplicate_record_ids(tmp_path):
    """The whole reason unique_id() exists - a repeated record_id in the
    source is a genuinely different ticket, not a line to drop. This is
    NOT a hypothetical: data/records.jsonl actually has these."""
    path = _write_jsonl(tmp_path, [
        {"record_id": "SHOPSENSE-00020", "raw_text": "first phrasing"},
        {"record_id": "SHOPSENSE-00020", "raw_text": "second phrasing"},
    ])
    records = load_records(path)
    assert len(records) == 2, "load_records must not silently drop a repeated record_id"
    assert [r["record_id"] for _, r in records] == ["SHOPSENSE-00020", "SHOPSENSE-00020"]


def test_unique_id_disambiguates_a_repeated_record_id():
    first = unique_id("SHOPSENSE-00020", line_no=11)
    second = unique_id("SHOPSENSE-00020", line_no=21)
    assert first != second


def test_unique_id_is_deterministic():
    assert unique_id("SHOPSENSE-1", 5) == unique_id("SHOPSENSE-1", 5)


def test_unique_id_embeds_the_original_record_id_for_traceability():
    assert "SHOPSENSE-1" in unique_id("SHOPSENSE-1", 5)


def test_end_to_end_with_real_shaped_duplicate_data(tmp_path):
    """Mirrors what data/records.jsonl actually contains: the same
    record_id appearing on two different lines with different content."""
    path = _write_jsonl(tmp_path, [
        {"record_id": "SHOPSENSE-00020", "raw_text": "first phrasing"},
        {"record_id": "SHOPSENSE-00020", "raw_text": "second phrasing"},
    ])
    records = load_records(path)
    assert len(records) == 2

    ids = [unique_id(r["record_id"], line_no) for line_no, r in records]
    assert len(set(ids)) == 2, "both records must get distinct graph identities"