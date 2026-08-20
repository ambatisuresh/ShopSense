"""
ShopSense M5 - Step 8 tests: the langgraph-independent half of
workflow/checkpointing.py (`thread_config`, `checkpoint_file_size`).

Split into its OWN file, separate from test_checkpointing_sqlite.py, for
the SAME concrete reason test_routing.py was split out of test_graph.py:
`pytest.importorskip` placed inside a module skips that ENTIRE module,
including test functions defined BEFORE the importorskip call - Python
cannot partially import a module. Confirmed again here directly: an
earlier single-file version of these tests reported "collected 0 items /
1 skipped" for every test below, none of which touch langgraph at all.

`thread_config()` and `checkpoint_file_size()` are plain functions with no
langgraph dependency - must run in every environment.
"""

import pytest

from workflow.checkpointing import checkpoint_file_size, thread_config


def test_thread_config_wraps_ticket_id_as_thread_id():
    assert thread_config("T-1") == {"configurable": {"thread_id": "T-1"}}


def test_thread_config_is_one_to_one_with_ticket_id():
    assert thread_config("A") != thread_config("B")


def test_thread_config_rejects_empty_ticket_id():
    with pytest.raises(ValueError):
        thread_config("")


def test_thread_config_rejects_none_ticket_id():
    with pytest.raises(ValueError):
        thread_config(None)


def test_checkpoint_file_size_none_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.sqlite"
    assert checkpoint_file_size(str(missing)) is None


def test_checkpoint_file_size_reports_bytes_for_an_existing_file(tmp_path):
    f = tmp_path / "exists.sqlite"
    f.write_bytes(b"\x00" * 42)
    assert checkpoint_file_size(str(f)) == 42