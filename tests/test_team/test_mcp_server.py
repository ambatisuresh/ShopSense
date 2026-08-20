"""Step 9 tests: mcp_server/contract_server.py, exercised over a REAL
stdio JSON-RPC connection to a real subprocess -- not a mock.

Run:
    pytest tests/test_team/test_mcp_server.py -v

Unlike Step 8's langgraph situation, the `mcp` SDK (server + client) is
genuinely installed in this sandbox, so this file was actually run and its
one real bug (see test_list_contracts_matches_repository's neighboring
comment) was caught and fixed here, the same way every step before Step 8
was verified.

No pytest-asyncio plugin is available wherever this runs, so rather than
depend on one, a single module-scoped fixture drives one asyncio.run() that
launches the server once and collects every response this file needs;
individual test functions are then plain sync assertions against that
shared result -- one subprocess for the whole file, not one per test.
"""
import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="mcp SDK not installed in this environment")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
SERVER_PATH = str(REPO_ROOT / "mcp_server" / "contract_server.py")

EXPECTED_CONTRACTS = {
    "vendor_payments_processor_agreement.md",
    "vendor_fulfillment_logistics_agreement.md",
    "vendor_warranty_repair_partner_agreement.md",
    "vendor_returns_processing_agreement.md",
}


def _error_text(result) -> str:
    return " ".join(getattr(b, "text", "") for b in result.content)


async def _gather() -> dict:
    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])
    out = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            out["tools"] = {t.name: t for t in (await session.list_tools()).tools}
            out["resources"] = {str(r.uri) for r in (await session.list_resources()).resources}

            list_result = await session.call_tool("list_contracts", {})
            out["list_contracts"] = list_result.structuredContent["result"]

            first = sorted(out["list_contracts"])[0]
            out["read_first_name"] = first
            out["read_first"] = await session.call_tool("read_contract", {"filename": first})

            out["escape_parent"] = await session.call_tool(
                "read_contract", {"filename": "../../../etc/passwd"})
            out["escape_absolute"] = await session.call_tool(
                "read_contract", {"filename": "/etc/passwd"})
            out["escape_nested"] = await session.call_tool(
                "read_contract", {"filename": "sub/../../escape.md"})
            out["not_found"] = await session.call_tool(
                "read_contract", {"filename": "does_not_exist.md"})

            out["playbook"] = await session.read_resource("contracts://playbook/negotiation")
    return out


@pytest.fixture(scope="module")
def server_session() -> dict:
    return asyncio.run(_gather())


# ---------------------------------------------------------------------------
# Discovery -- what the server advertises, with no hints from this test file
# ---------------------------------------------------------------------------

def test_exactly_the_two_intended_tools_exist(server_session):
    assert set(server_session["tools"]) == {"list_contracts", "read_contract"}


def test_no_write_or_delete_tool_exists(server_session):
    """The design rule this server follows: the strongest guard rail is a
    capability that was never exposed. There is no write_contract or
    delete_contract tool to guard -- it's simply absent."""
    names = set(server_session["tools"])
    assert "write_contract" not in names
    assert "delete_contract" not in names
    assert not any("delete" in n or "write" in n for n in names)


def test_negotiation_playbook_resource_is_advertised(server_session):
    assert "contracts://playbook/negotiation" in server_session["resources"]


def test_tool_docstrings_are_present_and_became_the_schema_description(server_session):
    # FastMCP turns the docstring into the JSON-RPC tool description --
    # a blank one would be a broken (silently-uncallable-correctly) tool.
    for tool in server_session["tools"].values():
        assert tool.description and len(tool.description.strip()) > 10


# ---------------------------------------------------------------------------
# Legitimate calls
# ---------------------------------------------------------------------------

def test_list_contracts_matches_repository(server_session):
    # Real bug caught running this against the live server: FastMCP returns
    # a list[str] tool result as MULTIPLE TextContent blocks plus a typed
    # structuredContent={"result": [...]}; the first version of this check
    # (and of scripts/run_mcp_server_selfcheck.py) tried json.loads() on the
    # space-joined content blocks and blew up with "Expecting value: line 1
    # column 1" -- structuredContent["result"] is the correct read.
    assert set(server_session["list_contracts"]) == EXPECTED_CONTRACTS


def test_read_contract_returns_real_contract_text(server_session):
    result = server_session["read_first"]
    assert not result.isError
    text = result.structuredContent["result"]
    assert len(text) > 100
    assert "clause" in text.lower()
    # Spot check: every contract in this repo opens with a level-1 heading.
    assert text.lstrip().startswith("#")


def test_negotiation_playbook_resource_reads_the_real_playbook(server_session):
    playbook = server_session["playbook"]
    text = "".join(getattr(c, "text", "") for c in playbook.contents)
    assert "Preferred" in text
    assert "Unacceptable" in text
    # The exact conflict Step 1's corpus was built to force into the open.
    assert "escalation-tone.md" in text and "refund-authority.md" in text


# ---------------------------------------------------------------------------
# The sandbox has to actually hold -- a model will eventually test it, so
# this test does first.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,must_contain", [
    ("escape_parent", "bare name"),
    ("escape_absolute", "bare name"),
    ("escape_nested", "bare name"),
    ("not_found", "not a file"),
])
def test_every_bad_filename_is_refused(server_session, key, must_contain):
    result = server_session[key]
    assert result.isError, f"{key} should have been refused but succeeded"
    assert must_contain in _error_text(result)


def test_refusal_does_not_leak_filesystem_content(server_session):
    """A refused read must never carry real file bytes back to the caller
    -- only the error message itself."""
    for key in ("escape_parent", "escape_absolute", "escape_nested"):
        text = _error_text(server_session[key])
        assert "root:" not in text  # /etc/passwd's tell-tale first line