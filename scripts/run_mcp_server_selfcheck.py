"""Step 9 self-check: launch mcp_server/contract_server.py as a REAL
subprocess and talk to it over real stdio JSON-RPC, using the official
`mcp` SDK's own client -- same shape as the Day3 Session 2 notebook's
sandbox self-check cell, translated from langchain-mcp-adapters (not
installed here) onto the plain `mcp.client` API (which is).

Unlike Step 8 (langgraph, uninstallable in this sandbox), this genuinely
runs end to end here: a second process is spawned, tools are discovered
with no hints from this script, both legitimate calls and four escape
attempts are made, and the resource is read.

Run:
    python3 scripts/run_mcp_server_selfcheck.py
"""
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = str(REPO_ROOT / "mcp_server" / "contract_server.py")

EXPECTED_CONTRACTS = {
    "vendor_payments_processor_agreement.md",
    "vendor_fulfillment_logistics_agreement.md",
    "vendor_warranty_repair_partner_agreement.md",
    "vendor_returns_processing_agreement.md",
}


def _tool_text(result) -> str:
    """Flatten a CallToolResult's content blocks into plain text/repr."""
    parts = []
    for block in result.content:
        parts.append(getattr(block, "text", str(block)))
    return " ".join(parts)


async def _attempt(session: ClientSession, tool_name: str, args: dict) -> str:
    """Call a tool and return the error text if the server refused it (via
    isError, the normal MCP path for a tool that raised) or if the call
    itself raised. Returns "" if the call was allowed through."""
    try:
        result = await session.call_tool(tool_name, args)
    except Exception as e:  # some transport errors surface as exceptions
        return f"{type(e).__name__}: {e}"
    if getattr(result, "isError", False):
        return _tool_text(result)
    return ""


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Discovery -- nobody told this script the tool names in advance.
            tools = (await session.list_tools()).tools
            tool_names = {t.name for t in tools}
            print("DISCOVERED TOOLS (no hints given in advance)")
            print("-" * 46)
            for t in tools:
                print(f"  {t.name}{tuple(t.inputSchema.get('properties', {}))}")
                print(f"      {t.description}")
            assert tool_names == {"list_contracts", "read_contract"}, \
                f"unexpected tool set: {tool_names}"
            assert "write_contract" not in tool_names, "a write tool must never exist here"
            assert "delete_contract" not in tool_names, "a delete tool must never exist here"

            resources = (await session.list_resources()).resources
            resource_uris = {str(r.uri) for r in resources}
            print(f"\nDISCOVERED RESOURCES: {resource_uris}")
            assert "contracts://playbook/negotiation" in resource_uris

            # 2. Legitimate calls. FastMCP returns list[str]/str results as
            # BOTH multiple TextContent blocks (content) and a typed
            # structuredContent = {"result": <the actual value>} -- use the
            # latter rather than parsing `content`, since a plain list[str]
            # is emitted as N separate text blocks, not one JSON string
            # (found by running this against the real server: my first
            # version tried json.loads() on the joined content text and blew
            # up with "Expecting value: line 1 column 1").
            list_result = await session.call_tool("list_contracts", {})
            assert not getattr(list_result, "isError", False), _tool_text(list_result)
            filenames = set(list_result.structuredContent["result"])
            print(f"\nlist_contracts() -> {sorted(filenames)}")
            assert filenames == EXPECTED_CONTRACTS, f"contract set mismatch: {filenames}"

            one = sorted(filenames)[0]
            read_result = await session.call_tool("read_contract", {"filename": one})
            assert not getattr(read_result, "isError", False), _tool_text(read_result)
            text = read_result.structuredContent["result"]
            print(f"read_contract({one!r}) -> {len(text)} chars, starts: {text[:40]!r}")
            assert len(text) > 100
            assert "clause" in text.lower()

            playbook = await session.read_resource("contracts://playbook/negotiation")
            playbook_text = "".join(
                getattr(c, "text", "") for c in playbook.contents
            )
            print(f"read_resource(playbook) -> {len(playbook_text)} chars")
            assert "Preferred" in playbook_text and "Unacceptable" in playbook_text

            # 3. Escape attempts -- the sandbox has to actually hold, because
            # a model will eventually test it for you.
            attacks = {
                "parent traversal":   ("read_contract", {"filename": "../../../etc/passwd"}),
                "absolute path":      ("read_contract", {"filename": "/etc/passwd"}),
                "nested traversal":   ("read_contract", {"filename": "sub/../../escape.md"}),
                "unknown file":       ("read_contract", {"filename": "does_not_exist.md"}),
            }
            print("\nESCAPE ATTEMPTS (all must be refused)")
            print("-" * 46)
            for label, (tool, args) in attacks.items():
                err = await _attempt(session, tool, args)
                assert err, f"SANDBOX HOLE: {label!r} was NOT refused"
                print(f"  refused  {label:<16} -> {err[:90]}")

    print("\nPASS - contract repository MCP server verified end to end: "
          "2 tools discovered, 1 resource discovered, legitimate reads "
          "succeeded, all 4 escape attempts refused, no write/delete tool "
          "exists.")


if __name__ == "__main__":
    asyncio.run(main())