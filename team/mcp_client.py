"""Step 10: the MCP client helper Extraction and Playbook RAG use to reach
mcp_server/contract_server.py, instead of reading data/contracts/ and
data/playbook/negotiation-playbook.md straight off local disk.

Two thin async functions, each spawning the server as a subprocess,
connecting, making ONE call, and shutting the server back down. This is a
per-call session, not a persistent one held open across a whole contract
review -- deliberately, per the Day3 Session 2 notebook's own guidance on
the trade-off: a persistent session pays for itself when the server holds
state between calls, uses sampling/elicitation, pushes notifications, or
setup itself is expensive. None of those apply here -- this server is
stateless, read-only, and a local subprocess spin-up is cheap -- so "fine
for a local subprocess and a handful of calls" (the notebook's own words)
applies directly, and per-call keeps the two node functions below simple:
neither has to manage a session's lifetime across multiple graph steps.
"""
from __future__ import annotations

import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = str(REPO_ROOT / "mcp_server" / "contract_server.py")


def _error_text(result) -> str:
    return " ".join(getattr(b, "text", "") for b in result.content)


async def _with_session(fn):
    """Spawn contract_server.py, initialize a session, run fn(session), and
    tear the server back down -- the one place this whole module touches
    the subprocess/handshake mechanics."""
    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


async def read_contract_via_mcp(filename: str) -> str:
    """Fetch one contract's text through the read_contract MCP tool.

    Args:
        filename: a bare filename, e.g. "vendor_payments_processor_agreement.md"
            -- matching what mcp_server/contract_server.py's _safe_path()
            requires (no directory components).

    Raises:
        ValueError if the server refused the call (unknown file, path
        rejected by the sandbox, etc.) -- the server's own error text is
        included, same as a local disk read raising FileNotFoundError would
        have told the caller what went wrong.
    """
    async def _run(session):
        result = await session.call_tool("read_contract", {"filename": filename})
        if result.isError:
            raise ValueError(f"read_contract MCP call failed for {filename!r}: {_error_text(result)}")
        return result.structuredContent["result"]

    return await _with_session(_run)


async def read_playbook_via_mcp() -> str:
    """Fetch the negotiation playbook's full text through the
    contracts://playbook/negotiation MCP resource -- a resource, not a tool,
    because Playbook RAG always needs this exact document; there's no
    judgment call for a model to make about whether to fetch it."""
    async def _run(session):
        result = await session.read_resource("contracts://playbook/negotiation")
        return "".join(getattr(c, "text", "") for c in result.contents)

    return await _with_session(_run)