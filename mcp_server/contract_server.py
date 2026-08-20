"""Step 9: an MCP server exposing ShopSense's vendor contract repository,
safely, over stdio. Contract repository only — the e-signature flow was
dropped from M6's scope early on per direction.

Run standalone with:  python3 mcp_server/contract_server.py
(it then sits waiting on stdin for JSON-RPC - that silence means it is
working; Ctrl+C to stop it.)

Built on the SAME sandbox pattern as the Day3 Session 2 notebook's
project_mcp_server.py, and imports FastMCP the same way the notebook does
(`from fastmcp import FastMCP`) when that standalone package is installed,
falling back to `mcp.server.fastmcp.FastMCP` -- the FastMCP implementation
bundled with the official `mcp` SDK -- when it isn't. The two expose the
same decorator API (`@mcp.tool()`, `@mcp.resource()`, `mcp.run()`), so
either import path runs the exact same code below unchanged.

This fallback exists because the two dev environments this project has run
in disagree: the build sandbox has `mcp` (with `mcp.server.fastmcp`) but
could not install the standalone `fastmcp` package at all; some `mcp`
package versions elsewhere ship `mcp.client.*` but not `mcp.server.fastmcp`,
in which case the standalone `fastmcp` package (`pip install fastmcp`) is
what's needed instead. Whichever one is actually on your machine, this file
should now import cleanly either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from fastmcp import FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

# Resolve every path from __file__, NOT os.getcwd() -- an MCP client launches
# this file as a SUBPROCESS and that subprocess's working directory is not
# guaranteed to be wherever the client itself was started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = PROJECT_ROOT / "data" / "contracts"
PLAYBOOK_PATH = PROJECT_ROOT / "data" / "playbook" / "negotiation-playbook.md"
MAX_READ_BYTES = 50_000  # generous headroom -- the largest contract today is ~3 KB

mcp = FastMCP("shopsense-contracts")


def _safe_path(filename: str, root: Path = CONTRACTS_DIR) -> Path:
    """Resolve `filename` inside `root`, or raise. THE security boundary of
    this server -- every tool that touches the filesystem goes through this.

    Four checks, each closing a hole the previous one leaves open:
      1. bare-filename rule   - reject anything with a path separator, or
                                 "", ".", ".." outright, before ever touching
                                 the filesystem. The contract repository is
                                 deliberately flat (no subdirectories), so a
                                 legitimate call never needs one.
      2. resolve()            - collapses '..' and symlinks into one real
                                 absolute path, so step 3 compares the actual
                                 destination rather than the string sent.
      3. is_relative_to()     - the destination must land inside root. This
                                 is what rejects both a smuggled absolute
                                 path and any traversal that survives step 1.
      4. is_symlink()         - a symlink INSIDE the sandbox can still point
                                 outside it. Refuse them outright.

    Raises ValueError rather than returning None -- under MCP an exception
    becomes a protocol-level tool error the client sees; a silent None
    becomes a confusing empty result instead.
    """
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        raise ValueError(f"filename must be a bare name, e.g. 'vendor_x_agreement.md': {filename!r}")
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the sandbox: {filename!r} is outside {root.name}/")
    if candidate.is_symlink():
        raise ValueError(f"symlinks are not allowed: {filename!r}")
    return candidate


@mcp.tool()
def list_contracts() -> list[str]:
    """List every vendor contract in the repository.

    Returns:
        Bare filenames (e.g. "vendor_payments_processor_agreement.md"),
        sorted alphabetically. Pass any of these straight to read_contract.
    """
    return sorted(
        p.name for p in CONTRACTS_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


@mcp.tool()
def read_contract(filename: str) -> str:
    """Read one vendor contract's full text from the repository.

    Args:
        filename: a bare filename from list_contracts(), e.g.
            "vendor_payments_processor_agreement.md". No directory
            components are accepted.

    Returns:
        The contract's raw Markdown text, truncated to 50000 bytes.
    """
    target = _safe_path(filename)
    if not target.is_file():
        raise ValueError(f"not a file: {filename!r}")
    return target.read_text(encoding="utf-8", errors="replace")[:MAX_READ_BYTES]


# THERE IS NO write_contract TOOL, AND NO delete TOOL. Not "guarded" --
# absent. This is a read-only repository: the e-signature / write-back flow
# was explicitly dropped from M6's scope, so there was never a legitimate
# reason for this server to accept writes, and a capability that doesn't
# exist can't be talked into misuse.


@mcp.resource("contracts://playbook/negotiation")
def negotiation_playbook() -> str:
    """ShopSense's negotiation playbook -- the Preferred/Fallback/Unacceptable
    position for every clause type, cited against real policy docs.

    A RESOURCE, not a tool: Playbook RAG always needs this exact document to
    do its job, so the application loads it directly (team/playbook_index.py
    already parses this same file from local disk) -- there's no judgment
    call for a model to make about whether to fetch it, unlike deciding
    which contract to read_contract(). Exposed here so an MCP client (in
    Step 10, or any other client, e.g. Claude Desktop) can browse it too.
    """
    return PLAYBOOK_PATH.read_text(encoding="utf-8", errors="replace")[:MAX_READ_BYTES]


if __name__ == "__main__":
    # NEVER print to stdout in a stdio server -- stdout IS the JSON-RPC
    # channel; one stray print() corrupts the stream and the client fails
    # with a parse error nowhere near the real cause. mcp.run() already
    # keeps its own banner/logging off stdout.
    mcp.run()