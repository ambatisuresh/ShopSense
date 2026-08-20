"""
ShopSense M5 - terminal-friendly equivalent of the notebook's own
`show_graph()` helper (Day3 Session1's "program-standard graph visualiser",
provided cell `ca494da7`, reused throughout Lab A/B).

WHY THE DEMO SCRIPTS NEVER SHOWED A PICTURE UNTIL NOW: the notebook's
show_graph() calls `IPython.display.Image` + `display(...)`. Those two only
do anything inside a running Jupyter kernel - in a plain `python3 -m
scripts.run_X` terminal script, `display()` either doesn't exist (no
IPython installed) or silently does nothing (IPython installed but no
notebook frontend attached to receive the display). None of Steps 5-8's
demo scripts ever called anything like it, so no graph picture ever
appeared - not a bug in those scripts, just a helper that was never built
for this context.

This module is the terminal-appropriate version of the SAME underlying
call the notebook uses (`graph.get_graph().draw_mermaid_png()` /
`.draw_mermaid()` - both ship with the base `langgraph` package, no extra
install needed, exactly as the notebook's setup-cell comment says). Instead
of `display(Image(...))`, it writes the PNG bytes to a file you open
yourself; instead of `display(Markdown(...))`, it just prints the Mermaid
source as plain text.
"""

from typing import Optional


def show_graph(graph, title: str = "", png_path: Optional[str] = "shopsense_graph.png") -> None:
    """`graph` is a COMPILED graph (the return value of `build_graph()` /
    `.compile()`), matching the notebook's own `show_graph(app, title)`
    signature - `app` there, `graph` here, same thing.

    Behavior, mirroring the notebook's try/except exactly:
        1. Try `graph.get_graph().draw_mermaid_png()` - renders via
           mermaid.ink over the network, same as the notebook. On success,
           write the bytes to `png_path` (default "shopsense_graph.png" in
           the current directory) rather than trying to `display()` it -
           open the file yourself to view it.
        2. If that raises (no network reaching mermaid.ink, or any other
           error), fall back to `graph.get_graph().draw_mermaid()` - the
           raw Mermaid source, ships with langgraph, works with zero
           network dependency - and print it as plain text. Paste it into
           https://mermaid.live or any Markdown viewer that renders Mermaid
           to see the picture.

    Pass `png_path=None` to always print the Mermaid source instead of
    writing a file, even if PNG rendering would have succeeded - useful in
    a headless CI environment where you don't want stray image files.
    """
    if title:
        print(title)
        print("-" * len(title))

    mermaid_graph = graph.get_graph()

    if png_path is not None:
        try:
            png_bytes = mermaid_graph.draw_mermaid_png()
            with open(png_path, "wb") as f:
                f.write(png_bytes)
            print(f"(graph picture written to {png_path} - open it to view)")
            return
        except Exception as e:
            print(f"(mermaid.ink unavailable - {type(e).__name__}; Mermaid source below)")

    print(mermaid_graph.draw_mermaid())
    print("(paste the above into https://mermaid.live to view it as a picture)")


def print_node_list(graph) -> None:
    """Matches the notebook's own habit of printing the node list right
    before/after show_graph() - `sorted(n for n in graph.get_graph().nodes
    if not n.startswith("__"))`, verbatim."""
    print("nodes:", sorted(n for n in graph.get_graph().nodes if not n.startswith("__")))