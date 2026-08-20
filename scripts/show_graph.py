"""
ShopSense M5 - standalone: build the default graph and show it, terminal
style. Run this any time you just want to SEE the graph shape without
running any tickets through it - the equivalent of the notebook's own
"### The finished graph" cell (`25145970`).

Needs the real `langgraph` package - not executable in the sandbox this was
built in.
"""

from workflow.graph import build_graph
from workflow.visualize import print_node_list, show_graph

if __name__ == "__main__":
    graph = build_graph()  # every dependency defaults to an offline fixture

    print_node_list(graph)
    print()
    show_graph(graph, "ShopSense M5 - the finished ticket-review workflow", png_path="shopsense_graph.png")

"""
EXPECTED OUTPUT
---------------
nodes: ['auto_approve', 'compare_to_playbook', 'draft_redline', 'extract', 'finalize', 'human_approval']

ShopSense M5 - the finished ticket-review workflow
----------------------------------------------------
(graph picture written to shopsense_graph.png - open it to view)

  - or, if mermaid.ink is unreachable from your network:
(mermaid.ink unavailable - <ExceptionType>; Mermaid source below)
<mermaid text>
(paste the above into https://mermaid.live to view it as a picture)

Either way, the picture should show: __start__ -> extract -> compare_to_playbook
-> draft_redline, DOTTED edges draft_redline -> auto_approve / human_approval
(the router), auto_approve -> finalize (solid), DOTTED edges human_approval ->
finalize / __end__ (the second router), finalize -> __end__.
"""