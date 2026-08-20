# ShopSense — Customer Care & Order Operations Assistant

## Overview
A multi-agent customer support assistant for a mid-size e-commerce marketplace.
Classifies and resolves order/refund/shipping tickets, answers policy questions
via RAG, executes order actions through tools, and escalates complex or
emotional cases to human agents.

## Tech Stack
- **LLM routing:** LiteLLM (provider-agnostic — Google AI Studio/Gemini and
  OpenAI both verified, switchable via `LLM_PROVIDER`)
- **Vector DB:** Qdrant
- **Retrieval:** BM25 (`rank_bm25`) + dense search, combined via Reciprocal
  Rank Fusion, re-scored with a cross-encoder (`sentence-transformers`) —
  see M4
- **Memory:** Supermemory
- **Observability:** Langfuse
- **Orchestration:** LangChain (M2's single-agent tool-calling loop) →
  LangGraph (M5's ticket-review workflow — conditional routing, real
  `interrupt()`-based human-in-the-loop approval, durable SQLite
  checkpointing; multi-agent orchestration is M6)

## Milestones
| # | Milestone | Status |
|---|-----------|--------|
| M1 | Provider-agnostic LLM client + structured intake | ✅ Completed |
| M2 | Tool-enabled single agent | ✅ Completed |
| M3 | Persistent memory + semantic index | ✅ Completed |
| M4 | Production RAG + evaluation baseline | ✅ Completed |
| M5 | Orchestrated LangGraph workflow with checkpointing | ✅ Completed |
| M6 | Multi-agent team + MCP integration | ⬜ Not started |
| M7 | Observability + reliability hardening | ⬜ Not started |
| M8 | End-to-end evaluation, guardrails & deployment | ⬜ Not started |

> **M5 note:** LangGraph ticket-review workflow — `extract` (M1 parser
> adapter) → `compare_to_playbook` (refund-cap + escalation-tone policy
> checks, M2/M4 adapters) → `draft_redline` (liability-language sanitizer)
> → conditional route → `auto_approve` (standard) or `human_approval` (real
> `interrupt()`, non_standard) → `finalize` (the only node allowed to
> commit — see below). Built in 9 steps, each with its own tests and demo
> script:
> - **State/nodes (Steps 1–4):** `workflow/state.py`'s control/audit field
>   split (`Annotated[list, add]` only on `audit_log`); `extract`,
>   `compare_to_playbook`, `draft_redline` — every external dependency
>   (M1 parser, M2 refund calc, M4 retrieval) wired via a `build_X_node(...)`
>   factory with an offline fixture default, so every node is testable and
>   demoable with zero live services.
> - **Routing + interrupt (Steps 5–6):** `workflow/routing.py`'s
>   `route_after_review`/`route_after_human` are pure `state -> str`
>   functions (routers can't write state — architecturally enforced by
>   LangGraph itself) and fail closed on any unrecognized/missing value.
>   `human_approval` uses a real `interrupt()`, lazily imported so the
>   module stays importable without `langgraph` installed.
> - **Finalize + the commit flag (Step 7):** `evaluate_refund` now takes a
>   `commit: bool` — `compare_to_playbook` always calls it with
>   `commit=False` (classify only), `finalize` is the *only* call site that
>   ever passes `commit=True`. This keeps pre-approval classification and
>   post-approval commit architecturally distinct, since it's unconfirmed
>   whether M2's real `process_refund` has side effects (flagged, not
>   guessed at — see `compare_to_playbook.py`'s "STEP 7 ADDITION" note).
> - **Durable checkpointing (Step 8):** `InMemorySaver` by default
>   (ephemeral, safe for tests); `build_graph(checkpoint_db_path=...)` opts
>   into a real `SqliteSaver` (`workflow/checkpointing.py`) for state that
>   survives a genuine process restart — proven via two *separate* `python3`
>   processes, not just a kernel-restart-in-one-session
>   (`scripts/run_checkpointing_pause.py` / `_resume.py`).
> - **Batch validation (Step 9):** `scripts/run_workflow.py` runs every
>   ticket in a records.jsonl file through the compiled graph end to end and
>   checks the batch against `workflow/checklist.py`'s 5 hard structural
>   invariants (no crashes, every ticket reaches a terminal state, complete
>   audit trail, JSON-serializable results, commit flag matches refund
>   need) — kept strictly separate from an *informational* classification-
>   vs-`requires_human` agreement metric, which is expected to diverge from
>   100% since this workflow's non_standard triggers are narrower in scope
>   than that ground-truth label.
>
> Test suite: `tests/test_workflow/` (18 files, 163 tests) — **all 163
> passing**, confirmed in the real `shopsensevenv` with `langgraph` +
> `langgraph-checkpoint-sqlite` installed. This includes `test_graph.py`
> and `test_checkpointing_sqlite.py` (the two files needing a real
> `langgraph` install, gated behind `pytest.importorskip` so the rest of
> the suite still runs without it) — both fully green, covering the
> compiled graph's routing/interrupt/resume behavior and real SQLite
> checkpoint durability end to end, not just the langgraph-independent
> ~150 tests. One test-assertion bug found along the way (byte-identical
> file size assumed on sqlite reattach, when SQLite is free to grow the
> file on reopen) was fixed and is included in that count.
>

> **M4 note:** production RAG over the 14-document Kartway policy corpus —
> clause-aware chunking, dense (Qdrant) + BM25 hybrid search (Reciprocal
> Rank Fusion), cross-encoder reranking, cited/grounded generation with
> prompt-injection defense, and a two-layer evaluation harness (retrieval
> metrics + citation-integrity/groundedness) against `data/eval/golden_set.json`.
> Implemented and covered by `tests/test_rag/` (9 files, all offline/
> deterministic, no live API calls). BM25 + cross-encoder reranking verified
> live against the real 20-item golden set: precision@5 0.376→0.482,
> recall@5 0.225→0.302, MRR 0.627→0.765 versus BM25 alone.

> **M3 note:** persistent per-customer memory (episodic + semantic, via
> Supermemory) is implemented and covered by `tests/test_memory.py`
> (fake Supermemory client, no live API calls) and `tests/test_agent.py`'s
> `TestM3MemoryIntegration`. 

> **M2 note:** all four tools and the agent loop are implemented and covered
> by a monkeypatched/fake-LLM test suite (`tests/test_tools.py`,
> `tests/test_agent.py`). A full live batch run against a real model via
> `scripts/run_agent.py` (no `--skip-parse`) is the remaining validation step
> before treating M2 as fully signed off.

## Setup

```bash
python3.12 -m venv shopsensevenv
source shopsensevenv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
LLM_PROVIDER=GEMINI

GEMINI_API_KEY=your_key_here
GEMINI_LLM_MODEL=gemini/gemini-2.0-flash

OPENAI_API_KEY=your_key_here
OPENAI_LLM_MODEL=openai/gpt-4o-mini

SUPERMEMORY_API_KEY=your_key_here

# M4 — required for rag/embeddings.py's real GeminiEmbedder (reads this
# specific var name via langchain-google-genai, separate from
# GEMINI_API_KEY above which LiteLLM's "gemini/..." calls use)
GOOGLE_API_KEY=your_key_here

# M4 — required for rag/qdrant_index.py (Qdrant Cloud collection)
QDRANT_URL=your_cluster_url_here
QDRANT_API_KEY=your_key_here
```

Only the API key and `*_LLM_MODEL` var matching your active `LLM_PROVIDER`
are required — the other provider's vars can stay unset until you switch.
`SUPERMEMORY_API_KEY` is required as of M3 — `scripts/run_agent.py` and
`memory/customer_memory.py` construct a `CustomerMemory` unconditionally
(including under `--skip-parse`), so it must be set even for dev-mode runs.
`GOOGLE_API_KEY` and `QDRANT_URL`/`QDRANT_API_KEY` are required as of M4 for
the live embedding and Qdrant paths — every `rag/` module also has an
offline fallback (see "Running the M4 RAG Pipeline") that needs neither.

## Project Structure

```
shopsense/
├── core/
│   ├── llm_client.py       # LiteLLM wrapper — sole call site for completion()
│   └── schema.py           # Pydantic schemas (SupportTicket, enums)
├── intake/
│   ├── reader.py           # JSONL reader — extracts ticket_id + raw_text
│   └── parser.py           # raw text -> validated SupportTicket, via LLMClient
├── tools/
│   ├── data_loader.py      # cached, indexed access to the mock data tables
│   ├── reliability.py      # retry + circuit breaker + call logging wrapper
│   ├── order_lookup.py     # mock order-lookup API
│   ├── shipping_tracker.py # mock shipping-tracker API (delay/lost-in-transit)
│   ├── refund_calculator.py # refund-amount calculator (policy math, pure fn)
│   ├── refund_replace.py   # mock refund/replace API (approval tiers, fraud checks)
│   └── customer_memory_tool.py # [M3] note_customer_preference tool factory —
│                                # customer_ref bound via closure per ticket, never model-supplied
├── memory/
│   └── customer_memory.py  # [M3] CustomerMemory — episodic + semantic memory via Supermemory,
│                            # container_tag-scoped per customer, retry/backoff on all external calls
├── agent/
│   ├── prompts.py          # system prompt + SupportTicket -> message formatting
│   │                        # [M3] +note_customer_preference tool doc, +customer-history rules,
│   │                        # +customer_context param on format_ticket_for_agent
│   └── support_agent.py    # bounded tool-calling loop (LangChain ChatLiteLLM)
│                            # [M3] run_support_agent(ticket, customer_ref, memory, max_iterations) —
│                            # tools now built per-ticket via build_tools() to scope the memory tool
├── rag/                     # [M4] production RAG over the policy corpus
│   ├── chunking.py           # clause-aware markdown splitter (bold + ATX + numeral-outside-bold headers)
│   ├── embeddings.py          # GeminiEmbedder (real) + FakeEmbedder (deterministic offline)
│   ├── qdrant_index.py        # Qdrant Cloud collection create/upsert/dense_search
│   ├── local_index.py         # in-memory dense index — offline stand-in for Qdrant
│   ├── bm25_index.py          # BM25 keyword search (rank_bm25, or a pure-Python fallback)
│   ├── hybrid.py               # Reciprocal Rank Fusion (dense + BM25)
│   ├── rerank.py               # CrossEncoderReranker (real) + FakeReranker (offline)
│   ├── generate.py             # cited, prompt-injection-resistant answer generation
│   ├── eval_retrieval.py       # precision@k / recall@k / MRR vs data/eval/golden_set.json
│   └── eval_groundedness.py    # citation-integrity + heuristic/LLM-judge faithfulness scoring
├── workflow/                 # [M5] LangGraph ticket-review workflow
│   ├── state.py                 # TicketReviewState — control fields (overwrite) vs audit_log (Annotated[list, add])
│   ├── nodes/
│   │   ├── extract.py               # M1 parser + M3 customer-resolution adapter
│   │   ├── compare_to_playbook.py   # refund-cap + escalation-tone concerns, classification, citations
│   │   ├── draft_redline.py         # liability-language sanitizer, deterministic + LLM composers
│   │   ├── human_approval.py        # real interrupt() node, lazily imported
│   │   └── finalize.py              # the ONLY node allowed to commit (evaluate_refund commit=True)
│   ├── routing.py                # route_after_review / route_after_human — pure state -> str, zero langgraph dep
│   ├── graph.py                  # build_graph() — wires every node, lazy langgraph import
│   ├── checkpointing.py           # build_sqlite_checkpointer(), thread_config() — durable checkpointing (Step 8)
│   ├── records_loader.py          # load_records()/unique_id() — records.jsonl loading, dedup-safe thread ids
│   ├── checklist.py                # pass/fail structural checks for a full batch run (Step 9)
│   └── visualize.py                # terminal-friendly graph picture (writes PNG / prints Mermaid source)
├── scripts/
│   ├── verify_llm_client.py   # smoke test for LLMClient.complete()
│   ├── run_intake.py          # batch runner: reads records.jsonl, parses each ticket
│   ├── run_agent.py           # batch runner: records.jsonl -> parser -> agent, end to end
│   │                            # [M3] +customer resolution stage (record.customer_ref preferred,
│   │                            # order_lookup fallback), +deterministic episodic write-back
│   ├── run_chunking.py         # [M4] demo: parse the real corpus into clause-level chunks
│   ├── run_embeddings.py       # [M4] demo: real embeddings + cosine similarity sanity check
│   ├── run_qdrant_index.py     # [M4] demo: build + upsert the Qdrant collection, sample dense_search
│   ├── run_bm25.py             # [M4] demo: BM25 top-k for a sample query
│   ├── run_hybrid.py           # [M4] demo: dense + BM25 -> RRF fusion side by side
│   ├── run_rerank.py           # [M4] demo: cross-encoder rerank of a fused shortlist
│   ├── run_generate.py         # [M4] demo: cited answer generation + a live prompt-injection probe
│   ├── run_eval_retrieval.py   # [M4] demo: precision@k/recall@k/MRR, bm25-only vs bm25+rerank
│   ├── run_eval_groundedness.py # [M4] demo: retrieve -> generate -> citation-integrity/groundedness report
│   ├── run_extract.py            # [M5] demo: extract node, 4 cases (primary/fallback/injection/empty)
│   ├── run_compare_to_playbook.py # [M5] demo: compare_to_playbook node, 5 scenarios
│   ├── run_draft_redline.py       # [M5] demo: draft_redline node, 3 cases
│   ├── run_human_approval.py      # [M5] demo: extract->compare->draft->human_approval, no langgraph needed
│   ├── run_graph.py                # [M5] demo: compiled graph end to end, standard + non_standard paths
│   ├── show_graph.py               # [M5] demo: just show the graph picture (PNG or Mermaid text)
│   ├── run_checkpointing_pause.py  # [M5] Step 8 demo pt.1: start a durable run, pause it — run as its own process
│   ├── run_checkpointing_resume.py # [M5] Step 8 demo pt.2: reattach in a NEW process, resume, finalize
│   └── run_workflow.py             # [M5] Step 9: full batch run vs records.jsonl + pass/fail checklist
├── tests/
│   ├── test_llm_client.py
│   ├── test_parser.py
│   ├── test_tools.py       # all four M2 tools, monkeypatched fixtures
│   ├── test_agent.py       # agent loop, fake-LLM stub (no live API calls)
│   │                        # [M3] +TestM3MemoryIntegration: tool availability/scoping,
│   │                        # context injection, closure-bound customer_ref
│   ├── test_memory.py      # [M3] CustomerMemory, fake Supermemory client + fake litellm
│   ├── test_rag/            # [M4] one file per rag/ module, all offline/deterministic
│   │   ├── test_chunking.py
│   │   ├── test_embeddings.py
│   │   ├── test_qdrant_index.py
│   │   ├── test_bm25.py
│   │   ├── test_hybrid.py
│   │   ├── test_rerank.py
│   │   ├── test_generate.py
│   │   ├── test_eval_retrieval.py
│   │   └── test_eval_groundedness.py
│   └── test_workflow/        # [M5] 18 files — see the M5 note above for the langgraph-gated vs executed split
│       ├── test_state.py
│       ├── test_extract.py
│       ├── test_extract_adapters.py
│       ├── test_compare_to_playbook.py
│       ├── test_draft_redline.py
│       ├── test_routing.py
│       ├── test_human_approval.py
│       ├── test_graph.py            # gated: pytest.importorskip("langgraph")
│       ├── test_finalize.py
│       ├── test_checkpointing.py    # thread_config/checkpoint_file_size — no langgraph needed
│       ├── test_checkpointing_sqlite.py  # gated: pytest.importorskip("langgraph.checkpoint.sqlite")
│       ├── test_checklist.py
│       └── test_records_loader.py
├── data/
│   ├── intake/
│   │   └── records.jsonl        # raw ticket intake records (ticket_id, raw_text, ground_truth, etc.)
│   ├── mock_api/
│   │   ├── orders.json
│   │   ├── shipments.json
│   │   ├── refunds.json
│   │   ├── customers.json
│   │   └── products.json
│   ├── corpus/                  # [M4] the 14 Kartway policy docs
│   │   ├── index.json              # doc slug/title/section -> markdown path, per document
│   │   ├── markdown/                # clause-source markdown for each policy doc
│   │   └── pdf/                      # source PDFs the markdown was authored from
│   └── eval/
│       └── golden_set.json      # [M4] 20-item eval set (factual/multi_hop/guardrail/injection/unanswerable)
├── outputs/                # run_agent.py batch results land here (gitignored)
├── .env
├── .gitignore
└── README.md
```

## Running Tests
```bash
pytest -v

# just the M3 memory suite
pytest tests/test_memory.py -v -s

# just the M4 RAG suite
pytest tests/test_rag/ -v

# just the M5 workflow suite (needs langgraph + langgraph-checkpoint-sqlite
# installed for test_graph.py / test_checkpointing_sqlite.py to run instead
# of skip — see requirements.txt's M5 section)
pytest tests/test_workflow/ -v
```

## Running the Agent
```bash
# one ticket, real M1 parse + M2/M3 agent (memory read/write included)
python scripts/run_agent.py --record-id SHOPSENSE-00004 --limit 1

# small batch
python scripts/run_agent.py --limit 20

# full records.jsonl
python scripts/run_agent.py

# fast dev iteration on agent/tool logic only — skips the M1 parse call,
# builds tickets from ground_truth instead (NOT a substitute for a real run).
# Customer resolution and memory read/write still run in this mode —
# SUPERMEMORY_API_KEY is required even with --skip-parse.
python scripts/run_agent.py --skip-parse --limit 20
```

Each ticket run resolves a `customer_ref` (preferring the record's own
`customer_ref`, falling back to `order_lookup` on the parsed `order_id`),
fetches that customer's memory context before the agent runs, and — when a
customer was resolved — writes an episodic summary back after the agent
finishes, regardless of what the agent chose to do with
`note_customer_preference`. A ticket with no resolvable customer (e.g. no
`order_id`) still runs normally; it just skips memory read/write for that
one run.

## Running the M4 RAG Pipeline
Each step has its own runnable demo, in order:

```bash
python3.12 -m scripts.run_chunking           # parse the corpus into clause-level chunks
python3.12 -m scripts.run_embeddings         # real embeddings + cosine similarity check — needs GOOGLE_API_KEY
python3.12 -m scripts.run_qdrant_index       # build + upsert the Qdrant collection — needs QDRANT_URL
python3.12 -m scripts.run_bm25               # BM25 top-k for a sample query — no external deps
python3.12 -m scripts.run_hybrid             # dense + BM25 -> RRF fusion — needs QDRANT_URL
python3.12 -m scripts.run_rerank             # cross-encoder rerank of a fused shortlist
python3.12 -m scripts.run_generate           # cited generation + a live prompt-injection probe — needs GOOGLE_API_KEY
python3.12 -m scripts.run_eval_retrieval     # precision@k/recall@k/MRR vs the golden set
python3.12 -m scripts.run_eval_groundedness [--sample N] [--judge]  # citation-integrity + groundedness report
```

`run_bm25.py`, `run_rerank.py`, `run_eval_retrieval.py`, and
`run_eval_groundedness.py` retrieve via BM25 + cross-encoder only, so they
don't require `QDRANT_URL` to be working. `run_generate.py` and
`run_eval_groundedness.py` call a real LLM (via LiteLLM) to generate cited
answers, so they need `GOOGLE_API_KEY` set.

## Running the M5 Workflow
Every step has its own runnable demo, in build order. All of them default to
offline fixtures (no LLM key, no order table, no corpus/index needed) unless
noted — swap in the production adapters (`make_m1_parser_adapter`,
`make_m2_refund_adapter`, `make_m4_retrieval_adapter`, `make_llm_compose_redline`)
for a real deployment:

```bash
python3.12 -m scripts.run_extract              # extract node: 4 cases
python3.12 -m scripts.run_compare_to_playbook  # compare_to_playbook node: 5 scenarios
python3.12 -m scripts.run_draft_redline        # draft_redline node: 3 cases
python3.12 -m scripts.run_human_approval       # extract->compare->draft->human_approval, no langgraph needed
python3.12 -m scripts.show_graph               # just the graph picture — needs langgraph
python3.12 -m scripts.run_graph                # compiled graph end to end — needs langgraph

# Step 8 — real cross-process durability. Run as TWO SEPARATE commands,
# not back to back in one script — the whole point is proving state
# survives a genuine process boundary, not just a kernel restart.
python3.12 -m scripts.run_checkpointing_pause    # process 1: pauses, writes to disk
python3.12 -m scripts.run_checkpointing_resume   # process 2: reattaches, resumes, finalizes

# Step 9 — full batch run + pass/fail checklist against real tickets.
# IMPORTANT: this project's records.jsonl lives at data/intake/records.jsonl
# (see Project Structure above) — pass that path explicitly; the script's
# own bundled 30-record sample at data/records.jsonl is a standalone-demo
# convenience only, not this repo's real dataset.
python3.12 -m scripts.run_workflow data/intake/records.jsonl
python3.12 -m scripts.run_workflow data/intake/records.jsonl 20   # optional: cap at first 20 records
```

`run_extract.py`, `run_compare_to_playbook.py`, `run_draft_redline.py`, and
`run_human_approval.py` need no `langgraph` install at all (every langgraph
primitive in this project is lazily imported — see `workflow/graph.py`'s
module docstring). `show_graph.py`, `run_graph.py`,
`run_checkpointing_pause.py`/`_resume.py`, and `run_workflow.py` need the
real `langgraph` (+ `langgraph-checkpoint-sqlite` for the checkpointing pair)
installed per `requirements.txt`'s M5 section.