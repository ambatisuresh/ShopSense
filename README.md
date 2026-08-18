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
  LangGraph (introduced in M5 for multi-agent orchestration + checkpointing)

## Milestones
| # | Milestone | Status |
|---|-----------|--------|
| M1 | Provider-agnostic LLM client + structured intake | ✅ Completed |
| M2 | Tool-enabled single agent | ✅ Completed |
| M3 | Persistent memory + semantic index | ✅ Completed |
| M4 | Production RAG + evaluation baseline | ✅ Completed |
| M5 | Orchestrated LangGraph workflow with checkpointing | ⬜ Not started |
| M6 | Multi-agent team + MCP integration | ⬜ Not started |
| M7 | Observability + reliability hardening | ⬜ Not started |
| M8 | End-to-end evaluation, guardrails & deployment | ⬜ Not started |

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
│   └── run_eval_groundedness.py # [M4] demo: retrieve -> generate -> citation-integrity/groundedness report
├── tests/
│   ├── test_llm_client.py
│   ├── test_parser.py
│   ├── test_tools.py       # all four M2 tools, monkeypatched fixtures
│   ├── test_agent.py       # agent loop, fake-LLM stub (no live API calls)
│   │                        # [M3] +TestM3MemoryIntegration: tool availability/scoping,
│   │                        # context injection, closure-bound customer_ref
│   ├── test_memory.py      # [M3] CustomerMemory, fake Supermemory client + fake litellm
│   └── test_rag/            # [M4] one file per rag/ module, all offline/deterministic
│       ├── test_chunking.py
│       ├── test_embeddings.py
│       ├── test_qdrant_index.py
│       ├── test_bm25.py
│       ├── test_hybrid.py
│       ├── test_rerank.py
│       ├── test_generate.py
│       ├── test_eval_retrieval.py
│       └── test_eval_groundedness.py
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