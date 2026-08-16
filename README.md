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
| M4 | Production RAG + evaluation baseline | ⬜ Not started (now includes M3's originally-scoped Qdrant/RAG work) |
| M5 | Orchestrated LangGraph workflow with checkpointing | ⬜ Not started |
| M6 | Multi-agent team + MCP integration | ⬜ Not started |
| M7 | Observability + reliability hardening | ⬜ Not started |
| M8 | End-to-end evaluation, guardrails & deployment | ⬜ Not started |

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
```

Only the API key and `*_LLM_MODEL` var matching your active `LLM_PROVIDER`
are required — the other provider's vars can stay unset until you switch.
`SUPERMEMORY_API_KEY` is required as of M3 — `scripts/run_agent.py` and
`memory/customer_memory.py` construct a `CustomerMemory` unconditionally
(including under `--skip-parse`), so it must be set even for dev-mode runs.

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
├── scripts/
│   ├── verify_llm_client.py   # smoke test for LLMClient.complete()
│   ├── run_intake.py          # batch runner: reads records.jsonl, parses each ticket
│   └── run_agent.py           # batch runner: records.jsonl -> parser -> agent, end to end
│                                # [M3] +customer resolution stage (record.customer_ref preferred,
│                                # order_lookup fallback), +deterministic episodic write-back
├── tests/
│   ├── test_llm_client.py
│   ├── test_parser.py
│   ├── test_tools.py       # all four M2 tools, monkeypatched fixtures
│   ├── test_agent.py       # agent loop, fake-LLM stub (no live API calls)
│   │                        # [M3] +TestM3MemoryIntegration: tool availability/scoping,
│   │                        # context injection, closure-bound customer_ref
│   └── test_memory.py      # [M3] CustomerMemory, fake Supermemory client + fake litellm
├── data/
│   ├── intake/
│   │   └── records.jsonl        # raw ticket intake records (ticket_id, raw_text, ground_truth, etc.)
│   └── mock_api/
│       ├── orders.json
│       ├── shipments.json
│       ├── refunds.json
│       ├── customers.json
│       └── products.json
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