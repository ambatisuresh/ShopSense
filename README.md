# ShopSense — Customer Care & Order Operations Assistant

## Overview
A multi-agent customer support assistant for a mid-size e-commerce marketplace.
Classifies and resolves order/refund/shipping tickets, answers policy questions
via RAG, executes order actions through tools, and escalates complex or
emotional cases to human agents.

## Tech Stack
- **LLM routing:** LiteLLM (provider-agnostic, currently Google AI Studio / Gemini)
- **Vector DB:** Qdrant
- **Memory:** Supermemory
- **Observability:** Langfuse
- **Orchestration:** LangGraph

## Milestones
| # | Milestone | Status |
|---|-----------|--------|
| M1 | Provider-agnostic LLM client + structured intake | ✅ Completed |
| M2 | Tool-enabled single agent | 🚧 In progress |
| M3 | Persistent memory + semantic index | ⬜ Not started |
| M4 | Production RAG + evaluation baseline | ⬜ Not started |
| M5 | Orchestrated LangGraph workflow with checkpointing | ⬜ Not started |
| M6 | Multi-agent team + MCP integration | ⬜ Not started |
| M7 | Observability + reliability hardening | ⬜ Not started |
| M8 | End-to-end evaluation, guardrails & deployment | ⬜ Not started |

## Setup

```bash
python3.12 -m venv shopsensevenv
source shopsensevenv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

GOOGLE_API_KEY=your_key_here

## Project Structure

```
shopsense/
├── core/
│   ├── llm_client.py       # LiteLLM wrapper — sole call site for completion()
│   └── schema.py           # Pydantic schemas (SupportTicket, enums)
├── intake/
│   ├── reader.py           # JSONL reader — extracts ticket_id + raw_text
│   └── parser.py           # raw text -> validated SupportTicket, via LLMClient
├── scripts/
│   ├── verify_llm_client.py   # smoke test for LLMClient.complete()
│   └── run_intake.py          # batch runner: reads records.jsonl, parses each ticket
├── tests/
│   ├── test_llm_client.py
│   ├── test_parser.py
│   └── first.py
├── data/intake/
│   └── records.jsonl        # raw ticket intake records (ticket_id, raw_text, ground_truth, etc.)
├── .gitignore
└── README.md
```


## Running Tests
```bash
pytest -v
```
