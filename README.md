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
| M1 | Provider-agnostic LLM client + structured intake | 🚧 In progress |
| M2 | Tool-enabled single agent | ⬜ Not started |
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
├── core/ # LLM client wrapper, config, Pydantic schemas
├── tests/ # pytest test suite
└── scripts/ # Verification / utility scripts
```


## Running Tests
```bash
pytest -v
```
