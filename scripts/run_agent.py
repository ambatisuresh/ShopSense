"""
End-to-end batch runner: raw ticket -> M1 intake parser -> SupportTicket -> M2 agent.

This is deliberately the FULL pipeline, not just the agent in isolation --
running records.jsonl through parse_ticket_safe() first means a bad M1 parse
(wrong issue_type, missed order_id, missed prompt-injection flag) shows up
here as a bad agent run, which is exactly the failure mode you want this
script to surface before M8's formal eval.

Usage:
    python scripts/run_agent.py                          # full batch
    python scripts/run_agent.py --limit 10                # first 10 tickets
    python scripts/run_agent.py --record-id SHOPSENSE-00004  # one ticket
    python scripts/run_agent.py --skip-parse --limit 20   # agent-only, dev iteration

--skip-parse bypasses M1's parser entirely and builds a ticket dict straight
from records.jsonl's ground_truth field. This is a DEV SHORTCUT ONLY: it
leaks ground truth into what should be inferred, and contains_suspicious_
instructions is never set (ground_truth doesn't carry it), so injection-
handling can't be exercised this way. Use it only to iterate quickly on
agent/tool logic without spending LLM calls on the parse step; never use it
for anything you'd report as a real pipeline result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed -- fine if env vars are set another way

from agent.support_agent import run_support_agent

DEFAULT_INPUT = Path("data/intake/records.jsonl")
DEFAULT_OUTPUT = Path("outputs/agent_run_results.json")


# ---------------------------------------------------------------------------
# Raw ticket reading -- deliberately a small local reader rather than
# importing intake/reader.py by guessing its exact function name; the
# behavior it needs to match (skip malformed lines, log and continue) is
# simple enough to duplicate here without that coupling risk.
# ---------------------------------------------------------------------------

def read_raw_records(path: Path) -> tuple[list[dict], list[str]]:
    """Returns (records, read_errors). Never raises on a single bad line."""
    records, errors = [], []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"line {line_num}: {e}")
    return records, errors


# ---------------------------------------------------------------------------
# Raw record -> ticket dict for the agent (two paths: real M1 parse, or the
# ground-truth dev shortcut)
# ---------------------------------------------------------------------------

def _model_to_dict(support_ticket: Any) -> dict:
    """SupportTicket is a Pydantic model; handle both v2 (.model_dump) and
    v1 (.dict), and stringify any Enum values so format_ticket_for_agent's
    f-strings print clean values instead of e.g. 'IssueType.REFUND'."""
    if hasattr(support_ticket, "model_dump"):
        raw = support_ticket.model_dump(mode="json")
    elif hasattr(support_ticket, "dict"):
        raw = support_ticket.dict()
    else:
        raw = dict(support_ticket)
    return {k: (v.value if hasattr(v, "value") else v) for k, v in raw.items()}


def parse_via_m1(record: dict, client) -> tuple[dict | None, list[str]]:
    from intake.parser import parse_ticket_safe

    support_ticket, errors = parse_ticket_safe(
        record["ticket_id"], record["raw_text"], client
    )
    if support_ticket is None:
        return None, errors
    return _model_to_dict(support_ticket), errors


def build_from_ground_truth(record: dict) -> dict:
    """DEV SHORTCUT -- see module docstring. Not a substitute for the real parse."""
    gt = record.get("ground_truth", {})
    return {
        "ticket_id": record.get("ticket_id"),
        "raw_text": record.get("raw_text", ""),
        "order_id": record.get("order_ref"),
        "issue_type": gt.get("intent", "unknown"),
        "sentiment": gt.get("sentiment", "neutral"),
        "urgency": "unknown",
        "claimed_refund_amount": gt.get("claimed_amount_inr"),
        "contains_suspicious_instructions": False,  # ground_truth doesn't carry this signal
        "confidence": None,
    }


# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------

def run_batch(
    input_path: Path,
    output_path: Path,
    limit: int | None,
    record_id_filter: str | None,
    skip_parse: bool,
    max_iterations: int,
    delay_seconds: float,
) -> None:
    records, read_errors = read_raw_records(input_path)
    for err in read_errors:
        print(f"  [read error] {err}")

    if record_id_filter:
        records = [r for r in records if r.get("record_id") == record_id_filter]
    if limit:
        records = records[:limit]

    if not records:
        print("No records to process (check --input, --record-id, --limit).")
        return

    client = None
    if not skip_parse:
        from core.llm_client import LLMClient
        client = LLMClient()

    print(f"Processing {len(records)} ticket(s){' [SKIP-PARSE DEV MODE]' if skip_parse else ''}...\n")

    results: list[dict] = []
    for i, record in enumerate(records, start=1):
        record_id = record.get("record_id", "unknown")
        ticket_id = record.get("ticket_id", "unknown")
        started = time.time()

        try:
            if skip_parse:
                ticket_dict = build_from_ground_truth(record)
                parse_errors: list[str] = []
            else:
                ticket_dict, parse_errors = parse_via_m1(record, client)

            if ticket_dict is None:
                elapsed = time.time() - started
                print(f"[{i}/{len(records)}] {record_id} ({ticket_id}) -- INTAKE PARSE FAILED: {parse_errors}")
                results.append({
                    "record_id": record_id, "ticket_id": ticket_id,
                    "stage": "intake_parse", "status": "failed",
                    "errors": parse_errors, "elapsed_seconds": round(elapsed, 2),
                })
                continue

            agent_result = run_support_agent(ticket_dict, max_iterations=max_iterations)
            elapsed = time.time() - started

            print(
                f"[{i}/{len(records)}] {record_id} ({ticket_id}) -- "
                f"{agent_result['resolution_status']} in {agent_result['iterations_used']} "
                f"iteration(s), {len(agent_result['tool_calls_made'])} tool call(s), "
                f"{elapsed:.1f}s"
            )

            results.append({
                "record_id": record_id, "ticket_id": ticket_id,
                "stage": "agent", "status": agent_result["resolution_status"],
                "order_id": ticket_dict.get("order_id"),
                "issue_type": ticket_dict.get("issue_type"),
                "final_answer": agent_result["final_answer"],
                "iterations_used": agent_result["iterations_used"],
                "tool_call_count": len(agent_result["tool_calls_made"]),
                "tool_calls_made": agent_result["tool_calls_made"],
                "parse_errors": parse_errors,
                "elapsed_seconds": round(elapsed, 2),
            })

        except Exception as e:
            elapsed = time.time() - started
            print(f"[{i}/{len(records)}] {record_id} ({ticket_id}) -- RUNTIME ERROR: {e}")
            results.append({
                "record_id": record_id, "ticket_id": ticket_id,
                "stage": "runtime", "status": "error",
                "error": str(e), "elapsed_seconds": round(elapsed, 2),
            })

        if delay_seconds and i < len(records):
            time.sleep(delay_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_summary(results, output_path)


def print_summary(results: list[dict], output_path: Path) -> None:
    status_counts = Counter(r["status"] for r in results)
    agent_runs = [r for r in results if r["stage"] == "agent"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tickets:        {len(results)}")
    for status, count in status_counts.most_common():
        print(f"  {status:<25} {count}")

    if agent_runs:
        avg_iterations = sum(r["iterations_used"] for r in agent_runs) / len(agent_runs)
        avg_tool_calls = sum(r["tool_call_count"] for r in agent_runs) / len(agent_runs)
        print(f"\nAvg iterations/ticket: {avg_iterations:.1f}")
        print(f"Avg tool calls/ticket: {avg_tool_calls:.1f}")

    flagged = [r for r in results if r["status"] in ("max_iterations_exceeded", "failed", "error")]
    if flagged:
        print(f"\n{len(flagged)} ticket(s) need manual review:")
        for r in flagged:
            print(f"  - {r['record_id']} ({r['ticket_id']}): {r['status']}")

    print(f"\nFull results written to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Run ShopSense tickets through the M1 parser + M2 agent end to end.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Path to records.jsonl (default: {DEFAULT_INPUT})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Path to write results JSON (default: {DEFAULT_OUTPUT})")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N tickets")
    p.add_argument("--record-id", type=str, default=None, help="Only process one ticket by its record_id (e.g. SHOPSENSE-00004)")
    p.add_argument("--skip-parse", action="store_true", help="DEV ONLY: bypass M1's parser, build ticket dict from ground_truth")
    p.add_argument("--max-iterations", type=int, default=6, help="Cap on agent tool-calling turns per ticket (default: 6)")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between tickets, rate-limit friendly (default: 1.0)")
    args = p.parse_args()

    run_batch(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        record_id_filter=args.record_id,
        skip_parse=args.skip_parse,
        max_iterations=args.max_iterations,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()