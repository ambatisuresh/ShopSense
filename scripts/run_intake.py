import sys
import time
from core.llm_client import LLMClient
from intake.reader import read_raw_tickets
from intake.parser import parse_ticket_safe

#Read data/intake/records.jsonl which consists of 200 records
def main(jsonl_path: str, limit: int = None):
    report = read_raw_tickets(jsonl_path)
    print(report.summary())

    client = LLMClient()
    tickets_to_process = report.tickets[:limit] if limit else report.tickets

    results = []
    for raw_ticket in tickets_to_process:
        ticket, errs = parse_ticket_safe(raw_ticket.ticket_id, raw_ticket.raw_text, client)
        results.append({"id": raw_ticket.ticket_id, "ticket": ticket, "errors": errs})

        if ticket:
            print(f"✅ {ticket.ticket_id}: {ticket.issue_type} | {ticket.sentiment} | "
                  f"{ticket.urgency} | order_id={ticket.order_id} | "
                  f"suspicious={ticket.contains_suspicious_instructions}")
        else:
            print(f"❌ {raw_ticket.ticket_id}: {errs}")

        time.sleep(1)  # stay under rate limits — adjust per your provider's tier

    success = sum(1 for r in results if r["ticket"] is not None)
    print(f"\nParsed {success}/{len(results)} records successfully.")
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/intake/records.jsonl"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(path, limit=limit)