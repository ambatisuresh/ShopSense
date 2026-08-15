import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RawTicket:
    ticket_id: str
    raw_text: str


@dataclass
class ReadReport:
    tickets: list[RawTicket] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.tickets) + len(self.errors)
        return f"Read {len(self.tickets)}/{total} tickets ({len(self.errors)} errors)"


def read_raw_tickets(path: str | Path) -> ReadReport:
    """
    Reads ticket_id + raw_text from a JSONL intake file.
    Skips and logs malformed lines rather than crashing the whole load.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intake file not found: {path}")

    report = ReadReport()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                report.errors.append({"line": line_num, "error": f"JSONDecodeError: {e}"})
                continue

            ticket_id = raw.get("ticket_id")
            raw_text = raw.get("raw_text")

            if not ticket_id or not raw_text:
                report.errors.append({
                    "line": line_num,
                    "error": "Missing required field ticket_id or raw_text",
                })
                continue

            report.tickets.append(RawTicket(ticket_id=ticket_id, raw_text=raw_text))

    return report