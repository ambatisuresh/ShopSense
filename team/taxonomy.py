"""The clause-type taxonomy shared by Extraction (Step 3) and Playbook RAG
(Step 4).

Each of these 23 clause types corresponds 1:1 to a numbered position in
data/playbook/negotiation-playbook.md. Extraction's whole job is mapping a
raw contract clause onto one of these labels — or leaving it unclassified
when no playbook position covers it, which is a real, expected outcome for
clauses like scope-of-services or a fee schedule, not a failure.
"""
from __future__ import annotations

CLAUSE_TYPES = [
    "refund_settlement_authority",
    "refund_settlement_timing",
    "delivery_sla_alignment",
    "delay_compensation_alignment",
    "lost_in_transit_threshold",
    "disputed_delivery_investigation",
    "carrier_liability_cap",
    "repair_turnaround",
    "replacement_turnaround",
    "warranty_void_criteria_alignment",
    "repair_vs_replacement_decision_authority",
    "claim_status_reporting_cadence",
    "return_intake_window",
    "condition_grading_alignment",
    "category_exclusions",
    "inspection_to_refund_buffer",
    "return_mishandling_liability",
    "termination_and_renewal",
    "limitation_of_liability",
    "indemnification",
    "governing_law_and_venue",
    "confidentiality_and_data_security",
    "force_majeure",
]

# Deterministic fallback classifier: title-keyword phrase -> clause_type.
# Checked in this order; first match wins. Deliberately title-only (not
# body-scanned) — every contract clause title in this corpus was written
# descriptively enough that title matching alone is reliable, and matching
# against body text risks false positives from incidental word overlap
# (e.g. "liability" appearing inside an unrelated clause's prose).
CLAUSE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "refund_settlement_authority": ["refund settlement authority", "settlement authority"],
    "refund_settlement_timing": ["settlement timing"],
    "delivery_sla_alignment": ["delivery service level", "delivery window"],
    "delay_compensation_alignment": ["delay compensation"],
    "lost_in_transit_threshold": ["lost-in-transit", "lost in transit"],
    "disputed_delivery_investigation": ["disputed delivery"],
    "carrier_liability_cap": ["liability for lost or damaged goods", "liability for lost"],
    "repair_turnaround": ["repair turnaround"],
    "replacement_turnaround": ["replacement turnaround"],
    "warranty_void_criteria_alignment": ["claim eligibility assessment", "eligibility assessment"],
    "repair_vs_replacement_decision_authority": [
        "repair vs. replacement determination",
        "repair vs replacement determination",
    ],
    "claim_status_reporting_cadence": ["claim status reporting"],
    "return_intake_window": ["return intake window"],
    "condition_grading_alignment": ["condition grading"],
    "category_exclusions": ["excluded categories"],
    "inspection_to_refund_buffer": ["inspection turnaround"],
    "return_mishandling_liability": ["liability for mishandled returns", "mishandled returns"],
    "termination_and_renewal": ["term and renewal", "termination for convenience", "termination"],
    "limitation_of_liability": ["limitation of liability"],
    "indemnification": ["indemnification"],
    "governing_law_and_venue": ["governing law"],
    "confidentiality_and_data_security": ["confidentiality", "data security"],
    "force_majeure": ["force majeure"],
}


def classify_clause_type_fallback(title: str, body: str = "") -> str | None:
    """Deterministic, LLM-free clause_type classifier.

    Returns None ("unclassified") when no keyword phrase matches the title —
    a real, expected outcome for clauses the playbook simply doesn't cover
    (scope-of-services, fee schedules, insurance, etc.), not a failure.
    `body` is accepted for interface symmetry with the LLM classifier but is
    not scanned, on purpose (see module docstring).
    """
    title_lower = title.lower()
    for clause_type, phrases in CLAUSE_TYPE_KEYWORDS.items():
        if any(phrase in title_lower for phrase in phrases):
            return clause_type
    return None