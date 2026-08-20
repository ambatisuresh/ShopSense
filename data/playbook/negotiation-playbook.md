# Kartway Vendor Contract Negotiation Playbook

**Purpose:** This playbook states Kartway's standard negotiating positions for vendor agreements that touch customer-facing commitments (refunds, shipping, warranty, returns). Every position below is derived from an existing Kartway policy document. Where a vendor's draft clause falls outside the Fallback range, it must be flagged for redline and Legal Reviewer sign-off before execution.

**1.1 Refund Settlement Authority**
Preferred: Vendor auto-settlement authority must not exceed INR 2,000 per refund, matching the agent auto-approval tier in refund-authority.md §4.1-4.2. Any refund above INR 2,000 requires documented Kartway approval before settlement.
Fallback: A USD-denominated auto-settlement ceiling is acceptable only if it is demonstrably at or below the INR 2,000 tier at the prevailing exchange rate, and is reviewed quarterly.
Unacceptable: An auto-settlement ceiling set with reference to escalation-tone.md §4.3.6's USD 50 figure without reconciling it against refund-authority.md's INR 2,000 tier — these two internal documents currently conflict, and a vendor contract must not silently inherit one over the other. Flag for Operations Manager sign-off per refund-authority.md §4.4.1 before this clause is executed.

**1.2 Refund Settlement Timing**
Preferred: Vendor settles authorized refunds within the fastest timeline Kartway has promised customers for any payment method — 24 hours for digital wallets per returns-policy.md §2.6b.
Fallback: Up to 3 business days, provided Kartway's customer-facing commitment for that payment method allows the buffer.
Unacceptable: Any settlement timing that would cause Kartway to miss its own published refund-processing commitments in returns-policy.md §2.6.

**2.1 Delivery SLA Alignment**
Preferred: Vendor delivery windows are equal to or faster than Kartway's published customer-facing SLA in shipping-policy.md §1.1 (metro standard: 3-5 business days; non-metro standard: 5-7 business days).
Fallback: Vendor windows may run up to 1 business day slower than the published SLA only if Kartway's customer-facing copy is updated to match, with Legal Reviewer approval.
Unacceptable: Vendor delivery windows that exceed Kartway's published customer SLA without a corresponding update to customer-facing commitments — this creates a standing risk that Kartway systematically misses the delivery promise made to customers.

**2.2 Delay Compensation Alignment**
Preferred: Vendor's compensation to Kartway for a delayed shipment is at least equal to Kartway's obligation to the customer under shipping-policy.md §3.1 (10% / 25% / 50% of shipping cost by delay tier).
Fallback: A single blended compensation rate is acceptable only if it is not lower than the weighted average of Kartway's tiered customer obligation across typical delay distributions.
Unacceptable: A flat compensation rate lower than Kartway's lowest customer-facing tier (10%) — this guarantees Kartway loses money on every delayed shipment regardless of severity.

**2.3 Lost-in-Transit Threshold**
Preferred: Vendor's lost-in-transit declaration threshold is at or below fifteen (15) calendar days, matching shipping-policy.md §4.1.
Fallback: Up to 17 calendar days, only with an interim customer-communication plan for the gap.
Unacceptable: Any threshold materially above 15 calendar days with no interim plan — Kartway would be unable to act on a customer's lost-parcel claim until the vendor agrees it is lost.

**2.4 Disputed Delivery Investigation Window**
Preferred: Vendor completes its investigation within five (5) business days, matching shipping-policy.md §5.1b.
Fallback: Up to 7 business days, only if Kartway's customer-facing commitment is updated accordingly.
Unacceptable: An investigation window longer than 7 business days with no update to the customer-facing commitment.

**2.5 Carrier Liability Cap**
Preferred: Liability for lost or damaged goods is based on the item's actual or declared value, uncapped or capped only at a level well above typical order value.
Fallback: A tiered cap by declared value, with a top tier sufficient to cover Kartway's typical electronics or furniture order value.
Unacceptable: A flat low-dollar cap, at or below USD 100 per package, applied uniformly regardless of item value.

**3.1 Repair Turnaround**
Preferred: Vendor commits to completing repairs within ten (10) business days, matching warranty-policy.md §6.1a.
Fallback: Up to 12 business days, only with a customer-notification mechanism for the gap.
Unacceptable: Any commitment longer than 12 business days, or one expressed only in vague terms.

**3.2 Replacement Turnaround**
Preferred: Vendor commits to shipping a replacement within five (5) business days of claim approval, matching warranty-policy.md §6.1b.
Fallback: Up to 7 business days, with a firm number stated in the contract.
Unacceptable: "Best efforts" or other language with no fixed turnaround commitment — this cannot be reconciled with the fixed 5-business-day promise Kartway makes to customers.

**3.3 Warranty Void Criteria Alignment**
Preferred: Vendor's claim-eligibility criteria match warranty-policy.md §4.1(a)-(e) exactly: unauthorized repair or modification, misuse/negligence/accident damage, no proof of purchase, failure to follow instructions, and unauthorized commercial use.
Fallback: Vendor criteria may be more detailed than §4.1 provided every additional exclusion is disclosed to Kartway and does not narrow coverage Kartway has promised customers.
Unacceptable: Vendor criteria that are broader or stricter than §4.1 without Kartway's sign-off — this risks the vendor denying claims Kartway has publicly promised to honor.

**3.4 Repair-vs-Replacement Decision Authority**
Preferred: The decision to repair or replace remains with Kartway, per warranty-policy.md §5.1(a).
Fallback: A joint decision process, with Kartway holding final sign-off.
Unacceptable: Sole vendor discretion over the repair-vs-replacement decision for Kartway Warranty claims.

**3.5 Claim Status Reporting Cadence**
Preferred: Vendor reports claim status no less often than every five (5) business days, matching warranty-policy.md §6.2b.
Fallback: Every 7 business days, only for claims not yet past their turnaround commitment.
Unacceptable: A reporting cadence slower than Kartway's own customer-update commitment of every 5 business days — Kartway cannot update a customer more often than it is itself updated.

**4.1 Return Intake Window**
Preferred: Vendor accepts items for processing if they arrive within Kartway's customer-facing shipping window (7 days from authorization, per returns-policy.md §2.8) plus a reasonable transit buffer of at least 5 additional days.
Fallback: A combined window of no less than 10 days from authorization.
Unacceptable: An intake window shorter than the 7-day customer shipping commitment itself — a customer who ships on day 7 as promised would already be logged as late on arrival.

**4.2 Condition Grading Alignment**
Preferred: Vendor grading maps directly to the three-tier schedule in returns-policy.md §2.5 — opened-unused (75% refund), opened-used (50% refund), heavily used or damaged (no refund).
Fallback: A finer-grained grading scale that can be deterministically mapped to the three published tiers.
Unacceptable: A binary "resalable / not resalable" grading with no mapping to the tiered refund schedule — Kartway cannot apply the partial-refund percentages it has published without this mapping.

**4.3 Category Exclusions**
Preferred: The vendor agreement explicitly excludes groceries, non-returnable per returns-policy.md §2.2b, and gift cards or downloadable digital content, non-returnable per §2.7a, from intake.
Fallback: A written operating procedure outside the contract itself confirming the vendor will reject these categories at intake.
Unacceptable: No exclusion language or procedure at all — the vendor may accept and process items Kartway is not permitted to refund.

**4.4 Inspection-to-Refund Buffer**
Preferred: Vendor inspection turnaround leaves Kartway enough time to still meet its fastest customer-facing refund commitment, 24 hours for digital wallets per returns-policy.md §2.6b, after grading is complete, or the digital-wallet commitment is explicitly carved out for vendor-processed returns.
Fallback: An inspection SLA of 1 business day or less.
Unacceptable: An inspection SLA that, combined with settlement time, cannot fit inside Kartway's fastest published customer refund commitment with no carve-out.

**4.5 Return Mishandling Liability**
Preferred: Vendor is liable for the item's retail value when loss, damage, or misgrading is the vendor's fault.
Fallback: A liability cap set at a multiple, for example 5x, of the per-item processing fee.
Unacceptable: Liability capped at the processing fee alone — this is typically a small fraction of the item's value and leaves Kartway to absorb the loss.

**5.1 Term and Auto-Renewal**
Preferred: Auto-renewal notice period of at least ninety (90) calendar days.
Fallback: Sixty (60) calendar days.
Unacceptable: Fewer than thirty (30) calendar days — too little runway to source or negotiate a replacement vendor before the term locks in again.

**5.2 Limitation of Liability**
Preferred: Liability cap of at least twelve (12) months' fees paid, or uncapped for data breach, gross negligence, or willful misconduct.
Fallback: Six (6) months' fees paid, with carve-outs for data breach and gross negligence.
Unacceptable: A cap of one (1) month's fees or less, or no carve-out for data breach or gross negligence.

**5.3 Indemnification**
Preferred: Mutual indemnification — each party indemnifies the other for claims arising from its own breach or negligence.
Fallback: Asymmetric indemnification is acceptable only where the narrower party's exposure is inherently limited, for example a vendor with no access to customer PII.
Unacceptable: One-sided indemnification requiring Kartway to indemnify the vendor for claims arising from the vendor's own service, with no reciprocal obligation.

**5.4 Governing Law and Venue**
Preferred: Governing law and venue in Kartway's home jurisdiction.
Fallback: A neutral third jurisdiction, with arbitration.
Note: This is a negotiation preference, not a customer-facing compliance risk. It does not block execution on its own and should be noted rather than escalated.

**5.5 Confidentiality and Data Security**
Preferred: Mutual confidentiality obligations, plus PCI-DSS or equivalent data-security commitments for any vendor handling payment or customer PII.
Acceptable as-is: Standard mutual confidentiality language with reasonable-care data handling, matching the baseline used across Kartway's existing vendor agreements.

**5.6 Force Majeure**
Acceptable as-is: Force majeure language consistent with shipping-policy.md §7 — natural disasters, severe weather, and other events outside a party's reasonable control.
