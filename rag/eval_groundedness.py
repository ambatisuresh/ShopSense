"""Groundedness evaluation — does a generated answer actually trace back to
the policy text it cites? This is M4's specific ask: "evaluate whether
cited clauses are actually grounded in the source playbook," not just
whether retrieval found the right document.

Two complementary, independently useful checks:

1. `citation_integrity` (no LLM, fully deterministic) — every [id] the
   answer references must be a chunk id that was actually offered in its
   retrieved context (an answer can't ground a claim in a source it was
   never given — a hallucinated citation), and the cited chunks'
   `doc_title`s must collectively cover `golden_set.json`'s `must_cite`
   list. This is the check that catches a fluent-sounding answer citing
   "[7]" when [7] was never in context.
2. Faithfulness scoring — does the *content* of the answer actually match
   the *content* of what it cited (as opposed to citing the right source
   but misstating it)? `heuristic_groundedness` is a lexical-overlap
   stand-in that needs no LLM (works fully offline, which is what this
   build environment and its test suite are limited to); `llm_judge_groundedness`
   is the real check for when a live LLM is available, mirroring the
   notebook's optional Lab C groundedness extension.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_cited_ids(answer: str) -> list[int]:
    return [int(m) for m in CITATION_RE.findall(answer)]


def citation_integrity(answer: str, context_ids: list[int], chunks_by_id: dict,
                        must_cite_titles: list[str]) -> dict:
    cited = extract_cited_ids(answer)
    context_set = set(context_ids)
    hallucinated = [cid for cid in cited if cid not in context_set]
    cited_titles = {chunks_by_id[cid]["doc_title"] for cid in cited if cid in chunks_by_id}
    missing_required = [t for t in must_cite_titles if t not in cited_titles]
    return {
        "cited_ids": cited,
        "hallucinated_citations": hallucinated,
        "cited_titles": sorted(cited_titles),
        "missing_required_citations": missing_required,
        "citations_grounded": len(hallucinated) == 0,
        "required_citations_satisfied": (len(missing_required) == 0) if must_cite_titles else True,
    }


def check_must_not_contain(answer: str, must_not_contain: list[str]) -> dict:
    lowered = answer.lower()
    violations = [phrase for phrase in must_not_contain if phrase.lower() in lowered]
    return {"violations": violations, "clean": len(violations) == 0}


def heuristic_groundedness(answer: str, context_ids: list[int], chunks_by_id: dict) -> float:
    """Lexical-overlap stand-in for an LLM-judge faithfulness score.

    Fraction of the answer's content words (len > 3, to skip stopword-ish
    noise) that also appear in the text of the chunks it actually cited.
    This can't detect a claim that *contradicts* its source (it only
    checks shared vocabulary, not truth), so it is not a substitute for
    `llm_judge_groundedness` — it exists so the eval pipeline is still
    testable end-to-end with no LLM and no API key.
    """
    cited = [cid for cid in extract_cited_ids(answer) if cid in context_ids]
    if not cited:
        return 0.0
    source_tokens: set[str] = set()
    for cid in cited:
        source_tokens |= set(chunks_by_id[cid]["text"].lower().split())
    answer_tokens = [t for t in re.findall(r"[a-zA-Z0-9%$]+", answer.lower()) if len(t) > 3]
    if not answer_tokens:
        return 0.0
    supported = sum(1 for t in answer_tokens if t in source_tokens)
    return supported / len(answer_tokens)


JUDGE_SYSTEM_PROMPT = (
    "You are a strict fact-checker. You will be shown a QUESTION, a generated ANSWER, and the "
    "SOURCE CLAUSES the answer cited. Score how well the answer is grounded in those source "
    "clauses: 1.0 means every factual claim in the answer is directly supported by the source "
    "text; 0.0 means the answer is unsupported or contradicts the sources. Respond with ONLY a "
    "JSON object: {\"score\": <float 0-1>, \"unsupported_claims\": [<string>, ...]}."
)


def llm_judge_groundedness(question: str, answer: str, context_ids: list[int], chunks_by_id: dict,
                            complete_fn: Callable[[str, str], str]) -> dict:
    """LLM-as-judge faithfulness score. `complete_fn(system, user) -> str`
    is injectable (see rag/generate.py's `default_complete` for the real
    LiteLLM implementation) so this is testable with a fake judge that
    returns canned JSON."""
    sources = "\n\n".join(
        f"[{cid}] {chunks_by_id[cid]['text']}" for cid in context_ids if cid in chunks_by_id
    )
    user = f"QUESTION: {question}\n\nANSWER: {answer}\n\nSOURCE CLAUSES:\n{sources}"
    raw = complete_fn(JUDGE_SYSTEM_PROMPT, user)
    try:
        parsed = json.loads(raw)
        return {
            "score": float(parsed.get("score", 0.0)),
            "unsupported_claims": parsed.get("unsupported_claims", []),
            "raw": raw,
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": None, "unsupported_claims": [], "raw": raw, "parse_error": True}


def evaluate_item(item: dict, answer: str, context_ids: list[int], chunks_by_id: dict,
                   judge_fn: Optional[Callable[[str, str], str]] = None) -> dict:
    """Full groundedness report for one golden_set.json item + its
    generated answer. `judge_fn` is optional -- omit it to skip the LLM
    judge pass and rely on `heuristic_groundedness` alone."""
    integrity = citation_integrity(answer, context_ids, chunks_by_id, item.get("must_cite") or [])
    not_contain = check_must_not_contain(answer, item.get("must_not_contain") or [])
    result = {
        "id": item["id"],
        "category": item.get("category"),
        "citation_integrity": integrity,
        "must_not_contain_check": not_contain,
        "heuristic_groundedness": heuristic_groundedness(answer, context_ids, chunks_by_id),
    }
    if judge_fn is not None:
        result["llm_groundedness"] = llm_judge_groundedness(
            item["question"], answer, context_ids, chunks_by_id, judge_fn)
    return result