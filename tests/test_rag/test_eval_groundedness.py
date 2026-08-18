"""Tests for rag/eval_groundedness.py -- the "are cited clauses actually
grounded in the source playbook" check M4 specifically asks for."""
from rag.eval_groundedness import (
    check_must_not_contain,
    citation_integrity,
    evaluate_item,
    extract_cited_ids,
    heuristic_groundedness,
    llm_judge_groundedness,
)

CHUNKS_BY_ID = {
    1: {"cid": 1, "doc_title": "Kartway Returns and Refunds Policy",
        "text": "Customers may return items within 30 days of receipt for a full refund."},
    2: {"cid": 2, "doc_title": "Category Policy Addendum: Electronics",
        "text": "Electronics returns must be initiated within 15 days of receipt."},
}


def test_extract_cited_ids_parses_bracketed_ids():
    assert extract_cited_ids("Per [1] and [2], this is allowed.") == [1, 2]


def test_citation_integrity_flags_hallucinated_citation():
    # [7] was never in context -- the model made it up. This is the
    # specific failure mode citation_integrity exists to catch.
    answer = "Returns are accepted within 30 days [1] and also see [7]."
    report = citation_integrity(answer, context_ids=[1, 2], chunks_by_id=CHUNKS_BY_ID, must_cite_titles=[])
    assert report["hallucinated_citations"] == [7]
    assert report["citations_grounded"] is False


def test_citation_integrity_passes_when_every_citation_was_in_context():
    answer = "Returns are accepted within 30 days [1]."
    report = citation_integrity(answer, context_ids=[1, 2], chunks_by_id=CHUNKS_BY_ID, must_cite_titles=[])
    assert report["citations_grounded"] is True


def test_citation_integrity_flags_missing_required_doc_title():
    answer = "Electronics get 15 days [2]."
    report = citation_integrity(
        answer, context_ids=[1, 2], chunks_by_id=CHUNKS_BY_ID,
        must_cite_titles=["Kartway Returns and Refunds Policy", "Category Policy Addendum: Electronics"],
    )
    assert report["missing_required_citations"] == ["Kartway Returns and Refunds Policy"]
    assert report["required_citations_satisfied"] is False


def test_citation_integrity_required_satisfied_true_when_nothing_required():
    # golden_set.json items with an empty must_cite (injection/unanswerable
    # categories) shouldn't be penalized for "missing" a citation nothing
    # ever required.
    report = citation_integrity("no citations here", context_ids=[1], chunks_by_id=CHUNKS_BY_ID, must_cite_titles=[])
    assert report["required_citations_satisfied"] is True


def test_must_not_contain_flags_a_present_phrase_case_insensitively():
    result = check_must_not_contain("Your REFUND APPROVED and on its way.", ["refund approved"])
    assert result["clean"] is False
    assert result["violations"] == ["refund approved"]


def test_must_not_contain_clean_when_absent():
    result = check_must_not_contain("We are reviewing your request.", ["refund approved"])
    assert result["clean"] is True


def test_heuristic_groundedness_scores_higher_for_answer_matching_its_source():
    grounded = "Returns within 30 days receipt full refund [1]"
    unrelated = "Bananas spaceship quantum unrelated nonsense [1]"
    g_score = heuristic_groundedness(grounded, context_ids=[1], chunks_by_id=CHUNKS_BY_ID)
    u_score = heuristic_groundedness(unrelated, context_ids=[1], chunks_by_id=CHUNKS_BY_ID)
    assert g_score > u_score


def test_heuristic_groundedness_zero_with_no_citations():
    assert heuristic_groundedness("no citations here", context_ids=[1], chunks_by_id=CHUNKS_BY_ID) == 0.0


def test_llm_judge_groundedness_parses_valid_json_response():
    def fake_judge(system, user):
        return '{"score": 0.9, "unsupported_claims": []}'

    result = llm_judge_groundedness("q", "a [1]", [1], CHUNKS_BY_ID, complete_fn=fake_judge)
    assert result["score"] == 0.9
    assert result["unsupported_claims"] == []


def test_llm_judge_groundedness_handles_malformed_json_gracefully():
    # A real LLM occasionally won't return clean JSON -- this must degrade
    # to a null score with a parse_error flag, not crash the eval run.
    def broken_judge(system, user):
        return "not json at all"

    result = llm_judge_groundedness("q", "a [1]", [1], CHUNKS_BY_ID, complete_fn=broken_judge)
    assert result["score"] is None
    assert result["parse_error"] is True


def test_evaluate_item_combines_all_checks_without_a_judge_fn():
    item = {
        "id": "SHOPSENSE-EV-901", "category": "guardrail",
        "must_cite": ["Kartway Returns and Refunds Policy"],
        "must_not_contain": ["refund approved"],
    }
    answer = "You have 30 days to return this item [1]."
    report = evaluate_item(item, answer, context_ids=[1], chunks_by_id=CHUNKS_BY_ID)
    assert report["citation_integrity"]["citations_grounded"] is True
    assert report["citation_integrity"]["required_citations_satisfied"] is True
    assert report["must_not_contain_check"]["clean"] is True
    assert "llm_groundedness" not in report


def test_evaluate_item_includes_llm_groundedness_when_judge_fn_given():
    item = {"id": "x", "category": "factual", "question": "q", "must_cite": [], "must_not_contain": []}

    def fake_judge(system, user):
        return '{"score": 0.75, "unsupported_claims": ["foo"]}'

    report = evaluate_item(item, "a [1]", context_ids=[1], chunks_by_id=CHUNKS_BY_ID, judge_fn=fake_judge)
    assert "llm_groundedness" in report
    assert report["llm_groundedness"]["score"] == 0.75