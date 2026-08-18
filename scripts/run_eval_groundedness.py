"""Step 9 demo: for a sample of data/eval/golden_set.json items, retrieve
real chunks (BM25 + your working cross-encoder), generate a real cited
answer, then score groundedness:
  - citation_integrity        (no LLM -- hallucinated / missing citations)
  - must_not_contain_check    (no LLM -- forbidden-phrase leakage)
  - heuristic_groundedness    (no LLM -- lexical-overlap faithfulness proxy)
  - llm_groundedness          (optional, --judge -- real LLM-as-judge pass)

No Qdrant dependency, same reasoning as step 8's script -- retrieval here
is BM25 + rerank only.

Usage:
    python3.12 -m scripts.run_eval_groundedness                  # first 5 golden items
    python3.12 -m scripts.run_eval_groundedness --sample 20      # the whole golden set
    python3.12 -m scripts.run_eval_groundedness --judge          # also run the LLM judge (2x LLM calls/item)
"""
import argparse
from pathlib import Path

from rag.bm25_index import BM25Index
from rag.chunking import build_chunks
from rag.eval_groundedness import evaluate_item
from rag.eval_retrieval import load_golden_set
from rag.generate import answer_from_ids, default_complete
from rag.rerank import CrossEncoderReranker

CORPUS_DIR = Path("data")
GOLDEN_PATH = Path("data/eval/golden_set.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=5, help="How many golden-set items to run (default 5).")
    parser.add_argument("--judge", action="store_true", help="Also run the LLM-as-judge pass (extra LLM call per item).")
    parser.add_argument("--pool", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    chunks = build_chunks(CORPUS_DIR)
    chunks_by_id = {c["cid"]: c for c in chunks}
    golden = load_golden_set(GOLDEN_PATH)[: args.sample]
    print(f"Loaded {len(chunks)} chunks. Running groundedness eval on {len(golden)} golden-set item(s).\n")

    bm25 = BM25Index(chunks)
    reranker = CrossEncoderReranker()
    judge_fn = default_complete if args.judge else None

    hallucination_count = 0
    missing_citation_count = 0
    violation_count = 0

    for item in golden:
        candidates = bm25.search(item["question"], k=args.pool)
        context_ids = reranker.rerank(item["question"], chunks_by_id, candidates, k=args.k)
        answer = answer_from_ids(item["question"], context_ids, chunks_by_id)  # real LLM call

        report = evaluate_item(item, answer, context_ids, chunks_by_id, judge_fn=judge_fn)
        integrity = report["citation_integrity"]
        not_contain = report["must_not_contain_check"]

        if not integrity["citations_grounded"]:
            hallucination_count += 1
        if not integrity["required_citations_satisfied"]:
            missing_citation_count += 1
        if not not_contain["clean"]:
            violation_count += 1

        print(f"[{item['id']}] ({item['category']}) {item['question'][:70]}")
        print(f"  answer: {answer[:150]}")
        print(f"  citations_grounded={integrity['citations_grounded']}  "
              f"required_citations_satisfied={integrity['required_citations_satisfied']}  "
              f"must_not_contain_clean={not_contain['clean']}  "
              f"heuristic_groundedness={report['heuristic_groundedness']:.2f}")
        if not integrity["citations_grounded"]:
            print(f"    !! hallucinated citations: {integrity['hallucinated_citations']}")
        if not integrity["required_citations_satisfied"]:
            print(f"    !! missing required citations: {integrity['missing_required_citations']}")
        if not not_contain["clean"]:
            print(f"    !! must_not_contain violations: {not_contain['violations']}")
        if args.judge and "llm_groundedness" in report:
            jg = report["llm_groundedness"]
            print(f"    llm_groundedness score: {jg['score']}  unsupported_claims: {jg['unsupported_claims']}")
        print()

    print("-" * 70)
    print(f"Summary over {len(golden)} item(s): "
          f"{hallucination_count} with hallucinated citations, "
          f"{missing_citation_count} missing a required citation, "
          f"{violation_count} with a must_not_contain violation.")


if __name__ == "__main__":
    """M4
    python3.12 -m scripts.run_eval_groundedness
python3.12 -m scripts.run_eval_groundedness --sample 20     # whole golden set
python3.12 -m scripts.run_eval_groundedness --judge          # also runs the real LLM-judge pass (2x LLM calls/item)"""
    main()