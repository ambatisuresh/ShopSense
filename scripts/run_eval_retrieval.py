"""Step 8 demo: run retrieval evaluation (precision@k / recall@k / MRR)
against data/golden_set.json.

Two variants, both using your real corpus and the now-working
CrossEncoderReranker -- neither needs Qdrant, so this isn't blocked by the
QDRANT_URL issue:
  - "bm25-only":    BM25Index.search() alone
  - "bm25+rerank":  BM25's top candidates re-scored by the real cross-encoder

Once QDRANT_URL is confirmed working, dense search (rag/qdrant_index.py)
and a true hybrid variant (rag/hybrid.py's rrf_fuse over dense+BM25) can be
added the same way -- same evaluate() call, just a different retrieve_fn.

Usage:
    python3.12 -m scripts.run_eval_retrieval
"""
from pathlib import Path

from rag.bm25_index import BM25Index
from rag.chunking import build_chunks
from rag.eval_retrieval import evaluate, load_golden_set
from rag.rerank import CrossEncoderReranker

CORPUS_DIR = Path("data")
GOLDEN_PATH = Path("data/eval/golden_set.json")


def main():
    chunks = build_chunks(CORPUS_DIR)
    chunks_by_id = {c["cid"]: c for c in chunks}
    golden = load_golden_set(GOLDEN_PATH)
    print(f"Loaded {len(chunks)} chunks, {len(golden)} golden-set items.\n")

    bm25 = BM25Index(chunks)
    reranker = CrossEncoderReranker()

    def bm25_only(q, k=5):
        return bm25.search(q, k=k)

    def bm25_plus_rerank(q, k=5, pool=10):
        candidates = bm25.search(q, k=pool)
        return reranker.rerank(q, chunks_by_id, candidates, k=k)

    variants = {
        "bm25-only": bm25_only,
        "bm25+rerank": bm25_plus_rerank,
    }

    k = 5
    print(f"{'variant':<16}{'precision@' + str(k):>14}{'recall@' + str(k):>14}{'mrr':>10}{'n_scored':>12}")
    result = None
    for name, fn in variants.items():
        result = evaluate(fn, golden, chunks, k=k)
        print(f"{name:<16}{result['precision@k']:>14.3f}{result['recall@k']:>14.3f}"
              f"{result['mrr']:>10.3f}{result['n_scored']:>12}")

    skipped = [item["id"] for item in golden if item["id"] not in result["scored_ids"]]
    print(f"\nSkipped (no resolvable must_cite -- injection/unanswerable categories, "
          f"scored separately by step 9's groundedness checks): {skipped}")


if __name__ == "__main__":
    main()