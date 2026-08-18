"""Step 4 demo: run BM25 keyword search over the real corpus, and show why
tokenization matters -- naive whitespace-split vs the punctuation-aware
version this module actually uses.

Usage:
    python3.12 -m scripts.run_bm25
"""
from pathlib import Path

from rag.bm25_index import BM25Index
from rag.chunking import build_chunks

CORPUS_DIR = Path("data/")


def main():
    chunks = build_chunks(CORPUS_DIR)
    chunks_by_id = {c["cid"]: c for c in chunks}
    bm25 = BM25Index(chunks)

    query = "How long is the return window for electronics?"
    print(f"Query: {query!r}")
    print(f"Naive tokens (plain .split()):        {query.lower().split()}")
    from rag.bm25_index import _tokenize
    print(f"Actual tokens (punctuation-stripped):  {_tokenize(query)}")
    print("-> notice 'electronics?' vs 'electronics' in the naive version; that trailing")
    print("   '?' would have made it silently fail to match the corpus's clean word.\n")

    results = bm25.search(query, k=5)
    print("Top 5 BM25 results:")
    for rank, cid in enumerate(results, 1):
        c = chunks_by_id[cid]
        marker = "  <-- electronics" if c["doc_slug"] == "category-electronics" else ""
        print(f"  {rank}. [{cid}] {c['doc_title']} clause {c['clause_number']}{marker}")


if __name__ == "__main__":
    main()