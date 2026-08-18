"""Step 5 demo: compare dense-only, BM25-only, and fused (hybrid) results
side by side for the electronics query that BM25 alone couldn't answer.

Usage:
    python3.12 -m scripts.run_hybrid
"""
from pathlib import Path

from dotenv import load_dotenv

from rag import qdrant_index as qi
from rag.bm25_index import BM25Index
from rag.chunking import build_chunks
from rag.embeddings import get_embedder
from rag.hybrid import rrf_fuse

CORPUS_DIR = Path("data/")
QUERY = "How long is the return window for electronics?"


def show(label, ids, chunks_by_id):
    print(label)
    for rank, cid in enumerate(ids, 1):
        c = chunks_by_id[cid]
        marker = "  <-- electronics" if c["doc_slug"] == "category-electronics" else ""
        print(f"  {rank}. [{cid}] {c['doc_title']} clause {c['clause_number']}{marker}")
    print()


def main():
    load_dotenv()

    chunks = build_chunks(CORPUS_DIR)
    chunks_by_id = {c["cid"]: c for c in chunks}
    embedder = get_embedder()
    client = qi.get_client()
    bm25 = BM25Index(chunks)

    dense_list = qi.dense_search(client, embedder.embed_query, QUERY, k=10)
    bm25_list = bm25.search(QUERY, k=10)
    fused = rrf_fuse([dense_list, bm25_list], k=10)

    print(f"Query: {QUERY!r}\n")
    show("Dense (Qdrant, real embeddings):", dense_list, chunks_by_id)
    show("BM25 (keyword):", bm25_list, chunks_by_id)
    show("Fused (RRF hybrid):", fused, chunks_by_id)


if __name__ == "__main__":
    main()