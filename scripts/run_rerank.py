"""Step 6 demo: rerank the fused shortlist from Step 5 with a real
cross-encoder, and compare the order before/after.

Needs: pip install sentence-transformers (downloads the model on first run).

Usage:
    python3.12 -m scripts.run_rerank
"""
from pathlib import Path

from dotenv import load_dotenv

from rag import qdrant_index as qi
from rag.bm25_index import BM25Index
from rag.chunking import build_chunks
from rag.embeddings import get_embedder
from rag.hybrid import rrf_fuse
from rag.rerank import CrossEncoderReranker

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

    show("Before reranking (fused shortlist from Step 5):", fused, chunks_by_id)

    print("Loading cross-encoder/ms-marco-MiniLM-L-6-v2 (first run downloads it)...\n")
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(QUERY, chunks_by_id, fused, k=5)

    show("After reranking:", reranked, chunks_by_id)


if __name__ == "__main__":
    main()