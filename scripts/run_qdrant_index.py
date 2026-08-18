"""Step 3 demo: embed the full corpus and load it into Qdrant Cloud, then
run one real dense search against it.

Needs QDRANT_URL and QDRANT_API_KEY in your .env, from a free cluster at
https://cloud.qdrant.io/ (see Day 2 Session 2's setup instructions).

Usage:
    python3.12 -m scripts.run_qdrant_index
"""
from pathlib import Path

from dotenv import load_dotenv

from rag.chunking import build_chunks
from rag.embeddings import get_embedder
from rag import qdrant_index as qi

CORPUS_DIR = Path("data/")


def main():
    load_dotenv()

    chunks = build_chunks(CORPUS_DIR)
    print(f"Parsed {len(chunks)} chunks.")

    embedder = get_embedder()
    vectors = embedder.embed_documents([c["text"] for c in chunks])
    dim = len(vectors[0])
    print(f"Embedded {len(vectors)} vectors (dim={dim}).")

    client = qi.get_client()
    qi.create_collection(client, dim=dim)
    qi.upsert_chunks(client, chunks, vectors)

    count = client.count(qi.COLLECTION_NAME, exact=True).count
    print(f"Indexed {count} points into Qdrant collection '{qi.COLLECTION_NAME}'.")
    assert count == len(chunks), f"expected {len(chunks)}, found {count}"

    query = "How long is the return window for electronics?"
    result_ids = qi.dense_search(client, embedder.embed_query, query, k=3)
    chunks_by_id = {c["cid"]: c for c in chunks}
    print(f"\nSample search: {query!r}")
    for cid in result_ids:
        c = chunks_by_id[cid]
        print(f"  [{cid}] {c['doc_title']} clause {c['clause_number']}: {c['text'][:80]}...")


if __name__ == "__main__":
    main()