"""Step 2 demo: embed a few real chunks and two similar/dissimilar
questions, so you can see real semantic closeness -- not the fake
placeholder numbers I had to use while building this.

Usage:
    python3.12 -m scripts.run_embeddings
"""
import math
from pathlib import Path

from dotenv import load_dotenv

from rag.chunking import build_chunks
from rag.embeddings import get_embedder

CORPUS_DIR = Path("data/")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main():
    load_dotenv()
    embedder = get_embedder()

    chunks = build_chunks(CORPUS_DIR)
    sample = next(c for c in chunks if c["doc_slug"] == "category-electronics" and c["clause_number"] == "1")
    vec = embedder.embed_query(sample["text"])
    print(f"Embedded 1 chunk -> vector length: {len(vec)}")
    print(f"First 8 numbers: {[round(x, 4) for x in vec[:8]]}\n")

    q1 = "How long is the return window for electronics?"
    q2 = "What is the electronics return period?"
    q3 = "Are grocery items returnable?"
    v1, v2, v3 = embedder.embed_query(q1), embedder.embed_query(q2), embedder.embed_query(q3)

    print(f"similarity(Q1, Q2) [same meaning, different words] = {cosine(v1, v2):.4f}")
    print(f"similarity(Q1, Q3) [different topic]                = {cosine(v1, v3):.4f}")


if __name__ == "__main__":
    main()