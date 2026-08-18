"""Step 1 demo: run the chunker over the real policy corpus and print
what it produces.

Usage:
    python3 demo.py
"""
from pathlib import Path

from rag.chunking import build_chunks

CORPUS_DIR = Path("data/")


def main():
    chunks = build_chunks(CORPUS_DIR)
    print(f"Total chunks produced: {len(chunks)}\n")

    # Show every chunk for one document, so you can see the full before -> after.
    doc_slug = "returns-policy"
    print(f"--- All chunks for '{doc_slug}' ---")
    for c in chunks:
        if c["doc_slug"] == doc_slug:
            print(f"  clause {c['clause_number']:<6} {c['clause_title']:<35} ({len(c['text'])} chars)")

    print()
    print("--- One chunk in full detail ---")
    sample = next(c for c in chunks if c["doc_slug"] == doc_slug and c["clause_number"] == "2.1")
    for key, value in sample.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()