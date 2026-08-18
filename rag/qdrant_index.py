"""Qdrant Cloud index for the policy corpus (Step 3).

One point per chunk: vector = its embedding, payload = the full chunk dict
(text + doc_slug/doc_title/section/clause metadata). Payload indexes on
`section` and `doc_slug` are created so metadata filtering works -- Qdrant
Cloud requires an explicit index before a payload field can be filtered on.
"""
from __future__ import annotations

import os
from typing import Optional

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "shopsense_policy_corpus")


def get_client():
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL is not set.")
    return QdrantClient(url=url, api_key=api_key)


def create_collection(client, dim: int, collection: str = COLLECTION_NAME, recreate: bool = True) -> None:
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        client.create_payload_index(collection_name=collection, field_name="section",
                                     field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name=collection, field_name="doc_slug",
                                     field_schema=PayloadSchemaType.KEYWORD)


def upsert_chunks(client, chunks: list[dict], vectors: list[list[float]], collection: str = COLLECTION_NAME) -> None:
    from qdrant_client.models import PointStruct

    client.upsert(collection, points=[
        PointStruct(id=c["cid"], vector=vectors[c["cid"]], payload=c) for c in chunks
    ])


def dense_search(client, embed_query_fn, query: str, k: int = 10, section: Optional[str] = None,
                  doc_slug: Optional[str] = None, collection: str = COLLECTION_NAME) -> list[int]:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must = []
    if section:
        must.append(FieldCondition(key="section", match=MatchValue(value=section)))
    if doc_slug:
        must.append(FieldCondition(key="doc_slug", match=MatchValue(value=doc_slug)))
    qfilter = Filter(must=must) if must else None
    hits = client.query_points(collection_name=collection, query=embed_query_fn(query),
                                query_filter=qfilter, limit=k).points
    return [h.id for h in hits]