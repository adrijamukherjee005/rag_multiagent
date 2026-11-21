"""
ChromaDB wrapper providing:
  - `get_document_store()`  -> the RAG document collection (PDF/PPTX/DOCX chunks)
  - `get_cache_store()`     -> a lightweight semantic cache of (query -> answer)

Kept as two separate collections so cache entries never pollute document retrieval.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from config.settings import settings
from llm.llm_factory import get_embeddings


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_document_store() -> Chroma:
    """LangChain Chroma vectorstore over ingested documents."""
    return Chroma(
        client=_client(),
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
    )


class SemanticCache:
    """
    Thin semantic cache on top of a dedicated Chroma collection. Stores the raw query
    text as the embedded document and the answer payload as metadata (JSON-encoded,
    since Chroma metadata values must be scalars).
    """

    def __init__(self) -> None:
        self._client = _client()
        self._embeddings = get_embeddings()
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_cache_collection
        )

    def lookup(self, query: str) -> dict[str, Any] | None:
        if self._collection.count() == 0:
            return None

        query_embedding = self._embeddings.embed_query(query)
        result = self._collection.query(query_embeddings=[query_embedding], n_results=1)

        if not result["ids"] or not result["ids"][0]:
            return None

        distance = result["distances"][0][0]
        # Chroma default distance is squared L2 for normalized embeddings ~ (1 - cosine_sim)*2
        # We normalize to an approximate cosine similarity for threshold comparison.
        similarity = 1 - (distance / 2)

        if similarity < settings.cache_similarity_threshold:
            return None

        metadata = result["metadatas"][0][0]
        payload = json.loads(metadata["payload"])
        payload["_cache_similarity"] = round(similarity, 4)
        payload["_cached_query"] = result["documents"][0][0]
        return payload

    def store(self, query: str, payload: dict[str, Any]) -> None:
        embedding = self._embeddings.embed_query(query)
        self._collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[query],
            metadatas=[{"payload": json.dumps(payload), "ts": time.time()}],
        )


_cache_singleton: SemanticCache | None = None


def get_cache_store() -> SemanticCache:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = SemanticCache()
    return _cache_singleton
