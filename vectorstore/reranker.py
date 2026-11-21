"""
Cross-encoder reranking stage.

A cross-encoder scores a (query, document) pair jointly and is far more accurate
than the bi-encoder similarity used in first-stage retrieval — but too slow to run
over the entire corpus, so it only re-scores a small candidate pool.

Flow:
  1. First stage (hybrid or vector) produces a ranked candidate pool of ids.
  2. `rerank()` fetches each candidate's text and scores it against the query
     with a cross-encoder (default: ms-marco-MiniLM-L-6-v2, ~80MB, fast on CPU).
  3. Returns the top_k most relevant documents with their relevance scores.

Set RERANK_ENABLED=false in .env to disable (returns first-stage order as-is).
"""
from __future__ import annotations

import logging

from config.settings import settings
from vectorstore.chroma_store import _client

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

_cross_encoder = None


def _get_cross_encoder():
    """Lazily load and cache the cross-encoder (downloads model on first use)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker: %s", settings.rerank_model)
        _cross_encoder = CrossEncoder(settings.rerank_model)
        logger.info("Reranker ready.")
    return _cross_encoder


def fetch_docs_by_ids(ids: list[str]) -> dict[str, dict]:
    """Return {id: {"text":..., "metadata":...}} for the given document ids."""
    if not ids:
        return {}
    collection = _client().get_or_create_collection(settings.chroma_collection)
    res = collection.get(ids=ids, include=["documents", "metadatas"])
    out = {}
    for i, doc_id in enumerate(res["ids"]):
        out[doc_id] = {
            "text": res["documents"][i] or "",
            "metadata": res["metadatas"][i] or {},
        }
    return out


def rerank(query: str, candidate_ids: list[str], top_k: int | None = None) -> list[tuple[str, float, dict]]:
    """
    Rerank candidate ids against the query using a cross-encoder.
    Returns a list of (id, relevance_score, metadata) sorted best-first, length <= top_k.
    If reranking is disabled or fails, falls back to the original candidate order.
    """
    k = top_k or settings.rerank_top_k

    if not candidate_ids:
        return []

    if not settings.rerank_enabled:
        docs = fetch_docs_by_ids(candidate_ids[:k])
        return [(cid, 0.0, docs.get(cid, {}).get("metadata", {})) for cid in candidate_ids[:k]]

    docs = fetch_docs_by_ids(candidate_ids)
    if not docs:
        return []

    try:
        model = _get_cross_encoder()
        pairs = [(query, docs[cid]["text"]) for cid in candidate_ids if cid in docs]
        ids_in_pair = [cid for cid in candidate_ids if cid in docs]
        scores = model.predict(pairs)

        ranked = sorted(zip(ids_in_pair, scores), key=lambda x: x[1], reverse=True)[:k]
        logger.info("Reranked %d candidates -> top %d (best score=%.4f)",
                    len(ids_in_pair), len(ranked), ranked[0][1] if ranked else 0.0)
        return [(cid, float(score), docs[cid]["metadata"]) for cid, score in ranked]

    except Exception as e:  # noqa: BLE001
        logger.warning("Reranking failed (%s); falling back to first-stage order.", e)
        return [(cid, 0.0, docs.get(cid, {}).get("metadata", {})) for cid in candidate_ids[:k]]
