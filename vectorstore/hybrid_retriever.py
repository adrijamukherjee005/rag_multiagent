"""
Hybrid retrieval: combines BM25 (lexical) + dense vector search (semantic), then
fuses the two ranked lists with Reciprocal Rank Fusion (RRF).

Why: vector search alone misses exact keyword matches (names, IDs, acronyms),
and BM25 alone misses synonyms/paraphrases. Fusion gets both.

- `bm25_scores()` returns a normalized BM25 score vector over the whole collection.
- `fuse_rrf()` merges two ranked id lists into a single ranking via RRF, which only
  depends on rank (not raw score scale), so BM25 and cosine scores can be merged safely.
- The BM25 index is built lazily from the Chroma collection and cached; it is rebuilt
  automatically when the collection count changes (e.g. after re-ingestion).
"""
from __future__ import annotations

import logging
from typing import Any

from rank_bm25 import BM25Okapi

from config.settings import settings
from llm.llm_factory import get_embeddings
from vectorstore.chroma_store import _client

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Cached BM25 index + the collection count it was built for.
_bm25_cache: tuple[int, BM25Okapi, list[str]] | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics; a lightweight, language-agnostic tokenizer."""
    return [w for w in __import__("re").findall(r"\w+", text.lower()) if w]


def _get_bm25_index(collection: Any) -> tuple[BM25Okapi, list[str]]:
    """
    Build (or reuse) a BM25 index over every document in the collection.
    Returns (bm25_index, ordered_ids) where ordered_ids[i] is the id of the doc
    whose scores are at position i.
    """
    global _bm25_cache

    data = collection.get(include=["documents", "metadatas"])
    count = len(data["ids"])

    if count == 0:
        return None, []

    if _bm25_cache is not None and _bm25_cache[0] == count:
        return _bm25_cache[1], _bm25_cache[2]

    logger.info("Building BM25 index over %d documents...", count)
    tokenized = [_tokenize(t or "") for t in data["documents"]]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache = (count, bm25, data["ids"])
    logger.info("BM25 index ready (cached for count=%d).", count)

    return bm25, data["ids"]


def bm25_rank(query: str, top_n: int) -> list[str]:
    """Return up to `top_n` document ids ranked by BM25 relevance (best first)."""
    collection = _client().get_or_create_collection(settings.chroma_collection)
    bm25, ordered_ids = _get_bm25_index(collection)
    if bm25 is None:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [ordered_ids[i] for i in ranked[:top_n] if scores[i] > 0]


def vector_rank(query: str, top_n: int) -> list[str]:
    """Return up to `top_n` document ids ranked by dense vector similarity (best first)."""
    collection = _client().get_or_create_collection(settings.chroma_collection)
    if collection.count() == 0:
        return []

    emb = get_embeddings().embed_query(query)
    res = collection.query(
        query_embeddings=[emb],
        n_results=min(top_n, collection.count()),
        include=["metadatas"],
    )
    return res["ids"][0] if res and res["ids"] else []


def fuse_rrf(ranked_lists: list[list[str]], k: int | None = None) -> list[str]:
    """
    Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank).
    Higher fused score = more relevant across both lists. Returns ids best-first.
    """
    k = k if k is not None else settings.rrf_k
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def hybrid_search(query: str, pool_size: int | None = None) -> list[str]:
    """
    Fuse BM25 + vector rankings into a single ranked id list (best first).
    `pool_size` controls how many candidates are pulled from each retriever.
    Falls back to plain vector search when HYBRID_SEARCH=false.
    """
    pool = pool_size or settings.candidate_pool_size

    if not settings.hybrid_search:
        return vector_rank(query, pool)

    bm25_ids = bm25_rank(query, pool)
    vec_ids = vector_rank(query, pool)

    if not bm25_ids:
        return vec_ids
    if not vec_ids:
        return bm25_ids

    fused = fuse_rrf([vec_ids, bm25_ids])
    logger.debug("Hybrid fusion: %d vector + %d bm25 candidates -> %d fused",
                 len(vec_ids), len(bm25_ids), len(fused))
    return fused


def retrieve(query: str, k: int | None = None) -> list[tuple[Any, float]]:
    """
    Full retrieval pipeline: hybrid (BM25+vector) first stage, then cross-encoder
    reranking. Returns a list of (Document, score) pairs, best-first, length <= k —
    the same shape `similarity_search_with_score` returns, so it is a drop-in
    replacement in rag_agent.py.
    """
    from langchain_core.documents import Document

    from vectorstore.reranker import fetch_docs_by_ids, rerank

    k = k or settings.rag_top_k
    pool = max(settings.candidate_pool_size, k)

    # First stage: fused BM25 + vector ranking -> candidate id pool
    candidate_ids = hybrid_search(query, pool_size=pool)

    if not candidate_ids:
        return []

    # Second stage: cross-encoder rerank the pool down to top_k
    # (rerank() no-ops to first-stage order when RERANK_ENABLED=false)
    ranked = rerank(query, candidate_ids, top_k=k)

    # Fetch full text for the final ids and build Document objects
    final_ids = [cid for cid, _, _ in ranked]
    final_docs = fetch_docs_by_ids(final_ids)

    results: list[tuple[Any, float]] = []
    for cid, score, _ in ranked:
        info = final_docs.get(cid)
        if not info:
            continue
        doc = Document(page_content=info["text"], metadata=info["metadata"])
        doc.metadata["retrieval_score"] = score
        results.append((doc, score))
    return results
