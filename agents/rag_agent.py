"""
Unstructured RAG Agent: retrieves relevant chunks from ChromaDB and synthesizes a
grounded natural-language answer with source citations.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from llm.llm_factory import get_chat_llm
from vectorstore.hybrid_retriever import retrieve

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a precise, grounded document Q&A assistant.
Answer the user's question using ONLY the provided context excerpts.
If the context does not contain enough information, say so plainly rather than guessing.
Cite sources inline like [source: filename, page/slide N] where applicable.
Keep the answer concise and directly responsive to the question.
"""


def run_rag_agent(query: str, top_k: int | None = None) -> dict:
    k = top_k or settings.rag_top_k

    results = retrieve(query, k=k)
    return answer_from_retrieved_context(query, results)


def answer_from_retrieved_context(
    query: str, results: list[tuple[Document, float]]
) -> dict:
    """Generate an answer from already retrieved documents."""
    if not results:
        return {
            "answer": "I couldn't find any relevant content in the ingested documents. "
                      "Try ingesting documents first with `python -m ingestion.document_ingest`.",
            "sources": [],
            "sector": "rag",
        }

    context_blocks = []
    sources = []
    for doc, score in results:
        meta = doc.metadata
        loc = meta.get("page") or meta.get("slide") or ""
        loc_label = f", loc {loc}" if loc else ""
        context_blocks.append(f"[source: {meta.get('source', 'unknown')}{loc_label}]\n{doc.page_content}")
        sources.append({
            "source": meta.get("source", "unknown"),
            "location": loc,
            "type": meta.get("type"),
            "relevance_score": round(float(score), 4),
        })

    context = "\n\n---\n\n".join(context_blocks)

    llm = get_chat_llm()
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {query}"),
    ]
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
        "sector": "rag",
    }
