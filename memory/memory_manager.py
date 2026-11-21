"""
Two layers of memory:

1. `get_checkpointer()` — a LangGraph SqliteSaver used by the StateGraph so full
   conversation state (messages, routing decisions, intermediate agent outputs)
   persists across turns for a given `thread_id`. This gives multi-turn context.

2. `SemanticCache` (in vectorstore/chroma_store.py) — a query -> answer cache checked
   BEFORE routing. This is what actually prevents redundant RAG/SQL execution for
   repeated or highly similar questions. Re-exported here for a single memory-layer
   entrypoint used by the orchestrator.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import settings
from vectorstore.chroma_store import SemanticCache, get_cache_store

__all__ = ["get_checkpointer", "get_cache_store", "SemanticCache", "lookup_or_none", "store_result"]


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    return SqliteSaver(conn)


def lookup_or_none(query: str) -> dict | None:
    """Convenience wrapper: returns a cached answer payload or None on a cache miss."""
    return get_cache_store().lookup(query)


def store_result(query: str, payload: dict) -> None:
    get_cache_store().store(query, payload)
