"""
Persistent chat history (separate from the semantic answer cache).

- `save_message()` / `load_thread()` store the full conversation (question + answer
  with its result payload: sector, SQL, sources, plot path, latency) so a thread
  survives browser refreshes, restarts, and can be reopened later.
- `list_threads()` / `delete_thread()` back the session picker in the GUI.

The semantic cache (vectorstore/chroma_store.py) answers *repeat questions* fast;
this store answers *"what did we talk about earlier"* — both layer in front of the
agents. Data lives in a small SQLite file (CHAT_DB_PATH).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.settings import settings

__all__ = ["save_message", "load_thread", "list_threads", "delete_thread"]

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _conn() -> sqlite3.Connection:
    Path(settings.chat_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.chat_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id  TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            meta       TEXT,
            created_at REAL    NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_thread ON chat_messages(thread_id, id)"
    )
    conn.commit()
    return conn


def save_message(thread_id: str, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
    with _lock:
        _conn().execute(
            "INSERT INTO chat_messages (thread_id, role, content, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, json.dumps(meta, default=str) if meta else None, time.time()),
        )
        _conn().commit()


def load_thread(thread_id: str) -> list[dict[str, Any]]:
    rows = (
        _conn()
        .execute(
            "SELECT role, content, meta, created_at FROM chat_messages "
            "WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        )
        .fetchall()
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        meta = json.loads(r["meta"]) if r["meta"] else None
        out.append({"role": r["role"], "content": r["content"], "meta": meta, "ts": r["created_at"]})
    return out


def list_threads() -> list[dict[str, Any]]:
    """Newest-first thread summaries for the session picker."""
    rows = (
        _conn()
        .execute(
            """
            SELECT m.thread_id,
                   MAX(m.created_at) AS updated_at,
                   COUNT(*)          AS n,
                   (SELECT content FROM chat_messages c
                     WHERE c.thread_id = m.thread_id AND c.role = 'user'
                     ORDER BY c.id LIMIT 1) AS title
            FROM chat_messages m
            GROUP BY m.thread_id
            ORDER BY updated_at DESC
            """
        )
        .fetchall()
    )
    out = []
    for r in rows:
        title = (r["title"] or r["thread_id"]).strip().replace("\n", " ")
        if len(title) > 42:
            title = title[:42] + "…"
        out.append({"thread_id": r["thread_id"], "title": title, "updated_at": r["updated_at"], "n": r["n"]})
    return out


def delete_thread(thread_id: str) -> None:
    with _lock:
        _conn().execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
        _conn().commit()
