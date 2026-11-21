"""
Professional Streamlit GUI for the RAG + RDBMS multi-agent system.

Run:
    streamlit run app.py

Everything (routing, memory/cache, SQL, charts, sources) is already handled by
graph.orchestrator.ask(); this app is a polished chat surface over it.
"""
from __future__ import annotations

import html
import time
import uuid

import pandas as pd
import streamlit as st

from config.settings import settings

st.set_page_config(
    page_title="RAG + RDBMS Multi-Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling
# A calm, light "workspace" theme. Each functionality is distinguished by a
# subtle accent colour + a soft tinted chip, never by loud blocks.
#   RAG   -> indigo    SQL -> teal    Cache -> amber    neutral -> slate
st.markdown(
    """
    <style>
      :root {
        --bg:#f5f7fb; --surface:#ffffff; --line:#e6eaf2;
        --ink:#1c2333; --muted:#69758f;
        --indigo:#5b6cdd; --indigo-soft:#eef0fd;
        --teal:#12988a;   --teal-soft:#e7f6f3;
        --amber:#c8891b;  --amber-soft:#fdf3e2;
        --slate:#64748b;  --slate-soft:#eef1f6;
        --rose:#d6456b;   --rose-soft:#fdeef2;
      }
      .stApp { background:var(--bg); color:var(--ink); }

      /* ---- header ---- */
      .hero {
        background:var(--surface); border:1px solid var(--line);
        border-radius:18px; padding:1.5rem 1.7rem; margin:.2rem 0 1.1rem;
        box-shadow:0 1px 2px rgba(28,35,51,.04);
      }
      .hero h1 { margin:0; font-size:1.75rem; font-weight:700; letter-spacing:-.3px; color:var(--ink); }
      .hero .sub { margin:.4rem 0 0; color:var(--muted); font-size:.95rem; }
      .hero .pill-row { margin-top:.9rem; }
      .pill { display:inline-block; padding:.2rem .65rem; border-radius:999px;
              font-size:.74rem; font-weight:600; margin-right:.4rem;
              border:1px solid transparent; }

      /* ---- chat bubbles ---- */
      .bubble-user {
        background:var(--indigo); color:#fff; padding:.7rem 1rem;
        border-radius:16px 16px 5px 16px; margin:.5rem 0 .5rem auto;
        max-width:72%; width:fit-content; text-align:right;
        box-shadow:0 1px 2px rgba(91,108,221,.25);
      }
      .card-bot {
        background:var(--surface); border:1px solid var(--line);
        border-left:4px solid var(--slate);
        border-radius:6px 16px 16px 6px; padding:1rem 1.15rem;
        margin:.5rem auto .5rem 0; max-width:88%;
        box-shadow:0 1px 2px rgba(28,35,51,.05);
      }
      .card-bot.rag  { border-left-color:var(--indigo); }
      .card-bot.sql  { border-left-color:var(--teal); }
      .card-bot.cache{ border-left-color:var(--amber); }
      .card-bot.err  { border-left-color:var(--rose); }

      /* ---- chips ---- */
      .chip { display:inline-block; padding:.16rem .6rem; border-radius:999px;
              font-size:.72rem; font-weight:700; letter-spacing:.2px;
              margin:0 .4rem .55rem 0; border:1px solid transparent; }
      .chip-rag   { background:var(--indigo-soft); color:#3a49b8; border-color:#d7dcfb; }
      .chip-sql   { background:var(--teal-soft);   color:#0d7d71; border-color:#c8ebe5; }
      .chip-cache { background:var(--amber-soft);  color:#9a6712; border-color:#f3e0bb; }
      .chip-slate { background:var(--slate-soft);  color:#55627a; border-color:#dde3ec; }
      .chip-rose  { background:var(--rose-soft);   color:#b23557; border-color:#f7d3dc; }

      .meta { color:var(--muted); font-size:.78rem; margin:.15rem 0 .2rem; }
      .src { display:inline-block; background:var(--slate-soft); border:1px solid var(--line);
             color:#55627a; padding:.14rem .58rem; border-radius:8px;
             font-size:.78rem; margin:.18rem .28rem .18rem 0; }

      /* ---- sidebar ---- */
      section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }
      section[data-testid="stSidebar"] * { color:var(--ink); }
      .side-label { font-size:.74rem; font-weight:700; letter-spacing:.6px;
                    text-transform:uppercase; color:var(--muted); margin:.2rem 0 .4rem; }

      /* ---- dividers inside chat details ---- */
      .hr { border:0; border-top:1px solid var(--line); margin:.6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = settings.default_thread_id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessions" not in st.session_state:
    st.session_state.sessions = {settings.default_thread_id: "default"}
if "history_loaded_for" not in st.session_state:
    st.session_state.history_loaded_for = None

_ask = None


def ask(query: str, thread_id: str):
    global _ask
    if _ask is None:
        from graph.orchestrator import ask as _a

        _ask = _a
    return _ask(query, thread_id=thread_id)


def load_history(thread_id: str) -> list[dict]:
    """Rebuild the on-screen chat from the persistent chat store."""
    from memory.chat_store import load_thread

    out = []
    for row in load_thread(thread_id):
        if row["role"] == "user":
            out.append({"role": "user", "content": row["content"]})
        else:
            meta = row["meta"] or {}
            out.append({
                "role": "assistant",
                "content": row["content"],
                "meta": {"result": meta, "latency": meta.get("_latency", 0.0)},
            })
    return out


def refresh_sessions() -> None:
    """Merge persisted threads (from chat store) into the session picker."""
    from memory.chat_store import list_threads

    for t in list_threads():
        st.session_state.sessions.setdefault(t["thread_id"], t["title"])
    # keep current thread visible even if it has no saved rows yet
    st.session_state.sessions.setdefault(st.session_state.thread_id, "New conversation")


# Hydrate picker + transcript from disk once per browser session
if st.session_state.history_loaded_for is None:
    from memory.chat_store import list_threads

    threads = list_threads()
    for t in threads:
        st.session_state.sessions.setdefault(t["thread_id"], t["title"])
    st.session_state.sessions.setdefault(st.session_state.thread_id, "New conversation")
    if threads:  # reopen the most recently active thread
        st.session_state.thread_id = threads[0]["thread_id"]
    st.session_state.messages = load_history(st.session_state.thread_id)
    st.session_state.history_loaded_for = st.session_state.thread_id


# ---------------------------------------------------------------- helpers
def _sector(result: dict) -> str:
    if result.get("_cache_similarity"):
        return "cache"
    s = result.get("sector", "")
    return s if s in ("rag", "sql") else "err"


def _chips(result: dict, sector: str) -> str:
    colors = {"rag": "chip-rag", "sql": "chip-sql", "cache": "chip-cache", "err": "chip-rose"}
    labels = {"rag": "RAG agent", "sql": "SQL agent", "cache": "Memory cache", "err": "Error"}
    chips = f'<span class="chip {colors[sector]}">{labels[sector]}</span>'
    if result.get("_cache_similarity"):
        chips += f'<span class="chip chip-slate">similarity {result["_cache_similarity"]:.2f}</span>'
    return chips


def _render_answer(result: dict, latency: float) -> None:
    sector = _sector(result)
    st.markdown(_chips(result, sector), unsafe_allow_html=True)
    st.markdown(
        f'<div class="card-bot {sector}">{result.get("answer","(no answer)")}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Details", expanded=False):
        if result.get("sql"):
            st.caption("SQL executed")
            st.code(result["sql"], language="sql")
        if result.get("plot"):
            st.caption("Generated chart")
            plot = result["plot"]
            if plot.get("data_uri"):
                st.image(plot["data_uri"])
            elif plot.get("file_path"):
                # restored from chat history: base64 was stripped, load from disk
                try:
                    from pathlib import Path as _P

                    if _P(plot["file_path"]).exists():
                        st.image(plot["file_path"])
                    else:
                        st.caption(f"(chart file no longer exists: {plot['file_path']})")
                except Exception:  # noqa: BLE001
                    st.caption(f"(chart: {plot['file_path']})")
            if plot.get("file_path"):
                st.caption(f"Saved to {plot['file_path']}")
        if result.get("table_preview"):
            st.caption("Result preview")
            st.dataframe(pd.DataFrame(result["table_preview"]), use_container_width=True)
        if result.get("sources"):
            st.caption("Sources")
            for s in result["sources"]:
                loc = f" · loc {s['location']}" if s.get("location") else ""
                st.markdown(
                    f'<span class="src">{html.escape(str(s.get("source","?")))}{html.escape(loc)}</span>',
                    unsafe_allow_html=True,
                )

    note = "served instantly from memory cache" if sector == "cache" else "routed + answered"
    st.markdown(f'<div class="meta">{note} · {latency:.2f}s</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="side-label">Sessions</div>', unsafe_allow_html=True)
    # unique labels -> thread ids (labels may repeat if titles match)
    items = [(tid, f"{title} · {tid[-4:]}" if sum(1 for _t in st.session_state.sessions.values() if _t == title) > 1 else title)
             for tid, title in st.session_state.sessions.items()]
    labels = [lab for _tid, lab in items]
    current_tid = st.session_state.thread_id
    idx = next((i for i, (tid, _lab) in enumerate(items) if tid == current_tid), 0)
    choice = st.selectbox("Conversation", labels, index=idx, label_visibility="collapsed")
    selected_tid = items[labels.index(choice)][0]

    if selected_tid != current_tid:
        st.session_state.thread_id = selected_tid
        st.session_state.messages = load_history(selected_tid)
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("New chat", use_container_width=True):
            tid = f"chat-{uuid.uuid4().hex[:8]}"
            st.session_state.sessions[tid] = f"Chat {len(st.session_state.sessions)+1}"
            st.session_state.thread_id = tid
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("Clear", use_container_width=True):
            from memory.chat_store import delete_thread

            delete_thread(st.session_state.thread_id)
            st.session_state.sessions.pop(st.session_state.thread_id, None)
            st.session_state.thread_id = settings.default_thread_id
            st.session_state.sessions[settings.default_thread_id] = "default"
            st.session_state.messages = load_history(settings.default_thread_id)
            st.rerun()

    st.divider()
    st.markdown('<div class="side-label">Status</div>', unsafe_allow_html=True)
    model = (settings.ollama_model if settings.llm_provider == "ollama"
             else settings.groq_model if settings.llm_provider == "groq"
             else settings.qwen_model)
    st.markdown(f"- **LLM** · {settings.llm_provider} / `{model}`")
    st.markdown(f"- **Embeddings** · {settings.embedding_provider}")
    st.markdown(f"- **Hybrid BM25+vector** · {'on' if settings.hybrid_search else 'off'}")
    st.markdown(f"- **Reranking** · {'on' if settings.rerank_enabled else 'off'}")
    st.markdown(f"- **Cache threshold** · {settings.cache_similarity_threshold}")

    st.divider()
    st.markdown(
        '<div class="side-label">How it works</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "1. Your question hits the **memory cache** first — a near-duplicate returns instantly.\n"
        "2. Otherwise the **router** picks the RAG (documents) or SQL (tabular) agent.\n"
        "3. The answer is written back to memory for fast reuse."
    )

# ---------------------------------------------------------------- main
st.markdown(
    """
    <div class="hero">
      <h1>RAG + RDBMS Multi-Agent Assistant</h1>
      <p class="sub">Ask anything about your documents (PDF/PPTX/DOCX) or tabular data
      (Excel/CSV). The router picks the right agent and remembers every answer.</p>
      <div class="pill-row">
        <span class="pill" style="background:var(--indigo-soft);color:#3a49b8;border-color:#d7dcfb">Documents</span>
        <span class="pill" style="background:var(--teal-soft);color:#0d7d71;border-color:#c8ebe5">Tabular data</span>
        <span class="pill" style="background:var(--amber-soft);color:#9a6712;border-color:#f3e0bb">Cached answers</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="bubble-user">{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
    else:
        _render_answer(msg["meta"]["result"], msg["meta"]["latency"])

query = st.chat_input("Ask a question about your documents or data…")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="bubble-user">{html.escape(query)}</div>', unsafe_allow_html=True)

    with st.spinner("Routing and answering…"):
        t0 = time.perf_counter()
        try:
            result = ask(query, thread_id=st.session_state.thread_id)
        except Exception as e:  # noqa: BLE001
            result = {"answer": f"Error: {e}", "sector": "error"}
        latency = time.perf_counter() - t0

    _render_answer(result, latency)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.get("answer", ""),
         "meta": {"result": result, "latency": latency}}
    )
