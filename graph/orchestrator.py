"""
LangGraph StateGraph wiring:

    START -> check_memory -> (cache_hit? -> END) -> route -> (rag | sql) -> save_memory -> END

`check_memory` queries the semantic cache before any routing/agent work happens.
On a cache hit, the graph short-circuits straight to END with the cached payload.
On a miss, `route` classifies intent, the chosen sector agent runs, and the fresh
result is written back into the semantic cache in `save_memory` before returning.

The graph is compiled with a SqliteSaver checkpointer so per-thread conversation
state (message history) also persists independently of the semantic cache.
"""
from __future__ import annotations

import logging
import time
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.rag_agent import run_rag_agent
from agents.router_agent import route_query
from agents.sql_agent import run_sql_agent
from config.settings import settings
from memory.chat_store import save_message
from memory.memory_manager import get_checkpointer, lookup_or_none, store_result

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    route: str
    result: dict[str, Any]
    from_cache: bool


def check_memory_node(state: GraphState) -> GraphState:
    cached = lookup_or_none(state["query"])
    if cached is not None:
        logger.info(
            "Cache hit (similarity=%s) for query: %s",
            cached.get("_cache_similarity"), state["query"],
        )
        return {"result": cached, "from_cache": True}
    return {"from_cache": False}


def route_after_memory(state: GraphState) -> Literal["route", "__end__"]:
    return END if state.get("from_cache") else "route"


def route_node(state: GraphState) -> GraphState:
    decision = route_query(state["query"])
    logger.info("Routed to '%s' (%s)", decision.route, decision.reasoning)
    return {"route": decision.route}


def route_after_classification(state: GraphState) -> Literal["rag_agent", "sql_agent"]:
    return "rag_agent" if state["route"] == "rag" else "sql_agent"


def rag_agent_node(state: GraphState) -> GraphState:
    result = run_rag_agent(state["query"])
    return {"result": result}


def sql_agent_node(state: GraphState) -> GraphState:
    result = run_sql_agent(state["query"])
    return {"result": result}


def save_memory_node(state: GraphState) -> GraphState:
    if not state.get("from_cache"):
        store_result(state["query"], state["result"])
    return {}


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("check_memory", check_memory_node)
    graph.add_node("route", route_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "check_memory")
    graph.add_conditional_edges("check_memory", route_after_memory, {"route": "route", END: END})
    graph.add_conditional_edges(
        "route", route_after_classification, {"rag_agent": "rag_agent", "sql_agent": "sql_agent"}
    )
    graph.add_edge("rag_agent", "save_memory")
    graph.add_edge("sql_agent", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=get_checkpointer())


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(query: str, thread_id: str | None = None) -> dict[str, Any]:
    """
    Single entrypoint used by main.py / app.py / any API layer.

    Persists the conversation (question + answer + result payload) to the chat
    store so full threads survive restarts; the semantic cache still short-circuits
    repeat/near-duplicate questions to a stored answer without re-running agents.
    """
    tid = thread_id or settings.default_thread_id
    graph = get_graph()
    config = {"configurable": {"thread_id": tid}}
    save_message(tid, "user", query)
    t0 = time.perf_counter()
    final_state = graph.invoke({"query": query, "messages": [("user", query)]}, config=config)
    result = final_state["result"]
    result["_latency"] = round(time.perf_counter() - t0, 3)

    # Persist the answer, but drop heavy base64 chart data (the PNG file path stays).
    to_store = dict(result)
    if isinstance(to_store.get("plot"), dict):
        plot = {k: v for k, v in to_store["plot"].items() if k not in ("base64_png", "data_uri")}
        to_store["plot"] = plot
    save_message(tid, "assistant", result.get("answer", ""), meta=to_store)
    return result
