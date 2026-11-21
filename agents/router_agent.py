"""
Router (intent classification) agent. Decides whether an incoming query should go to:
  - "rag" -> Unstructured RAG Agent (documents: PDF/PPT/DOCX)
  - "sql" -> Structured RDBMS Agent (tabular/analytics/Excel-derived data)

Uses a small, deterministic prompt with structured output so routing is fast and cheap
even on a small local model.
"""
from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config.settings import settings
from database.db_manager import get_schema_description
from llm.llm_factory import get_chat_llm

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    route: Literal["rag", "sql"] = Field(description="Target sector agent for this query")
    reasoning: str = Field(description="One short sentence explaining the decision")


ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a multi-agent system with two sectors:

1. "rag" — Unstructured Document Sector. Choose this for questions about the CONTENT,
   MEANING, or SUMMARY of text documents (PDF reports, PowerPoint decks, Word docs):
   policies, narratives, explanations, quotes, summaries, "what does X say about Y".

2. "sql" — Structured Data Sector. Choose this for questions requiring TABULAR /
   RELATIONAL / NUMERIC analysis over ingested spreadsheet data: aggregations, counts,
   trends, comparisons, filters, rankings, or any request for a chart/plot/graph.

Known SQL tables currently available (empty means no tables ingested yet):
{schema}

Respond with ONLY a JSON object matching this schema, no other text:
{{"route": "rag" | "sql", "reasoning": "<short reason>"}}
"""


def route_query(query: str) -> RouteDecision:
    llm = get_chat_llm(temperature=0.0)
    schema = get_schema_description()

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT.format(schema=schema)),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Be forgiving of models that wrap JSON in markdown fences.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        return RouteDecision(**data)
    except Exception as e:  # noqa: BLE001
        logger.warning("Router JSON parse failed (%s); falling back to keyword heuristic.", e)
        return _keyword_fallback(query)


_SQL_KEYWORDS = (
    "chart", "plot", "graph", "average", "sum", "total", "count", "trend",
    "compare", "top", "revenue", "sales", "region", "table", "rows", "column",
)


def _keyword_fallback(query: str) -> RouteDecision:
    q_lower = query.lower()
    if any(kw in q_lower for kw in _SQL_KEYWORDS):
        return RouteDecision(route="sql", reasoning="Keyword heuristic fallback matched SQL/analytics terms.")
    return RouteDecision(route="rag", reasoning="Keyword heuristic fallback defaulted to document search.")
