"""
Structured Data & Analytics Agent:
  1. Translates the natural-language question into SQL against the ingested schema.
  2. Safely executes the SQL via database.db_manager.run_sql.
  3. If the question asks for (or would benefit from) a visual, generates matplotlib/
     seaborn code against the result DataFrame and renders it via utils.plotting.
  4. Synthesizes a short natural-language summary of the result.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from database.db_manager import UnsafeSQLError, get_schema_description, run_sql
from llm.llm_factory import get_chat_llm
from utils.plotting import UnsafePlotCodeError, render_plot

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

SQL_GEN_SYSTEM_PROMPT = """You are an expert {dialect} SQL analyst.
Given the schema below and a natural-language question, write ONE syntactically correct
{dialect} SELECT query that answers it. Only use tables/columns that exist in the schema.
Never write INSERT/UPDATE/DELETE/DROP/ALTER statements.

SCHEMA:
{schema}

Respond with ONLY a JSON object, no markdown fences, no extra text:
{{"sql": "<the SELECT statement>", "needs_plot": true|false, "plot_rationale": "<why or why not>"}}

Set "needs_plot" to true if the question asks for or clearly implies a chart, plot,
graph, trend, comparison, or distribution.
"""

PLOT_GEN_SYSTEM_PROMPT = """You generate a single matplotlib/seaborn code snippet.
A pandas DataFrame named `df` is already available with the SQL query results.
Available names: df, pd, plt, sns. Do NOT import anything, do NOT call plt.show(),
do NOT call plt.savefig() (the caller handles saving). Build the chart onto the
current matplotlib figure/axes (e.g., via sns.barplot(...) or df.plot(...)).
Choose the chart type (bar/line/scatter/box/etc.) appropriate to the data and question.
Set a clear title, axis labels, and rotate x tick labels if they'd overlap.

Respond with ONLY the Python code, no markdown fences, no explanation.
"""

SUMMARY_SYSTEM_PROMPT = """You are a data analyst. Given a user's question, the SQL query
that was run, and a preview of the resulting data, write a concise (2-4 sentence)
natural-language summary of the answer. Mention concrete numbers from the data.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json") or text.lower().startswith("python"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
    return text.strip()


def _generate_sql(query: str) -> dict:
    llm = get_chat_llm(temperature=0.0)
    schema = get_schema_description()
    messages = [
        SystemMessage(
            content=SQL_GEN_SYSTEM_PROMPT.format(dialect=settings.sql_dialect, schema=schema)
        ),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    raw = _strip_fences(response.content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # last-resort: try to pull a SELECT statement out with regex
        match = re.search(r"(SELECT .*?)(;|$)", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return {"sql": match.group(1).strip(), "needs_plot": False, "plot_rationale": ""}
        raise ValueError(f"Could not parse SQL generation response: {raw[:300]}")


def _generate_plot_code(query: str, df_preview: str, columns: list[str]) -> str:
    llm = get_chat_llm(temperature=0.1)
    messages = [
        SystemMessage(content=PLOT_GEN_SYSTEM_PROMPT),
        HumanMessage(
            content=f"User question: {query}\nDataFrame columns: {columns}\n"
                    f"DataFrame preview:\n{df_preview}"
        ),
    ]
    response = llm.invoke(messages)
    return _strip_fences(response.content)


def _summarize(query: str, sql: str, df_preview: str) -> str:
    llm = get_chat_llm(temperature=0.2)
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {query}\nSQL: {sql}\nData preview:\n{df_preview}"),
    ]
    return llm.invoke(messages).content


def run_sql_agent(query: str) -> dict:
    try:
        gen = _generate_sql(query)
        sql = gen["sql"]
    except Exception as e:  # noqa: BLE001
        return {
            "answer": f"I couldn't translate that into a valid SQL query: {e}",
            "sql": None,
            "sector": "sql",
        }

    try:
        df = run_sql(sql)
    except UnsafeSQLError as e:
        return {"answer": f"Generated SQL was rejected for safety reasons: {e}", "sql": sql, "sector": "sql"}
    except Exception as e:  # noqa: BLE001
        return {"answer": f"SQL execution failed: {e}", "sql": sql, "sector": "sql"}

    df_preview = df.head(20).to_string(index=False)
    summary = _summarize(query, sql, df_preview)

    result = {
        "answer": summary,
        "sql": sql,
        "row_count": len(df),
        "columns": list(df.columns),
        "table_preview": df.head(20).to_dict(orient="records"),
        "sector": "sql",
        "plot": None,
    }

    if gen.get("needs_plot") and not df.empty:
        try:
            plot_code = _generate_plot_code(query, df_preview, list(df.columns))
            plot_result = render_plot(df, plot_code, title_hint=query[:40])
            result["plot"] = plot_result
        except (UnsafePlotCodeError, RuntimeError) as e:
            logger.warning("Plot generation/execution failed: %s", e)
            result["plot_error"] = str(e)

    return result
