"""
SQLAlchemy engine + safe SQL execution helpers for the Structured RDBMS Agent.

Safety measures in `run_sql`:
  1. Only SELECT statements are permitted unless ALLOW_WRITE_SQL=true.
  2. sqlglot parses the statement and rejects multi-statement input, DDL, and DML
     (INSERT/UPDATE/DELETE/DROP/ALTER/etc.) when writes are disabled.
  3. A LIMIT clause is injected automatically if the query doesn't already have one
     and the dialect supports LIMIT, capping result size at SQL_ROW_LIMIT.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd
import sqlglot
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from config.settings import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

_DISALLOWED_STATEMENT_TYPES = {
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "merge", "grant", "revoke", "attach", "pragma",
}


class UnsafeSQLError(Exception):
    pass


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(settings.sqlalchemy_url(), pool_pre_ping=True)


def get_schema_description() -> str:
    """Human/LLM-readable description of all tables and columns, for prompting."""
    engine = get_engine()
    inspector = inspect(engine)
    lines = []
    for table_name in inspector.get_table_names():
        cols = inspector.get_columns(table_name)
        col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
        lines.append(f"TABLE {table_name}: {col_desc}")
    if not lines:
        return "No tables found. Run ingestion.excel_to_sql to load data first."
    return "\n".join(lines)


def _validate_sql(sql: str) -> None:
    dialect = "sqlite" if settings.sql_dialect == "sqlite" else settings.sql_dialect
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as e:  # noqa: BLE001
        raise UnsafeSQLError(f"Could not parse SQL: {e}") from e

    if statements is None or len(statements) != 1:
        raise UnsafeSQLError("Only a single SQL statement is permitted per query.")

    stmt = statements[0]
    if stmt is None:
        raise UnsafeSQLError("Empty or unparseable SQL statement.")

    stmt_type = type(stmt).__name__.lower()

    if not settings.allow_write_sql:
        if stmt_type != "select":
            raise UnsafeSQLError(
                f"Only SELECT statements are permitted (got: {stmt_type}). "
                "Set ALLOW_WRITE_SQL=true in .env to allow writes."
            )
        sql_lower = sql.lower()
        for kw in _DISALLOWED_STATEMENT_TYPES:
            if kw in sql_lower.split():
                raise UnsafeSQLError(f"Disallowed keyword detected: {kw}")


def _inject_limit(sql: str) -> str:
    sql_stripped = sql.strip().rstrip(";")
    if "limit" in sql_stripped.lower():
        return sql_stripped
    return f"{sql_stripped} LIMIT {settings.sql_row_limit}"


def run_sql(sql: str, enforce_limit: bool = True) -> pd.DataFrame:
    """Validates and executes a SQL statement, returning results as a DataFrame."""
    _validate_sql(sql)
    final_sql = _inject_limit(sql) if (enforce_limit and not settings.allow_write_sql) else sql

    logger.info("Executing SQL: %s", final_sql)
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(final_sql))
        if result.returns_rows:
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        else:
            conn.commit()
            df = pd.DataFrame({"rows_affected": [result.rowcount]})
    return df


def list_tables() -> list[str]:
    return inspect(get_engine()).get_table_names()
