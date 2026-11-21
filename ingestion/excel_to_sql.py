"""
Loads Excel (.xlsx/.xls, all sheets) and CSV files into SQL tables, inferring column
types from pandas dtypes. Table name = <filename>__<sheet_name>, sanitized to be a
valid SQL identifier.

Usage:
    python -m ingestion.excel_to_sql --path data/excel
    python -m ingestion.excel_to_sql --path data/excel --file sales.xlsx --if-exists replace
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from config.settings import settings
from database.db_manager import get_engine

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def sanitize_identifier(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip())
    if re.match(r"^\d", name):
        name = f"t_{name}"
    return name.lower()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [sanitize_identifier(str(c)) for c in df.columns]
    return df


def load_excel(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl" if path.suffix == ".xlsx" else None)
    return {sanitize_identifier(name): clean_columns(df) for name, df in sheets.items()}


def load_csv(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    return {"data": clean_columns(df)}


def ingest_path(path: str, single_file: str | None = None, if_exists: str = "replace") -> list[str]:
    root = Path(path)
    if single_file:
        files = [root / single_file]
    elif root.is_file():
        files = [root]
    else:
        files = [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.warning("No supported spreadsheet/csv files found under %s", path)
        return []

    engine = get_engine()
    created_tables: list[str] = []

    for f in files:
        try:
            sheets = load_csv(f) if f.suffix.lower() == ".csv" else load_excel(f)
            for sheet_name, df in sheets.items():
                table_name = sanitize_identifier(f"{f.stem}__{sheet_name}") if len(sheets) > 1 or f.suffix.lower() != ".csv" and sheet_name != "data" else sanitize_identifier(f.stem)
                if f.suffix.lower() == ".csv":
                    table_name = sanitize_identifier(f.stem)

                df.to_sql(table_name, engine, if_exists=if_exists, index=False)
                created_tables.append(table_name)
                logger.info("Loaded %s (sheet=%s) -> table `%s` (%d rows, %d cols)",
                            f.name, sheet_name, table_name, len(df), len(df.columns))
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to ingest %s: %s", f, e)

    logger.info("SQL ingestion complete. Tables created/updated: %s", created_tables)
    return created_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Excel/CSV files into SQL tables")
    parser.add_argument("--path", default=settings.excel_dir, help="Directory or file to ingest")
    parser.add_argument("--file", default=None, help="Optional single filename within --path directory")
    parser.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"])
    args = parser.parse_args()
    ingest_path(args.path, args.file, args.if_exists)


if __name__ == "__main__":
    main()
