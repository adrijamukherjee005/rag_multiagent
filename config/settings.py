"""
Centralized configuration. Loads .env once and exposes a typed `settings` object
used across every module. Nothing else in the codebase should call os.getenv directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


def _get(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


@dataclass
class Settings:
    # --- LLM ---
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "ollama"))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.1))
    llm_max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 2048))

    ollama_base_url: str = field(default_factory=lambda: _get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _get("OLLAMA_MODEL", "llama3"))
    ollama_embed_model: str = field(default_factory=lambda: _get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    groq_api_key: str | None = field(default_factory=lambda: _get("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _get("GROQ_MODEL", "llama-3.1-70b-versatile"))

    qwen_base_url: str = field(default_factory=lambda: _get("QWEN_BASE_URL", "http://localhost:8000/v1"))
    qwen_model: str = field(default_factory=lambda: _get("QWEN_MODEL", "Qwen/Qwen2.5-72B-Instruct"))
    qwen_api_key: str = field(default_factory=lambda: _get("QWEN_API_KEY", "not-needed"))

    # --- Embeddings ---
    embedding_provider: str = field(default_factory=lambda: _get("EMBEDDING_PROVIDER", "sentence_transformers"))
    sentence_transformers_model: str = field(
        default_factory=lambda: _get("SENTENCE_TRANSFORMERS_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embed_base_url: str = field(default_factory=lambda: _get("EMBED_BASE_URL", "http://localhost:8001/v1"))
    embed_model: str = field(default_factory=lambda: _get("EMBED_MODEL", "BAAI/bge-large-en-v1.5"))

    # --- Vector store ---
    chroma_persist_dir: str = field(default_factory=lambda: _get("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    chroma_collection: str = field(default_factory=lambda: _get("CHROMA_COLLECTION", "documents"))
    chroma_cache_collection: str = field(default_factory=lambda: _get("CHROMA_CACHE_COLLECTION", "query_cache"))
    rag_top_k: int = field(default_factory=lambda: _get_int("RAG_TOP_K", 5))
    cache_similarity_threshold: float = field(
        default_factory=lambda: _get_float("CACHE_SIMILARITY_THRESHOLD", 0.92)
    )

    # --- Hybrid retrieval (BM25 + vector) ---
    hybrid_search: bool = field(default_factory=lambda: _get_bool("HYBRID_SEARCH", True))
    bm25_weight: float = field(default_factory=lambda: _get_float("BM25_WEIGHT", 0.5))
    rrf_k: int = field(default_factory=lambda: _get_int("RRF_K", 60))
    candidate_pool_size: int = field(default_factory=lambda: _get_int("CANDIDATE_POOL_SIZE", 30))

    # --- Reranking ---
    rerank_enabled: bool = field(default_factory=lambda: _get_bool("RERANK_ENABLED", True))
    rerank_top_k: int = field(default_factory=lambda: _get_int("RERANK_TOP_K", 5))
    rerank_model: str = field(
        default_factory=lambda: _get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # --- SQL database ---
    sql_dialect: str = field(default_factory=lambda: _get("SQL_DIALECT", "sqlite"))
    sqlite_path: str = field(default_factory=lambda: _get("SQLITE_PATH", "./data/sqlite/app.db"))

    postgres_host: str = field(default_factory=lambda: _get("POSTGRES_HOST", "localhost"))
    postgres_port: str = field(default_factory=lambda: _get("POSTGRES_PORT", "5432"))
    postgres_db: str = field(default_factory=lambda: _get("POSTGRES_DB", "ragdb"))
    postgres_user: str = field(default_factory=lambda: _get("POSTGRES_USER", "postgres"))
    postgres_password: str = field(default_factory=lambda: _get("POSTGRES_PASSWORD", "postgres"))

    mysql_host: str = field(default_factory=lambda: _get("MYSQL_HOST", "localhost"))
    mysql_port: str = field(default_factory=lambda: _get("MYSQL_PORT", "3306"))
    mysql_db: str = field(default_factory=lambda: _get("MYSQL_DB", "ragdb"))
    mysql_user: str = field(default_factory=lambda: _get("MYSQL_USER", "root"))
    mysql_password: str = field(default_factory=lambda: _get("MYSQL_PASSWORD", "root"))

    allow_write_sql: bool = field(default_factory=lambda: _get_bool("ALLOW_WRITE_SQL", False))
    sql_row_limit: int = field(default_factory=lambda: _get_int("SQL_ROW_LIMIT", 500))

    # --- Memory ---
    checkpoint_db_path: str = field(default_factory=lambda: _get("CHECKPOINT_DB_PATH", "./data/sqlite/checkpoints.db"))
    chat_db_path: str = field(default_factory=lambda: _get("CHAT_DB_PATH", "./data/sqlite/chat.db"))
    default_thread_id: str = field(default_factory=lambda: _get("DEFAULT_THREAD_ID", "default-session"))

    # --- Ingestion ---
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 120))
    documents_dir: str = field(default_factory=lambda: _get("DOCUMENTS_DIR", "./data/documents"))
    excel_dir: str = field(default_factory=lambda: _get("EXCEL_DIR", "./data/excel"))

    # --- Outputs ---
    plots_output_dir: str = field(default_factory=lambda: _get("PLOTS_OUTPUT_DIR", "./data/outputs"))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))

    def sqlalchemy_url(self) -> str:
        if self.sql_dialect == "sqlite":
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{self.sqlite_path}"
        if self.sql_dialect == "postgresql":
            return (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        if self.sql_dialect == "mysql":
            return (
                f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
                f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            )
        raise ValueError(f"Unsupported SQL_DIALECT: {self.sql_dialect}")

    def ensure_dirs(self) -> None:
        for d in (
            self.chroma_persist_dir,
            self.documents_dir,
            self.excel_dir,
            self.plots_output_dir,
            Path(self.checkpoint_db_path).parent,
            Path(self.sqlite_path).parent,
        ):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
