# RAG + RDBMS Multi-Agent System

A production-grade, model-agnostic multi-agent system built on **LangGraph** that routes
user queries between:

- **Unstructured RAG Agent** — semantic search over PDFs/PPTX/DOCX via ChromaDB
- **Structured RDBMS Agent** — Text-to-SQL + auto-generated matplotlib/seaborn charts over
  Excel/CSV data loaded into SQLite/PostgreSQL/MySQL

...with a **memory/cache layer** in front of the router so repeated or near-duplicate
queries are answered from history instead of re-running retrieval or SQL.

The LLM backend is fully pluggable via `.env`: **Ollama** (local dev), **Groq** (fast
API dev/test), or **vLLM/OpenAI-compatible Qwen2.5-72B-Instruct** (production, multi-GPU).

```
                         ┌─────────────────────┐
   user query ─────────▶ │  Memory / Cache      │──▶ cache hit ──▶ return cached answer
                         │  Layer (semantic)    │
                         └──────────┬───────────┘
                                    │ cache miss
                                    ▼
                         ┌─────────────────────┐
                         │  Router Agent        │
                         │  (intent classifier) │
                         └───┬─────────────┬────┘
                     doc/text│             │tabular/analytics
                             ▼             ▼
                  ┌────────────────┐ ┌───────────────────────┐
                  │ Unstructured   │ │ Structured RDBMS Agent │
                  │ RAG Agent      │ │ (Text-to-SQL + plots)  │
                  │ (ChromaDB)     │ │ (SQLite/Postgres/MySQL)│
                  └────────────────┘ └───────────────────────┘
                             │             │
                             └──────┬──────┘
                                    ▼
                          write result to memory
                                    │
                                    ▼
                              final answer
```

## 1. Project Structure

```
rag_rdbms_multiagent/
├── .env.example              # config template (LLM provider, DB, paths)
├── requirements.txt
├── main.py                   # CLI entrypoint / interactive loop
├── config/
│   └── settings.py           # loads & validates .env into a Settings object
├── llm/
│   └── llm_factory.py        # plug-and-play LLM interface (Ollama/Groq/Qwen-vLLM)
├── memory/
│   └── memory_manager.py     # LangGraph checkpointer + semantic query cache
├── vectorstore/
│   └── chroma_store.py       # ChromaDB client, embeddings, retriever
├── ingestion/
│   ├── document_ingest.py    # PDF/PPTX/DOCX -> chunks -> ChromaDB
│   └── excel_to_sql.py       # Excel/CSV -> SQL tables (schema inference + load)
├── database/
│   └── db_manager.py         # SQLAlchemy engine, safe SQL execution, schema introspection
├── agents/
│   ├── router_agent.py       # intent classification (RAG vs SQL)
│   ├── rag_agent.py          # retrieval + synthesis over ChromaDB
│   └── sql_agent.py          # NL -> SQL, execution, and NL -> chart generation
├── graph/
│   └── orchestrator.py       # LangGraph StateGraph wiring memory -> router -> agents
├── utils/
│   └── plotting.py           # safe matplotlib/seaborn execution + base64/PNG output
└── data/
    ├── documents/            # drop PDFs/PPTX/DOCX here for ingestion
    ├── excel/                # drop Excel/CSV files here for ingestion
    ├── chroma_db/            # persisted vector store
    └── sqlite/               # default SQLite database file
```

## 2. Quickstart (Local Dev — Ollama)

```bash
cd rag_rdbms_multiagent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: LLM_PROVIDER=ollama, OLLAMA_MODEL=llama3

# 1. pull a local model
ollama pull llama3
ollama pull nomic-embed-text     # embeddings model used by ChromaDB

# 2. ingest sample data
python -m ingestion.document_ingest --path data/documents
python -m ingestion.excel_to_sql --path data/excel

# 3. run
python main.py
```

## 3. Quickstart (Dev/Test — Groq API)

```bash
cp .env.example .env
# edit .env:
#   LLM_PROVIDER=groq
#   GROQ_API_KEY=gsk_xxx
#   GROQ_MODEL=llama-3.1-70b-versatile   # or mixtral-8x7b-32768

pip install -r requirements.txt
python -m ingestion.document_ingest --path data/documents
python -m ingestion.excel_to_sql --path data/excel
python main.py
```

Embeddings still run locally (via `sentence-transformers` or `OLLAMA_EMBED_MODEL`) even
when the chat LLM is Groq, since Groq does not serve an embeddings endpoint.

## 4. Swapping to Production — Qwen2.5-72B-Instruct (multi-GPU)

The production path assumes you're serving Qwen behind an **OpenAI-compatible** endpoint
(e.g., via `vllm` or `text-generation-inference`), so no code changes are required —
only `.env`.

1. Launch the server on your GPU cluster, e.g. with vLLM:

   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-72B-Instruct \
       --tensor-parallel-size 4 \
       --port 8000
   ```

2. Point the app at it:

   ```bash
   # .env
   LLM_PROVIDER=qwen_vllm
   QWEN_BASE_URL=http://<gpu-host>:8000/v1
   QWEN_MODEL=Qwen/Qwen2.5-72B-Instruct
   QWEN_API_KEY=not-needed-or-your-gateway-key
   ```

3. Restart `main.py` (or your API server). `llm/llm_factory.py` reads `LLM_PROVIDER`
   and instantiates `ChatOpenAI(base_url=QWEN_BASE_URL, ...)`, which talks to vLLM's
   OpenAI-compatible `/v1/chat/completions` route — same LangChain interface as
   Ollama/Groq, so agents and graph logic are untouched.

For embeddings at production scale, either keep a local `sentence-transformers` model
on a CPU node, or point `EMBEDDING_PROVIDER=openai_compatible` at a served embedding
model (e.g., `bge-large-en-v1.5` via vLLM/TEI) — see `.env.example`.

## 5. How Memory/Cache Avoids Redundant Work

`memory/memory_manager.py` maintains two things:

- **LangGraph `SqliteSaver` checkpointer** — full conversation state (turn history,
  routing decisions, agent outputs) keyed by `thread_id`, so multi-turn context persists.
- **Semantic query cache** — a small ChromaDB collection of `(query_embedding -> answer)`.
  Before routing, `orchestrator.py` embeds the incoming query and searches this cache. If
  cosine similarity is above `CACHE_SIMILARITY_THRESHOLD` (default `0.92`), the cached
  answer/citations are returned immediately and the router/agents are skipped entirely.
  Every fresh agent answer is written back into this cache.

## 6. Safety Notes on SQL Execution

`database/db_manager.py`'s `run_sql()`:
- only permits `SELECT` statements by default (`ALLOW_WRITE_SQL=false` in `.env`),
- runs each query through `sqlglot` to reject multi-statement/DDL/DML injection attempts,
- enforces a `SQL_ROW_LIMIT` (default 500) via automatic `LIMIT` injection,
- executes inside a read-only transaction where the driver supports it.

## 7. Testing the Two Sectors End-to-End

```
> What does the Q3 marketing report say about churn drivers?      # -> RAG agent
> Show me a bar chart of monthly revenue by region for 2025        # -> SQL agent + plot
> Compare average order value between the East and West regions    # -> SQL agent + plot
> Summarize the onboarding guide in three bullet points             # -> RAG agent
```

Generated charts are saved to `data/outputs/*.png` and also returned as base64 in the
agent response payload for API/UI consumption.

## 8. Measuring RAG Quality with RAGAS

The evaluation runner measures retrieval and answer quality separately using a
question set with manually supplied reference answers:

```bash
python -m pip install -r requirements.txt
python -m evaluation.ragas_eval
```

The starter benchmark in `evaluation/ragas_dataset.json` contains twelve questions
balanced across the three sample documents currently in `data/documents`: the AgentAI
survey, the agentic AI article, and the cyber-risk presentation. Each entry identifies
its expected `source` and includes a `question` and a manually supplied `ground_truth`.
To evaluate a different dataset or choose an output location:

```bash
python -m evaluation.ragas_eval --dataset path/to/questions.json --output data/outputs/ragas_eval_report.json
```

The report contains per-question answers, citations, and scores, plus dataset means.
The four scores are faithfulness (grounding), answer relevancy, context precision
(retrieved relevance), and context recall (retrieved coverage). Scores range from 0
to 1; RAGAS scores are LLM-judge estimates, not exact factual accuracy. Interpret
them alongside the per-question answers and manually validated reference answers.
The evaluation invokes the RAG agent directly and does not use routing or semantic
answer cache.
