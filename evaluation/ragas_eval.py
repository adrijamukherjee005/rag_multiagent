"""Run RAGAS metrics against the document RAG pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path(__file__).with_name("ragas_dataset.json")
DEFAULT_OUTPUT = Path(settings.plots_output_dir) / "ragas_eval_report.json"
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def load_dataset(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load question/reference/source examples and reject incomplete rows."""
    with path.open(encoding="utf-8") as dataset_file:
        rows = json.load(dataset_file)

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Evaluation dataset must be a non-empty JSON list: {path}")

    validated: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset entry {index} must be a JSON object.")
        question = row.get("question")
        ground_truth = row.get("ground_truth")
        source = row.get("source")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Dataset entry {index} has no non-empty 'question'.")
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError(f"Dataset entry {index} has no non-empty 'ground_truth'.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Dataset entry {index} has no non-empty 'source'.")
        validated.append({
            "question": question.strip(),
            "ground_truth": ground_truth.strip(),
            "source": source.strip(),
        })

    return validated[:limit] if limit is not None else validated


def collect_predictions(
    examples: list[dict[str, str]], top_k: int
) -> list[dict[str, Any]]:
    """Retrieve once per question and use the same documents for answer and scoring."""
    from agents.rag_agent import answer_from_retrieved_context
    from vectorstore.hybrid_retriever import retrieve

    samples = []
    for example in examples:
        question = example["question"]
        retrieved = retrieve(question, k=top_k)
        result = answer_from_retrieved_context(question, retrieved)
        samples.append({
            "user_input": question,
            "reference": example["ground_truth"],
            "expected_source": example["source"],
            "response": str(result["answer"]),
            "retrieved_contexts": [doc.page_content for doc, _ in retrieved],
            "sources": result.get("sources", []),
        })
    return samples


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def evaluate_dataset(
    samples: list[dict[str, Any]], output_path: Path
) -> dict[str, Any]:
    """Score generated RAG samples with RAGAS and save a machine-readable report."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from ragas.run_config import RunConfig

    from llm.llm_factory import get_chat_llm, get_embeddings

    ragas_llm = LangchainLLMWrapper(get_chat_llm())
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=metrics,
        run_config=RunConfig(timeout=600, max_workers=1),
        raise_exceptions=True,
        show_progress=True,
    )
    scored_rows = result.to_pandas()

    means = {
        metric: _as_float(scored_rows[metric].mean())
        for metric in METRIC_NAMES
        if metric in scored_rows
    }
    rows = []
    for index, sample in enumerate(samples):
        row_scores = {
            metric: _as_float(scored_rows.iloc[index][metric])
            for metric in METRIC_NAMES
            if metric in scored_rows
        }
        rows.append({
            "question": sample["user_input"],
            "expected_source": sample["expected_source"],
            "expected_source_found": any(
                source.get("source") == sample["expected_source"]
                for source in sample["sources"]
            ),
            "ground_truth": sample["reference"],
            "answer": sample["response"],
            "sources": sample["sources"],
            "scores": row_scores,
        })

    source_hits = sum(row["expected_source_found"] for row in rows)
    report = {
        "sample_count": len(rows),
        "metrics_mean": means,
        "expected_source_hits": source_hits,
        "expected_source_hit_rate": source_hits / len(rows) if rows else None,
        "metric_interpretation": {
            "faithfulness": "Whether answer claims are supported by retrieved context.",
            "answer_relevancy": "Whether the answer addresses the question.",
            "context_precision": "Whether retrieved context is relevant to the reference answer.",
            "context_recall": "Whether retrieved context contains information needed by the reference answer.",
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate document RAG answers and retrieval with RAGAS."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="JSON file containing question, ground_truth, and expected source fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save per-question and aggregate metrics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.rag_top_k,
        help="Number of retrieved documents per question.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally evaluate only the first N dataset rows.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    examples = load_dataset(args.dataset, limit=args.limit)
    samples = collect_predictions(examples, top_k=args.top_k)
    report = evaluate_dataset(samples, args.output)

    print(f"\nRAGAS results ({report['sample_count']} questions)")
    for metric, score in report["metrics_mean"].items():
        score_text = "n/a" if score is None else f"{score:.3f}"
        print(f"  {metric}: {score_text}")
    print(f"Detailed report: {args.output.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    main()
