"""
Interactive CLI for the RAG + RDBMS multi-agent system.

    python main.py                       # interactive loop, default thread
    python main.py --thread my-session   # named conversation thread (memory persists)
    python main.py --query "..."         # single non-interactive query
"""
from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from config.settings import settings
from graph.orchestrator import ask

logging.basicConfig(level=settings.log_level)
console = Console()


def render_result(result: dict) -> None:
    sector = result.get("sector", "cache")
    header = f"[bold cyan]sector: {sector}[/bold cyan]"
    if result.get("_cache_similarity"):
        header += f"  [dim](served from cache, similarity={result['_cache_similarity']})[/dim]"

    console.print(Panel(Markdown(result.get("answer", "(no answer)")), title=header))

    if result.get("sql"):
        console.print(f"[dim]SQL:[/dim] {result['sql']}")
    if result.get("plot"):
        console.print(f"[green]Chart saved to:[/green] {result['plot']['file_path']}")
    if result.get("sources"):
        console.print("[dim]Sources:[/dim]")
        for s in result["sources"]:
            loc = f" (loc {s['location']})" if s.get("location") else ""
            console.print(f"  - {s['source']}{loc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG + RDBMS multi-agent system")
    parser.add_argument("--thread", default=settings.default_thread_id, help="Conversation thread id")
    parser.add_argument("--query", default=None, help="Run a single query non-interactively")
    args = parser.parse_args()

    if args.query:
        result = ask(args.query, thread_id=args.thread)
        render_result(result)
        return

    console.print(Panel(
        "RAG + RDBMS Multi-Agent System\n"
        "Ask about your ingested documents or Excel/SQL data. Type 'exit' to quit.",
        title="Welcome",
    ))

    while True:
        try:
            query = console.input("[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not query.strip():
            continue
        if query.strip().lower() in ("exit", "quit"):
            console.print("Goodbye.")
            break

        try:
            result = ask(query, thread_id=args.thread)
            render_result(result)
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
