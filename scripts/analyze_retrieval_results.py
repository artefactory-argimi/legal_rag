#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0.0",
#     "rich>=13.0.0",
#     "sqlitedict>=2.0.0",
# ]
# ///
"""Analyze retrieval results to diagnose low recall issues.

This script compares qrels (ground truth) with run files (retrieval results)
to identify potential causes of low recall.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlitedict import SqliteDict


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    qrels_df: pd.DataFrame
    run_df: pd.DataFrame
    query_overlap: set[str]
    doc_overlap: set[str]
    hits_per_query: dict[str, dict] = field(default_factory=dict)
    indexed_chunk_ids: set[str] = field(default_factory=set)
    indexed_parent_ids: set[str] = field(default_factory=set)


def load_index_documents(index_path: Path) -> tuple[set[str], set[str]]:
    """Load indexed document IDs from PLAID index SQLite.

    Args:
        index_path: Path to the index directory containing the sqlite file.

    Returns:
        Tuple of (chunk_ids, parent_doc_ids).
    """
    sqlite_path = index_path / "documents_ids_to_plaid_ids.sqlite"
    if not sqlite_path.exists():
        return set(), set()

    chunk_ids: set[str] = set()
    parent_ids: set[str] = set()

    with SqliteDict(str(sqlite_path), flag="r") as db:
        for chunk_id in db.keys():
            chunk_ids.add(chunk_id)
            parts = chunk_id.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                parent_ids.add(parts[0])
            else:
                parent_ids.add(chunk_id)

    return chunk_ids, parent_ids


def load_qrels(path: Path) -> pd.DataFrame:
    """Load qrels file in TREC format: query_id 0 doc_id relevance."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str},
    )


def load_run(path: Path) -> pd.DataFrame:
    """Load run file in TREC format: query_id Q0 doc_id rank score run_name."""
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["query_id", "q0", "doc_id", "rank", "score", "run_name"],
        dtype={"query_id": str, "doc_id": str},
    )


def analyze_results(
    qrels_path: Path, run_path: Path, index_path: Path | None = None
) -> AnalysisResult:
    """Perform comprehensive analysis of retrieval results."""
    qrels_df = load_qrels(qrels_path)
    run_df = load_run(run_path)

    # Load indexed documents if path provided
    indexed_chunk_ids: set[str] = set()
    indexed_parent_ids: set[str] = set()
    if index_path:
        indexed_chunk_ids, indexed_parent_ids = load_index_documents(index_path)

    qrels_queries = set(qrels_df["query_id"].unique())
    run_queries = set(run_df["query_id"].unique())
    query_overlap = qrels_queries & run_queries

    qrels_docs = set(qrels_df["doc_id"].unique())
    run_docs = set(run_df["doc_id"].unique())
    doc_overlap = qrels_docs & run_docs

    hits_per_query: dict[str, dict] = {}
    for query_id in query_overlap:
        relevant_docs = set(
            qrels_df[qrels_df["query_id"] == query_id]["doc_id"].tolist()
        )
        retrieved_docs = run_df[run_df["query_id"] == query_id].sort_values("rank")

        hits_at_k: dict[int, bool] = {}
        first_hit_rank: int | None = None

        for k in [1, 5, 10, 20, 50, 100]:
            top_k_docs = set(retrieved_docs.head(k)["doc_id"].tolist())
            hits_at_k[k] = bool(relevant_docs & top_k_docs)
            if hits_at_k[k] and first_hit_rank is None:
                for _, row in retrieved_docs.iterrows():
                    if row["doc_id"] in relevant_docs:
                        first_hit_rank = int(row["rank"])
                        break

        hits_per_query[query_id] = {
            "relevant_docs": relevant_docs,
            "num_retrieved": len(retrieved_docs),
            "hits_at_k": hits_at_k,
            "first_hit_rank": first_hit_rank,
            "top_retrieved": retrieved_docs.head(5)["doc_id"].tolist(),
            "top_scores": retrieved_docs.head(5)["score"].tolist(),
        }

    return AnalysisResult(
        qrels_df=qrels_df,
        run_df=run_df,
        query_overlap=query_overlap,
        doc_overlap=doc_overlap,
        hits_per_query=hits_per_query,
        indexed_chunk_ids=indexed_chunk_ids,
        indexed_parent_ids=indexed_parent_ids,
    )


def print_report(result: AnalysisResult, console: Console) -> None:
    """Print comprehensive diagnostic report using rich."""
    console.print()
    console.rule("[bold blue]RETRIEVAL DIAGNOSTICS REPORT[/bold blue]", style="blue")
    console.print()

    # Basic Statistics Table
    stats_table = Table(
        title="Basic Statistics", show_header=True, header_style="bold cyan"
    )
    stats_table.add_column("Metric", style="white")
    stats_table.add_column("Qrels", justify="right", style="green")
    stats_table.add_column("Run", justify="right", style="yellow")

    stats_table.add_row(
        "Total entries",
        str(len(result.qrels_df)),
        str(len(result.run_df)),
    )
    stats_table.add_row(
        "Unique queries",
        str(result.qrels_df["query_id"].nunique()),
        str(result.run_df["query_id"].nunique()),
    )
    stats_table.add_row(
        "Unique documents",
        str(result.qrels_df["doc_id"].nunique()),
        str(result.run_df["doc_id"].nunique()),
    )
    console.print(stats_table)
    console.print()

    # Index Analysis (if available)
    qrels_docs = set(result.qrels_df["doc_id"].unique())
    if result.indexed_parent_ids:
        index_table = Table(
            title="Index Analysis", show_header=True, header_style="bold magenta"
        )
        index_table.add_column("Metric", style="white")
        index_table.add_column("Count", justify="right")
        index_table.add_column("Status", justify="center")

        index_table.add_row(
            "Total indexed chunks",
            str(len(result.indexed_chunk_ids)),
            "[dim]-[/dim]",
        )
        index_table.add_row(
            "Unique indexed documents",
            str(len(result.indexed_parent_ids)),
            "[dim]-[/dim]",
        )

        # Check how many qrels docs are in the index
        qrels_in_index = qrels_docs & result.indexed_parent_ids
        qrels_not_in_index = qrels_docs - result.indexed_parent_ids
        pct_in_index = len(qrels_in_index) / len(qrels_docs) * 100 if qrels_docs else 0

        status = "[green]✓[/green]" if pct_in_index > 90 else "[red]✗[/red]"
        index_table.add_row(
            "Relevant docs IN index",
            f"{len(qrels_in_index)} ({pct_in_index:.1f}%)",
            status,
        )
        index_table.add_row(
            "Relevant docs NOT in index",
            f"{len(qrels_not_in_index)} ({100 - pct_in_index:.1f}%)",
            "[red]✗[/red]" if qrels_not_in_index else "[green]✓[/green]",
        )

        console.print(index_table)
        console.print()

        # Show sample missing docs
        if qrels_not_in_index:
            missing_sample = list(qrels_not_in_index)[:5]
            console.print(
                Panel(
                    "\n".join(missing_sample),
                    title="[red]Sample Relevant Docs NOT in Index[/red]",
                    border_style="red",
                )
            )
            console.print()

    # Query & Document Overlap
    qrels_queries = set(result.qrels_df["query_id"].unique())
    run_queries = set(result.run_df["query_id"].unique())
    run_docs = set(result.run_df["doc_id"].unique())

    overlap_table = Table(
        title="ID Overlap Analysis", show_header=True, header_style="bold cyan"
    )
    overlap_table.add_column("Category", style="white")
    overlap_table.add_column("Count", justify="right")
    overlap_table.add_column("Status", justify="center")

    query_both = len(result.query_overlap)
    query_only_qrels = len(qrels_queries - run_queries)
    query_only_run = len(run_queries - qrels_queries)

    overlap_table.add_row(
        "Queries in both",
        str(query_both),
        "[green]✓[/green]" if query_both > 0 else "[red]✗[/red]",
    )
    overlap_table.add_row(
        "Queries only in qrels",
        str(query_only_qrels),
        "[yellow]![/yellow]" if query_only_qrels > 0 else "[green]✓[/green]",
    )
    overlap_table.add_row("Queries only in run", str(query_only_run), "[dim]-[/dim]")

    doc_both = len(result.doc_overlap)
    doc_only_qrels = len(qrels_docs - run_docs)
    doc_only_run = len(run_docs - qrels_docs)
    missing_pct = (doc_only_qrels / len(qrels_docs)) * 100 if qrels_docs else 0

    status = "[green]✓[/green]" if doc_both > len(qrels_docs) * 0.5 else "[red]✗[/red]"
    overlap_table.add_row("Docs in both", str(doc_both), status)
    overlap_table.add_row(
        "Relevant docs NOT retrieved",
        f"{doc_only_qrels} ({missing_pct:.1f}%)",
        "[red]✗[/red]" if missing_pct > 50 else "[yellow]![/yellow]",
    )
    overlap_table.add_row(
        "Retrieved docs NOT in qrels", str(doc_only_run), "[dim]-[/dim]"
    )

    console.print(overlap_table)
    console.print()

    # Document ID Format Comparison
    format_table = Table(
        title="Document ID Format Comparison",
        show_header=True,
        header_style="bold cyan",
    )
    format_table.add_column("Source", style="white")
    format_table.add_column("Sample IDs", style="dim")

    qrels_sample = list(qrels_docs)[:3]
    run_sample = list(run_docs)[:3]
    format_table.add_row("Qrels", ", ".join(qrels_sample))
    format_table.add_row("Run", ", ".join(run_sample))
    console.print(format_table)
    console.print()

    # Recall Metrics
    recall_table = Table(
        title="Recall Metrics", show_header=True, header_style="bold cyan"
    )
    recall_table.add_column("K", justify="right", style="white")
    recall_table.add_column("Hits", justify="right")
    recall_table.add_column("Total", justify="right")
    recall_table.add_column("Recall", justify="right")
    recall_table.add_column("Visual", style="blue")

    for k in [1, 5, 10, 20, 50, 100]:
        hits = sum(
            1 for q in result.hits_per_query.values() if q["hits_at_k"].get(k, False)
        )
        total = len(result.hits_per_query)
        recall = hits / total if total > 0 else 0
        bar_len = int(recall * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        color = "green" if recall > 0.5 else "yellow" if recall > 0.1 else "red"
        recall_table.add_row(
            str(k),
            str(hits),
            str(total),
            f"[{color}]{recall:.4f}[/{color}]",
            bar,
        )

    console.print(recall_table)
    console.print()

    # First Hit Rank Distribution
    first_hits = [
        q["first_hit_rank"]
        for q in result.hits_per_query.values()
        if q["first_hit_rank"] is not None
    ]

    if first_hits:
        rank_table = Table(
            title="First Hit Rank Distribution",
            show_header=True,
            header_style="bold cyan",
        )
        rank_table.add_column("Metric", style="white")
        rank_table.add_column("Value", justify="right", style="green")

        rank_table.add_row("Queries with hits", str(len(first_hits)))
        rank_table.add_row(
            "Mean first hit rank", f"{sum(first_hits) / len(first_hits):.1f}"
        )
        rank_table.add_row("Min rank", str(min(first_hits)))
        rank_table.add_row("Max rank", str(max(first_hits)))
        console.print(rank_table)
    else:
        console.print(
            Panel(
                "[red bold]No hits found in any query![/red bold]",
                title="First Hit Analysis",
            )
        )

    console.print()

    # Score Distribution
    score_table = Table(
        title="Score Distribution", show_header=True, header_style="bold cyan"
    )
    score_table.add_column("Metric", style="white")
    score_table.add_column("Value", justify="right", style="yellow")

    score_table.add_row("Min", f"{result.run_df['score'].min():.4f}")
    score_table.add_row("Max", f"{result.run_df['score'].max():.4f}")
    score_table.add_row("Mean", f"{result.run_df['score'].mean():.4f}")
    score_table.add_row("Std", f"{result.run_df['score'].std():.4f}")
    console.print(score_table)
    console.print()

    # Sample Miss Analysis
    console.rule("[bold yellow]Sample Miss Analysis[/bold yellow]", style="yellow")
    console.print()

    miss_count = 0
    for qid, data in result.hits_per_query.items():
        if not data["hits_at_k"].get(100, False) and miss_count < 5:
            miss_table = Table(show_header=False, box=None, padding=(0, 1))
            miss_table.add_column("Label", style="bold white")
            miss_table.add_column("Value", style="dim")

            relevant_doc = str(list(data["relevant_docs"])[0])
            miss_table.add_row("Query ID", qid)
            miss_table.add_row("Relevant doc", relevant_doc)
            miss_table.add_row("Top retrieved", ", ".join(data["top_retrieved"][:3]))

            # Check if relevant doc is in index
            if result.indexed_parent_ids:
                in_index = relevant_doc in result.indexed_parent_ids
                miss_table.add_row(
                    "In index?",
                    "[green]Yes[/green]" if in_index else "[red]No[/red]",
                )

            console.print(
                Panel(
                    miss_table,
                    title=f"[red]Miss #{miss_count + 1}[/red]",
                    border_style="red",
                )
            )
            miss_count += 1

    console.print()

    # Issues Summary
    console.rule("[bold red]Potential Issues[/bold red]", style="red")
    console.print()

    issues: list[Text] = []

    if query_only_qrels > 0:
        issues.append(
            Text(
                f"⚠ {query_only_qrels} queries in qrels missing from run",
                style="yellow",
            )
        )

    if missing_pct > 50:
        issues.append(
            Text(
                f"✗ {missing_pct:.1f}% of relevant docs never appear in results",
                style="red bold",
            )
        )

    if doc_both == 0:
        issues.append(
            Text(
                "✗ CRITICAL: Zero overlap between qrels docs and retrieved docs!",
                style="red bold",
            )
        )

    # Check if relevant docs are missing from index
    if result.indexed_parent_ids:
        qrels_not_in_index = qrels_docs - result.indexed_parent_ids
        if qrels_not_in_index:
            pct_missing = len(qrels_not_in_index) / len(qrels_docs) * 100
            issues.append(
                Text(
                    f"✗ {len(qrels_not_in_index)} ({pct_missing:.1f}%) relevant docs NOT IN INDEX",
                    style="red bold",
                )
            )

    # Check for potential ID format mismatch
    qrels_has_hyphen = (
        sum(1 for d in qrels_docs if "-" in d) / len(qrels_docs) if qrels_docs else 0
    )
    run_has_hyphen = (
        sum(1 for d in run_docs if "-" in d) / len(run_docs) if run_docs else 0
    )

    if abs(qrels_has_hyphen - run_has_hyphen) > 0.5:
        issues.append(
            Text(
                "⚠ Document ID format mismatch detected (hyphen patterns differ)",
                style="yellow",
            )
        )

    if issues:
        for issue in issues:
            console.print(f"  {issue}")
    else:
        console.print(
            "  [green]No obvious issues detected. Low recall may be due to model quality.[/green]"
        )

    console.print()
    console.rule(style="blue")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze retrieval results")
    parser.add_argument("qrels", type=Path, help="Path to qrels file")
    parser.add_argument("run", type=Path, help="Path to run file")
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Path to PLAID index directory (for index analysis)",
    )
    args = parser.parse_args()

    console = Console()
    result = analyze_results(args.qrels, args.run, args.index)
    print_report(result, console)


if __name__ == "__main__":
    main()
