import asyncio
import json
import os
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from model_builder.core.store import ProjectStore
from model_builder.core.models import NodeState

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def runs_command() -> None:
    """List all runs with status and key metrics."""
    asyncio.run(_runs_async())


async def _runs_async() -> None:
    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()
    runs = await store.list_runs()
    if not runs:
        console.print("[dim]No runs yet.[/dim]")
        return

    table = Table(title="Runs")
    table.add_column("Run", style="bold")
    table.add_column("Created")
    table.add_column("Nodes done")
    table.add_column("Best metric")

    for run in runs:
        node_runs = await store.get_all_node_runs(run.id)
        done = sum(1 for nr in node_runs if nr.state in (
            NodeState.SUCCEEDED, NodeState.APPROVED, NodeState.SKIPPED
        ))
        total = len(node_runs)

        eval_path = project_dir / "runs" / run.name / "eval_report.json"
        best = "—"
        if eval_path.exists():
            report = json.loads(eval_path.read_text())
            metrics = report.get("metrics", {})
            if metrics:
                key, val = next(iter(metrics.items()))
                best = f"{key}={val:.3f}"

        table.add_row(
            run.name,
            run.created_at.strftime("%Y-%m-%d %H:%M"),
            f"{done}/{total}",
            best,
        )

    console.print(table)


def compare_command(
    run_a: str = typer.Argument(..., help="First run name (e.g. run_001)"),
    run_b: str = typer.Argument(..., help="Second run name (e.g. run_002)"),
) -> None:
    """Diff eval metrics between two runs."""
    asyncio.run(_compare_async(run_a, run_b))


async def _compare_async(run_a: str, run_b: str) -> None:
    project_dir = _project_dir()

    def load_report(name: str) -> dict:
        path = project_dir / "runs" / name / "eval_report.json"
        if not path.exists():
            console.print(f"[red]eval_report.json not found for {name}[/red]")
            raise typer.Exit(code=1)
        return json.loads(path.read_text())

    report_a = load_report(run_a)
    report_b = load_report(run_b)

    metrics_a = report_a.get("metrics", {})
    metrics_b = report_b.get("metrics", {})
    all_keys = sorted(set(metrics_a) | set(metrics_b))

    table = Table(title=f"Comparing {run_a} vs {run_b}")
    table.add_column("Metric", style="bold")
    table.add_column(run_a, justify="right")
    table.add_column(run_b, justify="right")
    table.add_column("Δ", justify="right")

    for key in all_keys:
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        if val_a is not None and val_b is not None:
            delta = val_b - val_a
            color = "green" if delta > 0 else "red" if delta < 0 else "dim"
            table.add_row(key, f"{val_a:.4f}", f"{val_b:.4f}",
                          f"[{color}]{delta:+.4f}[/{color}]")
        else:
            table.add_row(
                key,
                f"{val_a:.4f}" if val_a else "—",
                f"{val_b:.4f}" if val_b else "—",
                "—",
            )

    console.print(table)
