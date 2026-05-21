"""model-builder tune — run hyperparameter optimization on latest run."""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def tune_command(
    trials: int = typer.Option(20, "--trials", "-n", help="Number of Optuna trials"),
    cv: int = typer.Option(3, "--cv", help="Cross-validation folds"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column name"),
    run: Optional[str] = typer.Option(None, "--run", help="Run name (default: latest)"),
) -> None:
    """Run hyperparameter tuning on the best-ranked plugin."""
    asyncio.run(_tune_async(trials, cv, target, run))


async def _tune_async(trials: int, cv: int, target: Optional[str],
                      run_name: Optional[str]) -> None:
    from ...core.store import ProjectStore
    from ...plugins.registry import PluginRegistry
    from ...plugins.core_protocol import CoreContext

    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()

    run = await store.get_run_by_name(run_name) if run_name else await store.get_latest_run()
    if run is None:
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(code=1)

    artifacts_dir = project_dir / "runs" / run.name / "artifacts"
    if not (artifacts_dir / "ranking.json").exists():
        console.print("[red]ranking.json not found. Run rank_algos step first.[/red]")
        raise typer.Exit(code=1)

    registry = PluginRegistry()
    registry.discover()
    ctx = CoreContext(
        run_dir=project_dir / "runs" / run.name,
        artifacts_dir=artifacts_dir,
        logs_dir=project_dir / "runs" / run.name / "logs",
        run_id=run.id,
        registry=registry,
        node_config={"n_trials": trials, "cv": cv, "target_col": target},
    )

    console.print(f"[bold]Tuning hyperparameters[/bold] — {trials} trials, cv={cv}")
    tuner = registry.get_core_plugin("core.automl_tuner")

    try:
        out = tuner.run(ctx)
        result = json.loads(out.read_text())

        table = Table(title=f"Best params — {result['plugin']}")
        table.add_column("Parameter", style="bold")
        table.add_column("Value")
        for k, v in result["best_params"].items():
            table.add_row(k, str(v))

        console.print(table)
        console.print(f"\n[green]Best score ({result['task']}):[/green] {result['best_value']:.4f}")
        console.print(f"Results saved: [dim]{out}[/dim]")
    except Exception as e:
        console.print(f"[red]Tuning failed:[/red] {e}")
        raise typer.Exit(code=1)

