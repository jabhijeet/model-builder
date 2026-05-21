"""model-builder models — view and update trained models."""
import asyncio
import json
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console = Console()
app = typer.Typer(name="models", help="View and update trained models")


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


@app.command("list")
def models_list() -> None:
    """List all trained models across all runs."""
    from ...model_registry import scan_models
    project_dir = _project_dir()
    entries = scan_models(project_dir)
    if not entries:
        console.print("[dim]No trained models found. Complete a training run first.[/dim]")
        return

    table = Table(title=f"Trained Models — {project_dir.name}")
    table.add_column("ID", style="bold")
    table.add_column("Plugin")
    table.add_column("Task")
    table.add_column("Features", justify="right")
    table.add_column("Artifact")

    for e in entries:
        table.add_row(
            e.id,
            e.plugin_name,
            e.task,
            str(len(e.features)),
            e.artifact_path.name,
        )
    console.print(table)


@app.command("info")
def models_info(
    model_id: str = typer.Argument(..., help="Model ID (run_name/node_id or run_name/plugin_name)"),
) -> None:
    """Show details for a specific model."""
    from ...model_registry import scan_models
    project_dir = _project_dir()
    entries = scan_models(project_dir)

    # Match by id or run/plugin
    parts = model_id.split("/")
    run_name = parts[0]
    plugin_or_node = parts[1] if len(parts) > 1 else ""

    match = next(
        (e for e in entries if e.run_name == run_name and
         (e.plugin_name == plugin_or_node or e.node_id == plugin_or_node)),
        None
    )
    if not match:
        console.print(f"[red]Model '{model_id}' not found.[/red]")
        console.print("Run [bold]model-builder models list[/bold] to see all models.")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]{match.plugin_name}[/bold]  ({match.run_name})")
    console.print(f"  Task: {match.task}")
    if match.features:
        console.print(f"  Features ({len(match.features)}): {', '.join(match.features[:8])}"
                      + ("..." if len(match.features) > 8 else ""))
    console.print(f"  Artifact: [dim]{match.artifact_path}[/dim]\n")

    for k, v in match.metadata.items():
        if k not in ("model", "state_dict", "X_cols", "task"):
            console.print(f"  {k}: {v}")


@app.command("update")
def models_update(
    model_id: Optional[str] = typer.Argument(
        None, help="Model ID (run_name/plugin_name). Omit to choose interactively."
    ),
    data: Optional[str] = typer.Option(
        None, "--data", "-d",
        help="Path to new data file (CSV/parquet). Default: data/raw/* in project."
    ),
    target: Optional[str] = typer.Option(None, "--target", help="Target column name"),
    n_estimators: int = typer.Option(50, "--n-estimators", help="New trees/rounds to add"),
    run_name: Optional[str] = typer.Option(None, "--run", help="Source run name"),
) -> None:
    """Update an existing model with new data."""
    asyncio.run(_update_async(model_id, data, target, n_estimators, run_name))


async def _update_async(
    model_id: Optional[str], data_path: Optional[str],
    target: Optional[str], n_estimators: int, run_name_override: Optional[str]
) -> None:
    from ...model_registry import scan_models, get_model
    from ...core.store import ProjectStore
    from ...plugins.registry import PluginRegistry
    from ...plugins.core_protocol import CoreContext

    project_dir = _project_dir()
    entries = scan_models(project_dir)

    if not entries:
        console.print("[red]No trained models found.[/red]")
        raise typer.Exit(code=1)

    # Select model
    if model_id is None:
        console.print("\n[bold]Available models:[/bold]")
        for i, e in enumerate(entries):
            console.print(f"  [{i+1}] {e.id}  ({e.plugin_name}, task={e.task})")
        choice = Prompt.ask("\nSelect model number", default="1")
        try:
            entry = entries[int(choice) - 1]
        except (ValueError, IndexError):
            console.print("[red]Invalid choice.[/red]")
            raise typer.Exit(code=1)
    else:
        parts = model_id.split("/")
        source_run = run_name_override or parts[0]
        plugin_name = parts[1] if len(parts) > 1 else parts[0]
        entry = get_model(project_dir, source_run, plugin_name)
        if entry is None:
            console.print(f"[red]Model '{model_id}' not found.[/red]")
            raise typer.Exit(code=1)

    console.print(f"\n[bold]Updating:[/bold] {entry.plugin_name} from {entry.run_name}")

    # Prepare new data
    if data_path:
        new_data_path = Path(data_path)
    else:
        raw_dir = project_dir / "data" / "raw"
        candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.parquet"))
        if not candidates:
            console.print("[red]No data files found in data/raw/. Use --data to specify a file.[/red]")
            raise typer.Exit(code=1)
        new_data_path = candidates[0]
        console.print(f"  Using data: [dim]{new_data_path}[/dim]")

    # Import into a temp run
    import pandas as pd
    from ...connectors.file import FileConnector

    store = ProjectStore(project_dir)
    await store.init()
    run_name = await store.next_run_name()
    run_id = await store.create_run(run_name, from_node_id=f"update_{entry.plugin_name}")
    run_dir = project_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir()
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()

    # Copy new data to artifacts
    if str(new_data_path).endswith(".parquet"):
        import shutil
        shutil.copy2(new_data_path, artifacts_dir / "merged.parquet")
    else:
        df = pd.read_csv(new_data_path)
        df.to_parquet(artifacts_dir / "merged.parquet", index=False)

    registry = PluginRegistry()
    registry.discover()

    ctx = CoreContext(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        run_id=run_id,
        registry=registry,
        node_config={
            "plugin_name": entry.plugin_name,
            "source_run": entry.run_name,
            "target_col": target,
            "n_estimators": n_estimators,
            "n_estimators_new": n_estimators,
            "learning_rate": 0.05,
        },
    )

    update_plugin = registry.get_core_plugin("core.model_update")
    try:
        out = update_plugin.run(ctx)
        receipt = json.loads(out.read_text())
        console.print(f"\n[green]Model updated successfully[/green] → {run_name}")
        console.print(f"  Updated artifact: [dim]{receipt['updated_artifact_path']}[/dim]")
        console.print(f"  Run: [bold]model-builder status[/bold] to see the update run")
    except Exception as e:
        console.print(f"[red]Update failed:[/red] {e}")
        raise typer.Exit(code=1)
