import asyncio
import json
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def export_command(
    format: str = typer.Option("pickle", "--format", "-f", help="Export format: pickle, onnx"),
    run: Optional[str] = typer.Option(None, "--run", help="Run name (default: latest)"),
    plugin: Optional[str] = typer.Option(None, "--plugin", help="Plugin name to export"),
) -> None:
    """Re-export a trained model in the specified format."""
    asyncio.run(_export_async(format, run, plugin))


async def _export_async(fmt: str, run_name: Optional[str], plugin_name: Optional[str]) -> None:
    from ...core.store import ProjectStore
    from ...plugins.registry import PluginRegistry
    from ...plugins.core_protocol import CoreContext

    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()

    run = await store.get_run_by_name(run_name) if run_name else await store.get_latest_run()
    if run is None:
        console.print("[red]No run found.[/red]")
        raise typer.Exit(code=1)

    registry = PluginRegistry()
    registry.discover()

    artifacts_dir = project_dir / "runs" / run.name / "artifacts"
    logs_dir = project_dir / "runs" / run.name / "logs"
    ctx = CoreContext(
        run_dir=project_dir / "runs" / run.name,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        run_id=run.id,
        registry=registry,
        node_config={"format": fmt, "plugin": plugin_name},
    )

    export_plugin = registry.get_core_plugin("core.export")
    try:
        out = export_plugin.run(ctx)
        console.print(f"[green]Exported:[/green] {out}")
    except Exception as e:
        console.print(f"[red]Export failed:[/red] {e}")
        raise typer.Exit(code=1)
