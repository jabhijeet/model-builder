import asyncio
import os
from pathlib import Path
import typer
from rich.console import Console
from model_builder.core.store import ProjectStore
from model_builder.core.dag import DAG
from model_builder.core.gates import GateManager
from model_builder.core.models import NodeRun, NodeState

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def approve_command(node_id: str = typer.Argument(..., help="GATE node ID to approve")) -> None:
    """Approve a GATE node to unblock downstream nodes."""
    asyncio.run(_approve_async(node_id))


async def _approve_async(node_id: str) -> None:
    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()
    dag = DAG.from_file(project_dir / "pipeline.yaml")
    run = await store.get_latest_run()
    if run is None:
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(code=1)
    node_def = dag.nodes.get(node_id)
    if node_def is None:
        console.print(f"[red]Node '{node_id}' not found in pipeline.[/red]")
        raise typer.Exit(code=1)
    mgr = GateManager(store)
    try:
        await mgr.approve(run.id, node_id, node_def)
        console.print(f"[green]Approved:[/green] {node_id}")
        console.print("  Resume: [bold]model-builder run[/bold]")
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def skip_command(node_id: str = typer.Argument(..., help="Node ID to skip")) -> None:
    """Skip a node and unblock downstream nodes."""
    asyncio.run(_skip_async(node_id))


async def _skip_async(node_id: str) -> None:
    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()
    run = await store.get_latest_run()
    if run is None:
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(code=1)
    mgr = GateManager(store)
    await mgr.skip(run.id, node_id)
    console.print(f"[yellow]Skipped:[/yellow] {node_id}")


def retry_command(node_id: str = typer.Argument(..., help="Failed node ID to retry")) -> None:
    """Reset a failed node to pending so the next run re-executes it."""
    asyncio.run(_retry_async(node_id))


async def _retry_async(node_id: str) -> None:
    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()
    run = await store.get_latest_run()
    if run is None:
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(code=1)
    nr = await store.get_node_run(run.id, node_id)
    if nr is None or nr.state != NodeState.FAILED:
        state = nr.state if nr else "not found"
        console.print(f"[red]Node '{node_id}' is not in failed state (state: {state})[/red]")
        raise typer.Exit(code=1)
    nr.state = NodeState.PENDING
    nr.error = None
    nr.started_at = None
    nr.finished_at = None
    await store.upsert_node_run(nr)
    console.print(f"[cyan]Reset to pending:[/cyan] {node_id}")
    console.print("  Resume: [bold]model-builder run[/bold]")
