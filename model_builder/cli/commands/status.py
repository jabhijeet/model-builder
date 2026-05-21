import asyncio
import os
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from model_builder.core.store import ProjectStore
from model_builder.core.dag import DAG
from model_builder.core.models import NodeState

console = Console()

_STATE_ICON = {
    NodeState.PENDING: (".", "dim"),
    NodeState.READY: ("~", "cyan"),
    NodeState.RUNNING: ("*", "yellow"),
    NodeState.SUCCEEDED: ("+", "green"),
    NodeState.FAILED: ("!", "red"),
    NodeState.AWAITING_HUMAN: ("?", "yellow"),
    NodeState.APPROVED: ("+", "green"),
    NodeState.SKIPPED: ("-", "dim"),
}


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def status_command() -> None:
    """Show DAG node states for the latest run."""
    asyncio.run(_status_async())


async def _status_async() -> None:
    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()

    latest_run = await store.get_latest_run()
    if latest_run is None:
        console.print("[dim]No runs yet. Run [bold]aimodelground run[/bold] to start.[/dim]")
        return

    dag = DAG.from_file(project_dir / "pipeline.yaml")
    node_runs = {nr.node_id: nr for nr in await store.get_all_node_runs(latest_run.id)}

    done = sum(1 for nr in node_runs.values() if nr.state in (
        NodeState.SUCCEEDED, NodeState.APPROVED, NodeState.SKIPPED
    ))
    total = len(dag.nodes)

    console.print(f"\n[bold]Pipeline:[/bold] {project_dir.name}  "
                  f"[dim]{latest_run.name}  {done}/{total} nodes done[/dim]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    for node_id in dag.nodes:
        nr = node_runs.get(node_id)
        state = nr.state if nr else NodeState.PENDING
        icon, color = _STATE_ICON.get(state, ("?", "white"))
        hint = ""
        if state == NodeState.AWAITING_HUMAN:
            hint = f"  [dim]→ aimodelground approve {node_id}[/dim]"
        elif state == NodeState.FAILED and nr and nr.error:
            hint = f"  [dim red]{nr.error[:60]}[/dim red]"
        table.add_row(
            f"[{color}]{icon}[/{color}]",
            f"[bold]{node_id}[/bold]",
            f"[{color}]{state.value}[/{color}]{hint}",
        )
    console.print(table)
    console.print()

