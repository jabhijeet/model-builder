import asyncio
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from model_builder.core.store import ProjectStore
from model_builder.core.dag import DAG
from model_builder.core.events import EventBus
from model_builder.core.scheduler import Scheduler
from model_builder.plugins.registry import PluginRegistry

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def run_command(
    from_node: Optional[str] = typer.Option(None, "--from", help="Re-run from this node"),
) -> None:
    """Start or resume DAG execution (creates a new run)."""
    asyncio.run(_run_async(from_node))


async def _run_async(from_node: Optional[str]) -> None:
    project_dir = _project_dir()
    pipeline_file = project_dir / "pipeline.yaml"
    if not pipeline_file.exists():
        console.print("[red]Error:[/red] pipeline.yaml not found. Run [bold]model-builder init[/bold] first.")
        raise typer.Exit(code=1)

    store = ProjectStore(project_dir)
    await store.init()

    dag = DAG.from_file(pipeline_file)
    registry = PluginRegistry()
    registry.discover()
    events = EventBus()

    run_name = await store.next_run_name()
    parent_run = await store.get_latest_run()
    parent_run_id = parent_run.id if parent_run and from_node else None
    run_id = await store.create_run(run_name, from_node_id=from_node, parent_run_id=parent_run_id)

    run_dir = project_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    if from_node and parent_run_id:
        await _reuse_upstream_artifacts(dag, from_node, parent_run_id, run_id, store)

    console.print(f"[bold]Starting {run_name}[/bold] {'(from ' + from_node + ')' if from_node else ''}")

    def on_event(run_id, node_id, event_type, payload):
        if event_type == "state_change":
            state = payload.get("state", "")
            icon = {"running": "*", "succeeded": "+", "failed": "!"}.get(state, ".")
            console.print(f"  {icon}  {node_id}  ->  {state}")
        elif event_type == "awaiting_human":
            console.print(f"\n[yellow]GATE:[/yellow] {node_id}")
            console.print(f"   {payload.get('message', '')}")
            console.print(f"   Run: [bold]model-builder approve {node_id}[/bold]\n")

    events.subscribe(on_event)

    scheduler = Scheduler(dag, store, registry, events, run_id, run_dir)
    await scheduler.run()
    console.print(f"\n[green]Run {run_name} paused or complete.[/green]")


async def _reuse_upstream_artifacts(
    dag: DAG, from_node: str, parent_run_id: int, new_run_id: int, store: ProjectStore
) -> None:
    upstream = dag.upstream_of(from_node)
    from model_builder.core.models import NodeRun, NodeState
    for nr in await store.get_all_node_runs(parent_run_id):
        if nr.node_id in upstream and nr.state.value in ("succeeded", "approved", "skipped"):
            reused = NodeRun(
                run_id=new_run_id,
                node_id=nr.node_id,
                state=NodeState(nr.state.value),
                started_at=nr.started_at,
                finished_at=nr.finished_at,
                output_path=nr.output_path,
            )
            await store.upsert_node_run(reused)
