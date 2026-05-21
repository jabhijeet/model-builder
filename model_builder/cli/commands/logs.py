import os
from pathlib import Path
import typer
from rich.console import Console

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def logs_command(
    node_id: str = typer.Argument(..., help="Node ID to show logs for"),
    lines: int = typer.Option(50, "--lines", "-n", help="Last N lines to show"),
) -> None:
    """Show logs for a pipeline node."""
    import asyncio
    asyncio.run(_logs_async(node_id, lines))


async def _logs_async(node_id: str, lines: int) -> None:
    from ...core.store import ProjectStore
    project_dir = _project_dir()
    store = ProjectStore(project_dir)
    await store.init()
    run = await store.get_latest_run()
    if run is None:
        console.print("[red]No runs found.[/red]")
        raise typer.Exit(code=1)

    log_file = project_dir / "runs" / run.name / "logs" / f"{node_id}.log"
    if not log_file.exists():
        console.print(f"[dim]No log file for node '{node_id}' in {run.name}.[/dim]")
        return

    all_lines = log_file.read_text().splitlines()
    shown = all_lines[-lines:] if len(all_lines) > lines else all_lines
    console.print(f"[dim]-- {log_file} (last {len(shown)} lines) --[/dim]")
    for line in shown:
        console.print(line)
