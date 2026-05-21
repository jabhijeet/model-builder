import asyncio
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.markdown import Markdown

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def deploy_command(
    run: Optional[str] = typer.Option(None, "--run", help="Run name (default: latest)"),
    regenerate: bool = typer.Option(False, "--regenerate", help="Re-run deploy advisor"),
) -> None:
    """Print deployment instructions. Use --regenerate to rebuild DEPLOY.md."""
    asyncio.run(_deploy_async(run, regenerate))


async def _deploy_async(run_name: Optional[str], regenerate: bool) -> None:
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

    deploy_path = project_dir / "runs" / run.name / "DEPLOY.md"

    if regenerate or not deploy_path.exists():
        registry = PluginRegistry()
        registry.discover()
        ctx = CoreContext(
            run_dir=project_dir / "runs" / run.name,
            artifacts_dir=project_dir / "runs" / run.name / "artifacts",
            logs_dir=project_dir / "runs" / run.name / "logs",
            run_id=run.id,
            registry=registry,
            node_config={},
        )
        advisor = registry.get_core_plugin("core.deploy_advisor")
        advisor.run(ctx)
        console.print(f"[green]Generated:[/green] {deploy_path}")

    if deploy_path.exists():
        console.print(Markdown(deploy_path.read_text()))
    else:
        console.print("[red]DEPLOY.md not found. Run a complete pipeline first.[/red]")
        raise typer.Exit(code=1)
