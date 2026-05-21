"""model-builder features — manage the project feature store."""
import asyncio
import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(name="features", help="Manage the project feature store")


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


@app.command("list")
def features_list() -> None:
    """List all feature sets in the store."""
    asyncio.run(_list_async())


async def _list_async() -> None:
    from ...feature_store.store import FeatureStore
    store = FeatureStore(_project_dir())
    await store.init()
    sets = await store.list()
    if not sets:
        console.print("[dim]No feature sets saved yet.[/dim]")
        return
    table = Table(title="Feature Store")
    table.add_column("Name", style="bold")
    table.add_column("Version", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Columns", justify="right")
    table.add_column("Created")
    table.add_column("Tags")
    for m in sets:
        table.add_row(
            m.name, str(m.version), str(m.row_count), str(m.column_count),
            m.created_at.strftime("%Y-%m-%d %H:%M"),
            ", ".join(m.tags) or "—",
        )
    console.print(table)


@app.command("info")
def features_info(
    name: str = typer.Argument(..., help="Feature set name"),
    version: Optional[int] = typer.Option(None, "--version", "-v"),
) -> None:
    """Show details for a feature set."""
    asyncio.run(_info_async(name, version))


async def _info_async(name: str, version: Optional[int]) -> None:
    from ...feature_store.store import FeatureStore
    store = FeatureStore(_project_dir())
    await store.init()
    try:
        meta = await store.get(name, version)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]{meta.name}[/bold]  v{meta.version}")
    console.print(f"  Rows: {meta.row_count}  |  Columns: {meta.column_count}")
    console.print(f"  Created: {meta.created_at.strftime('%Y-%m-%d %H:%M')}")
    if meta.run_id:
        console.print(f"  Run ID: {meta.run_id}")
    if meta.tags:
        console.print(f"  Tags: {', '.join(meta.tags)}")
    console.print(f"  Path: [dim]{meta.data_path}[/dim]\n")

    table = Table(show_header=True, box=None)
    table.add_column("Column")
    table.add_column("Type")
    for col, dtype in meta.columns.items():
        table.add_row(col, dtype)
    console.print(table)


@app.command("versions")
def features_versions(
    name: str = typer.Argument(..., help="Feature set name"),
) -> None:
    """List all versions of a feature set."""
    asyncio.run(_versions_async(name))


async def _versions_async(name: str) -> None:
    from ...feature_store.store import FeatureStore
    store = FeatureStore(_project_dir())
    await store.init()
    versions = await store.list_versions(name)
    if not versions:
        console.print(f"[red]No feature set named '{name}'[/red]")
        raise typer.Exit(code=1)
    table = Table(title=f"Versions: {name}")
    table.add_column("Version", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Run ID")
    table.add_column("Created")
    for m in versions:
        table.add_row(
            str(m.version), str(m.row_count),
            str(m.run_id) if m.run_id else "—",
            m.created_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@app.command("delete")
def features_delete(
    name: str = typer.Argument(..., help="Feature set name"),
    version: Optional[int] = typer.Option(None, "--version", "-v",
                                           help="Specific version (default: all)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a feature set (or specific version)."""
    ver_str = f" v{version}" if version else " (all versions)"
    if not yes:
        confirm = typer.confirm(f"Delete feature set '{name}'{ver_str}?")
        if not confirm:
            raise typer.Abort()
    asyncio.run(_delete_async(name, version))


async def _delete_async(name: str, version: Optional[int]) -> None:
    from ...feature_store.store import FeatureStore
    store = FeatureStore(_project_dir())
    await store.init()
    try:
        count = await store.delete(name, version)
        console.print(f"[green]Deleted {count} version(s) of '{name}'[/green]")
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
