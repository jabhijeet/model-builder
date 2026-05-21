import os
import webbrowser
from pathlib import Path
import typer
from rich.console import Console

console = Console()


def _project_dir() -> Path:
    env = os.environ.get("MB_PROJECT_DIR")
    return Path(env) if env else Path.cwd()


def ui_command(
    port: int = typer.Option(8765, "--port", help="Port to serve on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
) -> None:
    """Start the web UI for the current project."""
    import uvicorn
    from ...web.app import create_app

    project_dir = _project_dir()
    if not (project_dir / "pipeline.yaml").exists():
        console.print("[red]Error:[/red] pipeline.yaml not found. Run aimodelground init first.")
        raise typer.Exit(code=1)

    app = create_app(project_dir)
    url = f"http://localhost:{port}"
    console.print(f"[green]Starting UI:[/green] {url}")
    console.print(f"  Project: [bold]{project_dir.name}[/bold]")
    console.print("  Press Ctrl+C to stop\n")

    if not no_browser:
        webbrowser.open(url)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

