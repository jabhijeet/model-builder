import os
from pathlib import Path
import typer
from rich.console import Console

console = Console()

_PIPELINE_TEMPLATE = """\
nodes:
  - id: ingest_files
    type: task
    plugin: connectors.file
    config:
      paths: ["data/raw/*"]

  - id: validate
    type: task
    plugin: validators.schema
    depends_on: [ingest_files]

  - id: review_data
    type: gate
    depends_on: [validate]
    message: "Review data profile before training"

  - id: profile
    type: task
    plugin: core.profile
    depends_on: [review_data]

  - id: rank_algos
    type: task
    plugin: core.automl_ranker
    depends_on: [profile]

  - id: select_algos
    type: gate
    depends_on: [rank_algos]
    message: "Select algorithms to train"

  - id: export_model
    type: task
    plugin: core.export
    depends_on: [select_algos]

  - id: gen_deploy_instructions
    type: task
    plugin: core.deploy_advisor
    depends_on: [export_model]
"""

_CONFIG_TEMPLATE = """\
default_metric: f1
export_format: onnx
"""


def _base_dir() -> Path:
    env = os.environ.get("MB_BASE_DIR")
    return Path(env) if env else Path.cwd()


def init_command(name: str = typer.Argument(..., help="Project name")) -> None:
    """Create a new aimodelground project."""
    project_dir = _base_dir() / name
    if project_dir.exists():
        console.print(f"[red]Error:[/red] Project '{name}' already exists at {project_dir}")
        raise typer.Exit(code=1)

    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / "data" / "processed").mkdir(parents=True)
    (project_dir / "runs").mkdir()
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / "pipeline.yaml").write_text(_PIPELINE_TEMPLATE)
    (project_dir / ".modelbuilder" / "config.yaml").write_text(_CONFIG_TEMPLATE)

    console.print(f"[green]OK[/green] Created project [bold]{name}[/bold] at {project_dir}")
    console.print("  Edit [bold]pipeline.yaml[/bold] to configure your pipeline.")
    console.print("  Then run: [bold]aimodelground run[/bold]")

