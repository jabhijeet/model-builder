import asyncio
from pathlib import Path
import pytest
from typer.testing import CliRunner
from model_builder.cli.main import app
from model_builder.core.store import ProjectStore
from model_builder.core.models import NodeRun, NodeState

runner = CliRunner()


@pytest.fixture
def project_with_log(tmp_path: Path) -> Path:
    project_dir = tmp_path / "proj"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("")
    (project_dir / "pipeline.yaml").write_text(
        "nodes:\n  - id: ingest\n    type: task\n    plugin: connectors.file\n"
    )

    async def seed():
        store = ProjectStore(project_dir)
        await store.init()
        run_id = await store.create_run("run_001")
        await store.upsert_node_run(
            NodeRun(run_id=run_id, node_id="ingest", state=NodeState.SUCCEEDED)
        )

    asyncio.run(seed())

    log_dir = project_dir / "runs" / "run_001" / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "ingest.log").write_text("Reading data.csv\nLoaded 500 rows\nDone.")
    return project_dir


def test_logs_shows_content(project_with_log: Path):
    result = runner.invoke(app, ["logs", "ingest"],
                           env={"MB_PROJECT_DIR": str(project_with_log)})
    assert result.exit_code == 0
    assert "Loaded 500 rows" in result.output


def test_logs_missing_node_graceful(project_with_log: Path):
    result = runner.invoke(app, ["logs", "nonexistent"],
                           env={"MB_PROJECT_DIR": str(project_with_log)})
    assert result.exit_code == 0
    assert "No log file" in result.output


def test_logs_no_runs(tmp_path: Path):
    project_dir = tmp_path / "empty"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("")
    (project_dir / "pipeline.yaml").write_text("nodes:\n  - id: a\n    type: task\n")

    async def seed():
        store = ProjectStore(project_dir)
        await store.init()

    asyncio.run(seed())
    result = runner.invoke(app, ["logs", "a"],
                           env={"MB_PROJECT_DIR": str(project_dir)})
    assert result.exit_code != 0


def test_cli_help_shows_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["init", "run", "status", "approve", "skip", "retry",
                "runs", "compare", "ui", "logs", "export", "deploy"]:
        assert cmd in result.output
