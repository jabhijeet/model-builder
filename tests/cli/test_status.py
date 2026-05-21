import asyncio
from pathlib import Path
import pytest
from typer.testing import CliRunner
from model_builder.cli.main import app
from model_builder.core.store import ProjectStore
from model_builder.core.models import NodeRun, NodeState

runner = CliRunner()


@pytest.fixture
def project_with_run(tmp_path: Path) -> Path:
    project_dir = tmp_path / "proj"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / "data" / "processed").mkdir()
    (project_dir / "runs").mkdir()
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("")
    (project_dir / "pipeline.yaml").write_text("""
nodes:
  - id: ingest
    type: task
    plugin: connectors.file
  - id: review
    type: gate
    depends_on: [ingest]
    message: "Check data"
""")

    async def seed():
        store = ProjectStore(project_dir)
        await store.init()
        run_id = await store.create_run("run_001")
        await store.upsert_node_run(NodeRun(run_id=run_id, node_id="ingest", state=NodeState.SUCCEEDED))
        await store.upsert_node_run(NodeRun(run_id=run_id, node_id="review", state=NodeState.AWAITING_HUMAN))

    asyncio.run(seed())
    return project_dir


def test_status_shows_node_states(project_with_run: Path):
    result = runner.invoke(app, ["status"],
                           env={"MB_PROJECT_DIR": str(project_with_run)})
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "review" in result.output
    assert "succeeded" in result.output.lower() or "+" in result.output
    assert "awaiting" in result.output.lower() or "?" in result.output


def test_status_no_runs(tmp_path: Path):
    project_dir = tmp_path / "empty"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("")
    (project_dir / "pipeline.yaml").write_text("nodes:\n  - id: a\n    type: task\n")

    result = runner.invoke(app, ["status"],
                           env={"MB_PROJECT_DIR": str(project_dir)})
    assert result.exit_code == 0
    assert "No runs" in result.output
