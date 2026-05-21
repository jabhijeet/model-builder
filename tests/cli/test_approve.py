import asyncio
from pathlib import Path
import pytest
from typer.testing import CliRunner
from model_builder.cli.main import app
from model_builder.core.store import ProjectStore
from model_builder.core.models import NodeRun, NodeState

runner = CliRunner()


@pytest.fixture
def project_with_gate(tmp_path: Path) -> Path:
    project_dir = tmp_path / "proj"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("")
    (project_dir / "pipeline.yaml").write_text("""
nodes:
  - id: review
    type: gate
    message: "Check"
""")

    async def seed():
        store = ProjectStore(project_dir)
        await store.init()
        run_id = await store.create_run("run_001")
        await store.upsert_node_run(
            NodeRun(run_id=run_id, node_id="review", state=NodeState.AWAITING_HUMAN)
        )

    asyncio.run(seed())
    return project_dir


def test_approve_gate(project_with_gate: Path):
    result = runner.invoke(app, ["approve", "review"],
                           env={"MB_PROJECT_DIR": str(project_with_gate)})
    assert result.exit_code == 0
    assert "approved" in result.output.lower()

    async def check():
        store = ProjectStore(project_with_gate)
        await store.init()
        run = await store.get_latest_run()
        nr = await store.get_node_run(run.id, "review")
        assert nr.state == NodeState.APPROVED

    asyncio.run(check())


def test_skip_node(project_with_gate: Path):
    result = runner.invoke(app, ["skip", "review"],
                           env={"MB_PROJECT_DIR": str(project_with_gate)})
    assert result.exit_code == 0

    async def check():
        store = ProjectStore(project_with_gate)
        await store.init()
        run = await store.get_latest_run()
        nr = await store.get_node_run(run.id, "review")
        assert nr.state == NodeState.SKIPPED

    asyncio.run(check())


def test_retry_resets_failed_node(tmp_path: Path):
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
            NodeRun(run_id=run_id, node_id="ingest", state=NodeState.FAILED, error="connection refused")
        )

    asyncio.run(seed())

    result = runner.invoke(app, ["retry", "ingest"],
                           env={"MB_PROJECT_DIR": str(project_dir)})
    assert result.exit_code == 0

    async def check():
        store = ProjectStore(project_dir)
        await store.init()
        run = await store.get_latest_run()
        nr = await store.get_node_run(run.id, "ingest")
        assert nr.state == NodeState.PENDING

    asyncio.run(check())
