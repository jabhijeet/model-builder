import asyncio
from pathlib import Path
import pytest
from typer.testing import CliRunner
from model_builder.cli.main import app
from model_builder.core.store import ProjectStore
from model_builder.core.models import NodeState

runner = CliRunner()


@pytest.fixture
def project_with_simple_pipeline(tmp_path: Path) -> Path:
    project_dir = tmp_path / "test-proj"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / "data" / "processed").mkdir()
    (project_dir / "runs").mkdir()
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("default_metric: f1\n")
    (project_dir / "pipeline.yaml").write_text("""
nodes:
  - id: gate_only
    type: gate
    message: "Initial gate"
""")
    return project_dir


def test_run_creates_new_run(project_with_simple_pipeline: Path):
    result = runner.invoke(
        app, ["run"],
        catch_exceptions=False,
        env={"MB_PROJECT_DIR": str(project_with_simple_pipeline)},
    )
    assert result.exit_code == 0
    assert "run_001" in result.output


def test_run_pauses_at_gate(project_with_simple_pipeline: Path):
    runner.invoke(app, ["run"],
                  env={"MB_PROJECT_DIR": str(project_with_simple_pipeline)})

    store = ProjectStore(project_with_simple_pipeline)

    async def check():
        await store.init()
        run = await store.get_latest_run()
        nr = await store.get_node_run(run.id, "gate_only")
        assert nr.state == NodeState.AWAITING_HUMAN

    asyncio.run(check())


def test_run_second_time_creates_run_002(project_with_simple_pipeline: Path):
    env = {"MB_PROJECT_DIR": str(project_with_simple_pipeline)}
    runner.invoke(app, ["run"], env=env)
    runner.invoke(app, ["run"], env=env)

    store = ProjectStore(project_with_simple_pipeline)

    async def check():
        await store.init()
        runs = await store.list_runs()
        assert len(runs) == 2
        assert runs[1].name == "run_002"

    asyncio.run(check())
