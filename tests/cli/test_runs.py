import asyncio
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from model_builder.cli.main import app
from model_builder.core.store import ProjectStore

runner = CliRunner()


@pytest.fixture
def project_with_two_runs(tmp_path: Path) -> Path:
    project_dir = tmp_path / "proj"
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / ".modelbuilder").mkdir()
    (project_dir / ".modelbuilder" / "config.yaml").write_text("")
    (project_dir / "pipeline.yaml").write_text("nodes:\n  - id: a\n    type: task\n")

    run1_dir = project_dir / "runs" / "run_001"
    run1_dir.mkdir(parents=True)
    (run1_dir / "eval_report.json").write_text(json.dumps({
        "best_plugin": "random_forest",
        "metrics": {"f1": 0.82, "accuracy": 0.85}
    }))

    run2_dir = project_dir / "runs" / "run_002"
    run2_dir.mkdir(parents=True)
    (run2_dir / "eval_report.json").write_text(json.dumps({
        "best_plugin": "xgboost",
        "metrics": {"f1": 0.89, "accuracy": 0.91}
    }))

    async def seed():
        store = ProjectStore(project_dir)
        await store.init()
        for name in ["run_001", "run_002"]:
            await store.create_run(name)

    asyncio.run(seed())
    return project_dir


def test_runs_lists_all(project_with_two_runs: Path):
    result = runner.invoke(app, ["runs"],
                           env={"MB_PROJECT_DIR": str(project_with_two_runs)})
    assert result.exit_code == 0
    assert "run_001" in result.output
    assert "run_002" in result.output


def test_compare_shows_metrics(project_with_two_runs: Path):
    result = runner.invoke(
        app, ["compare", "run_001", "run_002"],
        env={"MB_PROJECT_DIR": str(project_with_two_runs)},
    )
    assert result.exit_code == 0
    assert "f1" in result.output
    assert "0.82" in result.output
    assert "0.89" in result.output


def test_compare_missing_run(project_with_two_runs: Path):
    result = runner.invoke(
        app, ["compare", "run_001", "run_999"],
        env={"MB_PROJECT_DIR": str(project_with_two_runs)},
    )
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "run_999" in result.output
