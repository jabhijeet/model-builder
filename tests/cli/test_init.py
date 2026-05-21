from pathlib import Path
from typer.testing import CliRunner
from model_builder.cli.main import app

runner = CliRunner()


def test_init_creates_project_folder(tmp_path: Path):
    result = runner.invoke(app, ["init", "my-model"], catch_exceptions=False,
                           env={"MB_BASE_DIR": str(tmp_path)})
    assert result.exit_code == 0
    project_dir = tmp_path / "my-model"
    assert project_dir.exists()
    assert (project_dir / "pipeline.yaml").exists()
    assert (project_dir / "data" / "raw").exists()
    assert (project_dir / "data" / "processed").exists()
    assert (project_dir / ".modelbuilder" / "config.yaml").exists()


def test_init_scaffold_pipeline_yaml(tmp_path: Path):
    runner.invoke(app, ["init", "test-proj"], catch_exceptions=False,
                  env={"MB_BASE_DIR": str(tmp_path)})
    pipeline = (tmp_path / "test-proj" / "pipeline.yaml").read_text()
    assert "nodes:" in pipeline
    assert "ingest_files" in pipeline


def test_init_rejects_existing_project(tmp_path: Path):
    runner.invoke(app, ["init", "my-model"], env={"MB_BASE_DIR": str(tmp_path)})
    result = runner.invoke(app, ["init", "my-model"], env={"MB_BASE_DIR": str(tmp_path)})
    assert result.exit_code != 0
    assert "already exists" in result.output
