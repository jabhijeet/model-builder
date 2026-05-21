"""Tests for model update — ModelRegistry, update() methods, ModelUpdatePlugin, CLI."""
import asyncio
import json
import pickle
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typer.testing import CliRunner
from model_builder.cli.main import app
from model_builder.model_registry import scan_models, get_model, ModelEntry
from model_builder.plugins.registry import PluginRegistry
from model_builder.plugins.core_protocol import CoreContext
from model_builder.plugins.protocol import TrainCallbacks, DataBundle, DataProfile

runner = CliRunner()


@pytest.fixture
def tabular_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.uniform(20, 60, n),
        "income": np.random.uniform(20000, 100000, n),
        "label": np.random.randint(0, 2, n),
    })


@pytest.fixture
def trained_project(tmp_project: Path, tabular_df: pd.DataFrame) -> Path:
    """Project with a pre-trained RandomForest model artifact."""
    from model_builder_classical.random_forest import RandomForestPlugin
    cb = TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)

    data_path = tmp_project / "data.parquet"
    tabular_df.to_parquet(data_path, index=False)

    run_dir = tmp_project / "runs" / "run_001"
    artifacts_dir = run_dir / "artifacts"
    (run_dir / "logs").mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)

    data_path_in_artifacts = artifacts_dir / "data.parquet"
    tabular_df.to_parquet(data_path_in_artifacts, index=False)

    profile = DataProfile(100, 3, {}, {}, ["tabular"], str(data_path_in_artifacts))
    bundle = DataBundle(profile=profile, data_path=str(data_path_in_artifacts))

    plugin = RandomForestPlugin()
    artifact = plugin.train(bundle, {"target_col": "label", "n_estimators": 10}, cb)

    # Write a log file so scan_models can detect the node
    (run_dir / "logs" / "train_random_forest.log").write_text("RandomForest: 100 rows")

    # Seed the store
    async def seed():
        from model_builder.core.store import ProjectStore
        s = ProjectStore(tmp_project)
        await s.init()
        await s.create_run("run_001")

    asyncio.run(seed())
    return tmp_project


# --- ModelRegistry ---

def test_scan_models_finds_artifact(trained_project: Path):
    entries = scan_models(trained_project)
    assert len(entries) >= 1
    assert any(e.plugin_name == "random_forest" for e in entries)


def test_scan_models_metadata(trained_project: Path):
    entries = scan_models(trained_project)
    rf = next(e for e in entries if e.plugin_name == "random_forest")
    assert rf.run_name == "run_001"
    assert rf.task in ("binary_classification", "regression", "unknown")


def test_get_model_found(trained_project: Path):
    entry = get_model(trained_project, "run_001", "random_forest")
    assert entry is not None
    assert entry.plugin_name == "random_forest"


def test_get_model_not_found(trained_project: Path):
    entry = get_model(trained_project, "run_001", "nonexistent")
    assert entry is None


def test_model_entry_id(trained_project: Path):
    entries = scan_models(trained_project)
    rf = next(e for e in entries if e.plugin_name == "random_forest")
    assert "/" in rf.id


# --- RandomForest update() ---

def test_rf_update_increases_estimators(tabular_df: pd.DataFrame, tmp_path: Path):
    from model_builder_classical.random_forest import RandomForestPlugin
    cb = TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)

    data_path = tmp_path / "data.parquet"
    tabular_df.to_parquet(data_path, index=False)
    profile = DataProfile(100, 3, {}, {}, ["tabular"], str(data_path))
    bundle = DataBundle(profile=profile, data_path=str(data_path))

    plugin = RandomForestPlugin()
    artifact = plugin.train(bundle, {"target_col": "label", "n_estimators": 10}, cb)

    with open(artifact.model_path, "rb") as f:
        saved_before = pickle.load(f)
    n_before = saved_before["model"].n_estimators

    updated = plugin.update(artifact, bundle, {"target_col": "label", "n_estimators_new": 5}, cb)

    with open(updated.model_path, "rb") as f:
        saved_after = pickle.load(f)
    assert saved_after["model"].n_estimators > n_before
    assert updated.metadata.get("updated") is True


def test_rf_update_evaluate_works(tabular_df: pd.DataFrame, tmp_path: Path):
    from model_builder_classical.random_forest import RandomForestPlugin
    cb = TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)

    data_path = tmp_path / "data.parquet"
    tabular_df.to_parquet(data_path, index=False)
    profile = DataProfile(100, 3, {}, {}, ["tabular"], str(data_path))
    bundle = DataBundle(profile=profile, data_path=str(data_path))

    plugin = RandomForestPlugin()
    artifact = plugin.train(bundle, {"target_col": "label", "n_estimators": 10}, cb)
    updated = plugin.update(artifact, bundle, {"target_col": "label", "n_estimators_new": 5}, cb)
    report = plugin.evaluate(updated, bundle)
    assert "accuracy" in report.metrics
    assert 0 <= report.metrics["accuracy"] <= 1


# --- XGBoost update() ---

def test_xgb_update_works(tabular_df: pd.DataFrame, tmp_path: Path):
    from model_builder_classical.xgboost_plugin import XGBoostPlugin
    cb = TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)

    data_path = tmp_path / "data.parquet"
    tabular_df.to_parquet(data_path, index=False)
    profile = DataProfile(100, 3, {}, {}, ["tabular"], str(data_path))
    bundle = DataBundle(profile=profile, data_path=str(data_path))

    plugin = XGBoostPlugin()
    artifact = plugin.train(bundle, {"target_col": "label", "n_estimators": 10}, cb)
    updated = plugin.update(artifact, bundle, {"target_col": "label", "n_estimators": 5}, cb)
    assert updated.metadata.get("updated") is True
    report = plugin.evaluate(updated, bundle)
    assert "accuracy" in report.metrics


# --- LightGBM update() ---

def test_lgbm_update_works(tabular_df: pd.DataFrame, tmp_path: Path):
    from model_builder_classical.lightgbm_plugin import LightGBMPlugin
    cb = TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)

    data_path = tmp_path / "data.parquet"
    tabular_df.to_parquet(data_path, index=False)
    profile = DataProfile(100, 3, {}, {}, ["tabular"], str(data_path))
    bundle = DataBundle(profile=profile, data_path=str(data_path))

    plugin = LightGBMPlugin()
    artifact = plugin.train(bundle, {"target_col": "label", "n_estimators": 10}, cb)
    updated = plugin.update(artifact, bundle, {"target_col": "label", "n_estimators": 5}, cb)
    assert updated.metadata.get("updated") is True
    report = plugin.evaluate(updated, bundle)
    assert "accuracy" in report.metrics


# --- ModelUpdatePlugin ---

def test_model_update_plugin_registered():
    reg = PluginRegistry()
    reg.register_built_ins()
    plugin = reg.get_core_plugin("core.model_update")
    assert plugin.name == "core.model_update"


def test_model_update_plugin_runs(trained_project: Path, tabular_df: pd.DataFrame):
    run2_dir = trained_project / "runs" / "run_002"
    artifacts_dir = run2_dir / "artifacts"
    (run2_dir / "logs").mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    tabular_df.to_parquet(artifacts_dir / "merged.parquet", index=False)

    reg = PluginRegistry()
    reg.discover()
    ctx = CoreContext(
        run_dir=run2_dir,
        artifacts_dir=artifacts_dir,
        logs_dir=run2_dir / "logs",
        run_id=2,
        registry=reg,
        node_config={
            "plugin_name": "random_forest",
            "source_run": "run_001",
            "target_col": "label",
            "n_estimators_new": 5,
        },
    )

    from model_builder.core_plugins.model_update_plugin import ModelUpdatePlugin
    out = ModelUpdatePlugin().run(ctx)
    assert out.exists()
    receipt = json.loads(out.read_text())
    assert receipt["plugin_name"] == "random_forest"
    assert receipt["source_run"] == "run_001"


# --- CLI ---

def test_models_list_shows_trained(trained_project: Path):
    result = runner.invoke(app, ["models", "list"],
                           env={"MB_PROJECT_DIR": str(trained_project)})
    assert result.exit_code == 0
    assert "random_forest" in result.output


def test_models_list_empty(tmp_project: Path):
    result = runner.invoke(app, ["models", "list"],
                           env={"MB_PROJECT_DIR": str(tmp_project)})
    assert result.exit_code == 0
    assert "No trained models" in result.output


def test_models_help():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    for cmd in ["list", "info", "update"]:
        assert cmd in result.output
