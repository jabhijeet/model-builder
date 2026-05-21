import pytest
from pathlib import Path
from model_builder_classical.xgboost_plugin import XGBoostPlugin
from model_builder.plugins.protocol import DataProfile, TrainCallbacks, ExportFormat


def _callbacks():
    return TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)


def test_detect_tabular(classification_bundle):
    assert XGBoostPlugin().detect(classification_bundle.profile) >= 0.8


def test_detect_non_tabular():
    profile = DataProfile(0, 0, {}, {}, ["text"], "")
    assert XGBoostPlugin().detect(profile) == 0.0


def test_train_classification(classification_bundle):
    artifact = XGBoostPlugin().train(
        classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks()
    )
    assert artifact.plugin_name == "xgboost"
    assert Path(artifact.model_path).exists()


def test_evaluate_metrics(classification_bundle):
    plugin = XGBoostPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    report = plugin.evaluate(artifact, classification_bundle)
    assert "accuracy" in report.metrics
    assert report.feature_importance is not None


def test_export_pickle(classification_bundle):
    plugin = XGBoostPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    out = plugin.export(artifact, ExportFormat("pickle"))
    assert out.exists()
