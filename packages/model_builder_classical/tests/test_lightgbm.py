import pytest
from pathlib import Path
from model_builder_classical.lightgbm_plugin import LightGBMPlugin
from model_builder.plugins.protocol import DataProfile, TrainCallbacks, ExportFormat


def _callbacks():
    return TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)


def test_detect_tabular(classification_bundle):
    assert LightGBMPlugin().detect(classification_bundle.profile) >= 0.8


def test_train_classification(classification_bundle):
    artifact = LightGBMPlugin().train(
        classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks()
    )
    assert artifact.plugin_name == "lightgbm"
    assert Path(artifact.model_path).exists()


def test_evaluate_metrics(classification_bundle):
    plugin = LightGBMPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    report = plugin.evaluate(artifact, classification_bundle)
    assert "accuracy" in report.metrics
    assert report.feature_importance is not None


def test_export_pickle(classification_bundle):
    plugin = LightGBMPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    out = plugin.export(artifact, ExportFormat("pickle"))
    assert out.exists()
