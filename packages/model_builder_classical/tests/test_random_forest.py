import pytest
from pathlib import Path
from model_builder_classical.random_forest import RandomForestPlugin
from model_builder.plugins.protocol import DataProfile, TrainCallbacks, ExportFormat


def _callbacks():
    return TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)


def test_detect_tabular_high_confidence(classification_bundle):
    assert RandomForestPlugin().detect(classification_bundle.profile) >= 0.7


def test_detect_non_tabular_zero():
    plugin = RandomForestPlugin()
    profile = DataProfile(0, 0, {}, {}, ["image"], "")
    assert plugin.detect(profile) == 0.0


def test_train_binary_classification(classification_bundle):
    artifact = RandomForestPlugin().train(
        classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks()
    )
    assert artifact.plugin_name == "random_forest"
    assert Path(artifact.model_path).exists()


def test_evaluate_returns_metrics(classification_bundle):
    plugin = RandomForestPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    report = plugin.evaluate(artifact, classification_bundle)
    assert report.plugin_name == "random_forest"
    assert "accuracy" in report.metrics
    assert 0 <= report.metrics["accuracy"] <= 1


def test_feature_importance_in_report(classification_bundle):
    plugin = RandomForestPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    report = plugin.evaluate(artifact, classification_bundle)
    assert report.feature_importance is not None
    assert len(report.feature_importance) == 3  # age, score, income


def test_export_pickle(classification_bundle):
    plugin = RandomForestPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    out = plugin.export(artifact, ExportFormat("pickle"))
    assert out.exists()
    assert out.suffix == ".pkl"


def test_export_onnx(classification_bundle):
    plugin = RandomForestPlugin()
    artifact = plugin.train(classification_bundle, {"n_estimators": 10, "target_col": "label"}, _callbacks())
    out = plugin.export(artifact, ExportFormat("onnx"))
    assert out.exists()
    assert out.suffix == ".onnx"
