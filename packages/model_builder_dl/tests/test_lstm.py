import pytest
from pathlib import Path
from model_builder.plugins.protocol import DataProfile, TrainCallbacks, ExportFormat
from model_builder_dl.lstm_tabular import LSTMTabularPlugin


def _cb():
    return TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)


def test_detect_tabular_high(tabular_bundle):
    assert LSTMTabularPlugin().detect(tabular_bundle.profile) >= 0.6


def test_detect_image_zero():
    profile = DataProfile(0, 0, {}, {}, ["image"], "")
    assert LSTMTabularPlugin().detect(profile) == 0.0


def test_train_binary_classification(tabular_bundle):
    plugin = LSTMTabularPlugin()
    artifact = plugin.train(
        tabular_bundle,
        {"target_col": "label", "epochs": 3, "hidden_size": 16, "num_layers": 1},
        _cb(),
    )
    assert artifact.plugin_name == "lstm_tabular"
    assert Path(artifact.model_path).exists()


def test_evaluate_returns_metrics(tabular_bundle):
    plugin = LSTMTabularPlugin()
    artifact = plugin.train(
        tabular_bundle,
        {"target_col": "label", "epochs": 3, "hidden_size": 16, "num_layers": 1},
        _cb(),
    )
    report = plugin.evaluate(artifact, tabular_bundle)
    assert "accuracy" in report.metrics
    assert 0 <= report.metrics["accuracy"] <= 1


def test_feature_importance_in_report(tabular_bundle):
    plugin = LSTMTabularPlugin()
    artifact = plugin.train(
        tabular_bundle,
        {"target_col": "label", "epochs": 3, "hidden_size": 16, "num_layers": 1},
        _cb(),
    )
    report = plugin.evaluate(artifact, tabular_bundle)
    assert report.feature_importance is not None
    assert len(report.feature_importance) == 3  # feat_a, feat_b, feat_c


def test_export_returns_path(tabular_bundle):
    plugin = LSTMTabularPlugin()
    artifact = plugin.train(
        tabular_bundle,
        {"target_col": "label", "epochs": 2, "hidden_size": 8, "num_layers": 1},
        _cb(),
    )
    out = plugin.export(artifact, ExportFormat("pickle"))
    assert out.exists()
