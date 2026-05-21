import pytest
from pathlib import Path
from model_builder.plugins.protocol import DataProfile, TrainCallbacks, ExportFormat
from model_builder_dl.cnn_image import CNNImagePlugin


def _cb():
    return TrainCallbacks(on_progress=lambda *_: None, on_log=lambda _: None)


def test_detect_image_high(image_bundle):
    assert CNNImagePlugin().detect(image_bundle.profile) >= 0.8


def test_detect_tabular_zero():
    profile = DataProfile(0, 0, {}, {}, ["tabular"], "")
    assert CNNImagePlugin().detect(profile) == 0.0


def test_train_classification(image_bundle, image_dir):
    plugin = CNNImagePlugin()
    artifact = plugin.train(
        image_bundle,
        {"image_dir": str(image_dir), "img_size": 8, "epochs": 2, "batch_size": 4},
        _cb(),
    )
    assert artifact.plugin_name == "cnn_image"
    assert Path(artifact.model_path).exists()
    assert artifact.metadata["num_classes"] == 2
    assert set(artifact.metadata["classes"]) == {"cat", "dog"}


def test_evaluate_returns_metrics(image_bundle, image_dir):
    plugin = CNNImagePlugin()
    artifact = plugin.train(
        image_bundle,
        {"image_dir": str(image_dir), "img_size": 8, "epochs": 2, "batch_size": 4},
        _cb(),
    )
    report = plugin.evaluate(artifact, image_bundle)
    assert report.plugin_name == "cnn_image"
    assert "accuracy" in report.metrics
    assert 0 <= report.metrics["accuracy"] <= 1


def test_train_missing_image_dir_raises(image_bundle):
    plugin = CNNImagePlugin()
    with pytest.raises((ValueError, FileNotFoundError)):
        plugin.train(image_bundle, {}, _cb())


def test_export_returns_existing_pkl(image_bundle, image_dir):
    plugin = CNNImagePlugin()
    artifact = plugin.train(
        image_bundle,
        {"image_dir": str(image_dir), "img_size": 8, "epochs": 1, "batch_size": 4},
        _cb(),
    )
    out = plugin.export(artifact, ExportFormat("pickle"))
    assert out.exists()
