"""Tests for LoRATextPlugin using a tiny local GPT-2 (no HF Hub download)."""
import pytest
from pathlib import Path
from model_builder.plugins.protocol import DataProfile, TrainCallbacks, ExportFormat
from model_builder_llm.lora_text import LoRATextPlugin


def _cb():
    logs = []
    return TrainCallbacks(
        on_progress=lambda *_: None,
        on_log=lambda msg: logs.append(msg),
    )


def test_detect_text_high(text_bundle):
    assert LoRATextPlugin().detect(text_bundle.profile) >= 0.8


def test_detect_tabular_zero():
    profile = DataProfile(0, 0, {}, {}, ["tabular"], "")
    assert LoRATextPlugin().detect(profile) == 0.0


def test_detect_image_zero():
    profile = DataProfile(0, 0, {}, {}, ["image"], "")
    assert LoRATextPlugin().detect(profile) == 0.0


def test_train_with_tiny_gpt2(text_bundle, tiny_gpt2_dir):
    """Train LoRA adapter on tiny local GPT-2 — no internet required."""
    plugin = LoRATextPlugin()
    artifact = plugin.train(
        text_bundle,
        {
            "base_model": str(tiny_gpt2_dir),
            "text_col": "review",
            "label_col": "sentiment",
            "epochs": 1,
            "batch_size": 4,
            "max_length": 32,
            "lora_r": 4,
            "lora_alpha": 8,
        },
        _cb(),
    )
    assert artifact.plugin_name == "lora_text"
    assert Path(artifact.model_path).exists()
    assert artifact.metadata["num_labels"] == 2
    assert set(artifact.metadata["label2id"].keys()) == {"negative", "positive"}


def test_evaluate_after_train(text_bundle, tiny_gpt2_dir):
    plugin = LoRATextPlugin()
    artifact = plugin.train(
        text_bundle,
        {
            "base_model": str(tiny_gpt2_dir),
            "text_col": "review", "label_col": "sentiment",
            "epochs": 1, "batch_size": 4, "max_length": 32,
            "lora_r": 4,
        },
        _cb(),
    )
    report = plugin.evaluate(artifact, text_bundle)
    assert report.plugin_name == "lora_text"
    assert "accuracy" in report.metrics
    assert "f1" in report.metrics
    assert 0 <= report.metrics["accuracy"] <= 1


def test_adapter_weights_saved(text_bundle, tiny_gpt2_dir):
    """LoRA adapter config must be saved alongside weights."""
    plugin = LoRATextPlugin()
    artifact = plugin.train(
        text_bundle,
        {
            "base_model": str(tiny_gpt2_dir),
            "text_col": "review", "label_col": "sentiment",
            "epochs": 1, "batch_size": 4, "max_length": 32,
        },
        _cb(),
    )
    model_dir = Path(artifact.model_path).parent
    assert (model_dir / "adapter_config.json").exists()
    assert (model_dir / "mb_meta.json").exists()


def test_export_returns_adapter_path(text_bundle, tiny_gpt2_dir):
    plugin = LoRATextPlugin()
    artifact = plugin.train(
        text_bundle,
        {
            "base_model": str(tiny_gpt2_dir),
            "text_col": "review", "label_col": "sentiment",
            "epochs": 1, "batch_size": 4, "max_length": 32,
        },
        _cb(),
    )
    out = plugin.export(artifact, ExportFormat("safetensors"))
    assert out.exists()


def test_find_text_col_heuristic(text_bundle, tiny_gpt2_dir):
    """Plugin auto-detects text column when not specified."""
    plugin = LoRATextPlugin()
    # No text_col in config — should auto-detect "review"
    artifact = plugin.train(
        text_bundle,
        {
            "base_model": str(tiny_gpt2_dir),
            "label_col": "sentiment",
            "epochs": 1, "batch_size": 4, "max_length": 32,
        },
        _cb(),
    )
    assert artifact.metadata["text_col"] == "review"


def test_entry_point_registered():
    """Plugin registers correctly via entry points."""
    from model_builder.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover()
    plugin = reg.get_ml_plugin("ml.llm.lora_text")
    assert plugin.name == "lora_text"
    assert "text" in plugin.data_types
