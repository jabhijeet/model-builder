import numpy as np
import pytest
from model_builder_llm._base import compute_metrics, detect_target_modules, get_lora_config


def test_compute_metrics_binary():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_detect_target_modules_gpt2():
    assert "c_attn" in detect_target_modules("gpt2")


def test_detect_target_modules_llama():
    mods = detect_target_modules("meta-llama/Llama-2-7b")
    assert "q_proj" in mods
    assert "v_proj" in mods


def test_detect_target_modules_mistral():
    mods = detect_target_modules("mistralai/Mistral-7B-v0.1")
    assert "q_proj" in mods


def test_lora_config_fields():
    cfg = get_lora_config(r=4, alpha=8, dropout=0.05)
    assert cfg.r == 4
    assert cfg.lora_alpha == 8
    assert cfg.lora_dropout == 0.05
