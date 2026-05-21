"""Shared utilities for LLM fine-tuning plugins."""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    avg = "binary" if len(set(y_true)) <= 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }


def get_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.1,
                    target_modules: list | None = None):
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["c_attn"],  # GPT-2 default; overridden per model
        bias="none",
    )


def detect_target_modules(model_name: str) -> list[str]:
    """Return LoRA target module names for known model families."""
    name = model_name.lower()
    if "llama" in name or "mistral" in name or "phi" in name:
        return ["q_proj", "v_proj"]
    if "gpt2" in name or "gpt-2" in name:
        return ["c_attn"]
    if "bert" in name or "roberta" in name:
        return ["query", "value"]
    return ["q_proj", "v_proj"]  # safe default for most transformer decoders
