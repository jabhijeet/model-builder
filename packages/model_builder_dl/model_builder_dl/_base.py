"""Shared utilities for DL plugins."""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error


def get_device(require_gpu: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if require_gpu:
        raise RuntimeError("GPU required but not available.")
    return torch.device("cpu")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict:
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {"rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred))}
    avg = "binary" if len(set(y_true)) == 2 else "weighted"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }


def gradient_importance(model: torch.nn.Module, X: torch.Tensor,
                        feature_names: list[str]) -> dict[str, float]:
    """Approximate feature importance via input gradient magnitude."""
    model.eval()
    X = X.clone().requires_grad_(True)
    out = model(X)
    if out.shape[1] > 1:
        out = out[:, 1]  # binary: use positive class
    out.sum().backward()
    importance = X.grad.abs().mean(dim=0).reshape(-1, len(feature_names)).mean(dim=0).detach().numpy()
    total = importance.sum() + 1e-8
    return {n: float(v / total) for n, v in zip(feature_names, importance)}
