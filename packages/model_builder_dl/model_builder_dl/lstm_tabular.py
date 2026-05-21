"""LSTM plugin for tabular/sequential data."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_builder.plugins.protocol import (
    DataProfile, DataBundle, Artifact, EvalReport, TrainCallbacks, ExportFormat
)
from ._base import get_device, compute_metrics, gradient_importance


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _detect_task(y: np.ndarray) -> str:
    if np.issubdtype(y.dtype, np.integer):
        return "binary" if len(np.unique(y)) == 2 else "multiclass"
    return "regression"


class LSTMTabularPlugin:
    name = "lstm_tabular"
    data_types = ["tabular"]
    requires_gpu = False

    def detect(self, profile: DataProfile) -> float:
        if "tabular" not in profile.data_types:
            return 0.0
        # LSTM useful for larger datasets or when temporal ordering matters
        return 0.70 if profile.row_count >= 50 else 0.3

    def train(self, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact:
        df = pd.read_parquet(data.data_path)
        target_col = config.get("target_col") or df.columns[-1]
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values.astype(np.float32)
        y_raw = df[target_col].values
        task = _detect_task(y_raw)

        seq_len = config.get("seq_len", 1)  # treat each row as seq_len=1 by default
        hidden = config.get("hidden_size", 64)
        layers = config.get("num_layers", 2)
        epochs = config.get("epochs", 20)
        lr = config.get("learning_rate", 1e-3)
        batch_size = config.get("batch_size", 32)
        device = get_device()

        if task == "regression":
            y = y_raw.astype(np.float32)
            num_classes = 1
            criterion = nn.MSELoss()
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y_raw).astype(np.int64)
            num_classes = len(le.classes_)
            criterion = nn.CrossEntropyLoss()

        # Reshape to (samples, seq_len, features)
        X_t = torch.tensor(X).unsqueeze(1)  # (N, 1, F) — each row = 1 time step
        y_t = torch.tensor(y)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = _LSTMNet(X.shape[1], hidden, layers, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        feature_names = list(df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns)
        callbacks.on_log(f"LSTM: {len(df)} rows, {X.shape[1]} features, task={task}, device={device}")

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                if task == "regression":
                    loss = criterion(out.squeeze(), yb)
                else:
                    loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(len(loader), 1)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                callbacks.on_progress(epoch + 1, epochs, {"loss": round(avg_loss, 4)})

        model_dir = Path(data.data_path).parent / "lstm_model"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "state_dict": model.state_dict(),
                "input_size": X.shape[1],
                "hidden_size": hidden,
                "num_layers": layers,
                "num_classes": num_classes,
                "task": task,
                "feature_names": feature_names,
                "target_col": target_col,
            }, f)

        return Artifact(
            model_path=str(model_path),
            plugin_name="lstm_tabular",
            metadata={"task": task, "num_classes": num_classes, "features": feature_names},
        )

    def evaluate(self, artifact: Artifact, data: DataBundle) -> EvalReport:
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)

        device = get_device()
        model = _LSTMNet(
            saved["input_size"], saved["hidden_size"],
            saved["num_layers"], saved["num_classes"]
        ).to(device)
        model.load_state_dict(saved["state_dict"])
        model.eval()

        df = pd.read_parquet(data.data_path)
        target_col = saved["target_col"]
        feature_names = saved["feature_names"]
        X = df[feature_names].values.astype(np.float32)
        y_raw = df[target_col].values
        task = saved["task"]

        X_t = torch.tensor(X).unsqueeze(1).to(device)
        with torch.no_grad():
            out = model(X_t)
            if task == "regression":
                y_pred = out.squeeze().cpu().numpy()
                y_true = y_raw.astype(np.float32)
            else:
                y_pred = out.argmax(dim=1).cpu().numpy()
                y_true = y_raw

        metrics = compute_metrics(y_true, y_pred, task)

        # Gradient-based importance
        importance: dict = {}
        try:
            sample = X_t[:min(32, len(X_t))]
            importance = gradient_importance(model, sample, feature_names)
        except Exception:
            pass

        return EvalReport(
            plugin_name="lstm_tabular",
            metrics=metrics,
            feature_importance=importance or None,
        )

    def export(self, artifact: Artifact, fmt: ExportFormat) -> Path:
        return Path(artifact.model_path)
