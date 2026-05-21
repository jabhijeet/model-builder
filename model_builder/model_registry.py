"""Scans project runs for trained model artifacts and provides a unified registry."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_DIR_TO_PLUGIN = {
    "rf": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "cnn": "cnn_image",
    "lstm": "lstm_tabular",
    "lora": "lora_text",
}


@dataclass
class ModelEntry:
    run_name: str
    node_id: str
    plugin_name: str
    model_dir: Path
    artifact_path: Path
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.run_name}/{self.node_id}"

    @property
    def task(self) -> str:
        return self.metadata.get("task", "unknown")

    @property
    def features(self) -> list[str]:
        return self.metadata.get("X_cols", self.metadata.get("features", []))


def scan_models(project_dir: Path) -> list[ModelEntry]:
    """Walk all run directories and find trained model artifacts."""
    entries: list[ModelEntry] = []
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return entries

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        artifacts_dir = run_dir / "artifacts"
        if not artifacts_dir.exists():
            continue

        # Each *_model directory is a trained plugin artifact
        for model_dir in sorted(artifacts_dir.glob("*_model")):
            if not model_dir.is_dir():
                continue
            artifact_path = model_dir / "model.pkl"
            if not artifact_path.exists():
                artifact_path = model_dir / "artifact.pkl"
            if not artifact_path.exists():
                continue

            raw_name = model_dir.name.replace("_model", "")
            plugin_name = _DIR_TO_PLUGIN.get(raw_name, raw_name)
            metadata = _read_metadata(artifact_path)

            # Try to find which node produced this model
            node_id = _guess_node_id(plugin_name, run_dir)

            entries.append(ModelEntry(
                run_name=run_dir.name,
                node_id=node_id,
                plugin_name=plugin_name,
                model_dir=model_dir,
                artifact_path=artifact_path,
                metadata=metadata,
            ))

    return entries


def get_model(project_dir: Path, run_name: str, plugin_name: str) -> Optional[ModelEntry]:
    """Get a specific model entry."""
    for entry in scan_models(project_dir):
        if entry.run_name == run_name and entry.plugin_name == plugin_name:
            return entry
    return None


def _read_metadata(artifact_path: Path) -> dict:
    try:
        with open(artifact_path, "rb") as f:
            saved = pickle.load(f)
        if isinstance(saved, dict):
            return {k: v for k, v in saved.items() if k != "model" and k != "state_dict"}
    except Exception:
        pass
    return {}


def _guess_node_id(plugin_name: str, run_dir: Path) -> str:
    """Try to match plugin name to a node ID from logs dir."""
    logs_dir = run_dir / "logs"
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            node_id = log_file.stem
            content = log_file.read_text(errors="ignore")
            if plugin_name.replace("_", "") in content.replace("_", "").lower():
                return node_id
    # Fallback: use convention train_<plugin>
    return f"train_{plugin_name}"
