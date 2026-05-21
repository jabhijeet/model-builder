import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from model_builder.plugins.protocol import (
    DataProfile, DataBundle, Artifact, EvalReport, TrainCallbacks, ExportFormat
)
from ._base import (
    TaskType, detect_task_type, split_features_target,
    compute_metrics, compute_shap, export_pickle, export_onnx_sklearn
)


class RandomForestPlugin:
    name = "random_forest"
    data_types = ["tabular"]
    requires_gpu = False

    def detect(self, profile: DataProfile) -> float:
        if "tabular" not in profile.data_types:
            return 0.0
        return 0.3 if profile.row_count < 10 else 0.85

    def train(self, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact:
        df = pd.read_parquet(data.data_path)
        X, y = split_features_target(df, config.get("target_col"))
        task = detect_task_type(y)

        callbacks.on_log(f"RandomForest: {len(df)} rows, task={task.value}")
        callbacks.on_progress(0, 1, {})

        n = config.get("n_estimators", 100)
        if task == TaskType.REGRESSION:
            model = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)

        model.fit(X, y)
        callbacks.on_progress(1, 1, {})

        model_dir = Path(data.data_path).parent / "rf_model"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "X_cols": list(X.columns), "task": task.value}, f)

        return Artifact(
            model_path=str(model_path),
            plugin_name="random_forest",
            metadata={"task": task.value, "n_estimators": n, "X_cols": list(X.columns)},
        )

    def evaluate(self, artifact: Artifact, data: DataBundle) -> EvalReport:
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        model, task, X_cols = saved["model"], TaskType(saved["task"]), saved["X_cols"]

        df = pd.read_parquet(data.data_path)
        X, y = split_features_target(df, None)
        X = X[X_cols]

        y_pred = model.predict(X)
        return EvalReport(
            plugin_name="random_forest",
            metrics=compute_metrics(y.values, y_pred, task),
            feature_importance=compute_shap(model, X),
        )

    def export(self, artifact: Artifact, fmt: ExportFormat) -> Path:
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        model, X_cols = saved["model"], saved["X_cols"]
        model_dir = Path(artifact.model_path).parent

        if str(fmt) == "onnx":
            sample = pd.DataFrame(np.zeros((1, len(X_cols)), dtype="float32"), columns=X_cols)
            out = model_dir / "model.onnx"
            export_onnx_sklearn(model, sample, out)
            return out

        out = model_dir / "model.pkl"
        export_pickle(model, out)
        return out

    def update(self, artifact: Artifact, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact:
        """Warm-start: load existing forest and add more trees."""
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        model, task, X_cols = saved["model"], TaskType(saved["task"]), saved["X_cols"]

        df = pd.read_parquet(data.data_path)
        X, y = split_features_target(df, config.get("target_col") or saved.get("target_col"))
        X = X[[c for c in X_cols if c in X.columns]]

        n_new = config.get("n_estimators_new", 50)
        model.n_estimators += n_new
        model.warm_start = True
        model.fit(X, y)
        callbacks.on_log(f"RandomForest warm-start: +{n_new} trees -> total={model.n_estimators}")
        callbacks.on_progress(1, 1, {"n_estimators": model.n_estimators})

        model_dir = Path(artifact.model_path).parent
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "X_cols": list(X.columns), "task": task.value}, f)
        return Artifact(model_path=str(model_path), plugin_name="random_forest",
                        metadata={"task": task.value, "n_estimators": model.n_estimators,
                                  "X_cols": list(X.columns), "updated": True})


