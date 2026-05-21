import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb
from model_builder.plugins.protocol import (
    DataProfile, DataBundle, Artifact, EvalReport, TrainCallbacks, ExportFormat
)
from ._base import (
    TaskType, detect_task_type, split_features_target,
    compute_metrics, compute_shap, export_pickle
)


class XGBoostPlugin:
    name = "xgboost"
    data_types = ["tabular"]
    requires_gpu = False

    def detect(self, profile: DataProfile) -> float:
        if "tabular" not in profile.data_types:
            return 0.0
        return 0.90

    def train(self, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact:
        df = pd.read_parquet(data.data_path)
        X, y = split_features_target(df, config.get("target_col"))
        task = detect_task_type(y)

        callbacks.on_log(f"XGBoost: {len(df)} rows, task={task.value}")
        callbacks.on_progress(0, 1, {})

        n = config.get("n_estimators", 100)
        lr = config.get("learning_rate", 0.1)
        depth = config.get("max_depth", 6)

        if task == TaskType.REGRESSION:
            model = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth,
                                      random_state=42, verbosity=0)
        elif task == TaskType.BINARY_CLASSIFICATION:
            model = xgb.XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth,
                                       random_state=42, verbosity=0, eval_metric="logloss")
        else:
            model = xgb.XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth,
                                       random_state=42, verbosity=0,
                                       objective="multi:softmax", num_class=y.nunique())

        model.fit(X, y)
        callbacks.on_progress(1, 1, {})

        model_dir = Path(data.data_path).parent / "xgb_model"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "X_cols": list(X.columns), "task": task.value}, f)

        return Artifact(
            model_path=str(model_path),
            plugin_name="xgboost",
            metadata={"task": task.value, "X_cols": list(X.columns)},
        )

    def evaluate(self, artifact: Artifact, data: DataBundle) -> EvalReport:
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        model, task, X_cols = saved["model"], TaskType(saved["task"]), saved["X_cols"]

        df = pd.read_parquet(data.data_path)
        X, y = split_features_target(df, None)
        X = X[X_cols]

        return EvalReport(
            plugin_name="xgboost",
            metrics=compute_metrics(y.values, model.predict(X), task),
            feature_importance=compute_shap(model, X),
        )

    def export(self, artifact: Artifact, fmt: ExportFormat) -> Path:
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        out = Path(artifact.model_path).parent / "model.pkl"
        export_pickle(saved["model"], out)
        return out

    def update(self, artifact: Artifact, data: DataBundle, config: dict, callbacks: TrainCallbacks) -> Artifact:
        """Incremental XGBoost training from existing booster checkpoint."""
        with open(artifact.model_path, "rb") as f:
            saved = pickle.load(f)
        old_model, task, X_cols = saved["model"], TaskType(saved["task"]), saved["X_cols"]

        df = pd.read_parquet(data.data_path)
        X, y = split_features_target(df, config.get("target_col") or saved.get("target_col"))
        X = X[[c for c in X_cols if c in X.columns]]

        n = config.get("n_estimators", 50)
        lr = config.get("learning_rate", old_model.learning_rate)
        depth = config.get("max_depth", old_model.max_depth)

        if task == TaskType.REGRESSION:
            new_model = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth,
                                          random_state=42, verbosity=0)
        else:
            new_model = xgb.XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth,
                                           random_state=42, verbosity=0, eval_metric="logloss")
        # Pass existing booster as starting point
        new_model.fit(X, y, xgb_model=old_model.get_booster())
        callbacks.on_log(f"XGBoost incremental: +{n} rounds from existing booster")
        callbacks.on_progress(1, 1, {})

        model_path = Path(artifact.model_path).parent / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": new_model, "X_cols": list(X.columns), "task": task.value}, f)
        return Artifact(model_path=str(model_path), plugin_name="xgboost",
                        metadata={"task": task.value, "X_cols": list(X.columns), "updated": True})
