from enum import Enum
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error,
)


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


def detect_task_type(label: pd.Series) -> TaskType:
    if label.dtype == object or str(label.dtype) == "bool":
        return TaskType.BINARY_CLASSIFICATION if label.nunique() == 2 else TaskType.MULTICLASS
    if str(label.dtype) in ("int32", "int64", "int"):
        n_unique = label.nunique()
        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION
        if n_unique <= 20:
            return TaskType.MULTICLASS
    return TaskType.REGRESSION


def split_features_target(df: pd.DataFrame, target_col: str | None) -> tuple:
    col = target_col if target_col and target_col in df.columns else df.columns[-1]
    return df.drop(columns=[col]), df[col]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: TaskType) -> dict:
    if task == TaskType.BINARY_CLASSIFICATION:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        }
    if task == TaskType.MULTICLASS:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse, "mae": float(mean_absolute_error(y_true, y_pred))}


def compute_shap(model, X: pd.DataFrame, max_rows: int = 200) -> dict:
    try:
        import shap
        sample = X.head(max_rows)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        importance = np.abs(shap_values).mean(axis=0)
        return {col: float(v) for col, v in zip(X.columns, importance)}
    except Exception:
        if hasattr(model, "feature_importances_"):
            return {col: float(v) for col, v in zip(X.columns, model.feature_importances_)}
        return {}


def export_pickle(model, output_path) -> str:
    import pickle
    with open(str(output_path), "wb") as f:
        pickle.dump(model, f)
    return str(output_path)


def export_onnx_sklearn(model, X_sample: pd.DataFrame, output_path) -> str:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(str(output_path), "wb") as f:
        f.write(onnx_model.SerializeToString())
    return str(output_path)
