import pandas as pd
import numpy as np
import pytest
from model_builder_classical._base import (
    TaskType, detect_task_type, split_features_target,
    compute_metrics, compute_shap
)


def test_detect_binary_classification():
    assert detect_task_type(pd.Series([0, 1, 0, 1, 0])) == TaskType.BINARY_CLASSIFICATION


def test_detect_multiclass():
    assert detect_task_type(pd.Series([0, 1, 2, 0, 1, 2])) == TaskType.MULTICLASS


def test_detect_regression():
    assert detect_task_type(pd.Series([1.5, 2.7, 3.1, 100.0, 0.5])) == TaskType.REGRESSION


def test_split_features_target():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": [0, 1]})
    X, y = split_features_target(df, target_col="label")
    assert list(X.columns) == ["a", "b"]
    assert list(y) == [0, 1]


def test_split_uses_last_col_if_no_target():
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "y": [0]})
    X, y = split_features_target(df, target_col=None)
    assert "y" not in X.columns
    assert list(y) == [0]


def test_compute_metrics_binary():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred, TaskType.BINARY_CLASSIFICATION)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_compute_metrics_regression():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    metrics = compute_metrics(y_true, y_pred, TaskType.REGRESSION)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["rmse"] >= 0
