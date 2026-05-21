import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from model_builder.plugins.protocol import DataProfile, DataBundle


@pytest.fixture
def classification_bundle(tmp_path: Path) -> DataBundle:
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.uniform(20, 60, n),
        "score": np.random.uniform(0, 1, n),
        "income": np.random.uniform(20000, 100000, n),
        "label": np.random.randint(0, 2, n),
    })
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path, index=False)
    profile = DataProfile(
        row_count=n, column_count=4,
        columns={"age": "float64", "score": "float64", "income": "float64", "label": "int64"},
        nulls={"age": 0, "score": 0, "income": 0, "label": 0},
        data_types=["tabular"],
        sample_path=str(data_path),
    )
    return DataBundle(profile=profile, data_path=str(data_path))


@pytest.fixture
def regression_bundle(tmp_path: Path) -> DataBundle:
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "x1": np.random.uniform(0, 10, n),
        "x2": np.random.uniform(0, 10, n),
        "target": np.random.uniform(0, 100, n),
    })
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path, index=False)
    profile = DataProfile(
        row_count=n, column_count=3,
        columns={"x1": "float64", "x2": "float64", "target": "float64"},
        nulls={"x1": 0, "x2": 0, "target": 0},
        data_types=["tabular"],
        sample_path=str(data_path),
    )
    return DataBundle(profile=profile, data_path=str(data_path))
