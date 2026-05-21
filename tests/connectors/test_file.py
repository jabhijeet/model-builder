import pytest
import pandas as pd
import json
from pathlib import Path
from model_builder.connectors.file import FileConnector
from model_builder.plugins.protocol import Connection


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"], "label": [0, 1, 0]})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    records = [{"a": 1.0, "b": "x", "label": 0}, {"a": 2.0, "b": "y", "label": 1}]
    p = tmp_path / "data.json"
    p.write_text(json.dumps(records))
    return p


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "label": [0, 1]})
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    return p


def test_connect_single_csv(csv_file: Path):
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(csv_file)]})
    assert conn.connector_name == "file"


def test_sample_csv_returns_dataframe(csv_file: Path):
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(csv_file)]})
    df = conn_obj.sample(conn, n=100)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b", "label"]
    assert len(df) == 3


def test_sample_respects_n(csv_file: Path):
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(csv_file)]})
    df = conn_obj.sample(conn, n=2)
    assert len(df) <= 2


def test_profile_csv(csv_file: Path):
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(csv_file)]})
    profile = conn_obj.profile(conn)
    assert profile.row_count == 3
    assert profile.column_count == 3
    assert "a" in profile.columns
    assert "tabular" in profile.data_types


def test_json_file(json_file: Path):
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(json_file)]})
    df = conn_obj.sample(conn, n=100)
    assert len(df) == 2
    assert "a" in df.columns


def test_parquet_file(parquet_file: Path):
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(parquet_file)]})
    df = conn_obj.sample(conn, n=100)
    assert len(df) == 2


def test_glob_pattern(tmp_path: Path):
    for i in range(3):
        pd.DataFrame({"x": [i]}).to_csv(tmp_path / f"part_{i}.csv", index=False)
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(tmp_path / "*.csv")]})
    df = conn_obj.sample(conn, n=100)
    assert len(df) == 3


def test_null_counts_in_profile(tmp_path: Path):
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
    p = tmp_path / "nulls.csv"
    df.to_csv(p, index=False)
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(p)]})
    profile = conn_obj.profile(conn)
    assert profile.nulls["a"] == 1
    assert profile.nulls["b"] == 1


async def test_stream_yields_chunks(csv_file: Path):
    conn_obj = FileConnector()
    conn = conn_obj.connect({"paths": [str(csv_file)], "chunk_size": 2})
    chunks = []
    async for chunk in conn_obj.stream(conn):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert sum(len(c.data) for c in chunks) == 3
