"""GCS connector tests — mock DuckDB to avoid real GCP credentials."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from model_builder.connectors.gcs import GCSConnector, _detect_format, _gcs_url


def test_gcs_url_construction():
    assert _gcs_url("my-bucket", "data/file.parquet") == "gs://my-bucket/data/file.parquet"
    assert _gcs_url("my-bucket", "/data/file.json") == "gs://my-bucket/data/file.json"


def test_detect_format_parquet():
    assert _detect_format("gs://b/data.parquet") == "parquet"


def test_connect_missing_bucket_raises():
    c = GCSConnector()
    with pytest.raises(ValueError, match="bucket"):
        c.connect({"path": "data.csv"})


def test_connect_missing_path_raises():
    c = GCSConnector()
    with pytest.raises(ValueError, match="path"):
        c.connect({"bucket": "my-bucket"})


def test_connect_stores_config():
    c = GCSConnector()
    conn = c.connect({
        "bucket": "ml-data",
        "path": "train/*.parquet",
        "hmac_access_key": "GOOG1234",
        "hmac_secret": "secret",
    })
    assert conn.connector_name == "gcs"
    assert conn.handle["bucket"] == "ml-data"
    assert conn.handle["format"] == "parquet"
    assert conn.handle["hmac_access_key"] == "GOOG1234"


def _mock_con(df: pd.DataFrame):
    result = MagicMock()
    result.df.return_value = df
    con = MagicMock()
    con.execute.return_value = result
    return con


def test_sample_returns_dataframe():
    c = GCSConnector()
    conn = c.connect({"bucket": "b", "path": "data.csv"})
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    with patch("duckdb.connect", return_value=_mock_con(df)):
        result = c.sample(conn, n=10)
    assert list(result.columns) == ["x", "y"]


def test_profile_data_types():
    c = GCSConnector()
    conn = c.connect({"bucket": "b", "path": "data.parquet"})
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with patch("duckdb.connect", return_value=_mock_con(df)):
        profile = c.profile(conn)
    assert "tabular" in profile.data_types
    assert profile.sample_path == "gs://b/data.parquet"


async def test_stream_yields_chunks():
    c = GCSConnector()
    conn = c.connect({"bucket": "b", "path": "data.json", "chunk_size": 3})
    df = pd.DataFrame({"v": range(7)})
    with patch("duckdb.connect", return_value=_mock_con(df)):
        chunks = []
        async for chunk in c.stream(conn):
            chunks.append(chunk)
    assert sum(len(ch.data) for ch in chunks) == 7


def test_service_account_key_path_stored():
    c = GCSConnector()
    conn = c.connect({
        "bucket": "b",
        "path": "data.csv",
        "service_account_key_path": "/path/to/key.json",
    })
    assert conn.handle["service_account_key_path"] == "/path/to/key.json"


def test_glob_pattern_gcs():
    c = GCSConnector()
    conn = c.connect({"bucket": "bucket", "path": "year=2024/**/*.parquet"})
    assert conn.handle["path"] == "year=2024/**/*.parquet"
    assert conn.handle["format"] == "parquet"
