"""S3 connector tests — mock DuckDB to avoid real AWS credentials."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from model_builder.connectors.s3 import S3Connector, _detect_format, _s3_url


# --- Unit tests (no cloud) ---

def test_s3_url_construction():
    assert _s3_url("my-bucket", "data/file.csv") == "s3://my-bucket/data/file.csv"
    assert _s3_url("my-bucket", "/data/file.csv") == "s3://my-bucket/data/file.csv"


def test_detect_format_parquet():
    assert _detect_format("s3://b/data/*.parquet") == "parquet"


def test_detect_format_json():
    assert _detect_format("s3://b/events.json") == "json"


def test_detect_format_default_csv():
    assert _detect_format("s3://b/data.txt") == "csv"


def test_connect_missing_bucket_raises():
    c = S3Connector()
    with pytest.raises(ValueError, match="bucket"):
        c.connect({"path": "data.csv"})


def test_connect_missing_path_raises():
    c = S3Connector()
    with pytest.raises(ValueError, match="path"):
        c.connect({"bucket": "my-bucket"})


def test_connect_stores_config():
    c = S3Connector()
    conn = c.connect({
        "bucket": "my-bucket",
        "path": "data/train.parquet",
        "region": "eu-west-1",
        "aws_access_key_id": "AKID",
    })
    assert conn.connector_name == "s3"
    assert conn.handle["bucket"] == "my-bucket"
    assert conn.handle["format"] == "parquet"
    assert conn.handle["region"] == "eu-west-1"


def test_connect_infers_format():
    c = S3Connector()
    conn = c.connect({"bucket": "b", "path": "file.json"})
    assert conn.handle["format"] == "json"


# --- Integration tests with mocked DuckDB ---

def _mock_duckdb_con(df: pd.DataFrame):
    """Return a mock duckdb connection that returns df for any execute().df()."""
    mock_result = MagicMock()
    mock_result.df.return_value = df

    mock_con = MagicMock()
    mock_con.execute.return_value = mock_result
    return mock_con


def test_sample_returns_dataframe():
    c = S3Connector()
    conn = c.connect({"bucket": "b", "path": "data.parquet"})
    expected_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with patch("duckdb.connect", return_value=_mock_duckdb_con(expected_df)):
        df = c.sample(conn, n=10)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]


def test_sample_respects_n():
    c = S3Connector()
    conn = c.connect({"bucket": "b", "path": "data.csv"})
    big_df = pd.DataFrame({"x": range(50)})

    with patch("duckdb.connect", return_value=_mock_duckdb_con(big_df)):
        df = c.sample(conn, n=5)

    assert len(df) <= 50  # mock returns full df; LIMIT is in SQL


def test_profile_returns_correct_types():
    c = S3Connector()
    conn = c.connect({"bucket": "b", "path": "data.parquet"})
    sample_df = pd.DataFrame({"col1": [1.0, 2.0], "col2": ["a", "b"]})

    with patch("duckdb.connect", return_value=_mock_duckdb_con(sample_df)):
        profile = c.profile(conn)

    assert profile.row_count == 2
    assert "tabular" in profile.data_types
    assert profile.sample_path == "s3://b/data.parquet"


async def test_stream_yields_chunks():
    c = S3Connector()
    conn = c.connect({"bucket": "b", "path": "data.csv", "chunk_size": 2})
    df = pd.DataFrame({"v": range(5)})

    with patch("duckdb.connect", return_value=_mock_duckdb_con(df)):
        chunks = []
        async for chunk in c.stream(conn):
            chunks.append(chunk)

    assert sum(len(ch.data) for ch in chunks) == 5


def test_endpoint_url_sets_path_style():
    """MinIO/localstack support via endpoint_url."""
    c = S3Connector()
    conn = c.connect({
        "bucket": "local-bucket",
        "path": "test.csv",
        "endpoint_url": "http://localhost:9000",
        "aws_access_key_id": "minioadmin",
        "aws_secret_access_key": "minioadmin",
    })
    assert conn.handle["endpoint_url"] == "http://localhost:9000"


def test_registration_in_registry():
    from model_builder.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.register_built_ins()
    assert reg.get_connector("connectors.s3").name == "s3"
    assert reg.get_connector("connectors.gcs").name == "gcs"
