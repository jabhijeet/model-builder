import pytest
from unittest.mock import patch, MagicMock
from model_builder.connectors.rest_poll import RestPollConnector
import pandas as pd


def test_connect_stores_config():
    c = RestPollConnector()
    conn = c.connect({
        "url": "http://example.com/data",
        "interval_seconds": 5,
        "max_batches": 3,
        "records_path": "data",
    })
    assert conn.connector_name == "rest_poll"
    assert conn.handle["url"] == "http://example.com/data"
    assert conn.handle["max_batches"] == 3


def test_sample_returns_dataframe():
    c = RestPollConnector()
    payload = {"data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response):
        conn = c.connect({"url": "http://x.com/api", "records_path": "data"})
        df = c.sample(conn, n=100)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "a" in df.columns


def test_profile_rest_poll():
    c = RestPollConnector()
    payload = {"data": [{"x": 1.0, "y": 2.0}]}
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response):
        conn = c.connect({"url": "http://x.com/api", "records_path": "data"})
        profile = c.profile(conn)

    assert profile.row_count == 1
    assert "stream" in profile.data_types


async def test_stream_yields_batches():
    c = RestPollConnector()
    payload = {"items": [{"val": i} for i in range(3)]}
    mock_response = MagicMock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response):
        conn = c.connect({
            "url": "http://x.com/api",
            "records_path": "items",
            "interval_seconds": 0,
            "max_batches": 2,
        })
        chunks = []
        async for chunk in c.stream(conn):
            chunks.append(chunk)

    assert len(chunks) == 2
    assert all(len(ch.data) == 3 for ch in chunks)
