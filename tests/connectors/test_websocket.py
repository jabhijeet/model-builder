import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from model_builder.connectors.websocket_conn import WebSocketConnector
import pandas as pd


def test_connect_stores_config():
    c = WebSocketConnector()
    conn = c.connect({"url": "ws://localhost:8765", "max_messages": 10})
    assert conn.connector_name == "websocket"
    assert conn.handle["url"] == "ws://localhost:8765"
    assert conn.handle["max_messages"] == 10


def test_profile_returns_stream_type():
    c = WebSocketConnector()
    conn = c.connect({"url": "ws://localhost:8765"})
    profile = c.profile(conn)
    assert "stream" in profile.data_types
    assert profile.sample_path == "ws://localhost:8765"


async def test_stream_yields_parsed_messages():
    c = WebSocketConnector()
    messages = [
        json.dumps({"sensor": "A", "value": 1.0}),
        json.dumps({"sensor": "B", "value": 2.0}),
        json.dumps({"sensor": "A", "value": 3.0}),
    ]

    async def fake_aiter(self):
        for m in messages:
            yield m

    mock_ws = MagicMock()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)
    mock_ws.__aiter__ = fake_aiter

    with patch("websockets.connect", return_value=mock_ws):
        conn = c.connect({"url": "ws://localhost:8765", "max_messages": 3, "batch_size": 2})
        chunks = []
        async for chunk in c.stream(conn):
            chunks.append(chunk)

    assert len(chunks) >= 1
    total_rows = sum(len(ch.data) for ch in chunks)
    assert total_rows == 3
