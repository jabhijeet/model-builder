"""Kafka connector tests — all use mocks (no broker needed)."""
import json
import pytest
from unittest.mock import MagicMock, patch
from model_builder.connectors.kafka_conn import KafkaConnector, _parse_message


# --- Unit tests (no Kafka) ---

def test_parse_json_message():
    result = _parse_message(json.dumps({"sensor": "A", "value": 1.5}).encode())
    assert result == {"sensor": "A", "value": 1.5}


def test_parse_non_json_message():
    result = _parse_message(b"plain text message")
    assert "raw" in result
    assert result["raw"] == "plain text message"


def test_parse_list_message():
    result = _parse_message(json.dumps([1, 2, 3]).encode())
    assert "value" in result


def test_connect_stores_config():
    c = KafkaConnector()
    conn = c.connect({
        "bootstrap_servers": "broker:9092",
        "topic": "events",
        "max_messages": 500,
    })
    assert conn.connector_name == "kafka"
    assert conn.handle["topic"] == "events"
    assert conn.handle["bootstrap_servers"] == "broker:9092"
    assert conn.handle["max_messages"] == 500


def test_connect_defaults():
    c = KafkaConnector()
    conn = c.connect({"topic": "my-topic"})
    assert conn.handle["bootstrap_servers"] == "localhost:9092"
    assert conn.handle["group_id"] == "model-builder"
    assert conn.handle["batch_size"] == 500


# --- Integration tests with mock KafkaConsumer ---

def _mock_consumer(messages: list[dict]):
    """Create a mock KafkaConsumer that yields pre-set messages."""
    mock_msgs = []
    for d in messages:
        m = MagicMock()
        m.value = json.dumps(d).encode()
        mock_msgs.append(m)

    mock_consumer = MagicMock()
    mock_consumer.__iter__ = MagicMock(return_value=iter(mock_msgs))
    mock_consumer.close = MagicMock()
    return mock_consumer


def test_sample_returns_dataframe():
    c = KafkaConnector()
    conn = c.connect({"topic": "events"})
    messages = [{"sensor": "A", "value": i} for i in range(10)]

    with patch("kafka.KafkaConsumer",
               return_value=_mock_consumer(messages)):
        df = c.sample(conn, n=10)

    assert len(df) == 10
    assert "sensor" in df.columns
    assert "value" in df.columns


def test_sample_respects_n():
    c = KafkaConnector()
    conn = c.connect({"topic": "events"})
    messages = [{"x": i} for i in range(20)]

    with patch("kafka.KafkaConsumer",
               return_value=_mock_consumer(messages)):
        df = c.sample(conn, n=5)

    assert len(df) <= 5


def test_profile_stream_data_type():
    c = KafkaConnector()
    conn = c.connect({"topic": "events"})
    messages = [{"a": 1, "b": 2} for _ in range(5)]

    with patch("kafka.KafkaConsumer",
               return_value=_mock_consumer(messages)):
        profile = c.profile(conn)

    assert "stream" in profile.data_types
    assert profile.row_count == 5
    assert "kafka://" in profile.sample_path


async def test_stream_yields_batches():
    c = KafkaConnector()
    conn = c.connect({"topic": "events", "batch_size": 3, "max_messages": 6})
    messages = [{"val": i} for i in range(6)]

    with patch("kafka.KafkaConsumer",
               return_value=_mock_consumer(messages)):
        chunks = []
        async for chunk in c.stream(conn):
            chunks.append(chunk)

    assert len(chunks) >= 1
    assert all(len(ch.data) > 0 for ch in chunks)


def test_registration_in_registry():
    from model_builder.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.register_built_ins()
    assert reg.get_connector("connectors.image").name == "image"
    assert reg.get_connector("connectors.audio").name == "audio"
    assert reg.get_connector("connectors.kafka").name == "kafka"
