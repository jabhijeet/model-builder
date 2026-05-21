"""Kafka connector — consumes messages and yields as batched DataFrames."""
import asyncio
import json
from typing import AsyncIterator
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk


def _parse_message(msg_value: bytes) -> dict:
    try:
        val = json.loads(msg_value)
        return val if isinstance(val, dict) else {"value": val}
    except (json.JSONDecodeError, TypeError):
        return {"raw": msg_value.decode("utf-8", errors="replace")}


class KafkaConnector:
    name = "kafka"

    def connect(self, config: dict) -> Connection:
        return Connection(
            connector_name="kafka",
            handle={
                "bootstrap_servers": config.get("bootstrap_servers", "localhost:9092"),
                "topic": config["topic"],
                "group_id": config.get("group_id", "model-builder"),
                "max_messages": config.get("max_messages", 10_000),
                "batch_size": config.get("batch_size", 500),
                "timeout_ms": config.get("timeout_ms", 5000),
                "auto_offset_reset": config.get("auto_offset_reset", "earliest"),
            },
        )

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        from kafka import KafkaConsumer
        h = conn.handle
        consumer = KafkaConsumer(
            h["topic"],
            bootstrap_servers=h["bootstrap_servers"],
            group_id=h["group_id"],
            auto_offset_reset=h["auto_offset_reset"],
            consumer_timeout_ms=h["timeout_ms"],
            value_deserializer=lambda m: m,
        )
        records = []
        try:
            for msg in consumer:
                records.append(_parse_message(msg.value))
                if len(records) >= n:
                    break
        finally:
            consumer.close()
        return pd.DataFrame(records)

    def profile(self, conn: Connection) -> DataProfile:
        df = self.sample(conn, n=100)
        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns={col: str(df[col].dtype) for col in df.columns},
            nulls={col: int(df[col].isna().sum()) for col in df.columns},
            data_types=["stream", "tabular"],
            sample_path=f"kafka://{conn.handle['bootstrap_servers']}/{conn.handle['topic']}",
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        h = conn.handle
        batch_size = h["batch_size"]
        max_messages = h["max_messages"]

        def _consume_batch(start_offset: int) -> list[dict]:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                h["topic"],
                bootstrap_servers=h["bootstrap_servers"],
                group_id=h["group_id"],
                auto_offset_reset=h["auto_offset_reset"],
                consumer_timeout_ms=h["timeout_ms"],
            )
            records = []
            try:
                for msg in consumer:
                    records.append(_parse_message(msg.value))
                    if len(records) >= batch_size:
                        break
            finally:
                consumer.close()
            return records

        consumed = 0
        seq = 0
        while consumed < max_messages:
            records = await asyncio.to_thread(_consume_batch, consumed)
            if not records:
                break
            yield Chunk(data=pd.DataFrame(records), sequence=seq)
            consumed += len(records)
            seq += 1
