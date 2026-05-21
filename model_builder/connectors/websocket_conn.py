import json
from typing import AsyncIterator
import pandas as pd
from ..plugins.protocol import Connection, DataProfile, Chunk


class WebSocketConnector:
    name = "websocket"

    def connect(self, config: dict) -> Connection:
        return Connection(
            connector_name="websocket",
            handle={
                "url": config["url"],
                "max_messages": config.get("max_messages", 1000),
                "batch_size": config.get("batch_size", 100),
                "headers": config.get("headers", {}),
            },
        )

    def sample(self, conn: Connection, n: int) -> pd.DataFrame:
        return pd.DataFrame()

    def profile(self, conn: Connection) -> DataProfile:
        return DataProfile(
            row_count=0,
            column_count=0,
            columns={},
            nulls={},
            data_types=["stream"],
            sample_path=conn.handle["url"],
        )

    async def stream(self, conn: Connection) -> AsyncIterator[Chunk]:
        import websockets
        handle = conn.handle
        batch_size = handle["batch_size"]
        max_messages = handle["max_messages"]
        received = 0
        batch: list = []
        seq = 0

        async with websockets.connect(handle["url"], additional_headers=handle["headers"]) as ws:
            async for raw in ws:
                try:
                    record = json.loads(raw)
                    if isinstance(record, dict):
                        batch.append(record)
                    elif isinstance(record, list):
                        batch.extend(record)
                    else:
                        batch.append({"value": record})
                except (json.JSONDecodeError, TypeError):
                    batch.append({"raw": str(raw)})

                received += 1
                if len(batch) >= batch_size:
                    yield Chunk(data=pd.DataFrame(batch), sequence=seq)
                    seq += 1
                    batch = []
                if received >= max_messages:
                    break

        if batch:
            yield Chunk(data=pd.DataFrame(batch), sequence=seq)
